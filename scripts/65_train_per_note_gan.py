"""Hybrid CE + GAN training for the per-note bass synth (per user request 2026-06-05).

The CE-only training plateaued at loss ~0.65 on Vytis Vol 1 v2 — the model
distinguishes timbres but outputs are still probability-averaged "muffled" tokens.
This script adds a token-domain discriminator: it learns to tell real DAC token
sequences from the generator's outputs, and provides a "be far from wrong"
gradient that complements CE's "be close to right" signal.

Architecture:
  - Generator G = TranslatorRVQ (warm-started from an existing per-note checkpoint
    if --warm-start is supplied; else fresh init).
  - Discriminator D = TokenDiscriminator (per_note_gan.py).
  - Loss balance:
      L_G = CE + λ_adv · g_lsgan_loss + λ_fm · feature_matching_loss
      L_D = d_lsgan_loss
  - Optimisers: separate AdamW for G and D, often with lower LR for G.

Per step:
  1. Sample a batch of notes.
  2. G forward (teacher forced) → logits → probs.
  3. D forward on real tokens, on probs → real_outs, fake_outs.
  4. Update D with d_lsgan_loss(real, fake.detach()).
  5. Update G with CE + λ_adv · g_lsgan_loss(fake) + λ_fm · fm_loss(real, fake).

Run:
  .venv/bin/python scripts/65_train_per_note_gan.py \\
    --dataset vytis_vol1_v2 \\
    --warm-start data/checkpoints/per_note/vytis_vol1_v2/per_note_best.pt \\
    --steps 5000 --batch-size 32 --lambda-adv 0.1 --lambda-fm 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.per_note_gan import (  # noqa: E402
    TokenDiscriminator,
    d_lsgan_loss,
    feature_matching_loss,
    g_lsgan_loss,
)
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import (  # noqa: E402
    CondConfig,
    TranslatorRVQ,
    TranslatorRVQConfig,
    ar_loss,
)

# Reuse the per-note sampler from script 62.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from importlib import util as _util
_spec = _util.spec_from_file_location("_train_per_note", str(REPO_ROOT / "scripts" / "62_train_per_note.py"))
_train_pn = _util.module_from_spec(_spec)
_spec.loader.exec_module(_train_pn)
PerNoteSampler = _train_pn.PerNoteSampler


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "per_note"))
    ap.add_argument("--warm-start", default=None,
                    help="path to an existing per-note .pt to initialise G from")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-g", type=float, default=1e-4,
                    help="generator LR (lower than CE-only since fine-tuning)")
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.1,
                    help="weight for G's adversarial loss term")
    ap.add_argument("--lambda-fm", type=float, default=1.0,
                    help="weight for G's feature-matching loss term")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    # Model arch (used if not warm-starting)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    # Discriminator arch
    ap.add_argument("--d-emb", type=int, default=32, help="discriminator's per-codebook emb dim")
    return ap.parse_args()


def _reconstruct_translator_config(d: dict) -> TranslatorRVQConfig:
    d = dict(d)
    if isinstance(d.get("cond"), dict):
        d["cond"] = CondConfig(**d["cond"])
    return TranslatorRVQConfig(**d)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device("auto")
    print(f"# per-note GAN · dataset={args.dataset} · device={device}")

    data_dir = Path(args.root) / args.dataset
    meta = json.loads((data_dir / "meta.json").read_text())
    tokens = np.load(data_dir / "tokens.npy")
    pa = np.load(data_dir / "pitch_active.npy")
    vb = np.load(data_dir / "velocity_bin.npy")
    bb = np.load(data_dir / "bend_bin.npy")
    op = np.load(data_dir / "onset_phase.npy")
    timbre_path = data_dir / "timbre_ids.npy"
    timbre_ids = np.load(timbre_path) if timbre_path.exists() else None
    n_timbres = int(meta.get("n_timbres", 0)) if timbre_ids is not None else 0
    N, W, K = tokens.shape
    V = int(meta["vocab_size"])
    print(f"  data:  N={N}  W={W}  K={K}  V={V}  timbres={n_timbres}")

    # ---- Build / load G ----
    if args.warm_start:
        print(f"  warm-start G from {args.warm_start}")
        ckpt = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        gcfg = _reconstruct_translator_config(ckpt["translator_config"])
    else:
        cond_cfg = meta["cond_cfg"]
        cc = CondConfig(
            n_pitches=int(cond_cfg["pitch_hi"]) - int(cond_cfg["pitch_lo"]) + 1,
            n_velocity_bins=int(cond_cfg["n_velocity_bins"]),
            n_bend_bins=int(cond_cfg["n_bend_bins"]),
            n_timbres=n_timbres,
        )
        gcfg = TranslatorRVQConfig(
            vocab_size=V, n_codebooks=K,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, d_ff=args.d_ff,
            dropout=0.0, max_seq_len=W + 16, cond=cc,
        )
    G = TranslatorRVQ(gcfg).to(device)
    if args.warm_start:
        G.load_state_dict(ckpt["model_state_dict"])
        print(f"    G loaded ({G.num_parameters():,} params)")
    print(f"  G: {G.num_parameters():,} params")

    # ---- Build D ----
    D = TokenDiscriminator(n_codebooks=K, vocab_size=V, d_emb=args.d_emb).to(device)
    print(f"  D: {D.num_parameters():,} params")

    sampler = PerNoteSampler(tokens, pa, vb, bb, op, timbre_ids=timbre_ids, seed=args.seed)
    opt_G = torch.optim.AdamW(G.parameters(), lr=args.lr_g)
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        REPO_ROOT / "data" / "checkpoints" / "per_note" / (args.dataset + "_gan")
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "results" / "per_note" / (args.dataset + "_gan")
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ckpts: {ckpt_dir}")
    print(f"  out:   {out_dir}")
    print(f"  losses: CE + {args.lambda_adv:.3g}·g_adv + {args.lambda_fm:.3g}·fm_loss")
    print()

    history = {"ce": [], "g_adv": [], "fm": [], "d": []}
    best_g_total = float("inf")
    t0 = time.time()
    G.train(); D.train()
    try:
        for step in range(1, args.steps + 1):
            x, cond = sampler.sample_pair(args.batch_size)
            x = x.to(device, non_blocking=True)
            cond = {k: v.to(device, non_blocking=True) for k, v in cond.items()}

            # ---- G forward: produce logits + probs ----
            logits = G(x, cond=cond)                          # (B, W, K, V)
            # Predictions are for frames 1..W given inputs 0..W-1. We discriminate
            # over the predicted positions: logits[:, :-1] correspond to targets
            # x[:, 1:].
            probs = torch.softmax(logits[:, :-1], dim=-1)     # (B, W-1, K, V)
            real_tokens = x[:, 1:]                              # (B, W-1, K)

            # ---- Train D ----
            d_real_logit, d_real_feats = D(tokens=real_tokens)
            d_fake_logit, d_fake_feats = D(probs=probs.detach())
            loss_D = d_lsgan_loss(d_real_logit, d_fake_logit)
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=args.grad_clip)
            opt_D.step()

            # ---- Train G ----
            # CE on the predicted positions
            B, Tm1, Kk, Vv = probs.shape
            ce = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, Vv),
                x[:, 1:].reshape(-1).long(),
            )
            # Re-run D with G's grad — we don't detach this time so adv flows to G.
            g_fake_logit, g_fake_feats = D(probs=probs)
            g_real_logit_for_fm, g_real_feats_for_fm = D(tokens=real_tokens)
            g_adv = g_lsgan_loss(g_fake_logit)
            fm = feature_matching_loss(g_real_feats_for_fm, g_fake_feats)
            loss_G = ce + args.lambda_adv * g_adv + args.lambda_fm * fm
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.grad_clip)
            opt_G.step()

            # ---- Log ----
            history["ce"].append(float(ce.detach().cpu()))
            history["g_adv"].append(float(g_adv.detach().cpu()))
            history["fm"].append(float(fm.detach().cpu()))
            history["d"].append(float(loss_D.detach().cpu()))

            if step % args.log_every == 0 or step == args.steps:
                ce_avg = sum(history["ce"][-args.log_every:]) / args.log_every
                ga_avg = sum(history["g_adv"][-args.log_every:]) / args.log_every
                fm_avg = sum(history["fm"][-args.log_every:]) / args.log_every
                d_avg = sum(history["d"][-args.log_every:]) / args.log_every
                elapsed = time.time() - t0
                sps = step / max(elapsed, 1e-9)
                eta = (args.steps - step) / max(sps, 1e-9)
                print(f"  step {step:>5d}/{args.steps}  "
                      f"CE={ce_avg:.4f}  G_adv={ga_avg:.4f}  fm={fm_avg:.4f}  D={d_avg:.4f}  "
                      f"elapsed={elapsed:6.1f}s  rate={sps:.2f}/s  eta={eta:6.1f}s",
                      flush=True)
                # Save best-CE checkpoint
                if ce_avg < best_g_total:
                    best_g_total = ce_avg
                    torch.save({
                        "model_state_dict": G.state_dict(),
                        "translator_config": {
                            "vocab_size": gcfg.vocab_size, "n_codebooks": gcfg.n_codebooks,
                            "d_model": gcfg.d_model, "n_layers": gcfg.n_layers,
                            "n_heads": gcfg.n_heads, "d_ff": gcfg.d_ff,
                            "dropout": gcfg.dropout, "max_seq_len": gcfg.max_seq_len,
                            "cond": {
                                "n_pitches": gcfg.cond.n_pitches,
                                "n_velocity_bins": gcfg.cond.n_velocity_bins,
                                "n_bend_bins": gcfg.cond.n_bend_bins,
                                "n_onset_phases": gcfg.cond.n_onset_phases,
                                "n_timbres": gcfg.cond.n_timbres,
                            },
                        },
                        "meta": meta,
                        "steps": step,
                        "best_ce": best_g_total,
                    }, ckpt_dir / "per_note_best.pt")
            if step % args.ckpt_every == 0:
                torch.save({"model_state_dict": G.state_dict(),
                            "discriminator_state_dict": D.state_dict(),
                            "step": step},
                           ckpt_dir / f"checkpoint_{step}.pt")
    except KeyboardInterrupt:
        print("\n[interrupted]")

    elapsed = time.time() - t0
    print(f"\n## done. {step} steps in {elapsed:.1f}s ({step/elapsed:.2f} steps/s)")
    print(f"  best CE: {best_g_total:.4f}")
    (out_dir / "history.json").write_text(json.dumps(history))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history["ce"], lw=0.5, label="CE", color="tab:blue")
        ax.plot(history["g_adv"], lw=0.5, label="G adv", color="tab:orange")
        ax.plot(history["fm"], lw=0.5, label="feature-matching", color="tab:green")
        ax.plot(history["d"], lw=0.5, label="D total", color="tab:red")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"per-note GAN — {args.dataset}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_loss.png", dpi=120)
        print(f"  loss plot: {out_dir/'training_loss.png'}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
