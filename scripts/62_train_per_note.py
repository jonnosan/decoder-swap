"""Train the per-note bass synth (per user's redirect 2026-06-04).

Each training sample is ONE note: 51-frame window = (1 silence-BOS frame) +
(50 note frames). The model is the same TranslatorRVQ + per-frame cond, but the
data layout eliminates within-song AR memorization by construction. Each note
starts from a silence token and the model must use cond to know what to play.

Run:
  .venv/bin/python scripts/62_train_per_note.py --dataset mayday_corpus --steps 10000
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

from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import (  # noqa: E402
    CondConfig,
    TranslatorRVQ,
    TranslatorRVQConfig,
    ar_loss,
)


class PerNoteSampler:
    """Yield random per-note (B, W, K) batches + parallel cond from cached arrays."""

    def __init__(
        self,
        tokens: np.ndarray,           # (N, W, K) int16
        pitch_active: np.ndarray,     # (N, W, P) uint8
        velocity_bin: np.ndarray,     # (N, W) int8
        bend_bin: np.ndarray,         # (N, W) int8
        onset_phase: np.ndarray,      # (N, W) int8
        timbre_ids: np.ndarray | None = None,   # (N,) int8 or None
        seed: int = 0,
    ):
        self.tokens = tokens
        self.pa = pitch_active
        self.vb = velocity_bin
        self.bb = bend_bin
        self.op = onset_phase
        self.timbre_ids = timbre_ids
        self.n_notes = tokens.shape[0]
        self.W = tokens.shape[1]
        self.K = tokens.shape[2]
        self.rng = np.random.default_rng(seed)

    def sample_pair(self, batch_size: int):
        idx = self.rng.integers(0, self.n_notes, size=int(batch_size))
        x = torch.from_numpy(self.tokens[idx].astype(np.int64))   # (B, W, K)
        cond = {
            "pitch_active": torch.from_numpy(self.pa[idx].astype(np.float32)),
            "velocity_bin": torch.from_numpy(self.vb[idx].astype(np.int64)),
            "bend_bin":     torch.from_numpy(self.bb[idx].astype(np.int64)),
            "onset_phase":  torch.from_numpy(self.op[idx].astype(np.int64)),
        }
        if self.timbre_ids is not None:
            cond["timbre_id"] = torch.from_numpy(self.timbre_ids[idx].astype(np.int64))
        return x, cond


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "per_note"))
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device("auto")
    print(f"# per-note translator · dataset={args.dataset} · device={device}")

    data_dir = Path(args.root) / args.dataset
    if not data_dir.exists():
        print(f"ERROR: per-note dataset not found at {data_dir}. "
              f"Run scripts/61_extract_per_note_data.py first.")
        return 1
    meta = json.loads((data_dir / "meta.json").read_text())
    tokens = np.load(data_dir / "tokens.npy")
    pa = np.load(data_dir / "pitch_active.npy")
    vb = np.load(data_dir / "velocity_bin.npy")
    bb = np.load(data_dir / "bend_bin.npy")
    op = np.load(data_dir / "onset_phase.npy")
    # Optional timbre_ids — if present, model gets a per-note timbre cond.
    timbre_path = data_dir / "timbre_ids.npy"
    timbre_ids = np.load(timbre_path) if timbre_path.exists() else None
    n_timbres = int(meta.get("n_timbres", 0)) if timbre_ids is not None else 0
    n_notes, W, K = tokens.shape
    P = pa.shape[-1]
    print(f"  notes: {n_notes:,}   window: {W} frames   codebooks: {K}   pitch_alphabet: {P}")
    if n_timbres > 0:
        print(f"  timbre cond enabled: {n_timbres} clusters")

    cond_cfg = meta["cond_cfg"]
    cc = CondConfig(
        n_pitches=int(cond_cfg["pitch_hi"]) - int(cond_cfg["pitch_lo"]) + 1,
        n_velocity_bins=int(cond_cfg["n_velocity_bins"]),
        n_bend_bins=int(cond_cfg["n_bend_bins"]),
        n_timbres=n_timbres,
    )
    tcfg = TranslatorRVQConfig(
        vocab_size=int(meta["vocab_size"]),
        n_codebooks=K,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=W + 16,
        cond=cc,
    )
    model = TranslatorRVQ(tcfg).to(device)
    print(f"  model: {model.num_parameters():,} params  ({model.num_parameters()/1e6:.2f} M)")

    sampler = PerNoteSampler(tokens, pa, vb, bb, op, timbre_ids=timbre_ids, seed=args.seed)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        REPO_ROOT / "data" / "checkpoints" / "per_note" / args.dataset
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "results" / "per_note" / args.dataset
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ckpts: {ckpt_dir}")
    print(f"  out:   {out_dir}")

    random_baseline = math.log(int(meta["vocab_size"]))
    print(f"  random baseline: {random_baseline:.4f}")
    print()

    losses: list[float] = []
    log_buf: list[float] = []
    best_window_loss = float("inf")
    t0 = time.time()
    model.train()
    try:
        for step in range(1, args.steps + 1):
            x, cond = sampler.sample_pair(args.batch_size)
            x = x.to(device, non_blocking=True)
            cond = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
            logits = model(x, cond=cond)
            loss = ar_loss(logits, x)
            lv = float(loss.detach().cpu())
            losses.append(lv); log_buf.append(lv)
            if torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optim.step()
            if step % args.log_every == 0 or step == args.steps:
                avg = sum(v for v in log_buf if v == v) / max(1, sum(1 for v in log_buf if v == v))
                log_buf.clear()
                elapsed = time.time() - t0
                sps = step / max(elapsed, 1e-9)
                eta = (args.steps - step) / max(sps, 1e-9)
                print(f"  step {step:>5d}/{args.steps}  loss={avg:.4f}  "
                      f"elapsed={elapsed:6.1f}s  rate={sps:.2f}/s  eta={eta:6.1f}s", flush=True)
                if avg < best_window_loss:
                    best_window_loss = avg
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "translator_config": {
                            "vocab_size": tcfg.vocab_size, "n_codebooks": tcfg.n_codebooks,
                            "d_model": tcfg.d_model, "n_layers": tcfg.n_layers,
                            "n_heads": tcfg.n_heads, "d_ff": tcfg.d_ff,
                            "dropout": tcfg.dropout, "max_seq_len": tcfg.max_seq_len,
                            "cond": {"n_pitches": cc.n_pitches,
                                     "n_velocity_bins": cc.n_velocity_bins,
                                     "n_bend_bins": cc.n_bend_bins,
                                     "n_onset_phases": cc.n_onset_phases,
                                     "n_timbres": cc.n_timbres},
                        },
                        "meta": meta,
                        "steps": step,
                        "best_window_loss": best_window_loss,
                    }, ckpt_dir / "per_note_best.pt")
            if step % args.ckpt_every == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "translator_config": {
                        "vocab_size": tcfg.vocab_size, "n_codebooks": tcfg.n_codebooks,
                        "d_model": tcfg.d_model, "n_layers": tcfg.n_layers,
                        "n_heads": tcfg.n_heads, "d_ff": tcfg.d_ff,
                        "dropout": tcfg.dropout, "max_seq_len": tcfg.max_seq_len,
                        "cond": {"n_pitches": cc.n_pitches,
                                 "n_velocity_bins": cc.n_velocity_bins,
                                 "n_bend_bins": cc.n_bend_bins,
                                 "n_onset_phases": cc.n_onset_phases},
                    },
                    "meta": meta,
                    "steps": step,
                }, ckpt_dir / "per_note.pt")
    except KeyboardInterrupt:
        print("\n[interrupted]")

    elapsed = time.time() - t0
    print(f"\n## done. {step} steps in {elapsed:.1f}s ({step/elapsed:.2f} steps/s)")
    print(f"  best window loss: {best_window_loss:.4f}")
    (out_dir / "train_losses.json").write_text(json.dumps(losses))
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, lw=0.5)
        ax.axhline(random_baseline, ls="--", c="grey", lw=1, label=f"random = {random_baseline:.2f}")
        ax.set_xlabel("step"); ax.set_ylabel("CE loss")
        ax.set_title(f"per-note translator — {args.dataset}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_loss.png", dpi=120)
        print(f"  loss plot: {out_dir/'training_loss.png'}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
