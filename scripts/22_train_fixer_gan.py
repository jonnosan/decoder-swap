"""HiFi-GAN-style adversarial training for the fixer.

Same generator (FixerUNet) as scripts/20_train_fixer.py, but with:
  - MPD + MSD discriminator providing adversarial pressure
  - Three-term generator loss: spectral reconstruction + adversarial + feature matching
  - Standard LSGAN losses (MSE on logits) for stability

Saves a fresh checkpoint alongside the non-GAN baseline so we can A/B compare.
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

from decoder_swap.fixer import FixerConfig, FixerUNet, multi_scale_spectral_loss  # noqa: E402
from decoder_swap.fixer_gan import (  # noqa: E402
    HiFiGANDiscriminator,
    discriminator_loss,
    generator_adv_loss,
    feature_matching_loss,
)
from decoder_swap.settings import resolve_device  # noqa: E402

DATA_DIR = REPO_ROOT / "data/fixer/vytis"
CKPT_DIR = REPO_ROOT / "data/checkpoints/fixer_gan"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class PairSampler:
    """Same as in scripts/20_train_fixer.py — random chunks from cached tracks."""

    def __init__(self, pairs, chunk_samples, seed=0):
        self.pairs = pairs
        self.chunk = chunk_samples
        self.rng = np.random.default_rng(seed)
        lens = np.array([len(c) for _, c in pairs], dtype=np.float64)
        self.probs = lens / lens.sum()

    def sample(self, batch_size):
        noisy = np.empty((batch_size, self.chunk), dtype=np.float32)
        clean = np.empty((batch_size, self.chunk), dtype=np.float32)
        for i in range(batch_size):
            ti = int(self.rng.choice(len(self.pairs), p=self.probs))
            n_arr, c_arr = self.pairs[ti]
            start = int(self.rng.integers(0, len(c_arr) - self.chunk + 1))
            noisy[i] = n_arr[start : start + self.chunk]
            clean[i] = c_arr[start : start + self.chunk]
        return (torch.from_numpy(noisy).unsqueeze(1),
                torch.from_numpy(clean).unsqueeze(1))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-seconds", type=float, default=1.0)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    # HiFi-GAN paper uses λ_recon=45, λ_fm=2, λ_adv=1. Our recon term is a
    # multi-scale STFT L1 (no Mel scale yet); slightly lower λ_recon than the
    # paper's Mel L1 works well empirically.
    ap.add_argument("--w-recon", type=float, default=10.0)
    ap.add_argument("--w-adv",   type=float, default=1.0)
    ap.add_argument("--w-fm",    type=float, default=2.0)
    # Warm-up: a few steps with recon-only before turning on the adversarial
    # signal, so D sees a generator that's better than identity.
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"device: {device}")
    SR = 24000
    chunk_samples = int(args.chunk_seconds * SR)

    print(f"loading pairs from {DATA_DIR} ...")
    pairs = []
    for clean_p in sorted(DATA_DIR.glob("*_clean.npy")):
        stem = clean_p.name[:-len("_clean.npy")]
        noisy_p = DATA_DIR / f"{stem}_noisy.npy"
        if not noisy_p.exists():
            continue
        clean = np.load(clean_p).astype(np.float32)
        noisy = np.load(noisy_p).astype(np.float32)
        pairs.append((noisy, clean))
        print(f"  {stem}: {len(clean)/SR/60:.1f} min")
    if not pairs:
        print(f"ERROR: no pairs in {DATA_DIR}.  Run scripts/19 first.")
        return 1

    torch.manual_seed(args.seed)
    sampler = PairSampler(pairs, chunk_samples, seed=args.seed)

    G = FixerUNet(FixerConfig()).to(device)
    D = HiFiGANDiscriminator().to(device)
    print(f"\ngenerator    : {G.num_parameters():,} params  ({G.num_parameters()/1e6:.2f} M)")
    print(f"discriminator: {D.num_parameters():,} params  ({D.num_parameters()/1e6:.2f} M)")

    # HiFi-GAN paper uses betas (0.8, 0.99) and lr 2e-4 for both. We follow.
    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.8, 0.99))

    losses = {k: [] for k in ("g_total", "g_recon", "g_adv", "g_fm", "d")}
    t0 = time.time()
    G.train()
    D.train()
    for step in range(1, args.steps + 1):
        noisy, clean = sampler.sample(args.batch_size)
        noisy = noisy.to(device); clean = clean.to(device)
        use_gan = step > args.warmup_steps

        # ----- D step -----
        if use_gan:
            with torch.no_grad():
                fake = G(noisy)
            real_outs = D(clean)
            fake_outs = D(fake.detach())
            d_loss = discriminator_loss(real_outs, fake_outs)
            optim_D.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optim_D.step()
            d_val = float(d_loss.detach())
        else:
            d_val = 0.0

        # ----- G step -----
        fake = G(noisy)
        g_recon = multi_scale_spectral_loss(fake, clean)
        if use_gan:
            real_outs_for_g = D(clean)         # for FM loss (features detached inside)
            fake_outs_for_g = D(fake)
            g_adv = generator_adv_loss(fake_outs_for_g)
            g_fm = feature_matching_loss(real_outs_for_g, fake_outs_for_g)
        else:
            g_adv = torch.zeros((), device=device)
            g_fm = torch.zeros((), device=device)
        g_total = args.w_recon * g_recon + args.w_adv * g_adv + args.w_fm * g_fm

        optim_G.zero_grad(set_to_none=True)
        g_total.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optim_G.step()

        losses["g_total"].append(float(g_total.detach()))
        losses["g_recon"].append(float(g_recon.detach()))
        losses["g_adv"].append(float(g_adv.detach()))
        losses["g_fm"].append(float(g_fm.detach()))
        losses["d"].append(d_val)

        if step % args.log_every == 0 or step == 1:
            def avg(k): return sum(losses[k][-args.log_every:]) / min(args.log_every, len(losses[k]))
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-9)
            phase = "GAN" if use_gan else "warmup"
            print(f"step {step:>5d}/{args.steps} [{phase}]  "
                  f"G(total={avg('g_total'):.3f}  recon={avg('g_recon'):.3f}  "
                  f"adv={avg('g_adv'):.3f}  fm={avg('g_fm'):.3f})  "
                  f"D={avg('d'):.3f}  "
                  f"rate={rate:.2f}/s  eta={(args.steps-step)/max(rate,1e-9):.0f}s",
                  flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = CKPT_DIR / "fixer_gan.pt"
            torch.save({
                "model_state_dict": G.state_dict(),    # same key as non-GAN — apply script can reuse
                "config": FixerConfig().__dict__,
                "args": vars(args),
                "step": step,
                "losses": losses,
            }, ckpt_path)
            print(f"  [save] step {step} -> {ckpt_path}", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s. ckpt: {CKPT_DIR/'fixer_gan.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
