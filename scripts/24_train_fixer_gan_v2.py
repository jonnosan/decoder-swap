"""GAN fixer v2: smaller discriminator + LM-output samples in the negative set.

Changes vs scripts/22_train_fixer_gan.py:

1. **SmallHiFiGANDiscriminator** (~4x fewer params) so we get back to ~3 steps/s.
2. **Two samplers**:
     - paired_sampler  → (noisy_real_roundtrip, clean_real)  used for recon + adv + fm
     - lm_sampler      → noisy_LM_output (no target)         used for adv + fm only
3. **D sees three negative kinds plus one positive** every step:
     real_clean              → "real"
     G(real_noisy)           → "fake"
     G(lm_noisy)             → "fake"        (NEW — addresses distribution shift)

Why this should work: the recon-only baseline is risk-averse on OOD input
(stays near identity). The previous GAN trained to be confident, which made
LM-output handling worse. Now D directly penalises bad outputs ON LM inputs,
so G learns "be cautious on LM outputs, be confident on real roundtrips."
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.fixer import FixerConfig, FixerUNet, multi_scale_spectral_loss  # noqa: E402
from decoder_swap.fixer_gan import (  # noqa: E402
    SmallHiFiGANDiscriminator,
    discriminator_loss,
    generator_adv_loss,
    feature_matching_loss,
)
from decoder_swap.settings import resolve_device  # noqa: E402

DATA_DIR = REPO_ROOT / "data/fixer/vytis"
LM_SAMPLES = DATA_DIR / "lm_samples.npy"
CKPT_DIR = REPO_ROOT / "data/checkpoints/fixer_gan_v2"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class PairSampler:
    def __init__(self, pairs, chunk_samples, seed=0):
        self.pairs = pairs
        self.chunk = chunk_samples
        self.rng = np.random.default_rng(seed)
        lens = np.array([len(c) for _, c in pairs], dtype=np.float64)
        self.probs = lens / lens.sum()

    def sample(self, batch_size):
        n = np.empty((batch_size, self.chunk), dtype=np.float32)
        c = np.empty((batch_size, self.chunk), dtype=np.float32)
        for i in range(batch_size):
            ti = int(self.rng.choice(len(self.pairs), p=self.probs))
            n_arr, c_arr = self.pairs[ti]
            s = int(self.rng.integers(0, len(c_arr) - self.chunk + 1))
            n[i] = n_arr[s : s + self.chunk]
            c[i] = c_arr[s : s + self.chunk]
        return (torch.from_numpy(n).unsqueeze(1),
                torch.from_numpy(c).unsqueeze(1))


class LMSampler:
    """Yields random chunks from the concatenated LM-output waveform."""

    def __init__(self, audio: np.ndarray, chunk_samples: int, seed=1):
        self.audio = audio
        self.chunk = chunk_samples
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):
        out = np.empty((batch_size, self.chunk), dtype=np.float32)
        for i in range(batch_size):
            s = int(self.rng.integers(0, len(self.audio) - self.chunk + 1))
            out[i] = self.audio[s : s + self.chunk]
        return torch.from_numpy(out).unsqueeze(1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=4)  # half batch since we do two passes
    ap.add_argument("--chunk-seconds", type=float, default=1.0)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--w-recon", type=float, default=10.0)
    ap.add_argument("--w-adv",   type=float, default=1.0)
    ap.add_argument("--w-fm",    type=float, default=2.0)
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

    # Load real-music paired data.
    print(f"loading paired data ...")
    pairs = []
    for clean_p in sorted(DATA_DIR.glob("*_clean.npy")):
        stem = clean_p.name[:-len("_clean.npy")]
        noisy_p = DATA_DIR / f"{stem}_noisy.npy"
        if not noisy_p.exists(): continue
        clean = np.load(clean_p).astype(np.float32)
        noisy = np.load(noisy_p).astype(np.float32)
        pairs.append((noisy, clean))
        print(f"  {stem}: {len(clean)/SR/60:.1f} min")
    if not pairs:
        print(f"ERROR: no pairs in {DATA_DIR}.")
        return 1

    # Load LM-output audio.
    if not LM_SAMPLES.exists():
        print(f"ERROR: {LM_SAMPLES} not found. Run scripts/23 first.")
        return 1
    lm_audio = np.load(LM_SAMPLES).astype(np.float32)
    print(f"LM samples: {len(lm_audio)/SR/60:.1f} min")

    torch.manual_seed(args.seed)
    paired_sampler = PairSampler(pairs, chunk_samples, seed=args.seed)
    lm_sampler = LMSampler(lm_audio, chunk_samples, seed=args.seed + 1)

    G = FixerUNet(FixerConfig()).to(device)
    D = SmallHiFiGANDiscriminator().to(device)
    print(f"\ngenerator    : {G.num_parameters():,} params  ({G.num_parameters()/1e6:.2f} M)")
    print(f"discriminator: {D.num_parameters():,} params  ({D.num_parameters()/1e6:.2f} M)")

    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.8, 0.99))

    losses = {k: [] for k in ("g_total", "g_recon", "g_adv_p", "g_adv_lm",
                              "g_fm_p", "g_fm_lm", "d")}
    t0 = time.time()
    G.train(); D.train()
    for step in range(1, args.steps + 1):
        noisy_p, clean_p = paired_sampler.sample(args.batch_size)
        noisy_lm = lm_sampler.sample(args.batch_size)
        noisy_p = noisy_p.to(device); clean_p = clean_p.to(device); noisy_lm = noisy_lm.to(device)

        use_gan = step > args.warmup_steps

        # ----- D step -----
        if use_gan:
            with torch.no_grad():
                fake_p = G(noisy_p)
                fake_lm = G(noisy_lm)
            real_outs   = D(clean_p)
            fake_p_outs = D(fake_p.detach())
            fake_lm_outs = D(fake_lm.detach())
            # Two "fake" terms — averages with the single "real" term.
            d_loss_p  = discriminator_loss(real_outs, fake_p_outs)
            d_loss_lm = discriminator_loss(real_outs, fake_lm_outs)
            d_loss = 0.5 * (d_loss_p + d_loss_lm)
            optim_D.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optim_D.step()
            d_val = float(d_loss.detach())
        else:
            d_val = 0.0

        # ----- G step -----
        fake_p = G(noisy_p)
        fake_lm = G(noisy_lm)
        g_recon = multi_scale_spectral_loss(fake_p, clean_p)
        if use_gan:
            real_outs    = D(clean_p)
            fake_p_outs  = D(fake_p)
            fake_lm_outs = D(fake_lm)
            g_adv_p  = generator_adv_loss(fake_p_outs)
            g_adv_lm = generator_adv_loss(fake_lm_outs)
            g_fm_p   = feature_matching_loss(real_outs, fake_p_outs)
            g_fm_lm  = feature_matching_loss(real_outs, fake_lm_outs)
        else:
            z = torch.zeros((), device=device)
            g_adv_p = g_adv_lm = g_fm_p = g_fm_lm = z
        g_total = (args.w_recon * g_recon
                   + args.w_adv * 0.5 * (g_adv_p + g_adv_lm)
                   + args.w_fm  * 0.5 * (g_fm_p + g_fm_lm))

        optim_G.zero_grad(set_to_none=True)
        g_total.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optim_G.step()

        losses["g_total"].append(float(g_total.detach()))
        losses["g_recon"].append(float(g_recon.detach()))
        losses["g_adv_p"].append(float(g_adv_p.detach()))
        losses["g_adv_lm"].append(float(g_adv_lm.detach()))
        losses["g_fm_p"].append(float(g_fm_p.detach()))
        losses["g_fm_lm"].append(float(g_fm_lm.detach()))
        losses["d"].append(d_val)

        if step % args.log_every == 0 or step == 1:
            def avg(k):
                return sum(losses[k][-args.log_every:]) / min(args.log_every, len(losses[k]))
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-9)
            phase = "GAN" if use_gan else "warmup"
            print(f"step {step:>5d}/{args.steps} [{phase}]  "
                  f"recon={avg('g_recon'):.3f}  "
                  f"adv_p={avg('g_adv_p'):.3f} adv_lm={avg('g_adv_lm'):.3f}  "
                  f"fm_p={avg('g_fm_p'):.4f} fm_lm={avg('g_fm_lm'):.4f}  "
                  f"D={avg('d'):.3f}  rate={rate:.2f}/s  "
                  f"eta={(args.steps-step)/max(rate,1e-9):.0f}s",
                  flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = CKPT_DIR / "fixer_gan_v2.pt"
            torch.save({
                "model_state_dict": G.state_dict(),
                "config": FixerConfig().__dict__,
                "args": vars(args),
                "step": step,
                "losses": losses,
            }, ckpt_path)
            print(f"  [save] step {step} -> {ckpt_path}", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s. ckpt: {CKPT_DIR/'fixer_gan_v2.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
