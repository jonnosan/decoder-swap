"""Train the fixer on cached (clean, noisy) pairs from scripts/19_."""
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

from decoder_swap.fixer import FixerConfig, FixerUNet, fixer_loss  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402

DATA_DIR = REPO_ROOT / "data/fixer/vytis"
CKPT_DIR = REPO_ROOT / "data/checkpoints/fixer"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class PairSampler:
    """Yields random fixed-length (noisy, clean) chunks from cached full-track arrays."""

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], chunk_samples: int, seed: int = 0):
        self.pairs = pairs
        self.chunk = chunk_samples
        self.rng = np.random.default_rng(seed)
        # Probability-by-length so longer tracks sample more often.
        lens = np.array([len(c) for _, c in pairs], dtype=np.float64)
        self.probs = lens / lens.sum()

    def sample(self, batch_size: int):
        noisy = np.empty((batch_size, self.chunk), dtype=np.float32)
        clean = np.empty((batch_size, self.chunk), dtype=np.float32)
        for i in range(batch_size):
            ti = int(self.rng.choice(len(self.pairs), p=self.probs))
            n_arr, c_arr = self.pairs[ti]
            T = len(c_arr)
            start = int(self.rng.integers(0, T - self.chunk + 1))
            noisy[i] = n_arr[start : start + self.chunk]
            clean[i] = c_arr[start : start + self.chunk]
        return (torch.from_numpy(noisy).unsqueeze(1),
                torch.from_numpy(clean).unsqueeze(1))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-seconds", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--spectral-weight", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"device: {device}")

    SR = 24000  # Mimi rate
    chunk_samples = int(args.chunk_seconds * SR)

    # Load all cached pairs into RAM (~190 min × 24kHz × 2 streams × 2 bytes ~ 1 GB).
    print(f"loading pairs from {DATA_DIR} ...")
    pairs = []
    stems = set()
    for p in sorted(DATA_DIR.glob("*_clean.npy")):
        stem = p.name[:-len("_clean.npy")]
        noisy_p = DATA_DIR / f"{stem}_noisy.npy"
        if not noisy_p.exists():
            continue
        clean = np.load(p).astype(np.float32)
        noisy = np.load(noisy_p).astype(np.float32)
        pairs.append((noisy, clean))
        stems.add(stem)
        print(f"  {stem}: {len(clean)/SR/60:.1f} min")
    if not pairs:
        print(f"ERROR: no pairs in {DATA_DIR}. Run scripts/19 first.")
        return 1

    torch.manual_seed(args.seed)
    sampler = PairSampler(pairs, chunk_samples, seed=args.seed)

    cfg = FixerConfig()
    model = FixerUNet(cfg).to(device)
    n = model.num_parameters()
    print(f"\nmodel: {n:,} params  ({n/1e6:.2f} M)")
    print(f"chunk: {args.chunk_seconds:.1f}s = {chunk_samples} samples @ {SR} Hz")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    losses_total: list[float] = []
    losses_wav: list[float] = []
    losses_spec: list[float] = []
    t0 = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        noisy, clean = sampler.sample(args.batch_size)
        noisy = noisy.to(device); clean = clean.to(device)
        pred = model(noisy)
        loss, parts = fixer_loss(pred, clean, spectral_weight=args.spectral_weight)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        losses_total.append(parts["total"])
        losses_wav.append(parts["wav"])
        losses_spec.append(parts["spec"])

        if step % args.log_every == 0 or step == 1:
            w_total = sum(losses_total[-args.log_every:]) / min(args.log_every, len(losses_total))
            w_wav = sum(losses_wav[-args.log_every:]) / min(args.log_every, len(losses_wav))
            w_spec = sum(losses_spec[-args.log_every:]) / min(args.log_every, len(losses_spec))
            elapsed = time.time() - t0
            rate = step / max(elapsed, 1e-9)
            eta = (args.steps - step) / max(rate, 1e-9)
            print(f"step {step:>5d}/{args.steps}  total={w_total:.4f}  wav={w_wav:.4f}  "
                  f"spec={w_spec:.4f}  rate={rate:.2f} steps/s  eta={eta:.0f}s",
                  flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = CKPT_DIR / "fixer.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
                "step": step,
                "losses_total": losses_total,
                "losses_wav": losses_wav,
                "losses_spec": losses_spec,
            }, ckpt_path)
            print(f"  [save] step {step} -> {ckpt_path}", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s. ckpt: {CKPT_DIR/'fixer.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
