"""M6.0 step 2: feasibility smoke for the token translator (issue #6).

Trains a tiny AR transformer for next-token CE on cached DAC tokens from CORPUS_NEW. The point
is NOT to produce a usable translator — it's to verify that DAC token sequences are
autoregressively predictable by a small model before committing to the full architecture's
design choices.

Verdict thresholds:
  - Random baseline (no learning):           ln(1024) ≈ 6.931
  - PASS (something is being learned):       last-window loss ≤ 5.5
  - STRONG (clearly capturing structure):    last-window loss ≤ 4.5

Run after caching tokens with scripts/07_cache_translator_tokens.py:
  uv run python scripts/08_train_translator_smoke.py
  uv run python scripts/08_train_translator_smoke.py --steps 500 --batch-size 8
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
from decoder_swap.translator import (  # noqa: E402
    FlatARTransformer,
    TranslatorConfig,
    ar_loss,
    flatten_codes,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", default="data/tokens_dac")
    ap.add_argument("--out-dir", default="results/m6_smoke")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window-seconds", type=float, default=3.0)
    ap.add_argument("--frame-rate", type=float, default=86.1328125,
                    help="DAC 44 kHz: 44100/512")
    ap.add_argument("--n-codebooks", type=int, default=9)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    return ap.parse_args()


def load_token_tracks(tokens_dir: Path) -> list[np.ndarray]:
    """Return a list of (n_codebooks, T_frames) int arrays — one per cached track."""
    paths = sorted(tokens_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy token caches under {tokens_dir} — "
                                 "run scripts/07_cache_translator_tokens.py first")
    tracks = [np.load(p) for p in paths]
    for p, t in zip(paths, tracks):
        print(f"  loaded {p.name}: shape {t.shape}  ({t.shape[-1]:,} frames)")
    return tracks


class TokenBatchSampler:
    """Yield (B, n_codebooks, window_frames) crops from the in-RAM token tracks."""

    def __init__(self, tracks: list[np.ndarray], window_frames: int, seed: int = 0):
        self.tracks = tracks
        self.window_frames = int(window_frames)
        for i, t in enumerate(tracks):
            if t.shape[-1] < self.window_frames:
                raise ValueError(f"track {i} has only {t.shape[-1]} frames < window {self.window_frames}")
        self.rng = np.random.default_rng(seed)
        lens = np.array([t.shape[-1] for t in tracks], dtype=np.float64)
        self.track_probs = lens / lens.sum()

    def sample(self, batch_size: int) -> torch.Tensor:
        B = int(batch_size)
        n_q = self.tracks[0].shape[0]
        out = np.empty((B, n_q, self.window_frames), dtype=np.int64)
        for i in range(B):
            ti = int(self.rng.choice(len(self.tracks), p=self.track_probs))
            T = self.tracks[ti].shape[-1]
            start = int(self.rng.integers(0, T - self.window_frames + 1))
            out[i] = self.tracks[ti][:, start : start + self.window_frames]
        return torch.from_numpy(out)


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print("# M6.0 step 2: translator feasibility smoke")
    print(f"device: {device}")

    torch.manual_seed(args.seed)

    tokens_dir = REPO_ROOT / args.tokens_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"tokens_dir: {tokens_dir}")
    print(f"out_dir:    {out_dir}")

    print("loading cached tokens ...")
    tracks = load_token_tracks(tokens_dir)
    total_frames = sum(t.shape[-1] for t in tracks)
    print(f"total: {len(tracks)} track(s) · {total_frames:,} frames · "
          f"{total_frames / args.frame_rate / 60:.1f} min of tokens")

    window_frames = int(round(args.window_seconds * args.frame_rate))
    flat_len = window_frames * args.n_codebooks
    print(f"window: {args.window_seconds:.1f} s = {window_frames} frames = {flat_len} flat tokens")

    cfg = TranslatorConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=flat_len + 16,
    )
    model = FlatARTransformer(cfg).to(device)
    n_params = model.num_parameters()
    print(f"model: {n_params:,} params  ({n_params/1e6:.2f} M)")

    sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=args.seed)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    random_baseline = math.log(args.vocab_size)
    print(f"random baseline loss (uniform 1/{args.vocab_size}): {random_baseline:.4f}")
    print(f"PASS threshold (last-window avg ≤): 5.5000")
    print(f"STRONG threshold:                    4.5000")
    print()

    losses: list[float] = []
    log_buf: list[float] = []
    t0 = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        codes = sampler.sample(args.batch_size).to(device, non_blocking=True)
        flat = flatten_codes(codes)  # (B, L) int64
        logits = model(flat)
        loss = ar_loss(logits, flat)

        lv = float(loss.detach().cpu())
        losses.append(lv)
        log_buf.append(lv)

        if not torch.isfinite(loss):
            print(f"  step {step:>4d}: loss=NaN — abort", flush=True)
            return 2

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        if step % args.log_every == 0 or step == args.steps:
            avg = sum(log_buf) / len(log_buf)
            log_buf.clear()
            elapsed = time.time() - t0
            sps = step / max(elapsed, 1e-9)
            eta = (args.steps - step) / max(sps, 1e-9)
            print(
                f"  step {step:>4d}/{args.steps}  loss={avg:.4f}  "
                f"elapsed={elapsed:6.1f}s  rate={sps:.2f} steps/s  eta={eta:6.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    log = args.log_every
    head = losses[:log]
    tail = losses[-log:]
    loss_first = sum(head) / len(head)
    loss_last = sum(tail) / len(tail)
    delta = loss_first - loss_last
    pct = delta / loss_first * 100 if loss_first > 0 else float("nan")

    if loss_last <= 4.5:
        verdict = "STRONG: clear AR structure captured — green-light full translator build"
        rc = 0
    elif loss_last <= 5.5:
        verdict = "PASS: meaningful descent — AR-on-DAC-tokens is viable, proceed to full translator"
        rc = 0
    elif loss_last < random_baseline - 0.5:
        verdict = ("WEAK: loss moved off random baseline but didn't pass threshold — "
                   "investigate (longer steps? different hparams? factorised per-codebook?)")
        rc = 1
    else:
        verdict = ("FAIL: loss stuck near random baseline — the flat-interleaved AR formulation "
                   "may be wrong for DAC tokens; try factorised per-codebook before issue #6 build")
        rc = 1

    print()
    print("## result")
    print(f"  steps                  : {args.steps}")
    print(f"  elapsed                : {elapsed:.1f} s  ({args.steps/max(elapsed,1e-9):.2f} steps/s)")
    print(f"  loss[first window avg] : {loss_first:.4f}")
    print(f"  loss[last window avg]  : {loss_last:.4f}")
    print(f"  improvement            : {delta:+.4f}  ({pct:+.1f}%)")
    print(f"  random baseline        : {random_baseline:.4f}")
    print(f"  VERDICT                : {verdict}")

    result = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "window_seconds": args.window_seconds,
        "window_frames": window_frames,
        "flat_len": flat_len,
        "n_params": n_params,
        "elapsed_seconds": elapsed,
        "loss_first_window": loss_first,
        "loss_last_window": loss_last,
        "improvement_pct": pct,
        "random_baseline": random_baseline,
        "verdict": verdict,
        "config": cfg.__dict__,
    }
    (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "smoke_losses.json").write_text(json.dumps(losses))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, lw=0.8, label="per-step CE loss")
        ax.axhline(random_baseline, color="gray", lw=0.8, ls="--",
                   label=f"random baseline ln({args.vocab_size})={random_baseline:.2f}")
        ax.axhline(5.5, color="orange", lw=0.8, ls=":", label="PASS threshold (5.5)")
        ax.axhline(4.5, color="green", lw=0.8, ls=":", label="STRONG threshold (4.5)")
        ax.set_xlabel("step")
        ax.set_ylabel("CE loss (nats)")
        ax.set_title("M6.0 translator feasibility smoke — AR next-token on DAC tokens")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / "smoke_loss.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  loss plot saved        : {plot_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
