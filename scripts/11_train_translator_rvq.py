"""M6.A v2: train the parallel factorised RVQ translator.

Replaces scripts/09 (flat-interleaved layout — see project memory entry
'm6a-flat-layout-dead-end' for why). Same data, smaller seq-len (frames not
flat tokens), same param budget as the working memorize-test architecture.

Run:
  uv run python scripts/11_train_translator_rvq.py                  # 1700 steps
  uv run python scripts/11_train_translator_rvq.py --steps 5000     # longer
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.train_translator_rvq import (  # noqa: E402
    TranslatorRVQTrainConfig,
    train_translator_rvq,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="techno")
    ap.add_argument("--codec", default="dac", choices=["dac", "mimi"],
                    help="which cached token set to train on (data/tokens_<codec>/<corpus>/)")
    ap.add_argument("--tokens-dir", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=1700)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window-seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr-min-ratio", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"# M6.A v2 (parallel RVQ) — corpus '{args.corpus}'")
    print(f"device: {device}")

    corpus = load_corpus(args.corpus)
    tokens_dir = (Path(args.tokens_dir) if args.tokens_dir
                  else corpus.tokens_dir(codec=args.codec))
    ckpt_dir = (Path(args.ckpt_dir) if args.ckpt_dir
                else corpus.translator_ckpt_dir() / f"rvq_{args.codec}")
    out_dir = (Path(args.out_dir) if args.out_dir
               else corpus.results_dir(f"m6a_v2_{args.codec}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Codec-specific conventions. DAC 44k = 86.13 fps, 1024 vocab. Mimi = 12.5 fps, 2048 vocab.
    codec_conv = {
        "dac":  {"frame_rate": 86.1328125, "vocab_size": 1024},
        "mimi": {"frame_rate": 12.5,       "vocab_size": 2048},
    }[args.codec]

    print(f"corpus:    {corpus.name}")
    print(f"codec:     {args.codec} (fps={codec_conv['frame_rate']}, "
          f"vocab={codec_conv['vocab_size']})")
    print(f"tokens:    {tokens_dir}")
    print(f"ckpts:     {ckpt_dir}")
    print(f"out:       {out_dir}")

    paths = sorted(tokens_dir.glob("*.npy"))
    if not paths:
        print(f"ERROR: no .npy {args.codec} tokens found in {tokens_dir}.")
        return 1
    tracks = []
    for p in paths:
        codes = np.load(p)
        print(f"  loaded {p.name}: shape {codes.shape}  ({codes.shape[-1]:,} frames)")
        tracks.append(codes)
    n_frames = sum(t.shape[-1] for t in tracks)
    n_q = tracks[0].shape[0]
    print(f"corpus: {len(tracks)} track(s) · {n_frames:,} frames · {n_q} codebooks")

    cfg = TranslatorRVQTrainConfig(
        n_codebooks=n_q,
        vocab_size=codec_conv["vocab_size"],
        frame_rate=codec_conv["frame_rate"],
        batch_size=args.batch_size,
        window_seconds=args.window_seconds,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        lr_min_ratio=args.lr_min_ratio,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=str(ckpt_dir),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    print(f"train cfg: {cfg}")
    print()

    result = train_translator_rvq(tracks, cfg, device)

    print()
    print("## result")
    print(f"  steps                  : {result.final_step}")
    print(f"  elapsed                : {result.elapsed_seconds:.1f} s  "
          f"({result.steps_per_second:.2f} steps/s)")
    print(f"  loss[first window avg] : {result.loss_first_window:.4f}")
    print(f"  loss[last window avg]  : {result.loss_last_window:.4f}")
    print(f"  nan steps              : {result.nan_steps}")
    print(f"  ckpt                   : {result.ckpt_path}")
    finite = [v for v in result.losses if v == v]
    delta = result.loss_first_window - result.loss_last_window
    pct = (delta / result.loss_first_window * 100) if finite and result.loss_first_window > 0 else float("nan")
    direction = "decreasing" if delta > 0 else "increasing"
    print(f"  improvement            : {delta:+.4f}  ({pct:+.1f}%)  ({direction})")
    print(f"  random baseline        : {math.log(cfg.vocab_size):.4f}")

    summary = {
        "steps": result.final_step,
        "elapsed_seconds": result.elapsed_seconds,
        "loss_first_window": result.loss_first_window,
        "loss_last_window": result.loss_last_window,
        "improvement_pct": pct,
        "random_baseline": math.log(cfg.vocab_size),
        "nan_steps": result.nan_steps,
        "ckpt_path": result.ckpt_path,
        "config": cfg.__dict__,
    }
    (out_dir / "train_result.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "train_losses.json").write_text(json.dumps(result.losses))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result.losses, lw=0.5, label="per-step CE loss")
        ax.axhline(math.log(cfg.vocab_size), ls="--", c="grey", lw=1,
                   label=f"uniform baseline = {math.log(cfg.vocab_size):.2f}")
        ax.set_xlabel("step")
        ax.set_ylabel("CE loss (nats)")
        ax.set_title(f"M6.A v2 (parallel RVQ) loss — {args.corpus}")
        ax.legend()
        plot_path = out_dir / "training_loss.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"  loss plot              : {plot_path}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
