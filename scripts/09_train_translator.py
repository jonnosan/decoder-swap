"""M6.A: train the scaled-up flat-AR transformer LM on cached DAC tokens (issue #6 Phase A).

Phase-A goal: a model that knows what techno token sequences look like, used later in Phase B
as the LM behind prefix-conditioned sampling. NO conditioning architecture here yet — that's
Phase B.

Run:
  uv run python scripts/09_train_translator.py                    # 5000 steps, default model
  uv run python scripts/09_train_translator.py --steps 2000       # shorter
  uv run python scripts/09_train_translator.py --d-model 512 --n-layers 8   # bigger
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
from decoder_swap.train_translator import (  # noqa: E402
    TranslatorTrainConfig,
    train_translator,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="techno",
                    help="corpus name (loads corpora/<name>.yaml). Default: techno.")
    ap.add_argument("--tokens-dir", default=None,
                    help="override corpus-default tokens dir (data/tokens_dac/<corpus>/)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="override corpus-default ckpt dir (data/checkpoints/translator/<corpus>/)")
    ap.add_argument("--out-dir", default=None,
                    help="override corpus-default results dir (results/m6a_<corpus>/)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window-seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear lr warmup over this many steps (0 = disabled)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"# M6.A: scaled-up LM training — corpus '{args.corpus}'")
    print(f"device: {device}")

    corpus = load_corpus(args.corpus)
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else corpus.tokens_dir(codec="dac")
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else corpus.translator_ckpt_dir()
    out_dir = Path(args.out_dir) if args.out_dir else corpus.results_dir("m6a")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"corpus:    {corpus.name}")
    print(f"tokens:    {tokens_dir}")
    print(f"ckpts:     {ckpt_dir}")
    print(f"out:       {out_dir}")
    paths = sorted(tokens_dir.glob("*.npy"))
    if not paths:
        print(f"no cached tokens under {tokens_dir} — "
              f"run: uv run python scripts/07_cache_translator_tokens.py --corpus {args.corpus}")
        return 1

    tracks = [np.load(p) for p in paths]
    for p, t in zip(paths, tracks):
        print(f"  loaded {p.name}: shape {t.shape}  ({t.shape[-1]:,} frames)")
    n_q = tracks[0].shape[0]
    total = sum(t.shape[-1] for t in tracks)
    print(f"corpus: {len(tracks)} track(s) · {total:,} frames · {n_q} codebooks")

    cfg = TranslatorTrainConfig(
        n_codebooks=n_q,
        batch_size=args.batch_size,
        window_seconds=args.window_seconds,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
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

    result = train_translator(tracks, cfg, device)

    print()
    print("## result")
    print(f"  steps                  : {result.final_step}")
    print(f"  elapsed                : {result.elapsed_seconds:.1f} s  "
          f"({result.steps_per_second:.2f} steps/s)")
    print(f"  loss[first window avg] : {result.loss_first_window:.4f}")
    print(f"  loss[last window avg]  : {result.loss_last_window:.4f}")
    print(f"  nan steps              : {result.nan_steps}")
    print(f"  ckpt                   : {result.ckpt_path}")
    delta = result.loss_first_window - result.loss_last_window
    finite = math.isfinite(delta)
    pct = (delta / result.loss_first_window * 100) if finite and result.loss_first_window > 0 else float("nan")
    print(f"  improvement            : {delta:+.4f}  ({pct:+.1f}%)  "
          f"({'decreasing' if finite and delta > 0 else ('NOT decreasing' if finite else 'INDETERMINATE — NaN')})")

    rb = math.log(1024)
    print(f"  random baseline        : {rb:.4f}")
    summary = {
        "steps": result.final_step,
        "elapsed_seconds": result.elapsed_seconds,
        "loss_first_window": result.loss_first_window,
        "loss_last_window": result.loss_last_window,
        "improvement_pct": pct,
        "random_baseline": rb,
        "nan_steps": result.nan_steps,
        "ckpt_path": result.ckpt_path,
        "config": cfg.__dict__,
    }
    (out_dir / "train_result.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "train_losses.json").write_text(json.dumps(result.losses))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(result.losses, lw=0.5, label="per-step CE loss")
        ax.axhline(rb, color="gray", lw=0.8, ls="--", label=f"random baseline ln(1024)={rb:.2f}")
        ax.axhline(5.5, color="orange", lw=0.6, ls=":", label="M6.0 PASS line (5.5)")
        ax.axhline(4.5, color="green", lw=0.6, ls=":", label="M6.0 STRONG line (4.5)")
        ax.set_xlabel("step")
        ax.set_ylabel("CE loss (nats)")
        ax.set_title(f"M6.A translator LM — d_model={args.d_model}, layers={args.n_layers}, "
                     f"heads={args.n_heads}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / "training_loss.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  loss plot              : {plot_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
