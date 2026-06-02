"""M3: fine-tune ONLY the decoder on CORPUS_NEW.

By default uses train.steps from config.yaml (which starts at 100 — a fast validation run).
Override with --steps for a longer real fine-tune.

Run:
  uv run python scripts/03_train_d2.py                    # validation run (100 steps)
  uv run python scripts/03_train_d2.py --steps 3000       # real fine-tune
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import load_codec  # noqa: E402
from decoder_swap.dataset import CorpusDataset  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402
from decoder_swap.train_decoder import TrainConfig, train_d2  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="override config.train.steps")
    ap.add_argument("--batch-size", type=int, default=None, help="override config.train.batch_size")
    ap.add_argument("--lr", type=float, default=None, help="override config.train.lr")
    ap.add_argument("--ckpt-dir", default=None, help="override config.train.ckpt_dir")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    device = resolve_device(settings.device)
    print("# decoder-swap M3: decoder fine-tune (D2)")
    print(f"device: {device}")

    codec = load_codec(
        name=settings.codec_name,
        model_type=settings.codec_model_type,
        model_tag=settings.codec_model_tag,
        model_path=settings.codec_model_path,
        device=device,
    )
    sr = codec.convention.sample_rate

    corpus_paths = settings.raw["corpora"]["new"]
    if not corpus_paths:
        print("config.corpora.new is empty — nothing to train on")
        return 1
    train_cfg_raw = dict(settings.raw["train"])
    if args.steps is not None:
        train_cfg_raw["steps"] = args.steps
    if args.batch_size is not None:
        train_cfg_raw["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg_raw["lr"] = args.lr
    if args.ckpt_dir is not None:
        train_cfg_raw["ckpt_dir"] = args.ckpt_dir

    segment_samples = int(round(train_cfg_raw["segment_seconds"] * sr))
    print(f"loading corpus: {len(corpus_paths)} file(s) …")
    corpus = CorpusDataset(
        paths=corpus_paths,
        target_sr=sr,
        segment_samples=segment_samples,
        seed=settings.raw.get("seed", 0),
    )
    print(f"corpus: {corpus.summary()}")

    cfg = TrainConfig(
        sample_rate=sr,
        batch_size=int(train_cfg_raw["batch_size"]),
        segment_seconds=float(train_cfg_raw["segment_seconds"]),
        lr=float(train_cfg_raw["lr"]),
        steps=int(train_cfg_raw["steps"]),
        log_every=int(train_cfg_raw["log_every"]),
        codebook_check_every=int(train_cfg_raw["codebook_check_every"]),
        ckpt_every=int(train_cfg_raw.get("ckpt_every", 200)),
        ckpt_dir=str(REPO_ROOT / train_cfg_raw["ckpt_dir"]),
        seed=settings.raw.get("seed", 0),
    )
    print(f"train cfg: {cfg}")
    print()

    result = train_d2(codec, corpus, cfg)

    import math
    print()
    print("## result")
    print(f"  steps                  : {result.final_step}")
    print(f"  elapsed                : {result.elapsed_seconds:.1f} s  ({result.steps_per_second:.2f} steps/s)")
    print(f"  loss[first window avg] : {result.loss_first_window:.4f}")
    print(f"  loss[last window avg]  : {result.loss_last_window:.4f}")
    print(f"  nan steps              : {result.nan_steps}")
    print(f"  ckpt                   : {result.ckpt_path}")
    delta = result.loss_first_window - result.loss_last_window
    finite = math.isfinite(delta)
    pct = (delta / result.loss_first_window * 100) if finite and result.loss_first_window > 0 else float("nan")
    print(f"  improvement            : {delta:+.4f}  ({pct:+.1f}%)  "
          f"({'decreasing' if finite and delta > 0 else ('NOT decreasing' if finite else 'INDETERMINATE — NaN')})")
    if not finite or delta <= 0:
        print("  FAIL: loss did not cleanly decrease — investigate before committing to a long run.")
        return 2
    if result.nan_steps > 0:
        print(f"  NOTE: {result.nan_steps} NaN step(s) were caught and skipped during training.")
    print("  PASS: loss decreased and frozen-codebook assertion held throughout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
