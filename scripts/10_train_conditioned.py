"""M7.B: train the jtxtok-conditioned translator on (jtxtok, DAC) pairs.

Fine-tunes a `ConditionedTranslator` initialised from the M6.A base LM checkpoint. The
training pairs are:
  conditioning = extractor's jtxtok stream (in data/jtxtok/<corpus>/<stem>.jtxtok)
  target       = corresponding DAC token cache (in data/tokens_dac/<corpus>/<stem>.npy)

Applies the two NON-NEGOTIABLE dropouts per spec §7.1:
  - CFG dropout (~15%): entire conditioning -> single PAD token
  - MT dropout (~30%): MT_* tokens stripped from conditioning

Synthetic MT supervision pairs (spec §7.2) are NOT included in this initial version —
they require jtx (PROMPT_1) which lives in jamtronix. Add later via a second dataset
loader if the corpus-only training fails to teach MT response.

Run:
  uv run python scripts/10_train_conditioned.py --base-lm data/checkpoints/translator/techno/translator_lm.pt
  uv run python scripts/10_train_conditioned.py --steps 3000 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.conditioned_translator import ConditionedTranslatorConfig  # noqa: E402
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.jtxtok_dataset import (  # noqa: E402
    JtxtokDacDataset,
    JtxtokDacDatasetConfig,
    discover_track_pairs,
)
from decoder_swap.jtxtok_vocab import DEFAULT_VOCAB, build_vocab  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.train_conditioned import (  # noqa: E402
    ConditionedTrainConfig,
    load_conditioned_with_base_lm,
    train_conditioned,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="techno")
    ap.add_argument("--tokens-dac-dir", default=None,
                    help="DAC tokens (default: data/tokens_dac/<corpus>/)")
    ap.add_argument("--jtxtok-dir", default=None,
                    help="jtxtok streams (default: data/jtxtok/<corpus>/)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="ckpts (default: data/checkpoints/translator/<corpus>/m7/)")
    ap.add_argument("--out-dir", default=None,
                    help="results (default: results/m7_<corpus>/)")
    ap.add_argument("--base-lm", default=None,
                    help="M6.A base LM ckpt (default: data/checkpoints/translator/<corpus>/translator_lm.pt)")

    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--window-seconds", type=float, default=3.0)
    ap.add_argument("--max-jtxtok-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--cfg-dropout", type=float, default=0.15)
    ap.add_argument("--mt-dropout", type=float, default=0.3)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"# M7.B: jtxtok-conditioned translator training — corpus '{args.corpus}'")
    print(f"device: {device}")

    corpus = load_corpus(args.corpus)
    tokens_dir = Path(args.tokens_dac_dir) if args.tokens_dac_dir else corpus.tokens_dir(codec="dac")
    jtxtok_dir = Path(args.jtxtok_dir) if args.jtxtok_dir else REPO_ROOT / f"data/jtxtok/{corpus.name}"
    ckpt_dir = (Path(args.ckpt_dir) if args.ckpt_dir
                else corpus.translator_ckpt_dir() / "m7")
    out_dir = Path(args.out_dir) if args.out_dir else corpus.results_dir("m7")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_lm_path = (Path(args.base_lm) if args.base_lm
                    else corpus.translator_ckpt_dir() / "translator_lm.pt")

    print(f"corpus:     {corpus.name}")
    print(f"DAC tokens: {tokens_dir}")
    print(f"jtxtok:     {jtxtok_dir}")
    print(f"base LM:    {base_lm_path}")
    print(f"ckpts:      {ckpt_dir}")
    print(f"out:        {out_dir}")

    if not base_lm_path.exists():
        print(f"ERROR: base LM checkpoint not found at {base_lm_path}. Run scripts/09 first.")
        return 1

    # Read decoder architecture from the base LM checkpoint so the conditioned model's
    # decoder shape matches the saved weights (load_from_unconditional copies layer-by-layer).
    import torch  # local import to keep top-level deps tidy
    base_state = torch.load(str(base_lm_path), map_location="cpu", weights_only=False)
    base_tc = base_state.get("translator_config", {})
    base_d_model = int(base_tc.get("d_model", 384))
    base_n_layers = int(base_tc.get("n_layers", 6))
    base_n_heads = int(base_tc.get("n_heads", 6))
    base_d_ff = int(base_tc.get("d_ff", 1536))
    print(f"base LM arch: d_model={base_d_model} n_layers={base_n_layers} "
          f"n_heads={base_n_heads} d_ff={base_d_ff}")

    pairs = discover_track_pairs(tokens_dir, jtxtok_dir)
    if not pairs:
        print(f"ERROR: no paired tracks found. Need both <stem>.npy in {tokens_dir} AND "
              f"<stem>.jtxtok+.json in {jtxtok_dir}.")
        return 1
    print(f"loaded {len(pairs)} paired track(s):")
    for p in pairs:
        print(f"  {p.stem}: DAC {p.dac.shape[-1]:,} frames, jtxtok {len(p.jtxtok_tokens):,} tokens, "
              f"{len(p.bar_starts_samples)} bars")

    vocab = DEFAULT_VOCAB
    print(f"vocabulary: {vocab.size} tokens (extractor contract defaults)")

    ds_cfg = JtxtokDacDatasetConfig(
        window_seconds=args.window_seconds,
        max_jtxtok_len=args.max_jtxtok_len,
        seed=args.seed,
    )
    dataset = JtxtokDacDataset(pairs, vocab, ds_cfg)
    print(f"window: {args.window_seconds:.1f} s = {dataset.window_frames} DAC frames, "
          f"jtxtok cap {args.max_jtxtok_len}")

    # ConditionedTranslator config — decoder arch is read from the M6.A base LM checkpoint
    # so load_from_unconditional can copy weights directly. Encoder arch stays at defaults.
    cond_cfg = ConditionedTranslatorConfig(
        dac_vocab_size=1024,
        d_model=base_d_model, n_layers=base_n_layers, n_heads=base_n_heads, d_ff=base_d_ff,
        max_dac_seq_len=dataset.window_frames * pairs[0].dac.shape[0] + 16,
        jtxtok_vocab_size=vocab.size, jtxtok_pad_id=vocab.pad_id,
        enc_d_model=256, enc_n_layers=3, enc_n_heads=4, enc_d_ff=1024,
        enc_max_seq_len=max(args.max_jtxtok_len + 16, 256),
    )
    model = load_conditioned_with_base_lm(cond_cfg, base_lm_path, device)
    n = model.num_parameters()
    print(f"model: {n:,} params ({n/1e6:.2f} M total)")

    train_cfg = ConditionedTrainConfig(
        steps=args.steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        cfg_dropout_prob=args.cfg_dropout,
        mt_dropout_prob=args.mt_dropout,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=str(ckpt_dir),
        seed=args.seed,
        base_lm_ckpt_path=str(base_lm_path),
    )
    print(f"train cfg: {train_cfg}")
    print()

    result = train_conditioned(dataset, model, vocab, train_cfg, device, args.batch_size)

    print()
    print("## result")
    print(f"  steps                  : {result.final_step}")
    print(f"  elapsed                : {result.elapsed_seconds:.1f} s")
    print(f"  loss[first window avg] : {result.loss_first_window:.4f}")
    print(f"  loss[last window avg]  : {result.loss_last_window:.4f}")
    print(f"  nan steps              : {result.nan_steps}")
    print(f"  CFG-dropout steps      : {result.cfg_drop_steps}  "
          f"({result.cfg_drop_steps/max(result.final_step,1)*100:.1f}%)")
    print(f"  MT-dropout steps       : {result.mt_drop_steps}  "
          f"({result.mt_drop_steps/max(result.final_step,1)*100:.1f}%)")
    print(f"  ckpt                   : {result.ckpt_path}")

    summary = {
        "steps": result.final_step,
        "elapsed_seconds": result.elapsed_seconds,
        "loss_first_window": result.loss_first_window,
        "loss_last_window": result.loss_last_window,
        "cfg_drop_steps": result.cfg_drop_steps,
        "mt_drop_steps": result.mt_drop_steps,
        "nan_steps": result.nan_steps,
        "ckpt_path": result.ckpt_path,
        "train_config": asdict(train_cfg),
        "model_config": asdict(cond_cfg),
        "tracks_used": [p.stem for p in pairs],
    }
    (out_dir / "train_result.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "train_losses.json").write_text(json.dumps(result.losses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
