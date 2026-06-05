"""Train the bass DAC-codec-LM conditioned on per-frame MIDI (issue #10 Phase 1B.1).

Reads:
  data/song_test/<slug>/stems_dac_tokens/<stem>.npy        — DAC tokens (n_q, T)
  data/song_test/<slug>/semantic/<stem>.json               — basic-pitch MIDI w/ bends

Builds frame-aligned conditioning at the DAC frame rate, then trains the
parallel-RVQ translator with optional conditioning summed into the input.

Two modes:
  --memorize           : trim to a single window and train repeatedly. Loss should
                         drive to ~0. Sanity-check that conditioning plumbing,
                         model, optimiser, save/load all work.
  (default)            : sample random windows from the full bass track and train.

Run:
  .venv/bin/python scripts/51_train_bass_translator.py --slug beltram_machine --memorize --steps 300
  .venv/bin/python scripts/51_train_bass_translator.py --slug beltram_machine --steps 5000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.midi_conditioning import FrameCondConfig, build_from_json, summarise  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.train_translator_rvq import (  # noqa: E402
    TranslatorRVQTrainConfig,
    train_translator_rvq,
)
from decoder_swap.translator_rvq import CondConfig  # noqa: E402

DAC_FPS = 86.1328125
DAC_VOCAB = 1024
DAC_N_CB = 9


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="primary song slug (also used as default ckpt/out dirname)")
    ap.add_argument("--extra-slugs", nargs="*", default=[],
                    help="additional song slugs to include in the training mix")
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--ckpt-dir", default=None,
                    help="default: data/checkpoints/bass_translator/<slug>/")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/bass_translator/<slug>/")
    # Memorize-test mode
    ap.add_argument("--memorize", action="store_true",
                    help="trim to a single window and train repeatedly (sanity test)")
    ap.add_argument("--memorize-start-frame", type=int, default=400,
                    help="frame offset for the memorize-test window (default: 400)")
    # Training
    ap.add_argument("--steps", type=int, default=2000)
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
    # Model
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # Conditioning
    ap.add_argument("--pitch-lo", type=int, default=24)
    ap.add_argument("--pitch-hi", type=int, default=50)
    ap.add_argument("--unconditional", action="store_true",
                    help="train without conditioning (control: should NOT memorize as cleanly)")
    ap.add_argument("--cond-shift-frames", type=int, default=0,
                    help="frames to shift cond forward (1 = cond[t] describes target frame t+1)")
    ap.add_argument("--cond-dropout-p", type=float, default=0.0,
                    help="probability of zeroing whole cond batch (classifier-free-guidance style)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"# bass translator · slug={args.slug} · stem={args.stem_name} · device={device}")

    def load_song(slug: str):
        sd = Path(args.root) / slug
        tp = sd / "stems_dac_tokens" / f"{args.stem_name}.npy"
        mp = sd / "semantic" / f"{args.stem_name}.json"
        if not tp.exists():
            print(f"ERROR: DAC tokens not found at {tp}. "
                  f"Run scripts/50_cache_bass_dac.py --slug {slug} first.")
            sys.exit(1)
        codes = np.load(tp)
        T = int(codes.shape[-1])
        print(f"  [{slug}] tokens: {codes.shape}  ({T:,} frames, {T/DAC_FPS/60:.2f} min)")
        if args.unconditional:
            return codes, None
        if not mp.exists():
            print(f"ERROR: MIDI JSON not found at {mp}.")
            sys.exit(1)
        cond = build_from_json(mp, fps=DAC_FPS, n_frames=T, cfg=cond_cfg)
        print(f"  [{slug}] conditioning: {summarise(cond)}")
        return codes, cond

    cond_cfg = FrameCondConfig(pitch_lo=args.pitch_lo, pitch_hi=args.pitch_hi)
    all_slugs = [args.slug] + list(args.extra_slugs)
    loaded = [load_song(s) for s in all_slugs]

    window_frames = int(round(args.window_seconds * DAC_FPS))

    if args.memorize:
        if len(all_slugs) > 1:
            print("ERROR: --memorize is for single-song single-window sanity tests. "
                  "Drop --extra-slugs or run without --memorize.")
            return 1
        codes, cond = loaded[0]
        T_full = int(codes.shape[-1])
        start = max(0, min(T_full - window_frames, args.memorize_start_frame))
        codes_use = codes[:, start : start + window_frames]
        if cond is not None:
            cond_use = type(cond)(
                pitch_active=cond.pitch_active[start : start + window_frames],
                velocity_bin=cond.velocity_bin[start : start + window_frames],
                bend_bin=cond.bend_bin[start : start + window_frames],
                onset_phase=cond.onset_phase[start : start + window_frames],
                cfg=cond.cfg,
            )
        else:
            cond_use = None
        tracks = [codes_use]
        conds = [cond_use] if cond_use is not None else None
        print(f"  MEMORIZE MODE: window=[{start}:{start+window_frames}] "
              f"({window_frames} frames, {args.window_seconds:.1f} s)")
        if conds is not None:
            print(f"    window cond: {summarise(conds[0])}")
    else:
        tracks = [c for c, _ in loaded]
        if args.unconditional:
            conds = None
        else:
            conds = [cd for _, cd in loaded]
        n_total_frames = sum(t.shape[-1] for t in tracks)
        print(f"  full-track training: {len(tracks)} song(s), "
              f"{n_total_frames:,} total frames ({n_total_frames/DAC_FPS/60:.1f} min), "
              f"{window_frames}-frame windows")

    ckpt_dir = (Path(args.ckpt_dir) if args.ckpt_dir
                else REPO_ROOT / "data" / "checkpoints" / "bass_translator" / args.slug)
    out_dir = (Path(args.out_dir) if args.out_dir
               else REPO_ROOT / "results" / "bass_translator" / args.slug)
    if args.memorize:
        ckpt_dir = ckpt_dir / "memorize"
        out_dir = out_dir / "memorize"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ckpts: {ckpt_dir}")
    print(f"  out:   {out_dir}")

    cond_ec = None
    if conds is not None:
        cond_ec = CondConfig(
            n_pitches=cond_cfg.n_pitches,
            n_velocity_bins=cond_cfg.n_velocity_bins,
            n_bend_bins=cond_cfg.n_bend_bins,
        )

    cfg = TranslatorRVQTrainConfig(
        n_codebooks=DAC_N_CB,
        vocab_size=DAC_VOCAB,
        frame_rate=DAC_FPS,
        batch_size=args.batch_size,
        window_seconds=args.window_seconds,
        cond_shift_frames=args.cond_shift_frames,
        cond_dropout_p=args.cond_dropout_p,
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
    print(f"train cfg: steps={cfg.steps} bs={cfg.batch_size} lr={cfg.lr} "
          f"window={cfg.window_seconds}s={window_frames}f")
    print()

    t0 = time.time()
    result = train_translator_rvq(tracks, cfg, device, conds=conds, cond_cfg=cond_ec)
    print()
    print("## result")
    print(f"  steps                  : {result.final_step}")
    print(f"  elapsed                : {result.elapsed_seconds:.1f} s  "
          f"({result.steps_per_second:.2f} steps/s)")
    print(f"  loss[first window avg] : {result.loss_first_window:.4f}")
    print(f"  loss[last window avg]  : {result.loss_last_window:.4f}")
    print(f"  random baseline        : {math.log(cfg.vocab_size):.4f}")
    print(f"  ckpt                   : {result.ckpt_path}")

    finite = [v for v in result.losses if v == v]
    delta = result.loss_first_window - result.loss_last_window
    pct = (delta / result.loss_first_window * 100) if finite and result.loss_first_window > 0 else float("nan")
    direction = "decreasing" if delta > 0 else "increasing"
    print(f"  improvement            : {delta:+.4f}  ({pct:+.1f}%)  ({direction})")

    summary = {
        "slug": args.slug,
        "stem": args.stem_name,
        "memorize": args.memorize,
        "unconditional": args.unconditional,
        "window_frames": window_frames,
        "n_frames_used": int(sum(t.shape[-1] for t in tracks)),
        "steps": result.final_step,
        "elapsed_seconds": result.elapsed_seconds,
        "loss_first_window": result.loss_first_window,
        "loss_last_window": result.loss_last_window,
        "improvement_pct": pct,
        "random_baseline": math.log(cfg.vocab_size),
        "ckpt_path": result.ckpt_path,
        "config": cfg.__dict__,
        "cond_cfg": cond_cfg.__dict__ if conds is not None else None,
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
        title = (f"bass translator — {args.slug} "
                 f"({'MEMORIZE' if args.memorize else 'full'}"
                 f"{', unconditional' if args.unconditional else ', conditioned'})")
        ax.set_title(title)
        ax.legend()
        plot_path = out_dir / "training_loss.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"  loss plot              : {plot_path}")
    except ImportError:
        pass

    print(f"\n  total wall clock: {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
