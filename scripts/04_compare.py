"""M4: produce S1 and S2 from an identical token grid, save them, compute metrics, print verdict.

Run:
  uv run python scripts/04_compare.py                 # uses corpora.heldout from config.yaml
  uv run python scripts/04_compare.py --input clip.wav --seconds 30

Acceptance:
  - results/m4_compare/S1.wav and S2.wav exist and are audibly different
  - results/m4_compare/metrics.json contains every §3 metric
  - the invariant check passed (the script aborts otherwise)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import soundfile as sf  # noqa: E402

from decoder_swap.measure import compute_metrics, derive_verdict  # noqa: E402
from decoder_swap.run_experiment import build_d1_d2, run_experiment  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="held-out clip path; defaults to corpora.heldout")
    ap.add_argument("--seconds", type=float, default=None, help="cap input length")
    ap.add_argument("--d2-ckpt", default=None, help="override D2 checkpoint path")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "m4_compare"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    device = resolve_device(settings.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    held_out = args.input or settings.raw["corpora"].get("heldout")
    if not held_out:
        print("no held-out clip configured (set corpora.heldout in config.yaml or pass --input)")
        return 1
    d2_path = args.d2_ckpt or str(
        REPO_ROOT / settings.raw["train"]["ckpt_dir"] / "d2_decoder.pt"
    )

    print("# decoder-swap M4: D1 vs D2 on identical token grid")
    print(f"device       : {device}")
    print(f"held-out clip: {held_out}")
    print(f"D2 ckpt      : {d2_path}")
    print()

    codec_d1, codec_d2, d2_meta = build_d1_d2(
        codec_name=settings.codec_name,
        codec_model_type=settings.codec_model_type,
        codec_model_tag=settings.codec_model_tag,
        codec_model_path=settings.codec_model_path,
        device=device,
        d2_ckpt_path=d2_path,
    )
    print("invariant check passed: D1 and D2 share token convention + byte-identical codebooks")
    print(f"D2 trained for {d2_meta.get('steps','?')} steps "
          f"({d2_meta.get('elapsed_seconds', 0)/60.0:.1f} min), "
          f"loss {d2_meta.get('loss_first_window', float('nan')):.4f} -> "
          f"{d2_meta.get('loss_last_window', float('nan')):.4f}")
    print()

    result = run_experiment(codec_d1, codec_d2, held_out, max_seconds=args.seconds, d2_meta=d2_meta)
    print(f"input loaded : {len(result.y_input)} samples ({len(result.y_input)/result.sample_rate:.2f} s mono)")
    print(f"tokens shape : {tuple(result.codes.shape)}  (B, n_codebooks, T_frames)")
    print()

    sr = result.sample_rate
    input_wav = out_dir / "input.wav"
    s1_wav = out_dir / "S1.wav"
    s2_wav = out_dir / "S2.wav"
    sf.write(input_wav, result.y_input, sr, subtype="PCM_16")
    sf.write(s1_wav, result.s1, sr, subtype="PCM_16")
    sf.write(s2_wav, result.s2, sr, subtype="PCM_16")
    print(f"wrote: {input_wav}")
    print(f"wrote: {s1_wav}")
    print(f"wrote: {s2_wav}")
    print()

    print("computing metrics …", flush=True)
    metrics = compute_metrics(result.y_input, result.s1, result.s2, sr)
    metrics["meta"] = {
        "input_path": result.input_path,
        "tokens_shape": list(result.codes.shape),
        "d2_train": {k: v for k, v in d2_meta.items() if k != "decoder_state_dict"},
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote: {metrics_path}")
    print()

    verdicts = derive_verdict(metrics)
    print("## verdict")
    for v in verdicts:
        print(f"  [{v.status:>40s}] {v.clause}")
        print(f"      {v.headline_metric} = {v.headline_value:+.4f}")
        if v.note:
            print(f"      {v.note}")
        print()

    print("## listen")
    print(f"  open '{input_wav}'   # original country (reference)")
    print(f"  open '{s1_wav}'      # D1: original decoder, same tokens")
    print(f"  open '{s2_wav}'      # D2: techno-trained decoder, same tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
