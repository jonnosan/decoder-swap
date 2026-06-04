"""Step 2 of the stems pivot: round-trip each stem (and the full mix baseline) through DAC.

DAC 44 kHz / 9 cb is the highest-quality option we have on hand — per the reframed pivot
design, the per-stem codec leg should be MAX quality (compression budget lives in the
later semantic-token layer, not here). DAC is mono, our stems are stereo, so we encode
left and right channels independently and recombine.

Inputs:  data/song_test/<slug>/stems/{drums,bass,other,vocals}.wav  (stereo 44.1 kHz)
         data/song_test/<slug>/original.wav                           (the full mix)
Outputs: data/song_test/<slug>/stems_dac/{drums,bass,other,vocals}.wav
         data/song_test/<slug>/full_dac/full.wav

Run:
  .venv/bin/python scripts/31_dac_roundtrip_stems.py --slug beltram_machine
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import decode_from_codes, encode_to_codes, load_codec  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk-seconds", type=float, default=30.0,
                    help="encode/decode chunk size (DAC eats memory at long sequences)")
    return ap.parse_args()


def roundtrip_channel(codec, audio: np.ndarray, chunk_samples: int, device: str) -> np.ndarray:
    """Encode then decode a mono float32 [N] array via DAC; return mono float32 [N']
    (length may differ from input by a small chunk-boundary amount; caller pads/trims).
    """
    out_chunks: list[np.ndarray] = []
    n = len(audio)
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        chunk = audio[start:end]
        if len(chunk) < codec.convention.hop_length:
            break
        x = torch.from_numpy(chunk).to(device).view(1, 1, -1)
        with torch.no_grad():
            codes = encode_to_codes(codec, x)
            y = decode_from_codes(codec, codes)
        out_chunks.append(y[0, 0].detach().cpu().numpy())
        if device == "mps":
            torch.mps.empty_cache()
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)


def roundtrip_stereo_wav(codec, in_path: Path, out_path: Path, chunk_samples: int, device: str) -> dict:
    """Roundtrip one stereo WAV through DAC (L and R independently). Write to out_path."""
    y, sr = sf.read(in_path, dtype="float32", always_2d=True)
    if sr != codec.convention.sample_rate:
        raise SystemExit(f"sample rate mismatch: {in_path} is {sr}, codec wants {codec.convention.sample_rate}")
    n_in = y.shape[0]
    n_ch = y.shape[1]
    print(f"    {in_path.name}: {n_in/sr:.1f}s · {n_ch}ch · RMS_in={float(np.sqrt(np.mean(y**2))):.4f}")

    t0 = time.time()
    channels_out: list[np.ndarray] = []
    for c in range(n_ch):
        ch_out = roundtrip_channel(codec, y[:, c], chunk_samples, device)
        channels_out.append(ch_out)
    # Align: DAC's chunked output length can differ from input by small chunk-boundary
    # amounts. Pad/trim every channel to the original input length so summing across
    # files later is sample-aligned.
    aligned = np.zeros((n_in, n_ch), dtype=np.float32)
    for c, ch_out in enumerate(channels_out):
        m = min(len(ch_out), n_in)
        aligned[:m, c] = ch_out[:m]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, aligned, sr, subtype="FLOAT")
    rms_out = float(np.sqrt(np.mean(aligned ** 2)))
    elapsed = time.time() - t0
    print(f"      → {out_path.name}  RMS_out={rms_out:.4f}  ({elapsed:.1f}s, "
          f"{elapsed/(n_in/sr)*60:.1f} s/min of audio)")
    return {"in": str(in_path), "out": str(out_path), "rms_in": float(np.sqrt(np.mean(y**2))),
            "rms_out": rms_out, "seconds": elapsed}


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"# DAC roundtrip for stems + full-mix baseline · slug={args.slug}")
    print(f"device: {device}")

    codec = load_codec(name="dac", model_type="44khz", device=device)
    sr = codec.convention.sample_rate
    n_q = codec.convention.n_codebooks
    fps = codec.convention.frame_rate
    print(f"codec: DAC 44 kHz · {n_q} codebooks · {fps:.2f} fps · hop {codec.convention.hop_length}")

    song_dir = Path(args.root) / args.slug
    stems_dir = song_dir / "stems"
    if not stems_dir.exists():
        raise SystemExit(f"no stems dir at {stems_dir} — run scripts/30_separate_stems.py first")

    chunk_samples = int(round(args.chunk_seconds * sr))

    print()
    print("## stems")
    out_stems = song_dir / "stems_dac"
    for stem_path in sorted(stems_dir.glob("*.wav")):
        roundtrip_stereo_wav(codec, stem_path, out_stems / stem_path.name, chunk_samples, device)

    print()
    print("## full-mix baseline")
    orig = song_dir / "original.wav"
    if orig.exists():
        roundtrip_stereo_wav(codec, orig, song_dir / "full_dac" / "full.wav", chunk_samples, device)
    else:
        print(f"  (no {orig} found — skipping full-mix baseline)")

    print()
    print(f"  artifacts in: {song_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
