"""Cache DAC tokens for an isolated bass stem (issue #10, Phase 1B.1).

The conditional codec-LM needs DAC tokens for the bass-only audio (post-Demucs).
This script is the per-stem analog of scripts/07_cache_translator_tokens.py but
points at a single song's `stems/bass.wav` and writes a single .npy.

Output: data/song_test/<slug>/stems_dac_tokens/bass.npy  shape (n_codebooks, T_frames) int16

Run:
  uv run python scripts/50_cache_bass_dac.py --slug beltram_machine
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import encode_to_codes, load_codec  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="song dir under data/song_test/")
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--chunk-seconds", type=float, default=30.0)
    ap.add_argument("--force", action="store_true",
                    help="re-encode even if output .npy already exists")
    return ap.parse_args()


def encode_track(codec, audio: np.ndarray, sr: int, chunk_samples: int, device: str) -> np.ndarray:
    all_codes: list[torch.Tensor] = []
    n = len(audio)
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        chunk = audio[start:end]
        if len(chunk) < codec.convention.hop_length:
            break
        x = torch.from_numpy(chunk).to(device).view(1, 1, -1)
        with torch.no_grad():
            codes = encode_to_codes(codec, x)
        all_codes.append(codes[0].cpu())
        if device == "mps":
            torch.mps.empty_cache()
    return torch.cat(all_codes, dim=-1).numpy().astype(np.int16)


def main() -> int:
    args = parse_args()
    settings = load_settings()
    device = resolve_device(settings.device)

    song_dir = Path(args.root) / args.slug
    stem_path = song_dir / "stems" / f"{args.stem_name}.wav"
    out_dir = song_dir / "stems_dac_tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.stem_name}.npy"

    print(f"# DAC-cache stem · slug={args.slug} · stem={args.stem_name}")
    print(f"input:  {stem_path}")
    print(f"output: {out_path}")
    print(f"device: {device}")

    if out_path.exists() and not args.force:
        existing = np.load(out_path)
        print(f"  [skip] already cached: shape {existing.shape} "
              f"({existing.shape[-1]:,} frames). Use --force to re-encode.")
        return 0

    codec = load_codec(name="dac", model_type="44khz", device=device)
    sr = codec.convention.sample_rate
    n_q = codec.convention.n_codebooks
    fps = codec.convention.frame_rate
    print(f"codec: DAC 44 kHz · {n_q} codebooks × {codec.convention.codebook_size} · "
          f"{fps:.2f} fps · hop {codec.convention.hop_length}")

    t_load = time.time()
    y, _ = librosa.load(str(stem_path), sr=sr, mono=True)
    y = y.astype(np.float32, copy=False)
    print(f"  loaded {len(y)/sr:.1f} s  ({time.time()-t_load:.1f} s)")

    chunk_samples = int(round(args.chunk_seconds * sr))
    t_enc = time.time()
    codes = encode_track(codec, y, sr, chunk_samples, device)
    enc_secs = time.time() - t_enc
    T = codes.shape[-1]
    print(f"  encoded: {codes.shape}  ({T} frames, {T/fps/60:.2f} min)  "
          f"({enc_secs:.1f} s · {T/enc_secs:.0f} frames/s)")
    np.save(out_path, codes)
    print(f"  saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
