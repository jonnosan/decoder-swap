"""M6.0 step 1: Encode CORPUS_NEW with DAC and cache the token stream to disk.

For the translator feasibility smoke we need a large slab of DAC tokens to do AR next-token
prediction over. Encoding the full Vytis corpus takes ~10 min on M4 Pro MPS; doing it once and
caching means the smoke trainer (and the eventual full translator trainer) just memory-maps these
arrays.

Output: one int16 .npy per input track in data/tokens_dac/<stem>.npy with shape (n_codebooks, T_frames).
Codebook size for DAC 44 kHz is 1024 — fits in int16.

Chunked encode (30 s) to stay inside M4 Pro's MPS budget. Small boundary effects at chunk seams
are acceptable noise for AR training (random crops will rarely straddle them and the model is
learning a distribution, not a faithful round-trip).

Run:
  uv run python scripts/07_cache_translator_tokens.py
  uv run python scripts/07_cache_translator_tokens.py --max-seconds 600   # quick test
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
    ap.add_argument("--out-dir", default="data/tokens_dac", help="where to write per-track .npy")
    ap.add_argument("--chunk-seconds", type=float, default=30.0, help="audio chunk size for encode")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only encode the first N seconds of each track (smoke / debug)")
    return ap.parse_args()


def encode_track(codec, audio: np.ndarray, sr: int, chunk_samples: int, device: str) -> np.ndarray:
    """Encode a full track in chunks; return concatenated codes (n_codebooks, T_frames) int64 (CPU)."""
    all_codes: list[torch.Tensor] = []
    n = len(audio)
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        chunk = audio[start:end]
        if len(chunk) < codec.convention.hop_length:
            break  # too short for one frame
        x = torch.from_numpy(chunk).to(device).view(1, 1, -1)
        with torch.no_grad():
            codes = encode_to_codes(codec, x)  # (1, n_q, T_frames)
        all_codes.append(codes[0].cpu())
        if device == "mps":
            torch.mps.empty_cache()
    return torch.cat(all_codes, dim=-1).numpy().astype(np.int16)


def main() -> int:
    args = parse_args()
    settings = load_settings()
    device = resolve_device(settings.device)
    print("# M6.0 step 1: cache DAC tokens for CORPUS_NEW")
    print(f"device: {device}")

    # Force DAC regardless of what's currently in config.yaml — translator first-attempt is DAC-only
    # (issue #6). Mimi is a different vocab/frame-rate and would need a separate cache.
    codec = load_codec(name="dac", model_type="44khz", device=device)
    sr = codec.convention.sample_rate
    n_q = codec.convention.n_codebooks
    fps = codec.convention.frame_rate
    print(f"codec: DAC 44 kHz · {n_q} codebooks × {codec.convention.codebook_size} entries · "
          f"{fps:.2f} fps · hop {codec.convention.hop_length}")

    corpus_paths = settings.raw["corpora"]["new"]
    if not corpus_paths:
        print("config.corpora.new is empty — nothing to cache")
        return 1

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    chunk_samples = int(round(args.chunk_seconds * sr))
    print(f"chunk: {args.chunk_seconds:.1f} s = {chunk_samples} samples")

    manifest: list[dict] = []
    t0 = time.time()
    total_frames = 0
    for p in corpus_paths:
        p = Path(p)
        stem = p.stem
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists():
            existing = np.load(out_path)
            print(f"  [skip] {stem}: {out_path} already exists ({existing.shape})")
            manifest.append({"stem": stem, "shape": list(existing.shape), "skipped": True})
            total_frames += existing.shape[-1]
            continue

        print(f"  loading {p.name} ...")
        t_load = time.time()
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        if args.max_seconds is not None:
            y = y[: int(args.max_seconds * sr)]
        y = y.astype(np.float32, copy=False)
        dur_min = len(y) / sr / 60.0
        print(f"    loaded {dur_min:.1f} min  ({time.time()-t_load:.1f} s)")

        t_enc = time.time()
        codes = encode_track(codec, y, sr, chunk_samples, device)
        enc_secs = time.time() - t_enc
        T = codes.shape[-1]
        print(f"    encoded {T} frames ({T/fps/60:.1f} min of tokens)  "
              f"({enc_secs:.1f} s · {T/enc_secs:.0f} frames/s)")
        np.save(out_path, codes)
        manifest.append({"stem": stem, "shape": list(codes.shape), "encoded_seconds": enc_secs})
        total_frames += T

    elapsed = time.time() - t0
    print()
    print("## summary")
    print(f"  total frames cached : {total_frames:,}  "
          f"({total_frames/fps/60:.1f} min of tokens)")
    print(f"  total flat tokens   : {total_frames * n_q:,}")
    print(f"  elapsed             : {elapsed:.1f} s")
    print(f"  cache dir           : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
