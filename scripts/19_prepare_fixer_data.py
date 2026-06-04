"""Pre-compute (noisy=Mimi-roundtripped, clean=original) audio pairs for the fixer.

For each Vytis track:
  1. Load original audio at 24 kHz mono (Mimi's native rate).
  2. Encode→decode through Mimi at 8 codebooks (same config the LM uses).
  3. Save both signals as int16 .npy under data/fixer/vytis/{stem}_clean.npy / {stem}_noisy.npy.

Saving the full track per file keeps the train script flexible (it can sample
random N-sample chunks at training time without storing thousands of files).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import decode_from_codes, encode_to_codes, load_codec  # noqa: E402
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402

OUT_DIR = REPO_ROOT / "data/fixer/vytis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SECONDS = 30.0  # encode in chunks to fit in MPS memory


def encode_decode_track(codec, audio_np, chunk_samples, device):
    """Run audio through Mimi encode → decode, chunk by chunk."""
    out_chunks = []
    n = len(audio_np)
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        chunk = audio_np[start:end]
        if len(chunk) < codec.convention.hop_length:
            break
        x = torch.from_numpy(chunk).float().to(device).view(1, 1, -1)
        with torch.no_grad():
            codes = encode_to_codes(codec, x)              # (1, K, T_frames)
            recon = decode_from_codes(codec, codes.long()) # (1, 1, T_samples)
        out_chunks.append(recon[0, 0].cpu().numpy())
        if device == "mps":
            torch.mps.empty_cache()
    return np.concatenate(out_chunks)


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    codec = load_codec(name="mimi", device=device, num_quantizers=8)
    sr = codec.convention.sample_rate
    print(f"Mimi @ {sr} Hz, {codec.convention.n_codebooks} codebooks")

    corpus = load_corpus("techno")
    chunk_samples = int(CHUNK_SECONDS * sr)
    t0 = time.time()
    for src in corpus.audio_paths:
        stem = Path(src).stem
        clean_path = OUT_DIR / f"{stem}_clean.npy"
        noisy_path = OUT_DIR / f"{stem}_noisy.npy"
        if clean_path.exists() and noisy_path.exists():
            print(f"[skip] {stem} already cached")
            continue

        print(f"\nloading {Path(src).name} ...")
        y_clean, _ = librosa.load(src, sr=sr, mono=True)
        y_clean = y_clean.astype(np.float32, copy=False)
        print(f"  loaded {len(y_clean)/sr/60:.1f} min")

        t_enc = time.time()
        y_noisy = encode_decode_track(codec, y_clean, chunk_samples, device).astype(np.float32)
        print(f"  round-tripped in {time.time()-t_enc:.1f}s")

        # Align lengths (Mimi may produce slightly fewer samples than input).
        n = min(len(y_clean), len(y_noisy))
        y_clean = y_clean[:n]
        y_noisy = y_noisy[:n]

        # Save as float16 to save disk (these are bounded ~ [-1, 1] so float16 is fine).
        np.save(clean_path, y_clean.astype(np.float16))
        np.save(noisy_path, y_noisy.astype(np.float16))
        print(f"  -> {clean_path.name}  ({n/sr/60:.1f} min)")
        print(f"  -> {noisy_path.name}")

    print(f"\ndone in {time.time()-t0:.0f}s. dir: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
