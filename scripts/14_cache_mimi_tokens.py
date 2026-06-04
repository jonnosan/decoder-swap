"""Encode a corpus's audio with Mimi (kyutai/mimi) and cache token streams.

Mirrors scripts/07_cache_translator_tokens.py but for Mimi. Mimi is 12.5 Hz
(~7x lower frame rate than DAC at 86 Hz), so sequences are much shorter and
the next-token modeling task should be far easier at the same model scale.

Output: one int16 .npy per input track in data/tokens_mimi/<corpus>/<stem>.npy
with shape (n_codebooks, T_frames). Mimi codebook_size is 2048 — still fits in int16.

Run:
  uv run python scripts/14_cache_mimi_tokens.py
  uv run python scripts/14_cache_mimi_tokens.py --max-seconds 600   # quick test
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
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="techno")
    ap.add_argument("--out-dir", default=None,
                    help="override default (data/tokens_mimi/<corpus>/)")
    ap.add_argument("--num-quantizers", type=int, default=8,
                    help="Mimi codebook count to use (default: 8 = 1 semantic + 7 acoustic)")
    ap.add_argument("--chunk-seconds", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    return ap.parse_args()


def encode_track(codec, audio, chunk_samples, device):
    all_codes = []
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
    print(f"# cache Mimi tokens for corpus '{args.corpus}'")
    print(f"device: {device}")

    codec = load_codec(name="mimi", model_tag=None, device=device,
                       num_quantizers=args.num_quantizers)
    sr = codec.convention.sample_rate
    n_q = codec.convention.n_codebooks
    fps = codec.convention.frame_rate
    print(f"codec: Mimi · {n_q} codebooks × {codec.convention.codebook_size} entries · "
          f"{fps:.2f} fps · hop {codec.convention.hop_length}")

    corpus = load_corpus(args.corpus)
    if not corpus.audio_paths:
        print(f"corpus '{corpus.name}' has no audio_paths")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else corpus.tokens_dir(codec="mimi")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"corpus:  {corpus.name} ({len(corpus.audio_paths)} track(s))")
    print(f"out_dir: {out_dir}")

    chunk_samples = int(round(args.chunk_seconds * sr))
    t0 = time.time()
    total_frames = 0
    for p in corpus.audio_paths:
        p = Path(p)
        stem = p.stem
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists():
            existing = np.load(out_path)
            print(f"  [skip] {stem}: {out_path.name} already exists ({existing.shape})")
            total_frames += existing.shape[-1]
            continue
        print(f"  loading {p.name} ...")
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        if args.max_seconds is not None:
            y = y[: int(args.max_seconds * sr)]
        y = y.astype(np.float32, copy=False)
        print(f"    loaded {len(y)/sr/60:.1f} min")
        t_enc = time.time()
        codes = encode_track(codec, y, chunk_samples, device)
        enc_secs = time.time() - t_enc
        T = codes.shape[-1]
        print(f"    encoded {T:,} frames ({T/fps/60:.1f} min of tokens) in {enc_secs:.1f}s")
        np.save(out_path, codes)
        total_frames += T

    elapsed = time.time() - t0
    print()
    print(f"  total frames cached : {total_frames:,}  ({total_frames/fps/60:.1f} min of tokens)")
    print(f"  total flat tokens   : {total_frames * n_q:,}")
    print(f"  elapsed             : {elapsed:.1f}s")
    print(f"  cache dir           : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
