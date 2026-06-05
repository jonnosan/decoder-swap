"""Chunked Demucs separation for long inputs (DJ-mix scale).

Demucs's internal chunking handles per-chunk inference but assembles the full
output as one big tensor at the end. For multi-hour inputs on MPS this silently
returns zeros (the full-mix Vol 1 case hit this 2026-06-05). This script slices
the input into N-minute chunks manually, runs Demucs on each, transfers to CPU,
and concatenates per-source CPU arrays.

Run:
  .venv/bin/python scripts/30b_separate_stems_chunked.py \\
      --in "/path/to/long_mix.mp3" --slug my_mix --chunk-min 8
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

from decoder_swap.settings import resolve_device  # noqa: E402


def load_audio_as_demucs_input(path: Path, target_sr: int, target_channels: int):
    import librosa
    y, sr = librosa.load(str(path), sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    if y.shape[0] == 1 and target_channels == 2:
        y = np.repeat(y, 2, axis=0)
    if y.shape[0] > target_channels:
        y = y[:target_channels]
    return torch.from_numpy(y.astype(np.float32))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--chunk-min", type=float, default=8.0,
                    help="chunk length in minutes (default 8)")
    ap.add_argument("--overlap-s", type=float, default=2.0,
                    help="seconds of overlap between chunks (smoothed with hann)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"# chunked stems · {args.model} · device={device}")

    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(args.model)
    model.to(device).eval()
    sr = int(model.samplerate)
    ch = int(model.audio_channels)
    sources = list(model.sources)
    print(f"model:  {args.model} · {sr} Hz · {ch}-ch · sources={sources}")

    in_path = Path(args.input_path).expanduser()
    out_dir = Path(args.out_root) / args.slug
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    print(f"input:  {in_path}")
    print(f"out:    {out_dir}")

    t0 = time.time()
    print("loading audio ...")
    wav = load_audio_as_demucs_input(in_path, target_sr=sr, target_channels=ch)
    N = wav.shape[-1]
    dur_s = N / sr
    print(f"  loaded {dur_s:.1f}s ({dur_s/60:.1f} min), shape {tuple(wav.shape)}")

    orig_out = out_dir / "original.wav"
    sf.write(orig_out, wav.T.numpy(), sr, subtype="FLOAT")
    print(f"  wrote {orig_out.name}")

    chunk_samples = int(args.chunk_min * 60 * sr)
    overlap_samples = int(args.overlap_s * sr)
    n_chunks = int(np.ceil(N / chunk_samples))
    print(f"chunking: {n_chunks} chunks of {args.chunk_min} min "
          f"+ {args.overlap_s}s overlap")

    # Pre-allocate per-source CPU output arrays.
    full_out = {name: np.zeros((ch, N), dtype=np.float32) for name in sources}
    weight = np.zeros(N, dtype=np.float32)

    for ci in range(n_chunks):
        cs = ci * chunk_samples
        ce = min(cs + chunk_samples + overlap_samples, N)
        chunk = wav[:, cs:ce].unsqueeze(0).to(device)
        L = ce - cs
        t_c = time.time()
        with torch.no_grad():
            out = apply_model(model, chunk, shifts=1, split=True, overlap=0.25,
                              progress=False, num_workers=0)
        out = out.squeeze(0).cpu().numpy()    # (sources, ch, samples)
        del chunk
        if device == "mps":
            torch.mps.empty_cache()
        # Build a smoothing window: ones in the middle, half-Hann tapers at the overlap edges.
        win = np.ones(L, dtype=np.float32)
        if ci > 0:
            tap = min(overlap_samples, L)
            win[:tap] = 0.5 * (1.0 - np.cos(np.pi * np.arange(tap) / tap))
        if ci < n_chunks - 1:
            tap = min(overlap_samples, L)
            win[-tap:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(tap) / tap))
        # Sanity check this chunk produced non-silent output (catch MPS dropouts)
        chunk_rms = float(np.sqrt(np.mean(out ** 2)))
        if chunk_rms < 1e-7:
            raise RuntimeError(
                f"chunk {ci+1}/{n_chunks} returned silence (rms={chunk_rms}) — "
                f"MPS allocator likely returned zeros. Try --chunk-min smaller."
            )
        for si, name in enumerate(sources):
            full_out[name][:, cs:ce] += out[si] * win[None, :]
        weight[cs:ce] += win
        dt = time.time() - t_c
        print(f"  chunk {ci+1}/{n_chunks}  [{cs/sr:.0f}-{ce/sr:.0f}s]  "
              f"sep {dt:.1f}s  rms={chunk_rms:.4f}", flush=True)

    # Normalize by overlap weights.
    weight = np.maximum(weight, 1e-3)
    for name in sources:
        full_out[name] = full_out[name] / weight[None, :]

    for name in sources:
        stem = full_out[name].T   # (samples, ch)
        path = stems_dir / f"{name}.wav"
        sf.write(path, stem, sr, subtype="FLOAT")
        rms = float(np.sqrt(np.mean(stem ** 2)))
        print(f"  wrote {path.name}  (RMS {rms:.4f})")

    elapsed = time.time() - t0
    print(f"\ntotal elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
