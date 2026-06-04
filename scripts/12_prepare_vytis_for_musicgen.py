"""Prepare Vytis MP3s for MusicGen fine-tuning.

Loads the two Vytis MP3s, resamples to 32 kHz mono (MusicGen's expected input),
splits into ~10-second non-overlapping chunks, saves each as a numpy array
plus a JSON metadata file with a fixed text description per track.

Output layout:
  data/musicgen/vytis/
    chunks/
      vol1_0001.npy
      vol1_0002.npy
      ...
    metadata.jsonl     # one line per chunk: {"path":..., "text":..., "duration":...}

Run:
  uv run python scripts/12_prepare_vytis_for_musicgen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data/musicgen/vytis"
CHUNK_DIR = OUT_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 32000
CHUNK_SECONDS = 10.0
SOURCE_FILES = [
    ("vol1", "/Users/jonno/Downloads/Vytis - Greatest Hits Vol. 1.mp3",
     "vytis lithuanian techno, deep, driving, hypnotic"),
    ("vol3", "/Users/jonno/Downloads/Vytis - Greatest Hits Vol. 3.mp3",
     "vytis lithuanian techno, deep, driving, hypnotic"),
]


def main() -> int:
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa not installed. Install with: uv add librosa", file=sys.stderr)
        return 1

    metadata_path = OUT_DIR / "metadata.jsonl"
    n_total = 0
    with metadata_path.open("w") as meta_f:
        for stem, src, text in SOURCE_FILES:
            src_path = Path(src)
            if not src_path.exists():
                print(f"ERROR: missing {src_path}", file=sys.stderr)
                return 1
            print(f"loading {src_path.name} ...")
            audio, sr_in = librosa.load(str(src_path), sr=SAMPLE_RATE, mono=True)
            print(f"  loaded: {len(audio):,} samples @ {SAMPLE_RATE}Hz = {len(audio)/SAMPLE_RATE/60:.1f} min")

            chunk_n = int(round(CHUNK_SECONDS * SAMPLE_RATE))
            n_chunks = len(audio) // chunk_n
            for i in range(n_chunks):
                chunk = audio[i*chunk_n : (i+1)*chunk_n]
                # Skip near-silent chunks (rare in techno but defensive).
                if float(np.abs(chunk).max()) < 1e-3:
                    continue
                rel = f"{stem}_{i+1:04d}.npy"
                np.save(CHUNK_DIR / rel, chunk.astype(np.float32))
                meta_f.write(json.dumps({
                    "path": str(CHUNK_DIR / rel),
                    "text": text,
                    "duration": float(CHUNK_SECONDS),
                }) + "\n")
                n_total += 1
            print(f"  wrote {n_chunks} chunks from {src_path.name}")

    print(f"\nTotal chunks: {n_total}")
    print(f"Metadata: {metadata_path}")
    print(f"Chunks dir: {CHUNK_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
