"""Post-processing denoise via classical spectral subtraction (noisereduce).

Applies non-stationary denoising at two strength levels to:
  - prompt_mimi_20s.wav (Mimi codec ceiling — shows how much codec noise can be removed)
  - the three sampling-tweak generations from script 17

Saves outputs alongside the originals as *_denoise50.wav (mild) and *_denoise80.wav (stronger).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import noisereduce as nr

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = REPO_ROOT / "results" / "mimi_gen_long"

INPUTS = [
    "prompt_mimi_20s.wav",
    "gen_topk50_t08.wav",
    "gen_topp95_t085.wav",
    "gen_topp95_t085_5s.wav",
    "gen_topp90_t075.wav",
]
STRENGTHS = [(0.5, "denoise50"), (0.85, "denoise85")]


def main() -> int:
    for fname in INPUTS:
        in_path = GEN_DIR / fname
        if not in_path.exists():
            print(f"[skip] {fname} not found")
            continue
        audio, sr = sf.read(str(in_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # to mono
        audio = audio.astype(np.float32)
        print(f"\n{fname}: {len(audio)} samples @ {sr} Hz")

        for prop, suffix in STRENGTHS:
            t0 = time.time()
            cleaned = nr.reduce_noise(
                y=audio, sr=sr,
                stationary=False,
                prop_decrease=prop,
            )
            dt = time.time() - t0
            cleaned = np.clip(cleaned, -1.0, 1.0)
            out_path = GEN_DIR / f"{in_path.stem}_{suffix}.wav"
            sf.write(str(out_path), cleaned, sr)
            print(f"  prop_decrease={prop:.2f}  -> {out_path.name}  ({dt:.1f}s)")

    print(f"\ndone. files in {GEN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
