"""Extract the original audio segment corresponding to the prompt used in
scripts/15_generate_mimi.py — bypasses the Mimi encode/decode round trip
so we can hear what the codec lost."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.train_translator_rvq import FrameBatchSampler  # noqa: E402

# Match scripts/15_generate_mimi.py exactly
PROMPT_FRAMES = 25
SEED = 42

# Mimi convention
SR = 24000
FRAME_RATE = 12.5
HOP = int(SR / FRAME_RATE)  # 1920 samples per frame

OUT_PATH = REPO_ROOT / "results" / "mimi_gen" / "prompt_orig.wav"


def main() -> int:
    corpus = load_corpus("techno")
    tokens_dir = corpus.tokens_dir(codec="mimi")
    token_paths = sorted(tokens_dir.glob("*.npy"))
    tracks = [np.load(p) for p in token_paths]

    # Replay the sampler to identify which track / start frame the prompt used.
    sampler = FrameBatchSampler(tracks, window_frames=PROMPT_FRAMES, seed=SEED)
    # FrameBatchSampler.sample for batch_size=1 calls rng.choice then rng.integers once.
    ti = int(sampler.rng.choice(len(tracks), p=sampler.track_probs))
    T = tracks[ti].shape[-1]
    start = int(sampler.rng.integers(0, T - PROMPT_FRAMES + 1))
    print(f"prompt was from track {ti} ({token_paths[ti].stem}) starting at frame {start}")
    print(f"  frame range: [{start}, {start + PROMPT_FRAMES})")

    # Map back to audio sample range, load original MP3, extract.
    start_sample = start * HOP
    n_samples = PROMPT_FRAMES * HOP
    print(f"  audio sample range: [{start_sample}, {start_sample + n_samples})  "
          f"({n_samples/SR:.2f}s @ {SR}Hz)")

    audio_paths = corpus.audio_paths
    src_mp3 = audio_paths[ti]
    print(f"  source: {src_mp3}")
    print("loading original audio...")
    y, _ = librosa.load(src_mp3, sr=SR, mono=True,
                        offset=start_sample / SR,
                        duration=n_samples / SR)
    print(f"  loaded {len(y)} samples")
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(OUT_PATH), y, SR)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
