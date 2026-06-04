"""Apply the trained fixer to existing WAVs (e.g., LM-generated audio or Mimi
round-trips). Processes in non-overlapping chunks since the U-Net is fully
convolutional and length-agnostic.

Run:
  uv run python scripts/21_apply_fixer.py             # default file set
  uv run python scripts/21_apply_fixer.py path/to.wav [path/to/other.wav ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.fixer import FixerConfig, FixerUNet  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402

CKPT = REPO_ROOT / "data/checkpoints/fixer/fixer.pt"
DEFAULT_INPUTS = [
    "results/mimi_gen_long/prompt_mimi_20s.wav",
    "results/mimi_gen_long/gen_topp95_t085_5s.wav",
    "results/mimi_gen_long/gen_topk50_t08.wav",
    "results/mimi_gen_long/gen_topp90_t075.wav",
]
CHUNK_SAMPLES = 24000 * 4  # process in 4-second chunks


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = FixerConfig(**ckpt["config"])
    model = FixerUNet(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"loaded fixer ckpt step={ckpt['step']}  params={model.num_parameters():,}")

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = [REPO_ROOT / p for p in DEFAULT_INPUTS]

    for src in paths:
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        audio, sr = sf.read(str(src))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        print(f"\n{src.name}: {len(audio)} samples @ {sr} Hz ({len(audio)/sr:.1f}s)")

        # Process in non-overlapping chunks; the U-Net is fully convolutional so
        # chunk boundaries are not visible in output (no boundary artifacts).
        out_chunks = []
        with torch.no_grad():
            for start in range(0, len(audio), CHUNK_SAMPLES):
                end = min(start + CHUNK_SAMPLES, len(audio))
                chunk = audio[start:end]
                x = torch.from_numpy(chunk).float().to(device).view(1, 1, -1)
                y = model(x)
                out_chunks.append(y[0, 0].cpu().numpy())
        cleaned = np.concatenate(out_chunks)
        # Trim to original length (U-Net may have produced ±1 sample due to padding).
        cleaned = cleaned[: len(audio)]
        cleaned = np.clip(cleaned, -1.0, 1.0)

        out_path = src.parent / f"{src.stem}_fixed.wav"
        sf.write(str(out_path), cleaned, sr)
        print(f"  -> {out_path.name}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
