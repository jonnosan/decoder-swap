"""Step 1 of the stems pivot: source-separate ONE song with Demucs (htdemucs, 4-stem).

Inputs:  one audio file (mp3/wav/...)
Outputs: data/song_test/<song_slug>/
           original.wav         (decoded source, stereo 44.1 kHz float32 PCM)
           stems/drums.wav      (44.1 kHz stereo)
           stems/bass.wav
           stems/other.wav
           stems/vocals.wav

Per project_stems_pivot_design.md, this is the first leg of the no-semantic-tokens
"does decomposition help at all" experiment. Subsequent scripts (31, 32) round-trip
each stem through DAC at max quality and compare summed stems vs full-mix baseline.

Run:
  .venv/bin/python scripts/30_separate_stems.py \
      --in "/path/to/song.mp3" --slug beltram_machine
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True,
                    help="path to the source audio file")
    ap.add_argument("--slug", required=True,
                    help="short name used as the output subdirectory (e.g. beltram_machine)")
    ap.add_argument("--model", default="htdemucs",
                    help="demucs model name (default: htdemucs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "song_test"))
    return ap.parse_args()


def load_audio_as_demucs_input(path: Path, target_sr: int, target_channels: int):
    """Read source audio and shape it for demucs: torch.Tensor [channels, samples] at target_sr."""
    import librosa
    y, sr = librosa.load(str(path), sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    if y.shape[0] == 1 and target_channels == 2:
        y = np.repeat(y, 2, axis=0)
    if y.shape[0] > target_channels:
        y = y[:target_channels]
    return torch.from_numpy(y.astype(np.float32))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"# stems separation · {args.model}")
    print(f"device: {device}")

    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(args.model)
    model.to(device).eval()
    sr = int(model.samplerate)
    ch = int(model.audio_channels)
    sources = list(model.sources)
    print(f"model:   {args.model} · {sr} Hz · {ch}-ch · sources={sources}")

    in_path = Path(args.input_path).expanduser()
    out_dir = Path(args.out_root) / args.slug
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    print(f"input:   {in_path}")
    print(f"out_dir: {out_dir}")

    t0 = time.time()
    print("  loading audio ...")
    wav = load_audio_as_demucs_input(in_path, target_sr=sr, target_channels=ch)
    dur_s = wav.shape[-1] / sr
    print(f"    loaded {dur_s:.1f} s ({dur_s/60:.1f} min), shape {tuple(wav.shape)}")

    orig_out = out_dir / "original.wav"
    sf.write(orig_out, wav.T.numpy(), sr, subtype="FLOAT")
    print(f"    wrote {orig_out.name}")

    print("  running demucs ...")
    t_sep = time.time()
    with torch.no_grad():
        # apply_model wants [batch, channels, samples]
        out = apply_model(model, wav.unsqueeze(0).to(device), shifts=1, split=True,
                          overlap=0.25, progress=True, num_workers=0)
        # out: [batch, sources, channels, samples]
    out = out.squeeze(0).cpu()
    sep_secs = time.time() - t_sep
    print(f"    separated in {sep_secs:.1f}s "
          f"({sep_secs / max(dur_s, 1e-6) * 60:.1f} s/min of audio)")

    for i, name in enumerate(sources):
        stem = out[i].numpy().T  # [samples, channels]
        path = stems_dir / f"{name}.wav"
        sf.write(path, stem, sr, subtype="FLOAT")
        rms = float(np.sqrt(np.mean(stem ** 2)))
        print(f"    wrote {path.name}  (RMS {rms:.4f})")

    elapsed = time.time() - t0
    print()
    print(f"  total elapsed: {elapsed:.1f}s")
    print(f"  artifacts in:  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
