"""Reassemble the song with the bass stem swapped for each available MIDI resynth variant.

Reads any/all of:
  stems_resynth/bass.wav   (exemplar pitch-shift,  produced by script 34)
  stems_synth/bass.wav     (numpy saw + ADSR + LP, produced by script 36)

Emits one swap mix per variant present, plus the original-stems control:
  mix_original_stems.wav     drums + bass         + other + vocals
  mix_swap_bass_<v>.wav      drums + bass_<v>     + other + vocals  (per variant)
  bass_original.wav          original bass stem in isolation
  bass_<v>.wav               each resynth variant in isolation

Run:
  .venv/bin/python scripts/35_reassemble_swap_bass.py --slug beltram_machine
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Resynth variants we know about: (variant_name, subdir, filename)
VARIANTS = [
    ("exemplar", "stems_resynth", "bass.wav"),
    ("synth",    "stems_synth",   "bass.wav"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results" / "stems_v1"))
    return ap.parse_args()


def load_stereo(path: Path, expected_sr: int | None = None) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if expected_sr is not None and sr != expected_sr:
        raise SystemExit(f"sr mismatch reading {path}: {sr} != {expected_sr}")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y, sr


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    stems_dir = song_dir / "stems"
    results_dir = Path(args.results_root) / args.slug / "swap_bass"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"# reassemble · slug={args.slug}")
    drums, sr = load_stereo(stems_dir / "drums.wav")
    bass_orig, _ = load_stereo(stems_dir / "bass.wav", sr)
    other, _ = load_stereo(stems_dir / "other.wav", sr)
    vocals, _ = load_stereo(stems_dir / "vocals.wav", sr)

    # Discover which resynth variants exist on disk
    variants: list[tuple[str, np.ndarray]] = []
    for vname, subdir, fname in VARIANTS:
        p = song_dir / subdir / fname
        if p.exists():
            y, _ = load_stereo(p, sr)
            variants.append((vname, y))
            print(f"  found variant: {vname}  ({p.relative_to(song_dir)})")
        else:
            print(f"  missing variant: {vname}  ({p.relative_to(song_dir)})")
    if not variants:
        raise SystemExit("no resynth variants found; run script 34 and/or 36 first")

    # Align everything to the shortest length
    n = min(drums.shape[0], bass_orig.shape[0], other.shape[0], vocals.shape[0],
            *(v.shape[0] for _, v in variants))
    drums, bass_orig, other, vocals = drums[:n], bass_orig[:n], other[:n], vocals[:n]
    variants = [(vname, v[:n]) for vname, v in variants]
    print(f"  aligned len: {n} samples ({n/sr/60:.2f} min)")

    # Original-stems control + every swap mix + every isolated bass
    mix_orig = drums + bass_orig + other + vocals
    sf.write(results_dir / "mix_original_stems.wav", mix_orig, sr, subtype="FLOAT")
    sf.write(results_dir / "bass_original.wav", bass_orig, sr, subtype="FLOAT")

    summary: list[tuple[str, np.ndarray]] = [
        ("mix_original_stems", mix_orig),
        ("bass_original",      bass_orig),
    ]
    for vname, bass_v in variants:
        mix_v = drums + bass_v + other + vocals
        mix_out = results_dir / f"mix_swap_bass_{vname}.wav"
        bass_out = results_dir / f"bass_{vname}.wav"
        sf.write(mix_out, mix_v, sr, subtype="FLOAT")
        sf.write(bass_out, bass_v, sr, subtype="FLOAT")
        summary.append((f"mix_swap_bass_{vname}", mix_v))
        summary.append((f"bass_{vname}",          bass_v))

    # Copy MIDI artifacts for inspection
    for name in ("bass.json", "bass.mid"):
        src = song_dir / "semantic" / name
        if src.exists():
            shutil.copy(src, results_dir / name)

    print()
    print("## RMS / peak per file")
    for label, audio in summary:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        warn = "  <-- clip" if peak >= 1.0 else ""
        print(f"  {label:<28}  RMS={rms:.4f}  peak={peak:.4f}{warn}")
    print()
    print(f"  listening files in: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
