"""Single-command driver for the stems-pivot bass pipeline.

Runs, in order:
  30 separate_stems         (Demucs htdemucs → 4 stems)
  33 bass_to_midi           (pYIN on bass.wav → bass.{json,mid})
  37 extract_bass_exemplars (rank + polish retrigger-clean candidates)
  34 midi_to_bass_exemplar  (exemplar pitch-shift resynth; uses script 37's top pick)
  36 midi_to_bass_synth     (numpy saw-bass-synth resynth)
  35 reassemble_swap_bass   (sum stems with both swap variants for A/B)

Each stage is skipped if its output already exists; pass --force to re-run.

Run:
  .venv/bin/python scripts/40_run_pipeline.py \
      --in "/path/to/song.mp3" --slug my_song
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPTS = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True,
                    help="path to the source audio file (mp3/wav/...)")
    ap.add_argument("--slug", required=True,
                    help="short name for the output subdirectory under data/song_test/")
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--force", action="store_true",
                    help="re-run every stage even if its output already exists")
    ap.add_argument("--candidate-idx", type=int, default=0,
                    help="which ranked exemplar from script 37 to feed into script 34 (0 = top)")
    return ap.parse_args()


def stems_exist(song_dir: Path) -> bool:
    needed = ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]
    return all((song_dir / "stems" / n).exists() for n in needed)


def run_stage(label: str, args: list[str], skip_if: bool) -> float:
    print()
    print(f"━━━ {label} ━━━")
    if skip_if:
        print(f"  [skip] outputs already exist")
        return 0.0
    print(f"  $ {' '.join(args)}")
    t0 = time.time()
    rc = subprocess.call(args)
    elapsed = time.time() - t0
    if rc != 0:
        raise SystemExit(f"{label} failed (rc={rc})")
    print(f"  done in {elapsed:.1f}s")
    return elapsed


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    force = args.force

    print(f"# stems pivot · pipeline driver")
    print(f"  input: {args.input_path}")
    print(f"  slug:  {args.slug}")
    print(f"  song dir: {song_dir}")

    t_total0 = time.time()
    timings: dict[str, float] = {}

    # 30 separate stems
    timings["30_separate_stems"] = run_stage(
        "30 separate_stems",
        [PY, str(SCRIPTS / "30_separate_stems.py"),
         "--in", args.input_path, "--slug", args.slug],
        skip_if=(not force) and stems_exist(song_dir),
    )

    # 33 bass → MIDI
    midi_done = (song_dir / "semantic" / "bass.json").exists()
    timings["33_bass_to_midi"] = run_stage(
        "33 bass_to_midi",
        [PY, str(SCRIPTS / "33_bass_to_midi.py"), "--slug", args.slug],
        skip_if=(not force) and midi_done,
    )

    # 37 rank + polish candidate samples
    candidates_done = (song_dir / "bass_samples" / "candidates.json").exists()
    timings["37_extract_bass_exemplars"] = run_stage(
        "37 extract_bass_exemplars",
        [PY, str(SCRIPTS / "37_extract_bass_exemplars.py"), "--slug", args.slug],
        skip_if=(not force) and candidates_done,
    )

    # 34 exemplar resynth (uses 37's top candidate by default)
    resynth_done = (song_dir / "stems_resynth" / "bass.wav").exists()
    timings["34_midi_to_bass_exemplar"] = run_stage(
        "34 midi_to_bass_exemplar",
        [PY, str(SCRIPTS / "34_midi_to_bass_exemplar.py"),
         "--slug", args.slug, "--candidate-idx", str(args.candidate_idx)],
        skip_if=(not force) and resynth_done,
    )

    # 36 synth resynth
    synth_done = (song_dir / "stems_synth" / "bass.wav").exists()
    timings["36_midi_to_bass_synth"] = run_stage(
        "36 midi_to_bass_synth",
        [PY, str(SCRIPTS / "36_midi_to_bass_synth.py"), "--slug", args.slug],
        skip_if=(not force) and synth_done,
    )

    # 35 reassemble (cheap, always re-run so the A/B set is current)
    timings["35_reassemble_swap_bass"] = run_stage(
        "35 reassemble_swap_bass",
        [PY, str(SCRIPTS / "35_reassemble_swap_bass.py"), "--slug", args.slug],
        skip_if=False,
    )

    total = time.time() - t_total0
    print()
    print("## pipeline complete")
    print(f"  total: {total:.1f}s")
    for k, v in timings.items():
        marker = " (skipped)" if v == 0.0 else ""
        print(f"    {k:<32} {v:>6.1f}s{marker}")
    print()
    print(f"## listening files")
    listen_root = REPO_ROOT / "results" / "stems_v1" / args.slug
    print(f"  A/B mixes:    {listen_root / 'swap_bass'}/")
    print(f"  bass samples: {listen_root / 'bass_samples'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
