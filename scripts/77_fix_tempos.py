"""Rewrite bass_auto.mid / drums_auto.mid / combined_auto.mid with the BPM
detected in bass_auto.json, so the tempo metadata matches the audio.

Without this fix, all MIDIs carry tempo=120 (the hybrid extractor's hardcoded
initial_tempo) regardless of what the auto-sweep actually detected. Note
POSITIONS are correct in seconds; this only updates the tempo metadata so
DAWs that read it (Ableton, Logic) display the right BPM and bars line up.

Run:
  .venv/bin/python scripts/77_fix_tempos.py --slug-pattern 'dmxkrew_*'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pretty_midi

REPO_ROOT = Path(__file__).resolve().parents[1]


def rewrite_with_tempo(midi_path: Path, bpm: float) -> bool:
    if not midi_path.exists():
        return False
    src = pretty_midi.PrettyMIDI(str(midi_path))
    out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for inst in src.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name,
        )
        new_inst.notes = list(inst.notes)
        new_inst.pitch_bends = list(inst.pitch_bends)
        new_inst.control_changes = list(inst.control_changes)
        out.instruments.append(new_inst)
    out.write(str(midi_path))
    return True


def fix_one(slug: str) -> dict:
    song_dir = REPO_ROOT / "data" / "song_test" / slug
    meta_path = song_dir / "semantic" / "bass_auto.json"
    if not meta_path.exists():
        return {"slug": slug, "error": "no bass_auto.json"}
    meta = json.loads(meta_path.read_text())
    bpm = float(meta.get("bpm", 120))
    fixed = {}
    for name in ("bass_auto.mid", "drums_auto.mid", "combined_auto.mid"):
        p = song_dir / "semantic" / name
        fixed[name] = rewrite_with_tempo(p, bpm)
    return {"slug": slug, "bpm": bpm, "fixed": fixed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--slug-pattern")
    args = ap.parse_args()

    slugs: list[str] = []
    if args.slug:
        slugs = [args.slug]
    elif args.slug_pattern:
        slugs = sorted([Path(p).name for p in glob.glob(str(REPO_ROOT / "data" / "song_test" / args.slug_pattern))])
    else:
        print("--slug or --slug-pattern", file=sys.stderr)
        return 1

    for slug in slugs:
        r = fix_one(slug)
        if "error" in r:
            print(f"  {slug}: {r['error']}")
        else:
            fixed = ", ".join(f"{k}={'Y' if v else '-'}" for k, v in r["fixed"].items())
            print(f"  {slug}: bpm={r['bpm']:.2f}  {fixed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
