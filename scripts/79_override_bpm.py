"""Manually override BPM for a track (and rewrite all MIDI files with that BPM).

For tracks where the auto-sweep + simple-double heuristic still gets BPM wrong
(e.g. dmxkrew_101_tonight where cross-corr found a non-musical period).

Updates: bass_auto.json (bpm field, marks override), bass_auto.mid, drums_auto.mid,
combined_auto.mid

Run:
  .venv/bin/python scripts/79_override_bpm.py --slug dmxkrew_101_tonight --bpm 143.55
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    args = ap.parse_args()

    song_dir = REPO_ROOT / "data" / "song_test" / args.slug
    meta_path = song_dir / "semantic" / "bass_auto.json"
    if not meta_path.exists():
        print(f"no bass_auto.json for {args.slug}", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text())
    old_bpm = meta.get("bpm")
    meta["bpm_before_override"] = old_bpm
    meta["bpm"] = args.bpm
    meta["bpm_user_override"] = True
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    print(f"  {args.slug}: bpm {old_bpm:.2f} -> {args.bpm:.2f} (user override)")
    for name in ("bass_auto.mid", "drums_auto.mid", "combined_auto.mid"):
        p = song_dir / "semantic" / name
        ok = rewrite_with_tempo(p, args.bpm)
        print(f"    {name}: {'updated' if ok else 'missing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
