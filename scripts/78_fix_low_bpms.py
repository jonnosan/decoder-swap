"""Double any BPM < 90 in bass_auto.json + rewrite the MIDI files with
the corrected tempo.

Auto-sweep's cross-correlation often finds a 2-bar (sometimes 4-bar)
periodicity and treats it as a 1-bar period, halving the BPM. User-supplied
heuristic: anything <90 BPM in this corpus is wrong; double it (repeat until
≥90 or hit safety cap at 200).

Updates: bass_auto.json (BPM field), bass_auto.mid, drums_auto.mid, combined_auto.mid

Run:
  .venv/bin/python scripts/78_fix_low_bpms.py --slug-pattern 'dmxkrew_*'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pretty_midi

REPO_ROOT = Path(__file__).resolve().parents[1]

MIN_BPM = 90.0
MAX_BPM = 200.0


def correct_bpm(bpm: float) -> float:
    while bpm < MIN_BPM and bpm * 2 <= MAX_BPM:
        bpm *= 2
    return bpm


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
        return {"slug": slug, "skipped": "no bass_auto.json"}
    meta = json.loads(meta_path.read_text())
    old_bpm = float(meta.get("bpm", 0))
    new_bpm = correct_bpm(old_bpm)
    if new_bpm == old_bpm:
        return {"slug": slug, "bpm": old_bpm, "unchanged": True}

    # Update JSON
    meta["bpm_original_autosweep"] = old_bpm
    meta["bpm"] = new_bpm
    meta["bpm_corrected"] = True
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    # Rewrite MIDI files
    fixed = {}
    for name in ("bass_auto.mid", "drums_auto.mid", "combined_auto.mid"):
        p = song_dir / "semantic" / name
        fixed[name] = rewrite_with_tempo(p, new_bpm)

    return {"slug": slug, "old_bpm": old_bpm, "new_bpm": new_bpm, "fixed": fixed}


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

    n_changed = 0
    for slug in slugs:
        r = fix_one(slug)
        if "skipped" in r:
            continue
        if r.get("unchanged"):
            print(f"  {slug}: {r['bpm']:.2f} (unchanged)")
        else:
            print(f"  {slug}: {r['old_bpm']:.2f} -> {r['new_bpm']:.2f}")
            n_changed += 1
    print(f"\n{n_changed} tracks corrected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
