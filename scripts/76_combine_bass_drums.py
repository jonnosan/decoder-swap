"""Merge bass_auto.mid + drums_auto.mid into combined_auto.mid for each track.

Both inputs are already on the same absolute timeline (extracted from the same
source audio). Combining is just: bass on channel 1, drums on channel 10
(is_drum=True), one PrettyMIDI with both instruments.

Output: data/song_test/<slug>/semantic/combined_auto.mid

Run:
  .venv/bin/python scripts/76_combine_bass_drums.py --slug-pattern 'dmxkrew_*'
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pretty_midi

REPO_ROOT = Path(__file__).resolve().parents[1]


def combine_one(slug: str) -> dict:
    song_dir = REPO_ROOT / "data" / "song_test" / slug
    bass_path = song_dir / "semantic" / "bass_auto.mid"
    drum_path = song_dir / "semantic" / "drums_auto.mid"
    out_path = song_dir / "semantic" / "combined_auto.mid"

    if not bass_path.exists():
        return {"slug": slug, "error": "missing bass_auto.mid"}
    if not drum_path.exists():
        return {"slug": slug, "error": "missing drums_auto.mid"}

    # Read the bass — preserve its tempo
    bass_pm = pretty_midi.PrettyMIDI(str(bass_path))
    drum_pm = pretty_midi.PrettyMIDI(str(drum_path))

    # Build new combined: tempo from bass (has the correct BPM from auto-sweep)
    initial_tempo = 120.0
    if bass_pm.get_tempo_changes()[1].size > 0:
        initial_tempo = float(bass_pm.get_tempo_changes()[1][0])

    combined = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

    for inst in bass_pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program if inst.program else 38,  # Synth Bass 1 default
            is_drum=False,
            name=inst.name or "bass",
        )
        new_inst.notes = list(inst.notes)
        combined.instruments.append(new_inst)

    for inst in drum_pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=0, is_drum=True, name=inst.name or "drums",
        )
        new_inst.notes = list(inst.notes)
        combined.instruments.append(new_inst)

    combined.write(str(out_path))

    n_bass = sum(len(i.notes) for i in bass_pm.instruments)
    n_drums = sum(len(i.notes) for i in drum_pm.instruments)
    return {
        "slug": slug,
        "n_bass_notes": n_bass,
        "n_drum_notes": n_drums,
        "tempo": initial_tempo,
        "out_midi": str(out_path),
    }


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

    results = []
    for slug in slugs:
        try:
            r = combine_one(slug)
        except Exception as e:
            r = {"slug": slug, "error": str(e)}
        results.append(r)
        if "error" in r:
            print(f"  {slug}: ERROR {r['error']}")
        else:
            print(f"  {slug}: bass={r['n_bass_notes']} + drums={r['n_drum_notes']} @ BPM {r['tempo']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
