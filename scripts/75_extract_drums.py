"""Extract drum MIDI from drums.wav using jtxtok_extractor's drum detector.

Output: GM Channel 10 (0-indexed=9) MIDI with the canonical mapping:
  kick → 36  (Bass Drum 1)
  snare → 38 (Acoustic Snare)
  hat → 42   (Closed Hi-Hat)
  ohat → 46  (Open Hi-Hat)
  clap → 39  (Hand Clap)

Saves to data/song_test/<slug>/semantic/drums_auto.mid

Run:
  .venv/bin/python scripts/75_extract_drums.py --slug dmxkrew_201_17_ways_to_break_my_heart
  .venv/bin/python scripts/75_extract_drums.py --slug-pattern 'dmxkrew_*'
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
JTXTOK_SRC = Path("/Users/jonno/src/jtxtok_extractor/src")

sys.path.insert(0, str(JTXTOK_SRC))
from jtxtok_extractor import drums as jt_drums  # noqa: E402

GM_DRUM_MAP = {
    "kick": 36,
    "snare": 38,
    "hat": 42,
    "ohat": 46,
    "clap": 39,
}


def extract_one(slug: str, method: str = "auto") -> dict:
    song_dir = REPO_ROOT / "data" / "song_test" / slug
    drum_path = song_dir / "stems" / "drums.wav"
    if not drum_path.exists():
        return {"slug": slug, "error": "missing drums.wav"}

    out_dir = song_dir / "semantic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_midi = out_dir / "drums_auto.mid"

    x, sr = sf.read(str(drum_path))
    if x.ndim > 1:
        x = x.mean(-1)
    x = x.astype(np.float32)

    onsets, method_used = jt_drums.extract_drums(x, sr, method=method)

    # Build GM CH10 MIDI
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    skipped = 0
    for o in onsets:
        pitch = GM_DRUM_MAP.get(o.drum_class)
        if pitch is None:
            skipped += 1
            continue
        start = o.sample / sr
        end = start + 0.05  # 50ms note (drums are short)
        # velocity from confidence (clipped, mapped to 60-120)
        vel = int(np.clip(60 + (o.confidence * 60), 1, 127))
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(out_midi))

    from collections import Counter
    class_counts = Counter(o.drum_class for o in onsets)
    return {
        "slug": slug,
        "method_used": method_used,
        "n_onsets": len(onsets),
        "n_skipped": skipped,
        "class_counts": dict(class_counts),
        "out_midi": str(out_midi),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="single track")
    ap.add_argument("--slug-pattern", help="glob (e.g. 'dmxkrew_*')")
    ap.add_argument("--method", default="auto", choices=["auto", "adt", "spectral"])
    args = ap.parse_args()

    slugs: list[str] = []
    if args.slug:
        slugs = [args.slug]
    elif args.slug_pattern:
        slugs = sorted([
            Path(p).name
            for p in glob.glob(str(REPO_ROOT / "data" / "song_test" / args.slug_pattern))
            if (Path(p) / "stems" / "drums.wav").exists()
        ])
    else:
        print("--slug or --slug-pattern", file=sys.stderr)
        return 1

    results = []
    for slug in slugs:
        try:
            r = extract_one(slug, method=args.method)
        except Exception as e:
            r = {"slug": slug, "error": str(e)}
        results.append(r)
        if "error" in r:
            print(f"  {slug}: ERROR {r['error']}")
        else:
            print(f"  {slug}: {r['n_onsets']} onsets ({r['method_used']}), {r['class_counts']}")

    import json
    out_path = REPO_ROOT / "results" / "drum_extract_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
