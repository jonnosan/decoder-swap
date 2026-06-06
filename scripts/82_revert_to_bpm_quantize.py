"""Revert each track's bass_auto.mid to BPM-quantize output (undo the corpus warp).

Reads bass_auto.json for the auto-sweep's chosen params (bpm, anchor_used, pre_merge_ms)
and re-quantizes from the raw bass.mid. Rebuilds combined_auto.mid and fixes tempos.

Special-case: skips tracks listed in --skip (e.g., '101' which user verified works under warp).

Run:
  .venv/bin/python scripts/82_revert_to_bpm_quantize.py --skip dmxkrew_101_tonight
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def revert_one(slug: str) -> dict:
    song_dir = REPO_ROOT / "data" / "song_test" / slug
    meta_path = song_dir / "semantic" / "bass_auto.json"
    raw_midi = song_dir / "semantic" / "bass.mid"
    out_midi = song_dir / "semantic" / "bass_auto.mid"
    if not meta_path.exists() or not raw_midi.exists():
        return {"slug": slug, "skipped": "missing inputs"}
    meta = json.loads(meta_path.read_text())

    # Use the bpm that was set after low-bpm correction / user override, NOT the bpm
    # from before warp. Fields written by previous scripts:
    #   bpm_original_autosweep: the raw auto-sweep result
    #   bpm_before_override: if user override happened
    #   bpm_before_warp: if warp happened
    #   bpm: current (which may be the warp's median-kick-interval bpm)
    bpm = (meta.get("bpm_before_warp")
           or meta.get("bpm_before_override")
           or meta.get("bpm"))
    anchor_used = meta.get("anchor_used", meta.get("anchor_s"))
    pre_merge_ms = meta.get("pre_merge_ms", 0)

    cmd = [
        ".venv/bin/python", "scripts/67_quantize_midi.py",
        "--midi", str(raw_midi), "--out", str(out_midi),
        "--bpm", str(bpm), "--grid", "16",
        "--pre-merge-ms", str(pre_merge_ms),
        "--pitch-lo", "24", "--pitch-hi", "50",
        "--mono", "--release-gap-ms", "30",
        "--anchor-s", str(anchor_used),
    ]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return {"slug": slug, "error": r.stderr[:200]}

    # Clear the warp marker in JSON
    meta["bpm"] = bpm
    meta.pop("warped_to_drum_grid", None)
    meta.pop("bpm_before_warp", None)
    meta["reverted_from_warp"] = True
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    # Rebuild combined
    subprocess.run([".venv/bin/python", "scripts/76_combine_bass_drums.py", "--slug", slug],
                   cwd=REPO_ROOT, capture_output=True)
    # Fix tempos (so all three have bpm metadata)
    subprocess.run([".venv/bin/python", "scripts/77_fix_tempos.py", "--slug", slug],
                   cwd=REPO_ROOT, capture_output=True)
    return {"slug": slug, "bpm": bpm, "anchor": anchor_used, "pre_merge_ms": pre_merge_ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="slug(s) to skip (keep current state)")
    ap.add_argument("--slug-pattern", default="dmxkrew_*")
    args = ap.parse_args()

    slugs = sorted([Path(p).name for p in glob.glob(str(REPO_ROOT / "data" / "song_test" / args.slug_pattern))])
    slugs = [s for s in slugs if (REPO_ROOT / "data" / "song_test" / s / "semantic" / "bass_auto.mid").exists()]
    print(f"Will revert {len(slugs)} tracks (skipping: {args.skip})")
    for slug in slugs:
        if slug in args.skip:
            print(f"  {slug}: SKIPPED (preserved)")
            continue
        r = revert_one(slug)
        if "error" in r:
            print(f"  {slug}: ERROR {r['error']}")
        elif "skipped" in r:
            print(f"  {slug}: {r['skipped']}")
        else:
            print(f"  {slug}: bpm={r['bpm']:.2f} anchor={r['anchor']:.3f} pre_merge={r['pre_merge_ms']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
