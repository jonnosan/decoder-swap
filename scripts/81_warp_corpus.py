"""Run drum-warp on all tracks with bass_auto.mid; measure impact + report.

For each track:
  1. Warp the raw bass.mid to the drum-kick grid (script 80 logic, inlined)
  2. Measure: how much did each bass note shift from raw extraction?
  3. Also measure: how much do warp vs old bass_auto.mid positions disagree?
  4. Replace bass_auto.mid with warped version
  5. Rebuild combined_auto.mid

Reports tracks sorted by impact so user can spot-check the most-changed.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi

REPO_ROOT = Path(__file__).resolve().parents[1]


def measure_disagreement(midi_a: Path, midi_b: Path) -> dict:
    """For each note in A, find nearest note in B by time; report distance stats."""
    if not midi_a.exists() or not midi_b.exists():
        return {}
    pm_a = pretty_midi.PrettyMIDI(str(midi_a))
    pm_b = pretty_midi.PrettyMIDI(str(midi_b))
    times_a = sorted([n.start for inst in pm_a.instruments for n in inst.notes])
    times_b = sorted([n.start for inst in pm_b.instruments for n in inst.notes])
    if not times_a or not times_b:
        return {}
    times_b_arr = np.array(times_b)
    shifts_ms = []
    for ta in times_a:
        nearest = times_b_arr[np.abs(times_b_arr - ta).argmin()]
        shifts_ms.append(abs(nearest - ta) * 1000)
    shifts = np.array(shifts_ms)
    return {
        "n_a": len(times_a),
        "n_b": len(times_b),
        "shift_mean_ms": float(shifts.mean()),
        "shift_median_ms": float(np.median(shifts)),
        "shift_p90_ms": float(np.percentile(shifts, 90)),
        "shift_max_ms": float(shifts.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug-pattern", default="dmxkrew_*")
    ap.add_argument("--subdivisions", type=int, default=4)
    args = ap.parse_args()

    slugs = sorted([Path(p).name for p in glob.glob(str(REPO_ROOT / "data" / "song_test" / args.slug_pattern))])
    # Only tracks with bass_auto.mid
    slugs = [s for s in slugs if (REPO_ROOT / "data" / "song_test" / s / "semantic" / "bass_auto.mid").exists()]

    print(f"Processing {len(slugs)} tracks")
    results = []
    for slug in slugs:
        song_dir = REPO_ROOT / "data" / "song_test" / slug
        # Run warp script
        r = subprocess.run(
            [".venv/bin/python", "scripts/80_warp_bass_to_drums.py",
             "--slug", slug, "--subdivisions", str(args.subdivisions)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            results.append({"slug": slug, "error": r.stderr[:200]})
            print(f"  {slug}: ERROR {r.stderr[:200]}")
            continue

        # Parse output for impact metric
        out = r.stdout
        # "warped 600/674 notes; max shift 54.8ms"
        import re
        m_warp = re.search(r"warped (\d+)/(\d+) notes.*max shift (\d+\.?\d*)ms", out)
        m_kicks = re.search(r"detected (\d+) kicks", out)
        m_bpm = re.search(r"approx BPM (\d+\.?\d*)", out)
        warped_count = int(m_warp.group(1)) if m_warp else 0
        total_notes = int(m_warp.group(2)) if m_warp else 0
        max_shift_ms = float(m_warp.group(3)) if m_warp else 0
        n_kicks = int(m_kicks.group(1)) if m_kicks else 0
        bpm = float(m_bpm.group(1)) if m_bpm else 0

        # Disagreement with old BPM-quantize bass_auto.mid
        disagree = measure_disagreement(
            song_dir / "semantic" / "bass_auto.mid",
            song_dir / "semantic" / "bass_warped.mid",
        )

        results.append({
            "slug": slug,
            "n_kicks": n_kicks,
            "bpm_median_kick": bpm,
            "total_notes": total_notes,
            "warped_count": warped_count,
            "warped_fraction": warped_count / total_notes if total_notes else 0,
            "max_shift_from_raw_ms": max_shift_ms,
            "vs_old_quantize": disagree,
        })
        print(f"  {slug}: kicks={n_kicks}, bpm={bpm:.1f}, warped {warped_count}/{total_notes}, "
              f"vs_old mean={disagree.get('shift_mean_ms', 0):.1f}ms max={disagree.get('shift_max_ms', 0):.1f}ms")

        # Replace bass_auto.mid with warped version
        import shutil
        shutil.copy(song_dir / "semantic" / "bass_warped.mid", song_dir / "semantic" / "bass_auto.mid")

        # Update JSON with new BPM
        meta_path = song_dir / "semantic" / "bass_auto.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["bpm_before_warp"] = meta.get("bpm")
            meta["bpm"] = bpm
            meta["warped_to_drum_grid"] = True
            meta_path.write_text(json.dumps(meta, indent=2, default=float))

        # Rebuild combined
        subprocess.run([".venv/bin/python", "scripts/76_combine_bass_drums.py", "--slug", slug],
                       cwd=REPO_ROOT, capture_output=True)
        # Fix tempos
        subprocess.run([".venv/bin/python", "scripts/77_fix_tempos.py", "--slug", slug],
                       cwd=REPO_ROOT, capture_output=True)

    # Save summary
    out_path = REPO_ROOT / "results" / "warp_corpus_summary.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out_path}")

    # Sort by impact (mean shift from old quantize)
    valid = [r for r in results if "vs_old_quantize" in r and r["vs_old_quantize"]]
    valid.sort(key=lambda r: -r["vs_old_quantize"].get("shift_mean_ms", 0))
    print("\n# Impact sorted (mean shift from old quantize → warp):")
    print(f"{'mean_ms':>8}  {'max_ms':>7}  {'warp%':>5}  {'BPM':>6}  slug")
    print("-" * 80)
    for r in valid:
        vs = r["vs_old_quantize"]
        wp = r["warped_fraction"] * 100
        bpm = r["bpm_median_kick"]
        print(f"{vs['shift_mean_ms']:8.1f}  {vs['shift_max_ms']:7.1f}  {wp:5.1f}  {bpm:6.2f}  {r['slug']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
