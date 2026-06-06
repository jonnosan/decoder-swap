"""Auto-extract bass MIDI with per-track parameter sweep.

Discovers the right pipeline parameters per-track by sweeping a small grid and
scoring by bar-to-bar rhythm consistency. Eliminates the per-track manual
tuning we did for Beltram / DMX 201 / Fresh 02.

Steps per track:
  1. BPM via cross-correlation on the drum stem (with librosa fallback).
  2. Anchor via librosa.beat.beat_track first-beat on the drum stem.
  3. Extract MIDI ONCE with permissive onset settings (delta 0.05, wait 40ms,
     band 1500-5000 Hz). False positives expected; the scoring step filters
     them implicitly.
  4. Sweep: pre-merge ∈ {0, 60} × anchor-offset ∈ {-1, 0, +1} 16th-steps.
  5. Score each combo by mean pairwise Jaccard similarity of bar-position-sets
     across the dense bass region. Higher = more consistent rhythm = cleaner
     extraction.
  6. Save the winning quantized MIDI + chosen params + score.

Output per track:
  data/song_test/<slug>/semantic/bass_auto.mid     winning quantized MIDI
  data/song_test/<slug>/semantic/bass_auto.json    {bpm, anchor, params, score, ...}

Run:
  .venv/bin/python scripts/71_auto_extract.py --slug dmxkrew_201_17_ways_to_break_my_heart
  .venv/bin/python scripts/71_auto_extract.py --slug-pattern 'dmxkrew_*'   # batch
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------- BPM + anchor detection -------------------------


def detect_bpm_anchor_from_drums(drum_path: Path) -> tuple[float, float, dict]:
    """Return (bpm, anchor_s, diagnostics)."""
    y, sr = sf.read(str(drum_path))
    if y.ndim > 1:
        y = y.mean(-1)

    # Find first loud drum second (skip intro silence)
    win = sr
    energy = np.array(
        [np.sqrt(np.mean(y[s:s + win] ** 2)) for s in range(0, len(y) - win, win)]
    )
    loud_thresh = energy.max() * 0.3
    drum_start_s = int(np.argmax(energy >= loud_thresh))

    # Cross-correlate an 8s chunk after drums-start against a 60s search window
    chunk_start_s = drum_start_s + 5
    chunk_dur_s = 8.0
    if chunk_start_s + chunk_dur_s + 60 > len(y) / sr:
        # Track too short — fall back
        chunk_start_s = max(0, drum_start_s)
        chunk_dur_s = min(8.0, (len(y) / sr - chunk_start_s) / 2)
    ref = y[int(chunk_start_s * sr): int((chunk_start_s + chunk_dur_s) * sr)]
    search_start_s = chunk_start_s + 1.5
    search_dur_s = min(60.0, len(y) / sr - search_start_s)
    search = y[int(search_start_s * sr): int((search_start_s + search_dur_s) * sr)]
    bpm_cross = None
    bar_s_cross = None
    if len(ref) > 0 and len(search) > len(ref):
        corr = correlate(search, ref, mode="valid")
        peaks, _ = find_peaks(corr, distance=int(0.5 * sr))
        if len(peaks) >= 2:
            peak_offsets_s = peaks / sr
            abs_offsets_s = search_start_s + peak_offsets_s - chunk_start_s
            order = np.argsort(corr[peaks])[::-1]
            top_offsets = sorted(abs_offsets_s[order[:8]])
            inter = np.diff(top_offsets)
            # smallest non-trivial period is likely the bar (or sub-bar)
            inter_valid = [i for i in inter if i > 0.5]
            if inter_valid:
                bar_s_cross = float(min(inter_valid))
                bpm_cross = 4 * 60 / bar_s_cross

    # Librosa fallback / sanity check
    tempo_lib, beats_lib = librosa.beat.beat_track(y=y, sr=sr)
    tempo_lib = float(np.atleast_1d(tempo_lib)[0])
    first_beat_lib = (
        float(librosa.frames_to_time(beats_lib[0], sr=sr)) if len(beats_lib) else 0.0
    )

    # Pick BPM: cross-corr if it's "reasonable" (40-220 BPM), else librosa
    if bpm_cross is not None and 40 < bpm_cross < 220:
        # Sanity: should be within 25% of librosa for half/double ambiguity check
        ratio = bpm_cross / max(tempo_lib, 1)
        # If ratio is far from 1, 2, or 0.5, prefer librosa
        if any(abs(ratio - r) < 0.05 for r in (0.5, 1.0, 2.0)):
            bpm_final = bpm_cross
        elif abs(ratio - 1.0) < 0.1:
            bpm_final = bpm_cross
        else:
            # cross-corr is suspicious vs librosa; pick whichever is in [80, 170]
            cands = [b for b in [bpm_cross, tempo_lib, tempo_lib * 2, tempo_lib / 2]
                     if 80 <= b <= 170]
            bpm_final = cands[0] if cands else tempo_lib
    else:
        bpm_final = tempo_lib

    return bpm_final, first_beat_lib, {
        "bpm_cross": bpm_cross,
        "bpm_librosa": tempo_lib,
        "bar_s_cross": bar_s_cross,
        "drum_start_s": drum_start_s,
        "anchor_source": "librosa_drums_first_beat",
    }


# ------------------------- Extraction wrapper -------------------------


def run_extraction(
    slug: str, band_lo: float, band_hi: float, delta: float, wait_ms: float,
) -> None:
    """Run script 69 to extract MIDI; overwrites data/song_test/<slug>/semantic/bass.mid."""
    cmd = [
        ".venv/bin/python", "scripts/69_extract_bass_hybrid.py",
        "--slug", slug,
        "--onset-band-lo", str(band_lo),
        "--onset-band-hi", str(band_hi),
        "--onset-delta", str(delta),
        "--onset-wait-ms", str(wait_ms),
    ]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"extraction failed: {r.stderr}")


def run_quantize(
    midi_in: Path, midi_out: Path, bpm: float, anchor_s: float,
    pre_merge_ms: float, pitch_lo: int = 24, pitch_hi: int = 50,
) -> None:
    """Run script 67 to quantize."""
    cmd = [
        ".venv/bin/python", "scripts/67_quantize_midi.py",
        "--midi", str(midi_in),
        "--out", str(midi_out),
        "--bpm", str(bpm),
        "--grid", "16",
        "--pre-merge-ms", str(pre_merge_ms),
        "--pitch-lo", str(pitch_lo),
        "--pitch-hi", str(pitch_hi),
        "--mono",
        "--release-gap-ms", "30",
        "--anchor-s", str(anchor_s),
    ]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"quantize failed: {r.stderr}")


# ------------------------- Scoring -------------------------


def find_dense_bass_region(bass_path: Path, anchor_s: float, bar_s: float, region_dur_s: float = 60) -> float:
    """Return bar-aligned start time of densest bass region."""
    x, sr = sf.read(str(bass_path))
    if x.ndim > 1:
        x = x.mean(-1)
    win = int(region_dur_s * sr)
    step = int(15 * sr)
    best_rms, best_start = 0.0, int(anchor_s * sr)
    for s in range(int(anchor_s * sr), max(int(anchor_s * sr) + 1, len(x) - win), step):
        chunk = x[s:s + win]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_start = s
    n_bars = round((best_start / sr - anchor_s) / bar_s)
    return anchor_s + n_bars * bar_s


def score_quantized(midi_path: Path, bpm: float, anchor_s: float,
                    region_start: float, region_dur_s: float) -> tuple[float, dict]:
    """Score by mean pairwise Jaccard similarity of bar-position-sets in the dense region."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if not pm.instruments or not pm.instruments[0].notes:
        return 0.0, {"n_bars": 0, "reason": "no_notes"}
    notes = pm.instruments[0].notes
    bar_s = 60.0 / bpm * 4
    step_s = bar_s / 16

    # Build per-bar position sets (rhythm only, ignore pitch)
    bars: list[set[int]] = []
    t = region_start
    region_end = region_start + region_dur_s
    while t + bar_s <= region_end:
        positions = set()
        for n in notes:
            if t <= n.start < t + bar_s:
                idx = round((n.start - t) / step_s) % 16
                positions.add(idx)
        bars.append(positions)
        t += bar_s

    if len(bars) < 4:
        return 0.0, {"n_bars": len(bars), "reason": "too_few_bars"}

    # Pairwise Jaccard
    sims = []
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            u = bars[i] | bars[j]
            if not u:
                continue
            sims.append(len(bars[i] & bars[j]) / len(u))
    if not sims:
        return 0.0, {"n_bars": len(bars), "reason": "no_pairs"}
    mean_sim = float(np.mean(sims))

    # Also report: histogram peakiness
    activity = [0] * 16
    for b in bars:
        for p in b:
            activity[p] += 1
    activity = [a / len(bars) for a in activity]
    n_high = sum(1 for a in activity if a >= 0.7)
    n_mid = sum(1 for a in activity if 0.3 <= a < 0.7)
    return mean_sim, {
        "n_bars": len(bars),
        "mean_jaccard": mean_sim,
        "n_high_activity": n_high,
        "n_mid_activity": n_mid,
        "activity": [round(a, 2) for a in activity],
    }


# ------------------------- Top-level auto-extract -------------------------


def auto_extract(slug: str, verbose: bool = True) -> dict:
    """Auto-extract MIDI for one track. Returns dict of results."""
    song_dir = REPO_ROOT / "data" / "song_test" / slug
    bass_path = song_dir / "stems" / "bass.wav"
    drum_path = song_dir / "stems" / "drums.wav"
    if not bass_path.exists() or not drum_path.exists():
        return {"slug": slug, "error": "missing stems"}

    out_dir = song_dir / "semantic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_midi = out_dir / "bass_auto.mid"
    out_meta = out_dir / "bass_auto.json"

    t0 = time.time()
    if verbose:
        print(f"\n=== {slug} ===")

    # 1) BPM + anchor
    bpm, anchor_s, bpm_diag = detect_bpm_anchor_from_drums(drum_path)
    bar_s = 60.0 / bpm * 4
    step_s = bar_s / 16
    if verbose:
        print(f"  BPM={bpm:.3f}  bar_s={bar_s:.4f}  anchor={anchor_s:.4f}")
        print(f"    diag: bpm_cross={bpm_diag.get('bpm_cross')}  bpm_librosa={bpm_diag.get('bpm_librosa'):.2f}")

    # 2) Find dense bass region
    region_start = find_dense_bass_region(bass_path, anchor_s, bar_s, 60)
    region_dur = 60.0
    if verbose:
        print(f"  dense region: {region_start:.2f}-{region_start + region_dur:.2f}s")

    # 3) Extract once with permissive settings
    try:
        run_extraction(slug, 1500, 5000, 0.05, 40)
    except Exception as e:
        return {"slug": slug, "error": f"extraction failed: {e}"}
    raw_midi = song_dir / "semantic" / "bass.mid"
    if verbose:
        try:
            pm = pretty_midi.PrettyMIDI(str(raw_midi))
            print(f"  raw extraction: {len(pm.instruments[0].notes)} notes")
        except Exception:
            pass

    # 4) Sweep
    best = {"score": -1.0}
    sweep_results = []
    for pre_merge_ms in [0, 60]:
        for offset_steps in [-1, 0, 1]:
            anchor_used = anchor_s + offset_steps * step_s
            tmp_midi = Path(f"/tmp/sweep_{slug}_pm{pre_merge_ms}_off{offset_steps}.mid")
            try:
                run_quantize(raw_midi, tmp_midi, bpm, anchor_used, pre_merge_ms)
                score, stats = score_quantized(tmp_midi, bpm, anchor_used, region_start, region_dur)
            except Exception as e:
                sweep_results.append({"pre_merge_ms": pre_merge_ms, "offset_steps": offset_steps, "error": str(e)})
                continue
            sweep_results.append({
                "pre_merge_ms": pre_merge_ms,
                "offset_steps": offset_steps,
                "score": round(score, 4),
                "n_bars": stats.get("n_bars"),
                "n_high_activity": stats.get("n_high_activity"),
                "n_mid_activity": stats.get("n_mid_activity"),
            })
            if score > best["score"]:
                best = {
                    "score": score, "pre_merge_ms": pre_merge_ms,
                    "offset_steps": offset_steps, "anchor_used": anchor_used,
                    "midi_path": tmp_midi, "stats": stats,
                }
    if "midi_path" not in best:
        return {"slug": slug, "error": "no valid sweep result", "sweep": sweep_results}

    # 5) Save winning MIDI + metadata
    import shutil
    shutil.copy(best["midi_path"], out_midi)
    meta = {
        "slug": slug,
        "bpm": bpm,
        "anchor_s": anchor_s,
        "anchor_used": best["anchor_used"],
        "anchor_offset_steps": best["offset_steps"],
        "pre_merge_ms": best["pre_merge_ms"],
        "region_start": region_start,
        "region_dur_s": region_dur,
        "score": round(best["score"], 4),
        "stats": best["stats"],
        "bpm_diag": {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in bpm_diag.items()},
        "sweep_all": sweep_results,
        "elapsed_s": round(time.time() - t0, 1),
        "out_midi": str(out_midi),
    }
    out_meta.write_text(json.dumps(meta, indent=2, default=float))
    if verbose:
        print(f"  WINNER: pre_merge={best['pre_merge_ms']}ms offset={best['offset_steps']} score={best['score']:.3f}")
        print(f"  stats: {best['stats'].get('n_high_activity')} high-activity positions, {best['stats'].get('n_mid_activity')} mid")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="single track slug")
    ap.add_argument("--slug-pattern", help="glob pattern (e.g. 'dmxkrew_*')")
    ap.add_argument("--out-summary", default=str(REPO_ROOT / "results" / "auto_extract_summary.json"))
    args = ap.parse_args()

    slugs: list[str] = []
    if args.slug:
        slugs = [args.slug]
    elif args.slug_pattern:
        for p in sorted(glob.glob(str(REPO_ROOT / "data" / "song_test" / args.slug_pattern))):
            slugs.append(Path(p).name)
    else:
        print("must supply --slug or --slug-pattern", file=sys.stderr)
        return 1

    results = []
    for slug in slugs:
        try:
            meta = auto_extract(slug)
        except Exception as e:
            meta = {"slug": slug, "error": str(e)}
        results.append(meta)

    # Summary
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(results, indent=2, default=float))
    print(f"\n# Summary saved to {args.out_summary}")
    print(f"# {len(results)} tracks processed")
    succeeded = [r for r in results if "score" in r]
    failed = [r for r in results if "error" in r]
    print(f"# {len(succeeded)} succeeded, {len(failed)} failed")
    if succeeded:
        scores = [r["score"] for r in succeeded]
        print(f"# scores: min={min(scores):.3f} median={sorted(scores)[len(scores)//2]:.3f} max={max(scores):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
