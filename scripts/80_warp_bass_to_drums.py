"""Warp bass MIDI onset times to a non-uniform grid derived from drum kick onsets.

Replaces the fixed-BPM quantize for tracks where tempo drifts slightly (e.g.,
old analog drum machines / sequencers). Each kick becomes a grid landmark; bass
notes snap to the nearest 16th subdivision WITHIN their kick interval. Bass and
drums stay aligned no matter how the master clock wobbles.

Algorithm:
  1. Detect kick onsets in drums.wav (bandpass 40-150 Hz onset_detect).
  2. Estimate subdivisions-per-kick-interval from kick density and target grid.
  3. For each bass note at time t:
       - Find kick interval (kicks[i], kicks[i+1]) containing t.
       - Compute fractional position frac = (t - kicks[i]) / (kicks[i+1] - kicks[i]).
       - Snap frac to nearest 1/subdivisions.
       - New time = kicks[i] + snapped_frac * (kicks[i+1] - kicks[i]).
  4. For bass notes outside the kick coverage, extend the grid using the nearest
     two kicks at the relevant edge.
  5. Optionally apply --release-gap and --mono after warping.

Output: data/song_test/<slug>/semantic/bass_warped.mid

Run:
  .venv/bin/python scripts/80_warp_bass_to_drums.py --slug dmxkrew_101_tonight
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_kicks(drum_path: Path, band_lo: float = 40, band_hi: float = 150,
                 delta: float = 0.2, wait_ms: float = 100) -> np.ndarray:
    y, sr = sf.read(str(drum_path))
    if y.ndim > 1:
        y = y.mean(-1)
    sos = butter(4, [band_lo / (sr / 2), band_hi / (sr / 2)], "band", output="sos")
    yk = sosfiltfilt(sos, y)
    hop = 512
    on = librosa.onset.onset_detect(
        y=yk, sr=sr, hop_length=hop, backtrack=True,
        pre_max=4, post_max=4, pre_avg=20, post_avg=20,
        delta=delta, wait=int(wait_ms / 1000 * sr / hop),
    )
    return librosa.frames_to_time(on, sr=sr, hop_length=hop)


def warp_note(t: float, kicks: np.ndarray, subdivisions: int) -> float:
    """Snap time t to the nearest subdivision of its kick interval."""
    if len(kicks) < 2:
        return t
    # Find interval index
    i = np.searchsorted(kicks, t, side="right") - 1
    if i < 0:
        # before first kick: extend grid backwards using kicks[0], kicks[1]
        ref0, ref1 = kicks[0], kicks[1]
        frac = (t - ref0) / (ref1 - ref0)
    elif i >= len(kicks) - 1:
        # after last kick: extend forwards using kicks[-2], kicks[-1]
        ref0, ref1 = kicks[-2], kicks[-1]
        frac = (t - ref0) / (ref1 - ref0)
    else:
        ref0, ref1 = kicks[i], kicks[i + 1]
        frac = (t - ref0) / (ref1 - ref0)
    # Snap frac to nearest 1/subdivisions
    k = round(frac * subdivisions)
    snapped_frac = k / subdivisions
    return ref0 + snapped_frac * (ref1 - ref0)


def auto_subdivisions(median_kick_interval_s: float,
                       bpm_lo: float = 110.0, bpm_hi: float = 160.0,
                       target_bpm: float = 130.0) -> tuple[int, float, str]:
    """Pick subdivisions and report inferred BPM from median kick spacing.

    Candidates assume kicks land on: half / quarter / 8th / 16th notes.
    We want bass quantized at 16th-note resolution → subdivisions per kick =
    (16ths per bar) / (kicks per bar).

    Returns (subdivisions, bpm, kick_interpretation).
    """
    I = median_kick_interval_s
    if I <= 0:
        return 4, 0.0, "invalid"
    candidates = [
        # (kicks_per_bar, label, implied_bpm)
        (2, "half-note",     120 / I),   # kicks on every half-note
        (4, "quarter-note",  60 / I),    # 4-on-floor
        (8, "eighth-note",   30 / I),    # kicks on every 8th
        (16, "sixteenth",    15 / I),
    ]
    # Prefer candidates in [bpm_lo, bpm_hi]; tiebreak by closeness to target_bpm
    in_range = [c for c in candidates if bpm_lo <= c[2] <= bpm_hi]
    if in_range:
        chosen = min(in_range, key=lambda c: abs(c[2] - target_bpm))
    else:
        # nothing in range — pick the closest candidate to target_bpm
        chosen = min(candidates, key=lambda c: abs(c[2] - target_bpm))
    kicks_per_bar, label, bpm = chosen
    subs = 16 // kicks_per_bar  # 16ths per bar / kicks per bar
    return subs, bpm, label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--subdivisions", type=int, default=None,
                    help="grid subdivisions per kick interval. Default auto-detect from kick spacing.")
    ap.add_argument("--bpm-range", type=str, default="110,160",
                    help="acceptable BPM range for auto-detect: 'lo,hi'")
    ap.add_argument("--target-bpm", type=float, default=130.0,
                    help="when multiple interpretations fit, pick the one closest to this")
    ap.add_argument("--mono", action="store_true", default=True,
                    help="enforce monophonic (truncate overlaps)")
    ap.add_argument("--release-gap-ms", type=float, default=30.0)
    ap.add_argument("--out-name", default="bass_warped.mid",
                    help="output filename in semantic/ dir")
    ap.add_argument("--input", default="bass.mid",
                    help="input MIDI filename in semantic/ dir (raw extraction)")
    args = ap.parse_args()

    song_dir = REPO_ROOT / "data" / "song_test" / args.slug
    drum_path = song_dir / "stems" / "drums.wav"
    bass_in = song_dir / "semantic" / args.input
    bass_out = song_dir / "semantic" / args.out_name

    if not drum_path.exists() or not bass_in.exists():
        print("missing inputs", file=sys.stderr)
        return 1

    print(f"# warp · slug={args.slug}")
    kicks = detect_kicks(drum_path)
    print(f"  detected {len(kicks)} kicks")
    if len(kicks) > 5:
        ki = np.diff(kicks)
        med = float(np.median(ki))
        print(f"  inter-kick interval: median={med*1000:.1f}ms, std={ki.std()*1000:.1f}ms")
    else:
        med = 0.5

    # Auto subdivisions selection
    bpm_lo, bpm_hi = (float(x) for x in args.bpm_range.split(","))
    auto_subs, auto_bpm, kick_label = auto_subdivisions(med, bpm_lo, bpm_hi, args.target_bpm)
    if args.subdivisions is None:
        subs_used = auto_subs
        print(f"  auto: kicks-as-{kick_label} → {subs_used} sub/kick → BPM {auto_bpm:.2f}")
    else:
        subs_used = args.subdivisions
        print(f"  manual: {subs_used} sub/kick (auto would have picked {auto_subs} → BPM {auto_bpm:.2f})")

    n_warped = 0
    max_shift_ms = 0.0
    for inst in pm.instruments:
        for n in inst.notes:
            dur = n.end - n.start
            new_start = warp_note(n.start, kicks, subs_used)
            shift = abs(new_start - n.start) * 1000
            max_shift_ms = max(max_shift_ms, shift)
            if shift > 1:
                n_warped += 1
            n.start = new_start
            n.end = new_start + dur
    total_notes = sum(len(i.notes) for i in pm.instruments)
    print(f"  warped {n_warped}/{total_notes} notes; max shift {max_shift_ms:.1f}ms")

    # mono + release gap
    if args.mono:
        gap_s = args.release_gap_ms / 1000.0
        n_trunc = 0
        for inst in pm.instruments:
            notes = sorted(inst.notes, key=lambda n: n.start)
            for i in range(len(notes) - 1):
                target_end = notes[i + 1].start - gap_s
                target_end = max(target_end, notes[i].start + 0.001)
                if notes[i].end > target_end:
                    notes[i].end = target_end
                    n_trunc += 1
            inst.notes = notes
        print(f"  mono: truncated {n_trunc} overlaps (gap {args.release_gap_ms:.0f}ms)")

    approx_bpm = auto_bpm if auto_bpm > 0 else 120.0
    out_pm = pretty_midi.PrettyMIDI(initial_tempo=approx_bpm)
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name,
        )
        new_inst.notes = list(inst.notes)
        out_pm.instruments.append(new_inst)
    out_pm.write(str(bass_out))
    print(f"  wrote {bass_out} (approx BPM {approx_bpm:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
