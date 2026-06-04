"""Diagnostic: show what pitches the bass stem ACTUALLY has, vs what pYIN reported.

For a chosen time window, compute a CQT (bass-friendly pitch representation) and
report the strongest 2-3 pitches per ~beat. Cross-reference against the pYIN MIDI.

This is for verifying the user's "the bass is pumping, not droning" observation
before committing to a basic-pitch install.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="beltram_machine")
    ap.add_argument("--start-s", type=float, default=100.0)
    ap.add_argument("--end-s", type=float, default=110.0)
    ap.add_argument("--beat-s", type=float, default=0.5,
                    help="report top pitches per this many seconds (≈ one beat at 120 BPM)")
    ap.add_argument("--top-k", type=int, default=3)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    song_dir = REPO_ROOT / "data" / "song_test" / args.slug
    stem_path = song_dir / "stems" / "bass.wav"
    midi_json = song_dir / "semantic" / "bass.json"

    y_st, sr = sf.read(stem_path, dtype="float32", always_2d=True)
    y = y_st.mean(axis=1).astype(np.float32) if y_st.shape[1] > 1 else y_st[:, 0]
    print(f"# bass pitch diagnostic · {args.start_s:.1f}-{args.end_s:.1f}s")
    print(f"  stem: {len(y)/sr:.1f}s · sr={sr}")

    i0 = int(args.start_s * sr)
    i1 = int(args.end_s * sr)
    clip = y[i0:i1]

    # CQT: 36 bins/octave, hop 256, fmin=C0 (16.35 Hz) → covers very low bass
    n_bins = 96  # 8 octaves × 12 semitones
    bins_per_octave = 12
    fmin_hz = librosa.note_to_hz("C0")
    cqt = np.abs(librosa.cqt(
        y=clip, sr=sr, hop_length=256, fmin=fmin_hz,
        n_bins=n_bins, bins_per_octave=bins_per_octave,
    ))
    # Bin i corresponds to MIDI note 12 + i (C0 is MIDI 12)
    midi_at_bin = np.arange(n_bins) + 12
    # Convert each frame to top-K MIDI notes by power
    frame_dur_s = 256 / sr
    n_frames = cqt.shape[1]

    print(f"  CQT: {n_bins} bins, {bins_per_octave}/oct, {n_frames} frames "
          f"({frame_dur_s*1000:.1f} ms/frame)")
    print()
    print(f"## ground-truth top-{args.top_k} pitches per {args.beat_s*1000:.0f} ms")
    print(f"  (CQT power; bass band MIDI 24-50 = C1-D3)")

    beat_frames = int(args.beat_s / frame_dur_s)
    pitch_counter: Counter = Counter()
    print(f"  {'t_s':>7}  {'top pitches (MIDI:note  pwr_norm)':<60}")
    for t0 in range(0, n_frames, beat_frames):
        t1 = min(t0 + beat_frames, n_frames)
        beat_power = cqt[:, t0:t1].mean(axis=1)
        # Restrict to bass range (MIDI 24..72 = C1..C5)
        bass_mask = (midi_at_bin >= 24) & (midi_at_bin <= 72)
        beat_power_bass = beat_power.copy()
        beat_power_bass[~bass_mask] = 0
        top_idx = np.argsort(beat_power_bass)[::-1][: args.top_k]
        total = float(beat_power_bass.sum()) + 1e-9
        parts = []
        for j in top_idx:
            p = int(midi_at_bin[j])
            norm = float(beat_power_bass[j]) / total
            if norm < 0.05:
                continue
            note = librosa.midi_to_note(p)
            parts.append(f"{p:>2}:{note}({norm:.2f})")
            pitch_counter[p] += 1
        clock = args.start_s + t0 * frame_dur_s
        print(f"  {clock:>7.2f}  {'  '.join(parts):<60}")

    print()
    print(f"## pitch histogram across the analyzed window")
    for p, count in sorted(pitch_counter.items(), key=lambda kv: -kv[1])[:8]:
        bar = "█" * count
        print(f"  MIDI {p:>3} ({librosa.midi_to_note(p):>3})  {count:>3}  {bar}")

    # Compare against pYIN MIDI for the same window
    if midi_json.exists():
        notes = json.loads(midi_json.read_text())
        in_window = [n for n in notes if args.start_s <= n["start_s"] <= args.end_s]
        pyin_pitches = Counter(int(n["pitch"]) for n in in_window)
        print()
        print(f"## pYIN-extracted MIDI in the same window ({len(in_window)} notes)")
        for p, count in sorted(pyin_pitches.items(), key=lambda kv: -kv[1])[:8]:
            bar = "█" * count
            print(f"  MIDI {p:>3} ({librosa.midi_to_note(p):>3})  {count:>3}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
