"""Hybrid bass extractor: librosa onset_detect for timing + pYIN for pitch.

Why hybrid: pYIN's pitch tracking is excellent (clean fundamental detection,
no harmonic confusion) but its note-boundary detection is based on pitch
transitions, which scatters onsets by ±10-50 ms relative to the attack
transient the listener perceives. librosa.onset.onset_detect locks to attack
transients directly — much tighter timing.

So we run them separately and combine:
  - librosa.onset.onset_detect on the audio → onset times
  - librosa.pyin on the audio → pitch contour
  - for each onset: pitch = median of pYIN contour in a small window after onset
  - velocity = local RMS in window after onset
  - note end = next onset (or fixed short length for last note)

Result: one MIDI note per perceived attack, with pYIN-clean pitch.

Run:
  .venv/bin/python scripts/69_extract_bass_hybrid.py --slug mayday_d1t02_beltram_machine
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pretty_midi


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Note:
    start_s: float
    end_s: float
    pitch: int
    velocity: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--fmin", type=float, default=30.0)
    ap.add_argument("--fmax", type=float, default=200.0)
    ap.add_argument("--pyin-frame-length", type=int, default=4096)
    ap.add_argument("--pyin-hop-length", type=int, default=512)
    ap.add_argument("--onset-pre-max", type=float, default=0.03,
                    help="pre-max window for onset peak picking (seconds)")
    ap.add_argument("--onset-post-max", type=float, default=0.03,
                    help="post-max window for onset peak picking (seconds)")
    ap.add_argument("--onset-pre-avg", type=float, default=0.10,
                    help="pre-avg window for onset peak picking (seconds)")
    ap.add_argument("--onset-post-avg", type=float, default=0.10,
                    help="post-avg window for onset peak picking (seconds)")
    ap.add_argument("--onset-delta", type=float, default=0.07,
                    help="onset peak-picking threshold; higher = fewer onsets")
    ap.add_argument("--onset-wait-ms", type=float, default=40.0,
                    help="minimum gap between onsets (ms)")
    ap.add_argument("--pitch-window-ms", type=float, default=80.0,
                    help="window after onset to compute pitch from (median of pYIN)")
    ap.add_argument("--velocity-window-ms", type=float, default=50.0,
                    help="window after onset to compute velocity from (RMS)")
    ap.add_argument("--onset-band-lo", type=float, default=None,
                    help="bandpass low Hz for onset detection. Synth bass with a constant tone has no overall-amplitude attacks, but filter-envelope hits show up as transient energy in 500-3000 Hz. Try 500.")
    ap.add_argument("--onset-band-hi", type=float, default=None,
                    help="bandpass high Hz for onset detection. Try 3000 for TB-303-style filter resonance.")
    return ap.parse_args()


def bandpass_filter(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Simple butterworth bandpass via scipy."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    sos = butter(N=4, Wn=[lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, y)


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    stem_path = song_dir / "stems" / f"{args.stem_name}.wav"
    out_dir = song_dir / "semantic"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem_name}.json"
    midi_path = out_dir / f"{args.stem_name}.mid"

    print(f"# hybrid bass→MIDI · slug={args.slug} · stem={args.stem_name}")
    print(f"  input: {stem_path}")

    y, sr = librosa.load(str(stem_path), sr=None, mono=True)
    print(f"  loaded {len(y)/sr:.1f}s · sr={sr}")

    # 1) Onset detection — optionally on a bandpassed copy of the signal
    onset_hop = 512
    if args.onset_band_lo and args.onset_band_hi:
        y_for_onsets = bandpass_filter(y, sr, args.onset_band_lo, args.onset_band_hi)
        print(f"  onset detection on bandpass {args.onset_band_lo:.0f}-{args.onset_band_hi:.0f} Hz")
    else:
        y_for_onsets = y
    pre_max = max(1, int(args.onset_pre_max * sr / onset_hop))
    post_max = max(1, int(args.onset_post_max * sr / onset_hop))
    pre_avg = max(1, int(args.onset_pre_avg * sr / onset_hop))
    post_avg = max(1, int(args.onset_post_avg * sr / onset_hop))
    wait = max(1, int(args.onset_wait_ms / 1000.0 * sr / onset_hop))
    onset_frames = librosa.onset.onset_detect(
        y=y_for_onsets, sr=sr, hop_length=onset_hop, backtrack=True,
        pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg,
        delta=args.onset_delta, wait=wait,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=onset_hop)
    print(f"  onsets detected: {len(onset_times)}")

    # 2) pYIN pitch contour
    print(f"  running pyin (fmin={args.fmin}, fmax={args.fmax}) ...")
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=args.fmin, fmax=args.fmax, sr=sr,
        frame_length=args.pyin_frame_length, hop_length=args.pyin_hop_length,
        fill_na=np.nan,
    )
    pyin_dt = args.pyin_hop_length / sr
    pyin_times = np.arange(len(f0)) * pyin_dt

    def pitch_at(t_s: float, win_ms: float) -> int:
        """Median pYIN-MIDI-pitch in [t_s, t_s + win_ms]. Returns -1 if no voiced frames."""
        t1 = t_s + win_ms / 1000.0
        mask = (pyin_times >= t_s) & (pyin_times <= t1)
        hz = f0[mask]
        vf = voiced_flag[mask]
        hz_good = hz[vf & np.isfinite(hz)]
        if len(hz_good) == 0:
            return -1
        median_hz = float(np.median(hz_good))
        midi = int(round(69.0 + 12.0 * np.log2(median_hz / 440.0)))
        return midi

    # 3) Velocity from local RMS
    full_rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    def velocity_at(t_s: float, win_ms: float) -> int:
        i0 = int(t_s * sr)
        i1 = min(len(y), i0 + int(win_ms / 1000.0 * sr))
        if i1 <= i0:
            return 1
        rms = float(np.sqrt(np.mean(y[i0:i1] ** 2)))
        return int(np.clip(round(100.0 * rms / full_rms), 1, 127))

    # 4) Build notes — one per onset, ending at next onset
    notes: list[Note] = []
    for i, t in enumerate(onset_times):
        end_t = float(onset_times[i+1]) if i + 1 < len(onset_times) else min(len(y)/sr, t + 0.5)
        pitch = pitch_at(float(t), args.pitch_window_ms)
        if pitch < 0:
            continue  # no pitch detected at this onset — drop
        vel = velocity_at(float(t), args.velocity_window_ms)
        notes.append(Note(start_s=float(t), end_s=float(end_t), pitch=pitch, velocity=vel))

    print(f"  built {len(notes)} notes (skipped {len(onset_times) - len(notes)} pitchless onsets)")

    # Stats
    if notes:
        from collections import Counter
        pc = Counter(n.pitch for n in notes)
        print("  top pitches:")
        for p, c in pc.most_common(6):
            name = librosa.midi_to_note(p)
            print(f"    MIDI {p:3d} ({name}): {c}")
        durs = np.array([n.end_s - n.start_s for n in notes])
        print(f"  duration: median {1000*np.median(durs):.0f} ms, min {1000*durs.min():.0f}, max {1000*durs.max():.0f}")

    # Write JSON
    json_path.write_text(json.dumps([asdict(n) for n in notes], indent=2))

    # Write MIDI
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=33, name=args.stem_name)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch,
            start=n.start_s, end=n.end_s,
        ))
    pm.instruments.append(inst)
    pm.write(str(midi_path))
    print(f"  wrote {json_path.name} and {midi_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
