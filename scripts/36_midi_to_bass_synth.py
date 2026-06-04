"""Resynthesize a bass stem from MIDI events using a simple subtractive synth.

This is the "clean reference" alternative to script 34 (exemplar pitch-shift). Goal:
isolate "is the MIDI extraction correct?" from "does the exemplar resynth fit the
corpus?". If this version sounds wrong rhythmically/melodically, the MIDI is the
problem; if it just sounds generic, the MIDI is fine and resynth quality is the issue.

Voice (canonical 90s techno bass):
  - Sawtooth oscillator at the note pitch (anti-aliased not necessary in bass range).
  - ADSR amplitude envelope (A=5ms, D=80ms, S=0.7, R=50ms by default).
  - Global low-pass filter (Butterworth, default 1200 Hz, order 4) applied to the
    summed render — gives the muffled "subby" character of a hardware analog bass.
  - Velocity scales amplitude.

Outputs:
  data/song_test/<slug>/stems_synth/bass.wav  (stereo, sample-aligned to stems)

Run:
  .venv/bin/python scripts/36_midi_to_bass_synth.py --slug beltram_machine
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--attack-ms", type=float, default=5.0)
    ap.add_argument("--decay-ms", type=float, default=80.0)
    ap.add_argument("--sustain", type=float, default=0.7)
    ap.add_argument("--release-ms", type=float, default=50.0)
    ap.add_argument("--lp-cutoff", type=float, default=1200.0,
                    help="global low-pass filter cutoff in Hz (Butterworth, order 4)")
    return ap.parse_args()


def saw_wave(n: int, freq_hz: float, sr: int) -> np.ndarray:
    """Bipolar sawtooth, n samples at constant freq_hz, starting at phase 0."""
    t = np.arange(n, dtype=np.float64) / sr
    phase = freq_hz * t
    return (2.0 * (phase - np.floor(phase + 0.5))).astype(np.float32)


def saw_wave_bent(n: int, base_freq_hz: float, sr: int,
                  bend_times_s: np.ndarray, bend_cents: np.ndarray) -> np.ndarray:
    """Bipolar sawtooth with time-varying pitch driven by a per-note bend curve.

    bend_times_s, bend_cents — sample points of a bend curve (cents offset from
    base_freq_hz) measured from the note start. Linearly interpolated to per-sample
    cents, converted to frequency multiplier, accumulated to instantaneous phase.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    cents_per_sample = np.interp(t, bend_times_s, bend_cents,
                                 left=float(bend_cents[0]),
                                 right=float(bend_cents[-1]))
    freq_mult = np.power(2.0, cents_per_sample / 1200.0)
    freq_per_sample = base_freq_hz * freq_mult
    # Phase accumulation: phase[k] = sum_{i<=k} freq[i] / sr
    phase = np.cumsum(freq_per_sample) / sr
    return (2.0 * (phase - np.floor(phase + 0.5))).astype(np.float32)


def adsr_envelope(n: int, sr: int, a_ms: float, d_ms: float,
                  sustain: float, r_ms: float) -> np.ndarray:
    a = int(a_ms / 1000.0 * sr)
    d = int(d_ms / 1000.0 * sr)
    r = int(r_ms / 1000.0 * sr)
    env = np.zeros(n, dtype=np.float32)
    pos = 0
    # Attack
    if a > 0:
        seg = min(a, n - pos)
        env[pos:pos + seg] = np.linspace(0.0, 1.0, seg, dtype=np.float32)
        pos += seg
    # Decay
    if d > 0 and pos < n:
        seg = min(d, n - pos)
        env[pos:pos + seg] = np.linspace(1.0, sustain, seg, dtype=np.float32)
        pos += seg
    # Sustain (constant) — everything between decay end and release start
    sustain_end = max(pos, n - r)
    if sustain_end > pos:
        env[pos:sustain_end] = sustain
        pos = sustain_end
    # Release
    if r > 0 and pos < n:
        seg = n - pos
        env[pos:] = np.linspace(env[pos - 1] if pos > 0 else sustain, 0.0, seg, dtype=np.float32)
    return env


def midi_to_hz(p: int) -> float:
    return float(440.0 * (2.0 ** ((p - 69) / 12.0)))


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    stem_path = song_dir / "stems" / f"{args.stem_name}.wav"
    midi_json = song_dir / "semantic" / f"{args.stem_name}.json"
    out_path = song_dir / "stems_synth" / f"{args.stem_name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# MIDI → synth bass · slug={args.slug}")
    print(f"input midi: {midi_json}")
    print(f"output:     {out_path}")

    # Use the original stem only for length + RMS-matching
    y_st, sr = sf.read(stem_path, dtype="float32", always_2d=True)
    n_samples = y_st.shape[0]
    y_mono = y_st.mean(axis=1).astype(np.float32) if y_st.shape[1] > 1 else y_st[:, 0]
    target_rms = float(np.sqrt(np.mean(y_mono ** 2)) + 1e-9)
    print(f"  length: {n_samples/sr:.1f}s · sr={sr}  target_rms={target_rms:.4f}")

    notes = json.loads(midi_json.read_text())
    if not notes:
        raise SystemExit("no notes in MIDI json")
    print(f"  loaded {len(notes)} notes")
    print(f"  synth: saw + ADSR(a={args.attack_ms}ms d={args.decay_ms}ms "
          f"s={args.sustain} r={args.release_ms}ms) + LP{args.lp_cutoff:.0f}Hz")

    out = np.zeros(n_samples, dtype=np.float32)
    n_with_bend = 0
    for note in notes:
        start = int(note["start_s"] * sr)
        end = int(note["end_s"] * sr)
        if end > n_samples:
            end = n_samples
        dur = end - start
        if dur <= 0:
            continue
        freq = midi_to_hz(int(note["pitch"]))
        pb = note.get("pitch_bends")
        if pb:
            bend_t = np.array([float(ev["time_s"]) for ev in pb], dtype=np.float64)
            bend_c = np.array([float(ev["cents"]) for ev in pb], dtype=np.float64)
            wave = saw_wave_bent(dur, freq, sr, bend_t, bend_c)
            n_with_bend += 1
        else:
            wave = saw_wave(dur, freq, sr)
        env = adsr_envelope(dur, sr, args.attack_ms, args.decay_ms,
                            args.sustain, args.release_ms)
        vel_scale = note["velocity"] / 100.0
        out[start:end] += (wave * env * vel_scale * 0.5).astype(np.float32)
    print(f"  rendered {len(notes)} notes ({n_with_bend} with pitchbend)")

    # Global LP filter (Butterworth order 4) → muffled techno-bass character
    sos = butter(4, args.lp_cutoff, btype="low", fs=sr, output="sos")
    out = sosfilt(sos, out).astype(np.float32)

    # Normalize RMS to match original stem
    actual_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
    out *= (target_rms / actual_rms)
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out *= 0.99 / peak
        print(f"  WARN: peak {peak:.3f} > 0.99, scaled down to avoid clipping")

    out_stereo = np.stack([out, out], axis=1)
    sf.write(out_path, out_stereo, sr, subtype="FLOAT")
    print(f"  wrote {out_path.name}  "
          f"final_rms={float(np.sqrt(np.mean(out**2))):.4f}  "
          f"peak={float(np.max(np.abs(out))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
