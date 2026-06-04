"""Resynthesize a bass stem from MIDI events + ONE exemplar audio sample.

Exemplar source (in priority order):
  1. Top-ranked polished candidate from bass_samples/candidates.json (script 37 output).
     This is the default: it's been scored on bass-band purity, pitch stability, etc.,
     and snapped to zero-crossings + fades so it retriggers cleanly.
  2. With --candidate-idx N, pick the Nth-ranked candidate instead.
  3. Fallback (--use-naive-picker, or candidates.json absent): the original "longest
     contiguous MIDI note" picker, which produced the "nasty" outro sustain on Beltram.

Algorithm:
  1. Load exemplar audio (polished from script 37 if available) + pitch.
  2. Load MIDI events from script 33.
  3. Pitch-shift the exemplar ONCE per unique target pitch (cache).
  4. For each note: stretch the pitched exemplar to the note duration, fade in/out,
     velocity-scale, place at start time.
  5. RMS-normalize to original bass stem, duplicate mono → stereo.

Outputs:
  data/song_test/<slug>/stems_resynth/bass.wav  (stereo, sample-aligned to stems)

Run:
  .venv/bin/python scripts/34_midi_to_bass_exemplar.py --slug beltram_machine
  .venv/bin/python scripts/34_midi_to_bass_exemplar.py --slug X --candidate-idx 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--fade-ms", type=float, default=10.0,
                    help="per-placed-note fade-in/out (in addition to exemplar's own fades)")
    ap.add_argument("--candidate-idx", type=int, default=0,
                    help="which rank from bass_samples/candidates.json to use as exemplar (0 = top)")
    ap.add_argument("--use-naive-picker", action="store_true",
                    help="ignore bass_samples/candidates.json and use the longest-MIDI-note picker")
    return ap.parse_args()


def fade_envelope(n_samples: int, fade_samples: int) -> np.ndarray:
    """Trapezoidal env: linear fade in, sustain at 1, linear fade out."""
    env = np.ones(n_samples, dtype=np.float32)
    f = min(fade_samples, n_samples // 2)
    if f > 0:
        env[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
        env[-f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
    return env


def fit_to_duration(audio: np.ndarray, target_n: int) -> np.ndarray:
    """Time-stretch audio (preserving pitch) to exactly target_n samples.
    librosa.effects.time_stretch operates on rate where rate>1 = faster (shorter).
    """
    src_n = len(audio)
    if src_n == 0 or target_n <= 0:
        return np.zeros(max(target_n, 0), dtype=np.float32)
    if abs(src_n - target_n) < 64:
        # Close enough — just pad/trim
        out = np.zeros(target_n, dtype=np.float32)
        m = min(src_n, target_n)
        out[:m] = audio[:m]
        return out
    rate = src_n / target_n
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) >= target_n:
        return stretched[:target_n].astype(np.float32, copy=False)
    out = np.zeros(target_n, dtype=np.float32)
    out[: len(stretched)] = stretched
    return out


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    stem_path = song_dir / "stems" / f"{args.stem_name}.wav"
    midi_json = song_dir / "semantic" / f"{args.stem_name}.json"
    out_path = song_dir / "stems_resynth" / f"{args.stem_name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# MIDI + exemplar → bass · slug={args.slug}")
    print(f"input stem: {stem_path}")
    print(f"input midi: {midi_json}")
    print(f"output:     {out_path}")

    # Load stem (need it for both exemplar source and length/RMS target)
    y_st, sr = sf.read(stem_path, dtype="float32", always_2d=True)
    n_samples = y_st.shape[0]
    y_mono = y_st.mean(axis=1).astype(np.float32) if y_st.shape[1] > 1 else y_st[:, 0]
    target_rms = float(np.sqrt(np.mean(y_mono ** 2)) + 1e-9)
    print(f"  stem: {n_samples/sr:.1f}s · sr={sr}  target_rms={target_rms:.4f}")

    notes = json.loads(midi_json.read_text())
    if not notes:
        raise SystemExit("no notes in MIDI json")
    print(f"  loaded {len(notes)} notes")

    # Pick exemplar: prefer script 37's polished candidates; fall back to naive picker.
    samples_dir = song_dir / "bass_samples"
    candidates_json = samples_dir / "candidates.json"
    exemplar: np.ndarray | None = None
    exemplar_pitch: int = -1
    exemplar_source: str = ""
    if candidates_json.exists() and not args.use_naive_picker:
        ranked = json.loads(candidates_json.read_text())
        if args.candidate_idx >= len(ranked):
            raise SystemExit(
                f"--candidate-idx {args.candidate_idx} but only {len(ranked)} candidates exist")
        chosen = ranked[args.candidate_idx]
        polished_meta = chosen.get("polished_sample")
        if polished_meta and (samples_dir / polished_meta["file"]).exists():
            ex_path = samples_dir / polished_meta["file"]
            ex_audio, ex_sr = sf.read(ex_path, dtype="float32", always_2d=False)
            if ex_sr != sr:
                raise SystemExit(f"exemplar sr {ex_sr} != stem sr {sr}")
            exemplar = ex_audio if ex_audio.ndim == 1 else ex_audio.mean(axis=1).astype(np.float32)
            exemplar_pitch = int(chosen["pitch"])
            exemplar_source = (f"polished candidate idx={args.candidate_idx} "
                              f"score={chosen['composite']:.3f} from {ex_path.name}")
        else:
            print("  WARN: candidates.json present but polished file missing; using naive picker")
    if exemplar is None:
        # Naive fallback: longest-duration MIDI note from the stem audio
        longest = max(notes, key=lambda n: n["end_s"] - n["start_s"])
        exi0 = int(longest["start_s"] * sr)
        exi1 = int(longest["end_s"] * sr)
        exemplar = y_mono[exi0:exi1].copy()
        exemplar_pitch = int(longest["pitch"])
        exemplar_source = (f"NAIVE longest-note picker "
                          f"(start {longest['start_s']:.2f}s, dur {(exi1-exi0)/sr:.2f}s)")
    print(f"  exemplar: pitch MIDI {exemplar_pitch} "
          f"({librosa.midi_to_note(exemplar_pitch)})  "
          f"dur {len(exemplar)/sr:.3f}s")
    print(f"    source: {exemplar_source}")

    # Pre-shift the exemplar to each unique target pitch (cache)
    unique_pitches = sorted({int(n["pitch"]) for n in notes})
    print(f"  unique target pitches: {len(unique_pitches)} "
          f"(MIDI {min(unique_pitches)}..{max(unique_pitches)})")

    print("  building pitch-shift cache ...")
    t0 = time.time()
    pitched: dict[int, np.ndarray] = {}
    for p in unique_pitches:
        n_steps = float(p - exemplar_pitch)
        if abs(n_steps) < 1e-3:
            pitched[p] = exemplar.copy()
        else:
            pitched[p] = librosa.effects.pitch_shift(exemplar, sr=sr, n_steps=n_steps)
    print(f"    cache built in {time.time()-t0:.1f}s "
          f"({len(unique_pitches)} pitches × ~{len(exemplar)/sr:.2f}s exemplar)")

    # Place each note
    fade_samples = int(args.fade_ms / 1000.0 * sr)
    out = np.zeros(n_samples, dtype=np.float32)
    n_clipped = 0
    for note in notes:
        start = int(note["start_s"] * sr)
        end = int(note["end_s"] * sr)
        if end > n_samples:
            end = n_samples
            n_clipped += 1
        dur = end - start
        if dur <= 0:
            continue
        src = pitched[int(note["pitch"])]
        seg = fit_to_duration(src, dur)
        env = fade_envelope(dur, fade_samples)
        vel_scale = note["velocity"] / 100.0  # script 33 normalized so 100 = full-track RMS
        out[start:end] += (seg * env * vel_scale).astype(np.float32)
    if n_clipped:
        print(f"  {n_clipped} notes clipped at stem-end (last-frame fencepost)")

    # Normalize RMS to original bass stem
    actual_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
    out *= (target_rms / actual_rms)
    # Light safety clip prevention
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out *= 0.99 / peak
        print(f"  WARN: peak {peak:.3f} > 0.99, scaled down to avoid clipping")

    # Duplicate to stereo (matches stem shape)
    out_stereo = np.stack([out, out], axis=1)
    sf.write(out_path, out_stereo, sr, subtype="FLOAT")
    print(f"  wrote {out_path.name}  "
          f"final_rms={float(np.sqrt(np.mean(out**2))):.4f}  "
          f"peak={float(np.max(np.abs(out))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
