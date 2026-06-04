"""Extract MIDI notes from an isolated bass stem.

Two backends:
  basic-pitch  Spotify's polyphonic audio→MIDI model (DEFAULT). Catches octave
               jumps / chord-tones; right tool for "pumping" basslines where
               pYIN collapses everything to one fundamental.
  pyin         librosa pyin, monophonic. Faster, no neural deps, but misses
               polyphony — only safe for clearly monophonic stems.

Outputs (regardless of backend):
  data/song_test/<slug>/semantic/bass.json   list of {start_s, end_s, pitch, velocity}
  data/song_test/<slug>/semantic/bass.mid    standard MIDI file

Run:
  .venv/bin/python scripts/33_bass_to_midi.py --slug beltram_machine
  .venv/bin/python scripts/33_bass_to_midi.py --slug X --detector pyin
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
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class Note:
    start_s: float
    end_s: float
    pitch: int       # MIDI pitch
    velocity: int    # 1..127
    # Per-note pitch-bend curve as a list of (time_s, cents) pairs. cents is offset
    # from the note's nominal pitch. Empty list = no bend (constant pitch).
    pitch_bends: list[dict] | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--detector", choices=["basic-pitch", "pyin"], default="basic-pitch",
                    help="audio→MIDI backend (basic-pitch is polyphonic; pyin is monophonic)")
    # pYIN options (only used when --detector pyin)
    ap.add_argument("--fmin", type=float, default=40.0)
    ap.add_argument("--fmax", type=float, default=400.0)
    ap.add_argument("--frame-length", type=int, default=4096)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--min-note-ms", type=float, default=50.0)
    # basic-pitch options
    ap.add_argument("--bp-onset-thr", type=float, default=0.5,
                    help="basic-pitch onset threshold (lower = more notes)")
    ap.add_argument("--bp-frame-thr", type=float, default=0.3,
                    help="basic-pitch frame threshold (lower = more notes / longer)")
    ap.add_argument("--bp-pitch-lo", type=int, default=24,
                    help="basic-pitch minimum MIDI pitch (C1 = 24, covers sub-bass)")
    ap.add_argument("--bp-pitch-hi", type=int, default=50,
                    help="basic-pitch maximum MIDI pitch (D3 = 50; bass rarely goes higher)")
    return ap.parse_args()


def hz_to_midi_int(hz: float) -> int:
    if hz <= 0 or not np.isfinite(hz):
        return -1
    return int(round(69.0 + 12.0 * np.log2(hz / 440.0)))


def rms_window(audio: np.ndarray, sr: int, t0_s: float, t1_s: float) -> float:
    i0 = max(0, int(t0_s * sr))
    i1 = min(len(audio), int(t1_s * sr))
    if i1 <= i0:
        return 0.0
    return float(np.sqrt(np.mean(audio[i0:i1] ** 2)))


def extract_notes_pyin(
    audio: np.ndarray, sr: int, fmin: float, fmax: float,
    frame_length: int, hop_length: int, min_note_ms: float,
) -> list[Note]:
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length, fill_na=np.nan,
    )
    midi_per_frame = np.array(
        [hz_to_midi_int(float(h)) if (v and np.isfinite(h)) else -1
         for h, v in zip(f0, voiced_flag, strict=False)],
        dtype=np.int32,
    )
    frame_dt_s = hop_length / sr
    rms_full = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    notes: list[Note] = []
    n = len(midi_per_frame)
    i = 0
    while i < n:
        p = int(midi_per_frame[i])
        if p < 0:
            i += 1
            continue
        j = i + 1
        while j < n and int(midi_per_frame[j]) == p:
            j += 1
        start_s = i * frame_dt_s
        end_s = j * frame_dt_s
        if (end_s - start_s) * 1000.0 < min_note_ms:
            i = j
            continue
        loudness = rms_window(audio, sr, start_s, end_s)
        velocity = int(np.clip(round(100.0 * loudness / rms_full), 1, 127))
        notes.append(Note(start_s=start_s, end_s=end_s, pitch=p, velocity=velocity))
        i = j
    return notes


def extract_notes_basic_pitch(
    audio_path: Path,
    onset_thr: float, frame_thr: float,
    pitch_lo: int, pitch_hi: int,
) -> list[Note]:
    """Run basic-pitch on the file, return polyphonic note list filtered to [pitch_lo, pitch_hi].

    Captures per-note pitchbend curves: basic-pitch's note_events[4] is a list of
    integer bin offsets (1/3 semitone per bin = ~33.33 cents) sampled at
    ANNOTATIONS_FPS (~86 Hz). We translate to (time_s, cents) pairs relative to
    the note start.
    """
    # Shim removed scipy API: basic-pitch 0.3 calls scipy.signal.gaussian which was moved
    # to scipy.signal.windows in scipy 1.13+. Patch before importing basic-pitch's predict.
    import scipy.signal
    if not hasattr(scipy.signal, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian
        scipy.signal.gaussian = _gaussian  # type: ignore[attr-defined]
    # The default ICASSP_2022_MODEL_PATH points at the TensorFlow saved-model, which fails
    # to load under TF 2.16 (Keras 3 incompatibility). Force the ONNX backend instead —
    # basic-pitch ships an ONNX model file in the same dir.
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.constants import ANNOTATIONS_FPS, CONTOURS_BINS_PER_SEMITONE
    from basic_pitch.inference import predict
    onnx_model = build_icassp_2022_model_path(FilenameSuffix.onnx)
    _model_output, _midi_pm, note_events = predict(
        str(audio_path),
        model_or_model_path=onnx_model,
        onset_threshold=onset_thr,
        frame_threshold=frame_thr,
        minimum_note_length=50.0,    # ms
        minimum_frequency=librosa.midi_to_hz(pitch_lo),
        maximum_frequency=librosa.midi_to_hz(pitch_hi),
        multiple_pitch_bends=True,   # extract bends for every note, not just one
    )
    frame_dt_s = 1.0 / ANNOTATIONS_FPS
    cents_per_bin = 100.0 / float(CONTOURS_BINS_PER_SEMITONE)
    # note_events: list of (start_s, end_s, pitch_midi, velocity[0,1], pitch_bends)
    out: list[Note] = []
    for ev in note_events:
        start_s = float(ev[0])
        end_s = float(ev[1])
        pitch = int(ev[2])
        amp = float(ev[3])  # 0..1
        bend_bins = ev[4] if len(ev) > 4 else None  # list[int] in 1/3-semitone units, or None
        if pitch < pitch_lo or pitch > pitch_hi:
            continue
        velocity = int(np.clip(round(amp * 127), 1, 127))
        pb_pairs: list[dict] | None = None
        if bend_bins:
            pb_pairs = [
                {"time_s": float(i * frame_dt_s), "cents": float(int(b) * cents_per_bin)}
                for i, b in enumerate(bend_bins)
            ]
        out.append(Note(
            start_s=start_s, end_s=end_s, pitch=pitch, velocity=velocity,
            pitch_bends=pb_pairs,
        ))
    out.sort(key=lambda n: (n.start_s, n.pitch))
    return out


def write_midi(notes: list[Note], path: Path, program: int = 33,
               pitch_bend_range_semitones: float = 2.0) -> None:
    """program=33 = Electric Bass (finger) in General MIDI.

    pitch_bend_range_semitones: standard MIDI assumes ±2 semitones unless the
    Pitch Bend Sensitivity RPN is set. We clamp |cents| to the chosen range and
    map to the 14-bit pitch-bend value (-8192..8191).
    """
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=program, is_drum=False, name="bass")
    pb_range_cents = pitch_bend_range_semitones * 100.0
    for n in notes:
        instr.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch, start=n.start_s, end=n.end_s,
        ))
        if not n.pitch_bends:
            continue
        for ev in n.pitch_bends:
            cents = float(ev["cents"])
            clamped = max(-pb_range_cents, min(pb_range_cents, cents))
            pb_val = int(round(clamped / pb_range_cents * 8191))
            instr.pitch_bends.append(pretty_midi.PitchBend(
                pitch=pb_val, time=n.start_s + float(ev["time_s"]),
            ))
    pm.instruments.append(instr)
    pm.write(str(path))


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    stem_path = song_dir / "stems" / f"{args.stem_name}.wav"
    out_dir = song_dir / "semantic"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem_name}.json"
    midi_path = out_dir / f"{args.stem_name}.mid"

    print(f"# bass → MIDI · slug={args.slug} · stem={args.stem_name} · detector={args.detector}")
    print(f"input:  {stem_path}")
    print(f"output: {json_path}, {midi_path}")

    if args.detector == "pyin":
        y, sr = sf.read(stem_path, dtype="float32", always_2d=True)
        y = y.mean(axis=1).astype(np.float32) if y.shape[1] > 1 else y[:, 0].astype(np.float32)
        print(f"  loaded {len(y)/sr:.1f} s · sr={sr} · mono")
        print("  running pyin ...")
        notes = extract_notes_pyin(
            y, sr, args.fmin, args.fmax,
            args.frame_length, args.hop_length, args.min_note_ms,
        )
    else:
        print("  running basic-pitch ...")
        notes = extract_notes_basic_pitch(
            stem_path, args.bp_onset_thr, args.bp_frame_thr,
            args.bp_pitch_lo, args.bp_pitch_hi,
        )

    if not notes:
        print("  WARN: no notes detected")
    else:
        pitches = [n.pitch for n in notes]
        durs_ms = [(n.end_s - n.start_s) * 1000.0 for n in notes]
        vels = [n.velocity for n in notes]
        from collections import Counter
        pcount = Counter(pitches)
        print(f"  {len(notes)} notes  "
              f"pitch range MIDI {min(pitches)}..{max(pitches)} "
              f"({librosa.midi_to_note(min(pitches))}..{librosa.midi_to_note(max(pitches))})")
        print(f"  pitch histogram (top 6):")
        for p, count in pcount.most_common(6):
            print(f"    MIDI {p:>3} ({librosa.midi_to_note(p):>3})  {count:>4}")
        print(f"  duration: median {np.median(durs_ms):.0f} ms, "
              f"min {np.min(durs_ms):.0f}, max {np.max(durs_ms):.0f}")
        print(f"  velocity: median {int(np.median(vels))}, "
              f"min {min(vels)}, max {max(vels)}")
        with_bend = [n for n in notes if n.pitch_bends]
        if with_bend:
            max_abs_cents = max(abs(ev["cents"]) for n in with_bend for ev in n.pitch_bends)
            print(f"  pitchbend: {len(with_bend)}/{len(notes)} notes have bend, "
                  f"max abs deviation {max_abs_cents:.0f} cents")
        else:
            print("  pitchbend: none captured")

    with json_path.open("w") as f:
        json.dump([asdict(n) for n in notes], f, indent=2)
    write_midi(notes, midi_path)
    print(f"  wrote {json_path.name} and {midi_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
