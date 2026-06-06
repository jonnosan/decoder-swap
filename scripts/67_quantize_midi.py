"""Quantize MIDI note onsets/durations to a tempo grid.

Basic-pitch (and any audio→MIDI extractor) returns onset times where the audio
attack transient actually starts. For rigidly-gridded music (techno, pop, hip-hop)
the *intended* grid position is what the listener perceives — small ms-level
drift reads as sloppy timing. This script snaps notes to the nearest grid
position at a chosen subdivision (16th note by default).

BPM source:
  --bpm <float>    explicit, takes precedence
  --audio <path>   auto-detect via librosa.beat.beat_track (returns global tempo)
  (else)           error — must supply one

Caveats:
  * librosa's tempo detector is known to be octave-confused on some material —
    if the auto-detected BPM sounds half/double-time vs. the audio, pass --bpm.
  * Note durations are snapped to integer multiples of the grid spacing (min 1).
  * Pitch-bend data is preserved as-is (we only move note_on / note_off times).

Run:
  .venv/bin/python scripts/67_quantize_midi.py \\
    --midi data/song_test/<slug>/semantic/bass.mid \\
    --audio data/song_test/<slug>/stems/bass.wav \\
    --grid 16 \\
    --out data/song_test/<slug>/semantic/bass_quantized.mid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, help="input MIDI file")
    ap.add_argument("--out", required=True, help="output quantized MIDI file")
    ap.add_argument("--bpm", type=float, default=None, help="explicit BPM (skips detection)")
    ap.add_argument("--audio", default=None, help="audio file to estimate BPM from (if --bpm not given)")
    ap.add_argument("--grid", type=int, default=16,
                    help="subdivisions per whole note (16 = sixteenth notes, 8 = eighth notes, 32 = thirty-seconds)")
    ap.add_argument("--anchor-s", type=float, default=0.0,
                    help="grid anchor in seconds — first grid position. 0 = song-start aligned")
    ap.add_argument("--pre-merge-ms", type=float, default=60.0,
                    help="before quantizing, merge consecutive same-pitch notes whose gap is < this (basic-pitch often splits sustained notes into 2)")
    ap.add_argument("--post-dedupe", action="store_true", default=True,
                    help="after quantizing, collapse duplicate (pitch, start) notes — keeps the longest")
    ap.add_argument("--pitch-lo", type=int, default=None,
                    help="drop notes below this MIDI pitch (filters extractor harmonic-jump artifacts)")
    ap.add_argument("--pitch-hi", type=int, default=None,
                    help="drop notes above this MIDI pitch")
    ap.add_argument("--mono", action="store_true", default=False,
                    help="enforce monophonic output: truncate any prior note that overlaps the next start")
    ap.add_argument("--release-gap-ms", type=float, default=20.0,
                    help="when --mono is on, leave this much silence before the next note (so synth release doesn't bleed into next attack)")
    return ap.parse_args()


def filter_pitch_range(pm: pretty_midi.PrettyMIDI, lo: int | None, hi: int | None) -> int:
    """Drop notes outside [lo, hi] (inclusive). Returns count dropped."""
    if lo is None and hi is None:
        return 0
    dropped = 0
    for inst in pm.instruments:
        kept = []
        for n in inst.notes:
            if (lo is not None and n.pitch < lo) or (hi is not None and n.pitch > hi):
                dropped += 1
                continue
            kept.append(n)
        inst.notes = kept
    return dropped


def enforce_mono(pm: pretty_midi.PrettyMIDI, release_gap_s: float = 0.0) -> int:
    """For each pair of overlapping/touching consecutive notes, truncate the earlier
    note's end so a silence gap of release_gap_s separates it from the next note.
    Returns count of notes affected."""
    truncated = 0
    for inst in pm.instruments:
        notes = sorted(inst.notes, key=lambda n: n.start)
        for i in range(len(notes) - 1):
            target_end = notes[i+1].start - release_gap_s
            # Keep at least 1ms of note even if the next is very close
            target_end = max(target_end, notes[i].start + 0.001)
            if notes[i].end > target_end:
                notes[i].end = target_end
                truncated += 1
        inst.notes = notes
    return truncated


def merge_close_same_pitch(pm: pretty_midi.PrettyMIDI, max_gap_s: float) -> int:
    """Pre-quantization cleanup: merge consecutive same-pitch notes where the gap
    between note1.end and note2.start is < max_gap_s. These are nearly always
    basic-pitch splitting a single sustained note in two.
    """
    merged_count = 0
    for inst in pm.instruments:
        # Sort by start time
        notes = sorted(inst.notes, key=lambda n: (n.pitch, n.start))
        out: list[pretty_midi.Note] = []
        for n in notes:
            if out and out[-1].pitch == n.pitch and (n.start - out[-1].end) < max_gap_s:
                # merge: extend previous note's end; keep max velocity
                out[-1].end = max(out[-1].end, n.end)
                out[-1].velocity = max(out[-1].velocity, n.velocity)
                merged_count += 1
            else:
                out.append(n)
        # Re-sort by start time for the instrument
        inst.notes = sorted(out, key=lambda n: n.start)
    return merged_count


def dedupe_grid_collisions(pm: pretty_midi.PrettyMIDI) -> int:
    """Post-quantization cleanup: if multiple notes share (pitch, start) after
    snapping, keep the longest one. Also merges (pitch, start, end) duplicates.
    """
    removed = 0
    for inst in pm.instruments:
        by_key: dict[tuple[int, float], pretty_midi.Note] = {}
        for n in inst.notes:
            key = (n.pitch, round(n.start, 4))
            if key in by_key:
                # keep the longer note
                if (n.end - n.start) > (by_key[key].end - by_key[key].start):
                    by_key[key] = n
                removed += 1
            else:
                by_key[key] = n
        inst.notes = sorted(by_key.values(), key=lambda n: n.start)
    return removed


def detect_bpm(audio_path: Path) -> float:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def quantize_pretty_midi(
    pm: pretty_midi.PrettyMIDI,
    bpm: float,
    grid_subdiv: int,
    anchor_s: float,
) -> pretty_midi.PrettyMIDI:
    """Snap every note's start and end to the nearest grid position."""
    # A "whole note" at this BPM = 4 beats. grid_subdiv=16 -> step = whole/16 = beat/4 = 16th note.
    beat_s = 60.0 / bpm
    step_s = (4.0 * beat_s) / grid_subdiv

    snap_count = 0
    total = 0
    max_shift_ms = 0.0
    for inst in pm.instruments:
        for n in inst.notes:
            total += 1
            # snap start
            k_start = round((n.start - anchor_s) / step_s)
            new_start = anchor_s + k_start * step_s
            # snap duration to at least 1 step
            dur = n.end - n.start
            k_dur = max(1, round(dur / step_s))
            new_end = new_start + k_dur * step_s
            shift = abs(new_start - n.start) * 1000.0
            max_shift_ms = max(max_shift_ms, shift)
            if shift > 1.0:
                snap_count += 1
            n.start = new_start
            n.end = new_end

    # Update tempo metadata so DAWs that read it show the right BPM
    # (pretty_midi keeps tempo in _tick_scales; easiest is to rewrite via initial_tempo)
    print(f"  quantized {snap_count}/{total} notes (those moved >1ms)")
    print(f"  max shift: {max_shift_ms:.1f} ms")
    return pm


def main() -> int:
    args = parse_args()
    midi_path = Path(args.midi)
    out_path = Path(args.out)

    if args.bpm is not None:
        bpm = args.bpm
        print(f"# quantize · BPM={bpm:.2f} (explicit)")
    elif args.audio:
        bpm = detect_bpm(Path(args.audio))
        print(f"# quantize · BPM={bpm:.2f} (detected from {args.audio})")
    else:
        print("ERROR: must supply --bpm or --audio", file=sys.stderr)
        return 1

    grid_s = (60.0 / bpm) * 4.0 / args.grid
    print(f"  grid: 1/{args.grid} note = {grid_s*1000:.2f} ms at {bpm:.2f} BPM")
    print(f"  anchor: {args.anchor_s:.3f} s")

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # 0) Pitch-range filter (drop extractor harmonic-jump artifacts)
    dropped = filter_pitch_range(pm, args.pitch_lo, args.pitch_hi)
    if dropped:
        print(f"  pitch-filter: dropped {dropped} notes outside [{args.pitch_lo}, {args.pitch_hi}]")

    # 1) Pre-quantization: merge basic-pitch's split-note artifacts
    if args.pre_merge_ms > 0:
        merged = merge_close_same_pitch(pm, max_gap_s=args.pre_merge_ms / 1000.0)
        print(f"  pre-merge: collapsed {merged} same-pitch notes within {args.pre_merge_ms:.0f} ms")

    # 2) Quantize to grid
    pm = quantize_pretty_midi(pm, bpm=bpm, grid_subdiv=args.grid, anchor_s=args.anchor_s)

    # 3) Post-quantization: dedupe grid collisions
    if args.post_dedupe:
        removed = dedupe_grid_collisions(pm)
        print(f"  post-dedupe: removed {removed} notes that collided on the grid")

    # 4) Optional: enforce monophonic output (truncate overlaps + leave release gap)
    if args.mono:
        truncated = enforce_mono(pm, release_gap_s=args.release_gap_ms / 1000.0)
        print(f"  mono: truncated {truncated} overlapping notes (release gap {args.release_gap_ms:.0f} ms)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
