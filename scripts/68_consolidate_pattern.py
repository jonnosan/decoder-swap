"""Collapse a noisy bar-by-bar MIDI extraction into a single canonical bar pattern, then tile.

basic-pitch transcribes audio frame-by-frame, so even when the underlying riff
is literally identical bar after bar (true for most electronic dance music),
the per-bar MIDI output is full of frame-noise — different onsets, different
pitches, missing hits, spurious hits. Listening to the result reveals
inconsistency the original audio doesn't have.

This script assumes the riff is a fixed-length looping pattern (a few bars,
default 1) and consolidates the noisy per-bar MIDI into a single canonical
pattern by voting per-grid-position across all bars in the loop region.

Approach:
  1. Divide the song into bars (BPM × beats-per-bar).
  2. For each grid position within the loop length (e.g. 16 positions per 1-bar
     loop at 16th-grid), collect the (pitch, presence) votes from every
     occurrence of that position across all bars.
  3. A position is "active" if it has notes in >= activity_threshold of bars.
  4. The pitch at an active position is the mode (most common) across all bars
     that voted active there.
  5. Tile the canonical pattern across the song from --start-s to --end-s.

Run:
  .venv/bin/python scripts/68_consolidate_pattern.py \\
    --midi data/song_test/<slug>/semantic/bass_quantized.mid \\
    --bpm 136 --grid 16 --loop-bars 1 \\
    --start-s 30 --end-s 180 \\
    --out data/song_test/<slug>/semantic/bass_consolidated.mid
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pretty_midi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--grid", type=int, default=16,
                    help="grid subdivisions per whole note (16=sixteenths)")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--loop-bars", type=int, default=1,
                    help="how many bars the loop pattern is — usually 1, sometimes 2 or 4")
    ap.add_argument("--start-s", type=float, required=True,
                    help="start of consolidation region (skip intro)")
    ap.add_argument("--end-s", type=float, default=None,
                    help="end of consolidation region (default: end of MIDI)")
    ap.add_argument("--activity-threshold", type=float, default=0.4,
                    help="fraction of bars that must have a note at a position for it to be 'active' (0..1)")
    ap.add_argument("--velocity", type=int, default=100,
                    help="fixed velocity for output notes")
    ap.add_argument("--note-len-steps", type=int, default=1,
                    help="output note length in grid steps (ignored if --sustain)")
    ap.add_argument("--sustain", action="store_true", default=False,
                    help="sustain each note until the next onset (no gaps). Right for continuous-growl bass.")
    ap.add_argument("--sustain-gap-ms", type=float, default=0.0,
                    help="when --sustain is on, leave this much silence before the next note (0 = touch, 20-30 = audible re-trigger).")
    ap.add_argument("--anchor-s", type=float, default=0.0,
                    help="grid anchor for bar/beat detection — same as quantize anchor")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    pm_in = pretty_midi.PrettyMIDI(args.midi)
    inst_in = pm_in.instruments[0]
    notes = sorted(inst_in.notes, key=lambda n: n.start)
    if args.end_s is None:
        args.end_s = pm_in.get_end_time()

    beat_s = 60.0 / args.bpm
    bar_s = beat_s * args.beats_per_bar
    step_s = (4.0 * beat_s) / args.grid
    steps_per_bar = int(round(args.beats_per_bar * args.grid / 4))
    loop_steps = steps_per_bar * args.loop_bars

    print(f"# consolidate · BPM={args.bpm} · {steps_per_bar} steps/bar · loop={args.loop_bars} bar(s) = {loop_steps} steps")
    print(f"  region: t={args.start_s:.2f}..{args.end_s:.2f}s")

    # Collect notes per loop-position
    pos_pitches: list[Counter] = [Counter() for _ in range(loop_steps)]
    pos_count: list[int] = [0] * loop_steps
    bars_seen = 0

    # Region is a multiple of loop length starting at start_s.
    t = args.start_s
    while t + args.loop_bars * bar_s <= args.end_s:
        bars_seen += args.loop_bars
        loop_start = t
        loop_end = t + args.loop_bars * bar_s
        loop_notes = [n for n in notes if loop_start <= n.start < loop_end]
        for n in loop_notes:
            rel_s = n.start - loop_start
            step_idx = int(round((rel_s - args.anchor_s % step_s) / step_s)) % loop_steps
            pos_pitches[step_idx][n.pitch] += 1
        # advance one position-count for every step in this loop (so threshold is denominated in loops)
        t += args.loop_bars * bar_s

    n_loops = bars_seen // args.loop_bars
    for i in range(loop_steps):
        pos_count[i] = n_loops  # every loop contributes one vote per position

    print(f"  bars seen: {bars_seen} ({n_loops} loops of {args.loop_bars} bar(s))")

    # Build canonical pattern: positions with activity > threshold get a note at their modal pitch
    canonical: list[tuple[int, int]] = []  # (step_idx, pitch)
    activity_by_step = {}
    for step in range(loop_steps):
        active_in_loops = sum(pos_pitches[step].values())
        if pos_count[step] == 0:
            continue
        fraction = active_in_loops / pos_count[step]
        activity_by_step[step] = fraction
        if fraction >= args.activity_threshold:
            modal_pitch = pos_pitches[step].most_common(1)[0][0]
            canonical.append((step, modal_pitch))

    # Drop adjacent same-pitch hits: keep the one with higher activity (the other is likely
    # an extraction-noise re-trigger of the same physical note).
    canonical_sorted = sorted(canonical, key=lambda sp: sp[0])
    keep = [True] * len(canonical_sorted)
    for i in range(len(canonical_sorted) - 1):
        s1, p1 = canonical_sorted[i]
        s2, p2 = canonical_sorted[i+1]
        if (s2 - s1) == 1 and p1 == p2:
            if activity_by_step[s1] >= activity_by_step[s2]:
                keep[i+1] = False
            else:
                keep[i] = False
    dropped = sum(1 for k in keep if not k)
    if dropped:
        print(f"  dropped {dropped} adjacent same-pitch hit(s) (likely split-note artifacts)")
    canonical = [sp for sp, k in zip(canonical_sorted, keep) if k]

    print(f"  canonical pattern ({len(canonical)} hits across {loop_steps} grid positions):")
    for step, pitch in canonical:
        beat = step // (args.grid // args.beats_per_bar) + 1
        within = step % (args.grid // args.beats_per_bar)
        # "1 e + a" naming for the 4 16ths within a beat
        sub_names = {0: ' ', 1: 'e', 2: '+', 3: 'a'}[within] if (args.grid // args.beats_per_bar) == 4 else f'+{within}'
        bar_index = step // steps_per_bar
        print(f"    step {step:2d} (bar{bar_index} beat{beat % args.beats_per_bar or args.beats_per_bar}-{sub_names}): pitch={pitch}  (modal of {dict(pos_pitches[step])})")

    # Tile the canonical pattern across the region (and any bars after, if end_s extends further)
    pm_out = pretty_midi.PrettyMIDI(initial_tempo=args.bpm)
    inst_out = pretty_midi.Instrument(program=inst_in.program, is_drum=inst_in.is_drum, name=inst_in.name)

    fixed_len_s = args.note_len_steps * step_s
    sustain_gap_s = args.sustain_gap_ms / 1000.0
    t = args.start_s
    loop_dur_s = args.loop_bars * bar_s
    out_notes = 0
    # Pre-compute next-step distances within the loop pattern (wrap to next loop's first hit)
    canonical_sorted = sorted(canonical, key=lambda sp: sp[0])
    next_step_in_loop = {}
    for idx, (step, _) in enumerate(canonical_sorted):
        nxt_idx = (idx + 1) % len(canonical_sorted)
        nxt_step = canonical_sorted[nxt_idx][0]
        gap_steps = (nxt_step - step) if nxt_idx > idx else (loop_steps - step + nxt_step)
        next_step_in_loop[step] = gap_steps

    while t < args.end_s:
        for step, pitch in canonical_sorted:
            start = t + step * step_s
            if start >= args.end_s:
                break
            if args.sustain:
                gap_steps = next_step_in_loop[step]
                end = start + gap_steps * step_s - sustain_gap_s
                end = max(end, start + 0.001)
            else:
                end = start + fixed_len_s
            inst_out.notes.append(pretty_midi.Note(
                velocity=args.velocity, pitch=pitch, start=start, end=end,
            ))
            out_notes += 1
        t += loop_dur_s

    pm_out.instruments.append(inst_out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pm_out.write(args.out)
    print(f"  wrote {args.out} ({out_notes} notes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
