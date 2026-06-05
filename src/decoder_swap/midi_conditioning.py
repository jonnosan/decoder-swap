"""Per-frame MIDI conditioning for the conditional codec-LM (issue #10, Phase 1B.1).

Reads a `bass.json` produced by `scripts/33_bass_to_midi.py` (basic-pitch) and
emits a frame-aligned dense conditioning tensor at the DAC frame rate
(86.13 fps). Four per-frame channels:

  pitch_active : multi-hot over [pitch_lo, pitch_hi]   — which pitches are sounding
  velocity_bin : int                                    — quantised max-active velocity
  bend_bin     : int                                    — quantised pitchbend in cents
  onset_phase  : int  (0=silent / 1=onset / 2=mid / 3=release)

Polyphony note: bass is mostly mono but root+octave / root+fifth happens. We
keep multi-hot pitch; velocity is the max of active notes; bend is taken from
the loudest active note (single scalar applies to the whole frame, matching
how a real bass synth's bend wheel works).

Design rationale: see project memory `project_stems_v1_findings` and the
GitHub issue #10 design note in the conversation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameCondConfig:
    """Knobs for the per-frame conditioning. Defaults match the bass/Beltram setup."""
    pitch_lo: int = 24                # C1 — covers sub-bass
    pitch_hi: int = 50                # D3 — bass rarely goes higher
    n_velocity_bins: int = 8          # bin 0 reserved for "silent"; 1..N-1 = note bins
    n_bend_bins: int = 16             # symmetric ±bend_range_cents, center bin = no bend
    bend_range_cents: float = 200.0   # clamp/scale range
    onset_window_frames: int = 3      # frames at the start of a note tagged as "onset"
    release_window_frames: int = 2    # frames at the end tagged as "release"

    @property
    def n_pitches(self) -> int:
        return self.pitch_hi - self.pitch_lo + 1


# Onset-phase enum (int).
ONSET_SILENT = 0
ONSET_ATTACK = 1
ONSET_MID = 2
ONSET_RELEASE = 3
N_ONSET_PHASES = 4


@dataclass
class FrameConditioning:
    """Per-frame multi-channel conditioning.

    Shapes (all length T = number of frames):
      pitch_active : (T, n_pitches)  uint8 multi-hot
      velocity_bin : (T,)            int8
      bend_bin     : (T,)            int8
      onset_phase  : (T,)            int8
    """
    pitch_active: np.ndarray
    velocity_bin: np.ndarray
    bend_bin: np.ndarray
    onset_phase: np.ndarray
    cfg: FrameCondConfig

    @property
    def n_frames(self) -> int:
        return int(self.pitch_active.shape[0])

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            pitch_active=self.pitch_active,
            velocity_bin=self.velocity_bin,
            bend_bin=self.bend_bin,
            onset_phase=self.onset_phase,
            pitch_lo=self.cfg.pitch_lo,
            pitch_hi=self.cfg.pitch_hi,
            n_velocity_bins=self.cfg.n_velocity_bins,
            n_bend_bins=self.cfg.n_bend_bins,
            bend_range_cents=self.cfg.bend_range_cents,
            onset_window_frames=self.cfg.onset_window_frames,
            release_window_frames=self.cfg.release_window_frames,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "FrameConditioning":
        d = np.load(path)
        cfg = FrameCondConfig(
            pitch_lo=int(d["pitch_lo"]),
            pitch_hi=int(d["pitch_hi"]),
            n_velocity_bins=int(d["n_velocity_bins"]),
            n_bend_bins=int(d["n_bend_bins"]),
            bend_range_cents=float(d["bend_range_cents"]),
            onset_window_frames=int(d["onset_window_frames"]),
            release_window_frames=int(d["release_window_frames"]),
        )
        return cls(
            pitch_active=d["pitch_active"],
            velocity_bin=d["velocity_bin"],
            bend_bin=d["bend_bin"],
            onset_phase=d["onset_phase"],
            cfg=cfg,
        )


def _velocity_to_bin(v: int, n_bins: int) -> int:
    """Map MIDI velocity (1..127) into bins 1..n_bins-1. Bin 0 = silent (caller decides)."""
    n_note_bins = n_bins - 1
    v = max(1, min(127, int(v)))
    return 1 + int((v - 1) * n_note_bins / 127)


def _bend_to_bin(cents: float, n_bins: int, bend_range_cents: float) -> int:
    """Map signed cents to a bin in [0, n_bins). Center bin is "no bend"."""
    cents = max(-bend_range_cents, min(bend_range_cents, float(cents)))
    # Map [-R, R] -> [0, n_bins) with the center at n_bins//2.
    norm = (cents + bend_range_cents) / (2.0 * bend_range_cents)   # in [0, 1]
    b = int(round(norm * (n_bins - 1)))
    return max(0, min(n_bins - 1, b))


def _bend_center_bin(n_bins: int) -> int:
    return _bend_to_bin(0.0, n_bins, 1.0)  # the value of bend_range_cents doesn't matter at 0


def build_from_notes(
    notes: list[dict],
    fps: float,
    n_frames: int,
    cfg: FrameCondConfig | None = None,
) -> FrameConditioning:
    """Construct a FrameConditioning of length n_frames at frame rate fps.

    notes: list of dicts as written by scripts/33_bass_to_midi.py — each has
      start_s, end_s, pitch (MIDI), velocity, pitch_bends (list of {time_s, cents}
      relative to note start). Notes outside [pitch_lo, pitch_hi] are clipped.

    Frames whose timestamps fall outside any note get pitch_active=0, vel=0,
    bend=center, onset_phase=SILENT.
    """
    cfg = cfg or FrameCondConfig()
    T = int(n_frames)
    n_pitches = cfg.n_pitches
    pitch_active = np.zeros((T, n_pitches), dtype=np.uint8)
    velocity_bin = np.zeros(T, dtype=np.int8)
    bend_bin = np.full(T, _bend_center_bin(cfg.n_bend_bins), dtype=np.int8)
    onset_phase = np.full(T, ONSET_SILENT, dtype=np.int8)

    if T == 0:
        return FrameConditioning(pitch_active, velocity_bin, bend_bin, onset_phase, cfg)

    # Track, per frame: list of (pitch_idx, velocity, bend_cents, frames_from_start, total_frames)
    # We accumulate then resolve at the end.
    # For speed on small N, iterate notes and fill ranges directly.
    frame_dt = 1.0 / fps

    # Per-frame "loudest active velocity" — used to pick bend source and onset_phase.
    max_vel_per_frame = np.zeros(T, dtype=np.int16)
    # Per-frame (start_frame_of_loudest_note, end_frame_of_loudest_note) and its bend curve.
    # We keep, per frame, the bend cents from the loudest active note.
    bend_cents_per_frame = np.zeros(T, dtype=np.float32)
    note_start_frame_per_frame = np.full(T, -1, dtype=np.int32)
    note_end_frame_per_frame = np.full(T, -1, dtype=np.int32)

    for note in notes:
        p = int(note["pitch"])
        if p < cfg.pitch_lo or p > cfg.pitch_hi:
            continue
        pi = p - cfg.pitch_lo
        v = int(note.get("velocity", 96))
        start_f = int(np.floor(float(note["start_s"]) / frame_dt))
        end_f = int(np.ceil(float(note["end_s"]) / frame_dt))
        start_f = max(0, start_f)
        end_f = min(T, end_f)
        if end_f <= start_f:
            continue

        # Multi-hot pitch: set this pitch active for [start_f, end_f).
        pitch_active[start_f:end_f, pi] = 1

        # Resolve "which note is loudest" frame-by-frame in this range.
        bends = note.get("pitch_bends") or []
        # Pre-render a per-frame bend curve for this note (relative to note start).
        # bends is a list of {time_s, cents}; if missing, treat as 0 cents throughout.
        if bends:
            bend_times = np.array([float(b["time_s"]) for b in bends], dtype=np.float32)
            bend_values = np.array([float(b["cents"]) for b in bends], dtype=np.float32)
        else:
            bend_times = np.array([0.0], dtype=np.float32)
            bend_values = np.array([0.0], dtype=np.float32)

        for f in range(start_f, end_f):
            if v <= max_vel_per_frame[f]:
                continue
            max_vel_per_frame[f] = v
            # Where in the note are we (seconds since start)?
            rel_t = f * frame_dt - float(note["start_s"])
            if rel_t < 0:
                rel_t = 0.0
            # piecewise-constant lookup: find last bend_time <= rel_t
            idx = int(np.searchsorted(bend_times, rel_t, side="right") - 1)
            idx = max(0, min(len(bend_values) - 1, idx))
            bend_cents_per_frame[f] = bend_values[idx]
            note_start_frame_per_frame[f] = start_f
            note_end_frame_per_frame[f] = end_f

    # Now fill velocity_bin, bend_bin, onset_phase from the per-frame loudest-note info.
    for f in range(T):
        v = int(max_vel_per_frame[f])
        if v == 0:
            continue  # silent — leave defaults
        velocity_bin[f] = _velocity_to_bin(v, cfg.n_velocity_bins)
        bend_bin[f] = _bend_to_bin(
            float(bend_cents_per_frame[f]), cfg.n_bend_bins, cfg.bend_range_cents
        )
        s = int(note_start_frame_per_frame[f])
        e = int(note_end_frame_per_frame[f])
        if f - s < cfg.onset_window_frames:
            onset_phase[f] = ONSET_ATTACK
        elif e - f - 1 < cfg.release_window_frames:
            onset_phase[f] = ONSET_RELEASE
        else:
            onset_phase[f] = ONSET_MID

    return FrameConditioning(pitch_active, velocity_bin, bend_bin, onset_phase, cfg)


def build_from_json(
    json_path: str | Path,
    fps: float,
    n_frames: int,
    cfg: FrameCondConfig | None = None,
) -> FrameConditioning:
    """Convenience: read scripts/33_bass_to_midi.py output and call build_from_notes."""
    with open(json_path) as f:
        notes = json.load(f)
    return build_from_notes(notes, fps, n_frames, cfg)


def summarise(c: FrameConditioning) -> str:
    """One-line summary string. For training logs / debugging."""
    nf = c.n_frames
    n_active = int((c.pitch_active.sum(axis=1) > 0).sum())
    n_attack = int((c.onset_phase == ONSET_ATTACK).sum())
    n_mid = int((c.onset_phase == ONSET_MID).sum())
    n_release = int((c.onset_phase == ONSET_RELEASE).sum())
    n_silent = int((c.onset_phase == ONSET_SILENT).sum())
    bend_center = _bend_center_bin(c.cfg.n_bend_bins)
    n_bend = int((c.bend_bin != bend_center).sum())
    return (
        f"frames={nf} active={n_active} ({100*n_active/max(1,nf):.0f}%) "
        f"phases attack={n_attack} mid={n_mid} release={n_release} silent={n_silent} "
        f"bent_frames={n_bend}"
    )
