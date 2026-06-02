"""Paired jtxtok ↔ DAC dataset for M7.B training (PROMPT_3 B1 corpus pairs).

Each track contributes:
  - cached DAC tokens at `data/tokens_dac/<corpus>/<stem>.npy`  (shape: n_q × T_frames)
  - cached jtxtok stream at `data/jtxtok/<corpus>/<stem>.jtxtok` (tokenised by extractor)
  - cached sidecar JSON with per-bar sample boundaries + the BAR token's index in the stream

Random crops are sampled aligned across both modalities:
  1. Pick a random DAC frame range [frame_start, frame_start + window_frames)
  2. Convert frames → samples: sample_start = frame_start * dac_hop, etc.
  3. Find jtxtok bars whose [bar_start, bar_end) sample range overlaps the window
  4. Extract the jtxtok token slice for those bars (the BAR tokens + their events)
  5. Tokenise + pad to a per-batch max length; build the parallel role-id array

This module does NOT apply CFG / MT dropout (that's a training decision; the trainer
applies them per step). It just produces raw aligned pairs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .jtxtok_vocab import JtxtokVocab


@dataclass
class TrackPair:
    """One (DAC, jtxtok) pair, fully loaded into RAM."""
    stem: str
    dac: np.ndarray                # (n_q, T_frames) int
    jtxtok_tokens: list[str]       # full stream BOS .. EOS
    bar_starts_samples: list[int]
    bar_ends_samples: list[int]
    bar_token_indices: list[int]   # parallel to bars: index of each BAR token in jtxtok_tokens
    sample_rate: int               # extractor's sample rate (audio frame of reference)
    duration_seconds: float


def load_track_pair(
    dac_path: Path | str,
    jtxtok_path: Path | str,
    sidecar_path: Path | str,
) -> TrackPair:
    """Load a single (DAC tokens, jtxtok, sidecar) triple from disk."""
    dac = np.load(str(dac_path))
    if dac.ndim != 2:
        raise ValueError(f"DAC tokens at {dac_path} have unexpected shape {dac.shape}")
    text = Path(jtxtok_path).read_text().split()
    sidecar = json.loads(Path(sidecar_path).read_text())
    if sidecar.get("vocabulary_version") != "jtxtok-v1":
        raise ValueError(f"unexpected vocabulary_version in {sidecar_path}")
    if "bar_starts_samples" not in sidecar:
        raise ValueError(
            f"sidecar {sidecar_path} predates the bar-boundaries change; re-extract this clip"
        )
    return TrackPair(
        stem=Path(dac_path).stem,
        dac=dac,
        jtxtok_tokens=text,
        bar_starts_samples=list(sidecar["bar_starts_samples"]),
        bar_ends_samples=list(sidecar["bar_ends_samples"]),
        bar_token_indices=list(sidecar["bar_token_indices"]),
        sample_rate=int(sidecar["sample_rate"]),
        duration_seconds=float(sidecar["duration_seconds"]),
    )


def discover_track_pairs(
    tokens_dac_dir: Path | str,
    jtxtok_dir: Path | str,
) -> list[TrackPair]:
    """Pair up <stem>.npy ↔ <stem>.jtxtok by filename stem. Skips tracks missing either side."""
    tokens_dac_dir = Path(tokens_dac_dir)
    jtxtok_dir = Path(jtxtok_dir)
    dac_files = sorted(tokens_dac_dir.glob("*.npy"))
    pairs: list[TrackPair] = []
    for dp in dac_files:
        stem = dp.stem
        # jtxtok files may use a different naming convention if the user specified --output.
        # Try a few common spellings.
        candidates = [
            jtxtok_dir / f"{stem}.jtxtok",
            jtxtok_dir / f"{stem.replace(' - ', '-')}.jtxtok",
            jtxtok_dir / f"{stem.replace(' ', '-')}.jtxtok",
        ]
        jt_path = next((c for c in candidates if c.exists()), None)
        if jt_path is None:
            continue
        sc_path = jt_path.with_suffix(".json")
        if not sc_path.exists():
            continue
        pairs.append(load_track_pair(dp, jt_path, sc_path))
    return pairs


@dataclass
class JtxtokDacDatasetConfig:
    dac_hop_length: int = 512                    # DAC 44 kHz convention
    dac_sample_rate: int = 44100
    window_seconds: float = 3.0
    max_jtxtok_len: int = 256                    # token slice padded/truncated to this length
    seed: int = 0


class JtxtokDacDataset:
    """Yields aligned (DAC tokens, jtxtok tokens, role ids) crops from in-RAM track pairs."""

    def __init__(self, tracks: list[TrackPair], vocab: JtxtokVocab, cfg: JtxtokDacDatasetConfig):
        if not tracks:
            raise ValueError("no tracks loaded for JtxtokDacDataset")
        self.tracks = tracks
        self.vocab = vocab
        self.cfg = cfg
        self.window_frames = int(round(cfg.window_seconds * cfg.dac_sample_rate / cfg.dac_hop_length))
        self.window_samples = self.window_frames * cfg.dac_hop_length
        self.rng = np.random.default_rng(cfg.seed)
        # Sample track index weighted by track DAC length so longer tracks contribute proportionally.
        lens = np.array([t.dac.shape[-1] for t in tracks], dtype=np.float64)
        self.track_probs = lens / lens.sum()

    def _slice_jtxtok_for_window(
        self, track: TrackPair, sample_start: int, sample_end: int,
    ) -> tuple[list[int], list[int]]:
        """Find jtxtok bars overlapping [sample_start, sample_end), return their token slice
        as (token_ids, role_ids). Includes BOS/EOS only if those bars are at track boundaries
        (so most crops won't include them — the model sees mid-stream context, like real training)."""
        v = self.vocab
        starts = track.bar_starts_samples
        ends = track.bar_ends_samples
        token_idxs = track.bar_token_indices
        tokens = track.jtxtok_tokens

        # Bars overlapping the window
        first_bar = None
        last_bar = None
        for i, (s, e) in enumerate(zip(starts, ends)):
            if e <= sample_start:
                continue
            if s >= sample_end:
                break
            if first_bar is None:
                first_bar = i
            last_bar = i

        if first_bar is None or last_bar is None:
            return [v.pad_id], [v.role_ids()[v.pad_id]]

        # Token slice: tokens[token_idxs[first_bar] : token_idxs[last_bar+1] OR end]
        start_idx = token_idxs[first_bar]
        if last_bar + 1 < len(token_idxs):
            end_idx = token_idxs[last_bar + 1]
        else:
            # Last bar — slice to end of stream (excludes EOS, which lives at the very end)
            end_idx = len(tokens)
            # If the last token is EOS, drop it (it's a sentinel for end-of-track, not the window)
            if tokens[end_idx - 1] == "EOS":
                end_idx -= 1

        slice_tokens = tokens[start_idx:end_idx]
        try:
            ids = v.encode(slice_tokens)
        except KeyError as e:
            raise ValueError(
                f"jtxtok stream contains token {e} that's not in the configured vocabulary; "
                f"check that the extractor's contract matches this consumer's vocab config"
            )
        role_lookup = v.role_ids()
        roles = [role_lookup[i] for i in ids]
        return ids, roles

    def sample_batch(self, batch_size: int) -> dict:
        """Return a batch dict with:
            dac_ids        : (B, n_q, window_frames) int64
            jtxtok_ids     : (B, L_padded) int64
            jtxtok_roles   : (B, L_padded) int64
        where L_padded = max sequence length across the batch, capped at max_jtxtok_len.
        """
        B = int(batch_size)
        v = self.vocab

        dac_out = np.empty((B, self.tracks[0].dac.shape[0], self.window_frames), dtype=np.int64)
        jtxtok_lists: list[list[int]] = []
        role_lists: list[list[int]] = []

        for i in range(B):
            ti = int(self.rng.choice(len(self.tracks), p=self.track_probs))
            track = self.tracks[ti]
            T_frames = track.dac.shape[-1]
            if T_frames < self.window_frames:
                raise ValueError(f"track {track.stem} too short ({T_frames} < {self.window_frames})")
            frame_start = int(self.rng.integers(0, T_frames - self.window_frames + 1))
            sample_start = frame_start * self.cfg.dac_hop_length
            sample_end = sample_start + self.window_samples
            dac_out[i] = track.dac[:, frame_start : frame_start + self.window_frames]

            ids, roles = self._slice_jtxtok_for_window(track, sample_start, sample_end)
            if len(ids) > self.cfg.max_jtxtok_len:
                # Truncate from the right (keep the earliest content; the rest gets dropped).
                ids = ids[: self.cfg.max_jtxtok_len]
                roles = roles[: self.cfg.max_jtxtok_len]
            jtxtok_lists.append(ids)
            role_lists.append(roles)

        # Right-pad to batch max length.
        L = max(len(x) for x in jtxtok_lists) if jtxtok_lists else 1
        pad_role = v.role_ids()[v.pad_id]
        jtxtok_arr = np.full((B, L), v.pad_id, dtype=np.int64)
        role_arr = np.full((B, L), pad_role, dtype=np.int64)
        for i, (ids, roles) in enumerate(zip(jtxtok_lists, role_lists)):
            jtxtok_arr[i, : len(ids)] = ids
            role_arr[i, : len(roles)] = roles

        return {
            "dac_ids": torch.from_numpy(dac_out),
            "jtxtok_ids": torch.from_numpy(jtxtok_arr),
            "jtxtok_roles": torch.from_numpy(role_arr),
        }
