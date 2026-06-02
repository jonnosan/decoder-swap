"""Tests for jtxtok_dataset + dropout helpers in train_conditioned."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from decoder_swap.jtxtok_dataset import (
    JtxtokDacDataset,
    JtxtokDacDatasetConfig,
    TrackPair,
    load_track_pair,
)
from decoder_swap.jtxtok_vocab import DEFAULT_VOCAB
from decoder_swap.train_conditioned import _apply_cfg_dropout, _apply_mt_dropout


def _synthetic_track(stem: str = "test", n_bars: int = 4) -> TrackPair:
    """Build a small in-memory track pair: 4 bars of mock DAC tokens + mock jtxtok stream."""
    sr = 44100
    hop = 512
    fps = sr / hop                           # 86.13
    # Each bar covers 1 second of audio = 44100 samples = 86 DAC frames.
    bar_samples = sr
    bar_starts = [i * bar_samples for i in range(n_bars)]
    bar_ends = [(i + 1) * bar_samples for i in range(n_bars)]
    total_samples = bar_ends[-1]
    n_frames = total_samples // hop
    dac = np.random.RandomState(42).randint(0, 1024, size=(9, n_frames), dtype=np.int64).astype(np.int16)

    # Build a simple jtxtok stream: BOS, then for each bar: BAR KEY_0 POS_0 DRUM_KICK POS_4 DRUM_KICK ...
    tokens = ["BOS"]
    bar_token_indices = []
    for b in range(n_bars):
        bar_token_indices.append(len(tokens))
        tokens.append("BAR")
        if b == 0:
            tokens.append("KEY_0")
        tokens.extend(["POS_0", "DRUM_KICK", "BASS_ON", "POS_4", "DRUM_KICK", "MT_+3", "POS_8", "DRUM_KICK"])
    tokens.append("EOS")

    return TrackPair(
        stem=stem,
        dac=dac,
        jtxtok_tokens=tokens,
        bar_starts_samples=bar_starts,
        bar_ends_samples=bar_ends,
        bar_token_indices=bar_token_indices,
        sample_rate=sr,
        duration_seconds=total_samples / sr,
    )


def test_dataset_sample_returns_correct_shapes():
    track = _synthetic_track(n_bars=8)
    cfg = JtxtokDacDatasetConfig(window_seconds=1.0, max_jtxtok_len=64, seed=0)
    ds = JtxtokDacDataset([track], DEFAULT_VOCAB, cfg)
    batch = ds.sample_batch(4)
    assert batch["dac_ids"].shape == (4, 9, ds.window_frames)
    # Variable jtxtok length per batch, must be padded — all rows same length.
    assert batch["jtxtok_ids"].shape[0] == 4
    assert batch["jtxtok_ids"].shape == batch["jtxtok_roles"].shape


def test_dataset_jtxtok_slice_contains_relevant_bars():
    """Window covering the second bar should include BAR token + drum events from that bar."""
    v = DEFAULT_VOCAB
    track = _synthetic_track(n_bars=4)
    cfg = JtxtokDacDatasetConfig(window_seconds=1.0, max_jtxtok_len=64, seed=0)
    ds = JtxtokDacDataset([track], v, cfg)
    # Manually call the slicer with a window that lands in bar 1 (samples 44100..88200).
    ids, roles = ds._slice_jtxtok_for_window(track, sample_start=44100, sample_end=88200)
    decoded = v.decode(ids)
    # Bar 1's tokens: BAR (no key, since key was emitted only on bar 0), POS_0 DRUM_KICK BASS_ON ...
    assert "BAR" in decoded
    assert "DRUM_KICK" in decoded
    # Should NOT include the BOS from track start
    assert "BOS" not in decoded


def test_cfg_dropout_collapses_to_single_pad():
    v = DEFAULT_VOCAB
    pad_role = ("struct", "drum", "bass", "key", "mt", "pad").index("pad")
    jtxtok_ids = torch.tensor([[5, 10, 12, 0]])
    jtxtok_roles = torch.tensor([[1, 2, 3, 5]])
    new_ids, new_roles = _apply_cfg_dropout(
        jtxtok_ids, jtxtok_roles, pad_id=v.pad_id, pad_role=pad_role,
    )
    assert new_ids.shape == (1, 1)
    assert new_ids[0, 0].item() == v.pad_id
    assert new_roles[0, 0].item() == pad_role


def test_mt_dropout_strips_only_mt_tokens():
    """MT_* tokens (role=='mt') become pad; everything else unchanged."""
    v = DEFAULT_VOCAB
    roles_order = ("struct", "drum", "bass", "key", "mt", "pad")
    mt_role = roles_order.index("mt")
    struct_role = roles_order.index("struct")
    drum_role = roles_order.index("drum")
    pad_role = roles_order.index("pad")

    jtxtok_ids = torch.tensor([[
        v.token_to_id["BAR"],
        v.token_to_id["POS_0"],
        v.token_to_id["DRUM_KICK"],
        v.token_to_id["MT_+3"],
        v.token_to_id["DRUM_HAT"],
        v.token_to_id["MT_-2"],
    ]])
    jtxtok_roles = torch.tensor([[
        struct_role, struct_role, drum_role, mt_role, drum_role, mt_role,
    ]])

    new_ids, new_roles = _apply_mt_dropout(jtxtok_ids, jtxtok_roles, v)

    # Non-MT positions unchanged
    assert new_ids[0, 0].item() == v.token_to_id["BAR"]
    assert new_ids[0, 1].item() == v.token_to_id["POS_0"]
    assert new_ids[0, 2].item() == v.token_to_id["DRUM_KICK"]
    assert new_ids[0, 4].item() == v.token_to_id["DRUM_HAT"]
    # MT positions: replaced with pad
    assert new_ids[0, 3].item() == v.pad_id
    assert new_ids[0, 5].item() == v.pad_id
    # And their roles change to pad
    assert new_roles[0, 3].item() == pad_role
    assert new_roles[0, 5].item() == pad_role


def test_load_track_pair_reads_sidecar(tmp_path):
    """End-to-end disk roundtrip: write a DAC .npy + .jtxtok + sidecar, load it back."""
    track = _synthetic_track(n_bars=2)
    dac_path = tmp_path / "test.npy"
    jtxtok_path = tmp_path / "test.jtxtok"
    sidecar_path = tmp_path / "test.json"

    np.save(dac_path, track.dac)
    jtxtok_path.write_text(" ".join(track.jtxtok_tokens))
    sidecar_path.write_text(json.dumps({
        "audio_path": "test.mp3",
        "duration_seconds": track.duration_seconds,
        "sample_rate": track.sample_rate,
        "tempo_bpm": 120.0,
        "bar_count": len(track.bar_starts_samples),
        "mean_grid_confidence": 1.0,
        "grid_method": "test",
        "drum_detector": "test",
        "bass_detector": "test",
        "drum_counts": {"kick": 1, "snare": 0, "hat": 0, "ohat": 0, "clap": 0},
        "bass_count": 0,
        "bar_starts_samples": track.bar_starts_samples,
        "bar_ends_samples": track.bar_ends_samples,
        "bar_token_indices": track.bar_token_indices,
        "config": {},
        "vocabulary_version": "jtxtok-v1",
    }))

    loaded = load_track_pair(dac_path, jtxtok_path, sidecar_path)
    assert loaded.dac.shape == track.dac.shape
    assert loaded.jtxtok_tokens == track.jtxtok_tokens
    assert loaded.bar_starts_samples == track.bar_starts_samples
    assert loaded.sample_rate == 44100
