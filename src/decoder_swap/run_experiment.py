"""M4: encode a held-out clip to T with the frozen front-end, decode T with D1 and D2 → S1, S2.

The cardinal rule: D1 and D2 must differ in nothing except decoder weights. We assert this BEFORE
producing S1/S2. If the invariant fails, the experiment is invalid and we raise — better to abort
than silently produce evidence that doesn't actually test the hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch

from .codec_io import Codec, decode_from_codes, encode_to_codes, load_codec
from .freeze import remove_weight_norm_recursive
from .invariants import assert_codec_invariants_match
from .train_decoder import load_d2_into


@dataclass
class ExperimentResult:
    input_path: str
    sample_rate: int
    y_input: np.ndarray              # the resampled mono input (numpy float32)
    codes: torch.Tensor              # (1, n_codebooks, T_frames) — the token grid T, identical for D1/D2
    s1: np.ndarray                   # D1 decoder output (numpy float32), trimmed to len(y_input)
    s2: np.ndarray                   # D2 decoder output (numpy float32), trimmed to len(y_input)
    d2_train_meta: dict              # metadata loaded from the D2 checkpoint


def load_audio_mono(path: str | Path, target_sr: int, max_seconds: float | None) -> np.ndarray:
    duration = max_seconds if max_seconds is not None else None
    y, _ = librosa.load(str(path), sr=target_sr, mono=True, duration=duration)
    return y.astype(np.float32, copy=False)


def build_d1_d2(
    codec_name: str,
    codec_model_type: str,
    codec_model_tag: str | None,
    codec_model_path: str | None,
    device: str,
    d2_ckpt_path: str | Path,
    num_quantizers: int | None = None,
) -> tuple[Codec, Codec, dict]:
    """Load two independent codecs. D1 is the pretrained baseline (decoder untouched). D2 has the
    same pretrained encoder+quantizer, but its decoder is replaced with the fine-tuned D2 weights.

    Returns (codec_d1, codec_d2, d2_metadata). Calls the pairwise invariant check before returning;
    if codebooks aren't byte-identical the function raises.
    """
    codec_d1 = load_codec(
        codec_name, codec_model_type, codec_model_tag, codec_model_path, device, num_quantizers
    )
    codec_d2 = load_codec(
        codec_name, codec_model_type, codec_model_tag, codec_model_path, device, num_quantizers
    )

    # D2 was trained with weight_norm removed from the decoder (MPS-safe backward — see freeze.py).
    # The saved state_dict therefore has plain `weight` keys not `weight_g`/`weight_v`. We must
    # mirror that on the receiving decoder before load_state_dict, else key mismatch.
    n_removed = remove_weight_norm_recursive(codec_d2.decoder)
    d2_meta = load_d2_into(codec_d2, d2_ckpt_path)
    codec_d2.decoder.to(device)

    # The single most important guard in the whole experiment.
    assert_codec_invariants_match(codec_d1, codec_d2)

    return codec_d1, codec_d2, {"decoder_weight_norm_removed": n_removed, **d2_meta}


def _mps_empty_cache() -> None:
    try:
        torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def run_experiment(
    codec_d1: Codec,
    codec_d2: Codec,
    audio_path: str | Path,
    max_seconds: float | None = None,
    d2_meta: dict | None = None,
    chunk_seconds: float = 30.0,
) -> ExperimentResult:
    """Encode `audio_path` through the frozen front-end, decode with D1 and D2 from the SAME z.

    Audio is processed in fixed-size chunks to bound MPS memory — DAC's decoder activations grow
    with output length and a 4+ min song at 44.1 kHz blows the M4 Pro's 30 GB MPS budget if done
    in one pass. Each chunk encodes independently (so encoder receptive field doesn't span chunks);
    the resulting per-chunk artifacts at boundaries are IDENTICAL in S1 and S2 (same encoder, same
    tokens) and don't bias the D1/D2 comparison. Output is the concatenation of per-chunk decodes.
    """
    assert_codec_invariants_match(codec_d1, codec_d2)

    sr = codec_d1.convention.sample_rate
    hop = codec_d1.convention.hop_length
    device = next(codec_d1.encoder.parameters()).device

    y = load_audio_mono(audio_path, target_sr=sr, max_seconds=max_seconds)
    total_samples = len(y)

    # Chunk size rounded down to a multiple of hop_length so each chunk's frame count is exact.
    chunk_samples = max(int(round(chunk_seconds * sr)) // hop * hop, hop)

    codes_chunks: list[torch.Tensor] = []
    s1_chunks: list[np.ndarray] = []
    s2_chunks: list[np.ndarray] = []

    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    for ci, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        chunk_np = y[start:end]
        chunk_len = len(chunk_np)
        x = torch.from_numpy(chunk_np)[None, None, :].to(device)

        with torch.no_grad():
            codes = encode_to_codes(codec_d1, x)
            s1_t = decode_from_codes(codec_d1, codes)
            s2_t = decode_from_codes(codec_d2, codes)

            codes_chunks.append(codes.detach().cpu())
            s1_chunks.append(s1_t.squeeze().detach().cpu().float().numpy()[:chunk_len])
            s2_chunks.append(s2_t.squeeze().detach().cpu().float().numpy()[:chunk_len])

        # Release MPS tensors before the next chunk's allocations.
        del x, codes, s1_t, s2_t
        _mps_empty_cache()
        print(f"  chunk {ci+1}/{n_chunks}: samples [{start}:{end}]", flush=True)

    s1 = np.concatenate(s1_chunks)[:total_samples]
    s2 = np.concatenate(s2_chunks)[:total_samples]
    codes_all = torch.cat(codes_chunks, dim=-1)

    return ExperimentResult(
        input_path=str(audio_path),
        sample_rate=sr,
        y_input=y,
        codes=codes_all,
        s1=s1,
        s2=s2,
        d2_train_meta=d2_meta or {},
    )
