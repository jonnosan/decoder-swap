"""Tests for the M7 jtxtok-conditioned translator scaffolding.

The load-bearing test is `test_zero_init_means_conditioned_equals_unconditional` — it
verifies that a `ConditionedTranslator` freshly built and seeded with a `FlatARTransformer`
checkpoint produces EXACTLY the same logits as the source unconditional model on the same
DAC input, when given empty conditioning. That property is what lets M7 fine-tune resume
from M6.A's base LM without ever destabilising it.

All tests run on CPU to avoid contention with the background M6.A training run.
"""
from __future__ import annotations

import math

import pytest
import torch

from decoder_swap.conditioned_translator import (
    ConditionedTranslator,
    ConditionedTranslatorConfig,
)
from decoder_swap.conditioning_encoder import (
    ConditioningEncoderConfig,
    JtxtokConditioningEncoder,
    N_ROLES,
)
from decoder_swap.jtxtok_vocab import DEFAULT_VOCAB, PAD_TOKEN, build_vocab
from decoder_swap.translator import FlatARTransformer, TranslatorConfig


# -------------------------------------------------------------------------------------------------
# Vocabulary

def test_default_vocab_size_matches_extractor_contract_defaults():
    """Extractor's contract defaults (5-class drums, bass=onset, vel=off, key=on) = 54 tokens.
    Our consumer side adds a pad token, so 55."""
    v = DEFAULT_VOCAB
    # 1 pad + 3 (BOS/EOS/BAR) + 16 (POS) + 17 (MT) + 5 (drums) + 1 (BASS_ON) + 12 (KEY) = 55.
    assert v.size == 55
    assert v.pad_id == 0
    assert v.tokens[0] == PAD_TOKEN


def test_vocab_encode_decode_roundtrip_for_spec_example():
    v = DEFAULT_VOCAB
    stream = ["BOS", "BAR", "KEY_9", "POS_0", "DRUM_KICK", "DRUM_HAT", "POS_2", "DRUM_HAT", "EOS"]
    assert v.decode(v.encode(stream)) == stream


def test_role_mask_keeps_struct_and_pad_always():
    v = DEFAULT_VOCAB
    mask = v.role_mask(set())   # keep nothing optional
    kept = {t for t, k in zip(v.tokens, mask) if k}
    # struct + pad must remain regardless of what's kept
    for t in (PAD_TOKEN, "BOS", "EOS", "BAR", "POS_0", "POS_15"):
        assert t in kept, f"{t} should always be kept (struct/pad)"
    # drums, bass, key, mt are dropped
    for t in ("DRUM_KICK", "BASS_ON", "KEY_0", "MT_+0"):
        assert t not in kept, f"{t} should be dropped"


def test_role_mask_rhythm_not_melody_keeps_drum_bass_drops_key():
    v = DEFAULT_VOCAB
    mask = v.role_mask({"drum", "bass"})
    kept = {t for t, k in zip(v.tokens, mask) if k}
    assert "DRUM_KICK" in kept
    assert "BASS_ON" in kept
    assert "KEY_0" not in kept


def test_pitched_bass_mode_adds_12_tokens():
    v_onset = build_vocab(bass_mode="onset")
    v_pitch = build_vocab(bass_mode="pitch")
    assert v_pitch.size == v_onset.size + 12
    for pc in range(12):
        assert f"BASS_P_{pc}" in v_pitch.token_to_id


# -------------------------------------------------------------------------------------------------
# Conditioning encoder

def test_conditioning_encoder_forward_shape():
    v = DEFAULT_VOCAB
    cfg = ConditioningEncoderConfig(vocab_size=v.size, d_model=64, n_layers=2, n_heads=4, d_ff=128,
                                     max_seq_len=64, pad_id=v.pad_id)
    enc = JtxtokConditioningEncoder(cfg)
    B, L = 2, 10
    tok_ids = torch.randint(low=1, high=v.size, size=(B, L))
    role_ids = torch.zeros((B, L), dtype=torch.long)  # all "struct" role
    ctx, pad_mask = enc(tok_ids, role_ids)
    assert ctx.shape == (B, L, cfg.d_model)
    assert pad_mask.shape == (B, L)
    assert pad_mask.dtype == torch.bool


def test_conditioning_encoder_pad_mask_correct():
    v = DEFAULT_VOCAB
    cfg = ConditioningEncoderConfig(vocab_size=v.size, d_model=32, n_layers=1, n_heads=2, d_ff=64,
                                     max_seq_len=32, pad_id=v.pad_id)
    enc = JtxtokConditioningEncoder(cfg)
    tok_ids = torch.tensor([[1, 2, 3, 0, 0]])  # last two are pad
    role_ids = torch.zeros_like(tok_ids)
    _, pad_mask = enc(tok_ids, role_ids)
    assert pad_mask.tolist() == [[False, False, False, True, True]]


# -------------------------------------------------------------------------------------------------
# Conditioned translator structural

def test_conditioned_translator_forward_shape():
    cfg = ConditionedTranslatorConfig(
        dac_vocab_size=1024, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_dac_seq_len=128,
        jtxtok_vocab_size=DEFAULT_VOCAB.size, enc_d_model=32, enc_n_layers=2, enc_n_heads=4,
        enc_d_ff=64, enc_max_seq_len=64,
    )
    model = ConditionedTranslator(cfg)
    B = 2
    dac_ids = torch.randint(0, 1024, (B, 32))
    jtxtok_ids = torch.randint(1, DEFAULT_VOCAB.size, (B, 16))
    role_ids = torch.zeros_like(jtxtok_ids)
    logits = model(dac_ids, jtxtok_ids, role_ids)
    assert logits.shape == (B, 32, 1024)


def test_cross_attn_out_proj_is_zero_init():
    cfg = ConditionedTranslatorConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                                       jtxtok_vocab_size=10, enc_d_model=64)
    model = ConditionedTranslator(cfg)
    for block in model.decoder.layers:
        assert torch.all(block.multihead_attn.out_proj.weight == 0), \
            "cross-attention out_proj should be zero-init"
        if block.multihead_attn.out_proj.bias is not None:
            assert torch.all(block.multihead_attn.out_proj.bias == 0)


# -------------------------------------------------------------------------------------------------
# The critical test: load M6.A weights + run with empty conditioning == unconditional output

def _build_matched_pair():
    """Build a small unconditional + conditioned model with compatible decoder config."""
    arch = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    uncond_cfg = TranslatorConfig(vocab_size=1024, max_seq_len=128, **arch)
    cond_cfg = ConditionedTranslatorConfig(
        dac_vocab_size=1024, max_dac_seq_len=128, **arch,
        jtxtok_vocab_size=DEFAULT_VOCAB.size,
        enc_d_model=32, enc_n_layers=2, enc_n_heads=4, enc_d_ff=64, enc_max_seq_len=64,
    )
    uncond = FlatARTransformer(uncond_cfg)
    cond = ConditionedTranslator(cond_cfg)
    return uncond, cond


def test_load_from_unconditional_succeeds_and_reports_mapping():
    uncond, cond = _build_matched_pair()
    info = cond.load_from_unconditional(uncond.state_dict())
    assert "mapped_summary" in info
    assert any("embed.weight" in line for line in info["mapped_summary"])
    assert any("self_attn" in line for line in info["mapped_summary"])


def test_zero_init_means_conditioned_with_empty_jtxtok_equals_unconditional():
    """LOAD-BEARING: cross-attn out_proj=0 means a freshly-loaded conditioned model is
    output-identical to the unconditional model regardless of the jtxtok context."""
    torch.manual_seed(42)
    uncond, cond = _build_matched_pair()
    cond.load_from_unconditional(uncond.state_dict())

    uncond.eval()
    cond.eval()

    B, L = 2, 24
    dac_ids = torch.randint(0, 1024, (B, L))
    # ANY jtxtok context works — cross-attn out_proj=0 means the contribution is zero
    # regardless of what the encoder produces.
    jtxtok_ids = torch.randint(1, DEFAULT_VOCAB.size, (B, 16))
    role_ids = torch.zeros_like(jtxtok_ids)

    with torch.no_grad():
        uncond_logits = uncond(dac_ids)
        cond_logits = cond(dac_ids, jtxtok_ids, role_ids)

    diff = (uncond_logits - cond_logits).abs().max().item()
    assert diff < 1e-5, (
        f"conditioned-with-empty-conditioning should equal unconditional after weight load; "
        f"max abs diff was {diff} (expected <1e-5)"
    )


def test_param_count_in_realistic_phase_b_config():
    """Sanity check: at the Phase-A defaults the conditioned model is in the 11–20 M range."""
    cfg = ConditionedTranslatorConfig()   # defaults
    model = ConditionedTranslator(cfg)
    n = model.num_parameters()
    # Encoder is small; decoder dominates; M6.A unconditional was ~11M. Cross-attn adds ~2-3M.
    assert 10_000_000 < n < 25_000_000, f"unexpected param count {n}"
