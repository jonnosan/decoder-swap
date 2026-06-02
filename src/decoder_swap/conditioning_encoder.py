"""Conditioning encoder over jtxtok streams (M7.A — PROMPT_3 Part A).

Reads a jtxtok stream + per-token role tags and produces a context tensor that the
decoder's cross-attention consumes.

Architecture (small by design — the vocabulary is small and the patterns are simple):

    token_ids -> token embed (vocab -> d_model)
    role_ids  -> role embed  (n_roles -> d_model)
    pos       -> sinusoidal pos enc (max_len -> d_model)

    h = (tok_embed + role_embed) * sqrt(d_model) + pos_enc
    for layer in encoder: h = layer(h, src_key_padding_mask=pad_mask)
    return h

Output shape (B, L, d_model) — the decoder reads this via cross-attention; pad positions
are masked out so they don't influence attention weights.

Design notes:
- **Bidirectional** (no causal mask) — the entire jtxtok sequence is observed at once;
  the decoder is the AR side.
- **Role embedding is additive on token embedding** so role and identity contribute
  jointly to each token's representation. Embedding scaling by sqrt(d_model) matches
  the decoder side (translator.py M6.A fix) so neither signal dominates pos_enc.
- **Voice filter at inference (PROMPT_3 Part C, Axis 1)** is implemented at the input:
  filtered-out tokens are replaced with the pad token, and the pad mask follows so the
  decoder's cross-attention ignores them. The encoder NEVER learns to "skip" a token;
  filtering happens upstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .jtxtok_vocab import JtxtokVocab

N_ROLES = 6  # struct/drum/bass/key/mt/pad (matches jtxtok_vocab.JtxtokVocab.role_ids)


@dataclass
class ConditioningEncoderConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    max_seq_len: int = 1024     # at 25 fps tokens * ~10 events/bar * say 100 bars + scaffolding
    pad_id: int = 0


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class JtxtokConditioningEncoder(nn.Module):
    """Bidirectional transformer encoder over jtxtok streams. (B, L) ids -> (B, L, d_model)."""

    def __init__(self, cfg: ConditioningEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.role_embed = nn.Embedding(N_ROLES, cfg.d_model)
        self.register_buffer("pos_enc", _sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(cfg.d_model)

        # GPT-style init on embeddings (matches translator.py M6.0 fix).
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_embed.weight, mean=0.0, std=0.02)
        # Depth-aware residual scaling (matches translator.py M6.A fix).
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.encoder.layers:
            nn.init.normal_(block.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)
            nn.init.normal_(block.linear2.weight, mean=0.0, std=residual_std)
            if block.linear2.bias is not None:
                nn.init.zeros_(block.linear2.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        token_ids: torch.Tensor,        # (B, L)
        role_ids: torch.Tensor,         # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of jtxtok streams. Returns (context, pad_mask) where:
            context  : (B, L, d_model) — cross-attention K/V source
            pad_mask : (B, L) bool, True at pad positions (for src_key_padding_mask)
        """
        B, L = token_ids.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(f"jtxtok sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}")
        pad_mask = token_ids == self.cfg.pad_id   # (B, L) — True where padded
        h = self.token_embed(token_ids) + self.role_embed(role_ids)
        h = h * math.sqrt(self.cfg.d_model) + self.pos_enc[:L].unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.out_norm(h)
        return h, pad_mask
