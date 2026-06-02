"""Small autoregressive transformer over flat-interleaved DAC tokens.

For the M6.0 feasibility smoke (issue #6 first step) we need to verify that DAC token sequences
are AR-predictable with a tiny model — *before* committing to the full translator architecture's
design choices (per-codebook vs flat vocab, conditioning recipe). This module is intentionally
minimal:

  - vocab = DAC codebook_size (1024)
  - input sequence = the 9 RVQ codes per frame interleaved into one stream
    so a 3 s crop is 3 * 86.13 * 9 ≈ 2326 tokens
  - causal self-attention, sinusoidal positional encoding (no learnable pos params),
    weight-tied input embedding and output head

Param budget: with d_model=256, n_layers=4, n_heads=4, ffn=1024 this is ~3.2 M params.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TranslatorConfig:
    vocab_size: int = 1024
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    max_seq_len: int = 4096


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FlatARTransformer(nn.Module):
    """Causal next-token transformer. Input: (B, L) token ids. Output: (B, L, vocab) logits."""

    def __init__(self, cfg: TranslatorConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
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
        self.norm = nn.LayerNorm(cfg.d_model)
        # Weight-tied output head: reuse self.embed.weight as the classifier matrix.
        # GPT-style init — keeps logits' variance ≈ O(1) at init so the initial cross-entropy
        # sits near ln(vocab) (the random-uniform baseline) instead of blowing up because a
        # LayerNormed hidden state dotted with N(0,1)-scaled embedding weights gives logits
        # of magnitude ~sqrt(d_model). With std=0.02, init logits ≈ N(0, 0.02·sqrt(d)), and
        # the loss starts near ln(vocab) as expected.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}")
        h = self.embed(x) + self.pos_enc[:L].unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.encoder(h, mask=causal, is_causal=True)
        h = self.norm(h)
        logits = h @ self.embed.weight.t()  # (B, L, vocab)
        return logits


def flatten_codes(codes: torch.Tensor) -> torch.Tensor:
    """(B, n_codebooks, T_frames) int -> (B, n_codebooks * T_frames) int by interleaving per frame.

    Order within a frame is codebook 0 first, then 1, ..., then n-1, then move to next frame.
    Matches what the trainer learns and what an eventual decoder-side de-interleaver will undo.
    """
    B, n_q, T = codes.shape
    return codes.transpose(1, 2).reshape(B, T * n_q)


def ar_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy.

    `logits`: (B, L, vocab) — logits at every position predicting THAT position's token (teacher-forced).
    `target`: (B, L) — token ids. The shift is done here: predict target[:, 1:] from logits[:, :-1].
    """
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        target[:, 1:].reshape(-1).long(),
    )
