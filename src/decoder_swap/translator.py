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
        # Embedding init: GPT-style std=0.02. Without this, the LayerNormed hidden state
        # dotted with N(0,1)-scaled embedding weights produces logits of magnitude
        # ~sqrt(d_model), so the M6.0 smoke saw initial loss ~56 instead of ~ln(vocab)=6.93.
        # With std=0.02, initial logits stay O(1) and loss starts at the random baseline.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Depth-aware residual scaling (GPT-2 style). The output of each pre-norm block
        # adds to the residual stream; if every layer contributes with O(1) variance, the
        # stream's variance grows linearly with depth and the loss landscape at init goes
        # flat — exactly the pathology we saw in M6.A (11 M-param Phase A model glued at
        # random baseline for 5000 steps; issue #9). Scaling the *output* projection of
        # each block (MHA's out_proj + FFN's second linear) by 1/sqrt(2 * n_layers) keeps
        # each block's contribution to the residual stream O(1/sqrt(n_layers)), so the
        # sum across n_layers has O(1) variance regardless of depth.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.encoder.layers:
            # `block` is a TransformerEncoderLayer. The two residual-output projections are:
            #   block.self_attn.out_proj — MultiheadAttention's output projection
            #   block.linear2            — the second linear in the FFN
            nn.init.normal_(block.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)
            nn.init.normal_(block.linear2.weight, mean=0.0, std=residual_std)
            if block.linear2.bias is not None:
                nn.init.zeros_(block.linear2.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {L} exceeds max_seq_len {self.cfg.max_seq_len}")
        # Scale the embedded signal by sqrt(d_model) before adding positional info.
        # With embed.weight ~ N(0, 0.02), un-scaled embed(x) has per-element magnitude
        # ~0.02 while the sinusoidal positional encoding has magnitude ~1 — so position
        # dominates and the model can't learn token-conditional predictions (the M6.A
        # regression in issue #9). Standard transformer practice (Vaswani et al. 2017
        # §3.4) scales embeddings by sqrt(d_model) so token and position contribute on
        # comparable scales. At d_model=384 this multiplies embed(x) by ~19.6.
        h = self.embed(x) * math.sqrt(self.cfg.d_model) + self.pos_enc[:L].unsqueeze(0)
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
