"""Parallel factorised RVQ translator — M6.A v2.

The sequence axis is FRAMES (not flat-interleaved codebook tokens). At each frame
position the model handles all K=9 codebooks in parallel:

  - Input:  (B, T_frames, K) of token ids, each in [0, vocab)
  - Output: (B, T_frames, K, vocab) of logits

Per frame:
  - K independent embedding tables (each vocab_size x d_model); sum them to get the
    per-frame input vector.
  - Standard causal transformer trunk over frame positions.
  - K weight-tied output heads (head k reuses embedding table k as the classifier).

Loss is mean cross-entropy across the K codebooks at each predicted frame.

This avoids the per-position codebook-routing shortcut that the flat-interleaved
layout encouraged (see project memory: m6a-flat-layout-dead-end).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .midi_conditioning import N_ONSET_PHASES


@dataclass
class CondConfig:
    """Per-frame conditioning embedding shape. Set on TranslatorRVQConfig.cond
    to enable conditioning. None = unconditional behaviour (memorize-test baseline).
    """
    n_pitches: int = 27           # pitch_hi - pitch_lo + 1; matches FrameCondConfig defaults
    n_velocity_bins: int = 8
    n_bend_bins: int = 16
    n_onset_phases: int = N_ONSET_PHASES
    n_timbres: int = 0            # 0 = no timbre cond (back-compat); >0 enables a per-NOTE
                                  # timbre embedding broadcast across all frames of the window.


@dataclass
class TranslatorRVQConfig:
    vocab_size: int = 1024        # per codebook
    n_codebooks: int = 9
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    max_seq_len: int = 512         # max number of FRAMES (not flat tokens)
    cond: CondConfig | None = None  # optional per-frame conditioning


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TranslatorRVQ(nn.Module):
    """Parallel-codebook causal LM over RVQ frames.

    Forward: (B, T, K) ints -> (B, T, K, V) logits, where logits[b, t, k] predicts
    the codebook-k token at frame t+1 (teacher forced from frame t).
    """

    def __init__(self, cfg: TranslatorRVQConfig):
        super().__init__()
        self.cfg = cfg
        # K independent embedding tables — one per codebook.
        self.embeds = nn.ModuleList([
            nn.Embedding(cfg.vocab_size, cfg.d_model) for _ in range(cfg.n_codebooks)
        ])
        self.register_buffer(
            "pos_enc",
            _sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
        )

        # Optional per-frame MIDI conditioning embeddings. When cfg.cond is None
        # the model behaves identically to the unconditional memorize-test baseline.
        if cfg.cond is not None:
            cc = cfg.cond
            self.pitch_embed = nn.Embedding(cc.n_pitches, cfg.d_model)
            self.velocity_embed = nn.Embedding(cc.n_velocity_bins, cfg.d_model)
            self.bend_embed = nn.Embedding(cc.n_bend_bins, cfg.d_model)
            self.onset_embed = nn.Embedding(cc.n_onset_phases, cfg.d_model)
            self.timbre_embed = (nn.Embedding(cc.n_timbres, cfg.d_model)
                                 if cc.n_timbres > 0 else None)
        else:
            self.pitch_embed = None
            self.velocity_embed = None
            self.bend_embed = None
            self.onset_embed = None
            self.timbre_embed = None

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
        # Output heads are weight-tied to the corresponding embedding table:
        # logits[k] = norm(h) @ embeds[k].weight.t().

        # Init: GPT-style 0.02 std embeddings + depth-aware residual projection init.
        for emb in self.embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        if cfg.cond is not None:
            for emb in (self.pitch_embed, self.velocity_embed, self.bend_embed, self.onset_embed):
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            if self.timbre_embed is not None:
                nn.init.normal_(self.timbre_embed.weight, mean=0.0, std=0.02)
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
        x: torch.Tensor,
        cond: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """x: (B, T, K) int -> logits (B, T, K, V).

        When the model was built with cfg.cond set, pass `cond` as a dict with:
          pitch_active : (B, T, n_pitches) float/uint multi-hot
          velocity_bin : (B, T) long
          bend_bin     : (B, T) long
          onset_phase  : (B, T) long
        The conditioning embedding is summed into the per-frame input before pos enc.
        """
        B, T, K = x.shape
        if K != self.cfg.n_codebooks:
            raise ValueError(f"expected {self.cfg.n_codebooks} codebooks, got {K}")
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        if (cond is None) != (self.cfg.cond is None):
            raise ValueError(
                "cond must be provided iff cfg.cond was set when building the model"
            )

        # Sum the K per-codebook embeddings to get one d_model vector per frame.
        # Scale by sqrt(d_model) so the embedded signal magnitude matches the positional
        # encoding (matches Vaswani et al. and the working memorize-test setup).
        h = sum(self.embeds[k](x[:, :, k]) for k in range(K)) * math.sqrt(self.cfg.d_model)

        if cond is not None:
            # Multi-hot pitch: matmul against the embedding table.
            pa = cond["pitch_active"].to(self.pitch_embed.weight.dtype)   # (B, T, n_pitches)
            pitch_sum = pa @ self.pitch_embed.weight                       # (B, T, d_model)
            h = h + (
                pitch_sum
                + self.velocity_embed(cond["velocity_bin"].long())
                + self.bend_embed(cond["bend_bin"].long())
                + self.onset_embed(cond["onset_phase"].long())
            )
            # Per-NOTE timbre: a single int per sample, embedded once and broadcast.
            if self.timbre_embed is not None and "timbre_id" in cond:
                t_emb = self.timbre_embed(cond["timbre_id"].long())   # (B, d_model)
                h = h + t_emb.unsqueeze(1)                             # broadcast over T

        h = h + self.pos_enc[:T].unsqueeze(0)

        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=causal, is_causal=True)
        h = self.norm(h)

        # K weight-tied output heads.
        logits = torch.stack(
            [h @ self.embeds[k].weight.t() for k in range(K)],
            dim=2,
        )  # (B, T, K, V)
        return logits


def ar_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Teacher-forced next-frame cross-entropy.

    logits: (B, T, K, V) — predictions at every position.
    target: (B, T, K)    — token ids.

    Standard next-frame: position t's logits predict frame t+1's K tokens.
    Returns the mean CE over all (B, T-1, K) predicted slots.
    """
    B, T, K, V = logits.shape
    return F.cross_entropy(
        logits[:, :-1, :, :].reshape(-1, V),
        target[:, 1:, :].reshape(-1).long(),
    )
