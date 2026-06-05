"""Token-domain discriminator + LSGAN losses for per-note GAN training.

The per-note generator (TranslatorRVQ) outputs a probability distribution over
DAC codebook tokens at every (frame, codebook) slot. To train against a
discriminator that says "is this a real bass token sequence?" we need:

  - REAL inputs to D: ground-truth DAC tokens (B, T, K) ints
  - FAKE inputs to D: the generator's predicted distribution, transformed into
    a representation compatible with the same D — but in a way that gradients
    still flow back into the generator.

Approach: each codebook gets its own embedding lookup table inside D. For real
data, we use a hard embedding lookup. For fake data, we use the soft expected
embedding (probs @ embedding_weights) — differentiable in the generator's logits.

Output of D: a per-batch scalar score + a list of intermediate feature maps for
feature-matching loss.

This is dramatically simpler than the audio-domain HiFi-GAN scaffolding (which
would require decoding through DAC at every training step) and trains quickly
on MPS. The architecture is inspired by HiFi-GAN's MSD but operates on token
embeddings rather than waveform samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _norm(m: nn.Module) -> nn.Module:
    return weight_norm(m)


class TokenDiscriminator(nn.Module):
    """1D-conv discriminator over per-codebook token embeddings.

    Accepts either:
      - tokens: (B, T, K) long ids (hard lookup) — for real data
      - probs:  (B, T, K, V) softmax distribution (soft expected lookup) — for fake data

    Returns (logit, features) where logit is (B,) and features is a list of
    intermediate activation maps for use in feature-matching loss.
    """

    def __init__(
        self,
        n_codebooks: int = 9,
        vocab_size: int = 1024,
        d_emb: int = 32,
        channels: tuple[int, ...] = (128, 128, 256, 256),
        kernel_size: int = 5,
        stride: int = 2,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        # Per-codebook embedding tables (independent — the same token id in
        # codebook 0 and codebook 1 mean different things).
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab_size, d_emb) for _ in range(n_codebooks)
        ])
        nn.init.normal_(self.embeds[0].weight, mean=0.0, std=0.02)
        for emb in self.embeds[1:]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        in_c = n_codebooks * d_emb
        self.convs = nn.ModuleList()
        cur_c = in_c
        for i, out_c in enumerate(channels):
            s = stride if i < len(channels) - 1 else 1
            self.convs.append(_norm(nn.Conv1d(
                cur_c, out_c, kernel_size=kernel_size, stride=s,
                padding=(kernel_size - 1) // 2,
            )))
            cur_c = out_c
        self.head = _norm(nn.Conv1d(cur_c, 1, kernel_size=3, padding=1))

    def _embed_hard(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T, K) long -> (B, K*d_emb, T)."""
        B, T, K = tokens.shape
        parts = [self.embeds[k](tokens[:, :, k]) for k in range(K)]   # K tensors of (B, T, d_emb)
        stacked = torch.cat(parts, dim=-1)                              # (B, T, K*d_emb)
        return stacked.permute(0, 2, 1)                                 # (B, K*d_emb, T)

    def _embed_soft(self, probs: torch.Tensor) -> torch.Tensor:
        """probs: (B, T, K, V) softmax -> (B, K*d_emb, T)."""
        B, T, K, V = probs.shape
        parts = []
        for k in range(K):
            # (B, T, V) @ (V, d_emb) = (B, T, d_emb)
            parts.append(probs[:, :, k] @ self.embeds[k].weight)
        stacked = torch.cat(parts, dim=-1)                              # (B, T, K*d_emb)
        return stacked.permute(0, 2, 1)                                 # (B, K*d_emb, T)

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert (tokens is None) != (probs is None), \
            "pass exactly one of tokens=(B,T,K) ints or probs=(B,T,K,V) softmax"
        h = self._embed_hard(tokens) if tokens is not None else self._embed_soft(probs)

        features: list[torch.Tensor] = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        h = self.head(h)                                                # (B, 1, T')
        features.append(h)
        logit = h.flatten(1).mean(dim=1)                                # (B,)
        return logit, features

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------------
# Adversarial losses (LSGAN — MSE on logits, more stable than vanilla BCE for audio).

def d_lsgan_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    """Push real -> 1, fake -> 0."""
    return ((real_logit - 1.0) ** 2).mean() + (fake_logit ** 2).mean()


def g_lsgan_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Generator wants D to say its outputs are real (logit -> 1)."""
    return ((fake_logit - 1.0) ** 2).mean()


def feature_matching_loss(
    real_features: list[torch.Tensor],
    fake_features: list[torch.Tensor],
) -> torch.Tensor:
    """L1 between real and fake intermediate features. Detach real so the loss only
    influences the generator.
    """
    loss = 0.0
    n = 0
    for r, f in zip(real_features, fake_features, strict=False):
        loss = loss + F.l1_loss(f, r.detach())
        n += 1
    return loss / max(n, 1)
