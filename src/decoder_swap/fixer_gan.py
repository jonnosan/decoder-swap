"""HiFi-GAN-style discriminators + adversarial losses for the fixer.

Two discriminator families, both standard in modern audio GANs:

  Multi-Period Discriminator (MPD):
    Reshape the (B, 1, T) waveform into (B, 1, T//P, P) for several periods
    P ∈ {2, 3, 5, 7, 11} and run 2D convs. Each sub-D catches the patterns
    that repeat at that period (kick at every 2nd sample of a hi-hat grid,
    pitched-tone cycles, etc.).

  Multi-Scale Discriminator (MSD):
    Run 1D convs on the raw waveform at the original rate, /2, and /4.
    Catches non-periodic features at different time scales.

The combined discriminator is the concatenation of all sub-Ds. Each sub-D
returns (logit, intermediate features). The generator loss uses:
  • adversarial term: MSE so sub-Ds output 1 for fake (LSGAN, more stable
    than vanilla BCE for audio).
  • feature-matching term: L1 between real and fake intermediate features
    (per-layer match) — gives the generator a strong, dense gradient
    signal even when the adv term is weak.
  • plus the existing reconstruction loss from fixer.py.

Conventions follow Kong et al. HiFi-GAN (2020), the de-facto reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _norm(m: nn.Module) -> nn.Module:
    return weight_norm(m)


class PeriodDiscriminator(nn.Module):
    """2D-conv discriminator on (T // P, P) reshape of a 1D waveform.

    The width dim (P) is the period. Kernel is (k, 1) so it only spans the
    time axis; each channel acts independently across phase within the period.
    """

    def __init__(self, period: int, channels=(32, 128, 512, 1024, 1024),
                 kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        in_c = 1
        for out_c in channels:
            self.convs.append(_norm(nn.Conv2d(
                in_c, out_c, kernel_size=(kernel_size, 1),
                stride=(stride, 1), padding=((kernel_size - 1) // 2, 0),
            )))
            in_c = out_c
        # Output head — 1 logit per spatial location, then we mean-pool.
        self.head = _norm(nn.Conv2d(channels[-1], 1, kernel_size=(3, 1),
                                    stride=1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        """x: (B, 1, T) -> (logit, features) where features is a list of intermediate maps."""
        B, _, T = x.shape
        # Pad T up to a multiple of period.
        pad = (self.period - (T % self.period)) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
            T = T + pad
        x = x.view(B, 1, T // self.period, self.period)

        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.head(x)
        features.append(x)
        logit = x.flatten(1).mean(dim=1)  # (B,) scalar per example
        return logit, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.subs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        return [sub(x) for sub in self.subs]


class ScaleDiscriminator(nn.Module):
    """1D-conv discriminator on the raw waveform at one resolution."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            _norm(nn.Conv1d(1,   128, kernel_size=15, stride=1, padding=7)),
            _norm(nn.Conv1d(128, 128, kernel_size=41, stride=4, padding=20, groups=4)),
            _norm(nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20, groups=16)),
            _norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=16)),
            _norm(nn.Conv1d(512, 512, kernel_size=5,  stride=1, padding=2)),
        ])
        self.head = _norm(nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.head(x)
        features.append(x)
        logit = x.flatten(1).mean(dim=1)
        return logit, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3):
        super().__init__()
        self.subs = nn.ModuleList([ScaleDiscriminator() for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        outs = []
        for sub in self.subs:
            outs.append(sub(x))
            x = self.pool(x)
        return outs


class HiFiGANDiscriminator(nn.Module):
    """MPD + MSD wrapped as a single module."""

    def __init__(self, periods=(2, 3, 5, 7, 11), n_scales: int = 3):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.msd = MultiScaleDiscriminator(n_scales)

    def forward(self, x):
        return self.mpd(x) + self.msd(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _SmallPeriodDisc(nn.Module):
    """Lightweight variant: same shape, ~4x fewer params per sub-D."""

    def __init__(self, period: int, channels=(16, 64, 128, 256), k: int = 5):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        in_c = 1
        for out_c in channels:
            self.convs.append(_norm(nn.Conv2d(
                in_c, out_c, kernel_size=(k, 1), stride=(3, 1),
                padding=((k - 1) // 2, 0),
            )))
            in_c = out_c
        self.head = _norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        B, _, T = x.shape
        pad = (self.period - (T % self.period)) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
            T += pad
        x = x.view(B, 1, T // self.period, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.head(x)
        feats.append(x)
        return x.flatten(1).mean(dim=1), feats


class _SmallScaleDisc(nn.Module):
    """Lightweight variant of ScaleDiscriminator (halved widths)."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            _norm(nn.Conv1d(1,   64, kernel_size=15, stride=1, padding=7)),
            _norm(nn.Conv1d(64,  64, kernel_size=41, stride=4, padding=20, groups=4)),
            _norm(nn.Conv1d(64,  128, kernel_size=41, stride=4, padding=20, groups=16)),
            _norm(nn.Conv1d(128, 128, kernel_size=5,  stride=1, padding=2)),
        ])
        self.head = _norm(nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.head(x)
        feats.append(x)
        return x.flatten(1).mean(dim=1), feats


class SmallHiFiGANDiscriminator(nn.Module):
    """~4x lighter than the full HiFi-GAN discriminator. Sufficient for our
    scale + 4-5x faster training on MPS."""

    def __init__(self, periods=(2, 3, 5, 7), n_scales: int = 2):
        super().__init__()
        self.subs_p = nn.ModuleList([_SmallPeriodDisc(p) for p in periods])
        self.subs_s = nn.ModuleList([_SmallScaleDisc() for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        outs = [d(x) for d in self.subs_p]
        for sub in self.subs_s:
            outs.append(sub(x))
            x = self.pool(x)
        return outs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------------
# Adversarial losses (LSGAN — Mean Squared Error on logits)

def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    """Push real logits → 1, fake logits → 0. Standard LSGAN form."""
    loss = 0.0
    for (r_logit, _), (f_logit, _) in zip(real_outs, fake_outs):
        loss = loss + ((r_logit - 1.0) ** 2).mean() + (f_logit ** 2).mean()
    return loss / max(len(real_outs), 1)


def generator_adv_loss(fake_outs) -> torch.Tensor:
    """Generator wants D to call its outputs real (logit → 1)."""
    loss = 0.0
    for f_logit, _ in fake_outs:
        loss = loss + ((f_logit - 1.0) ** 2).mean()
    return loss / max(len(fake_outs), 1)


def feature_matching_loss(real_outs, fake_outs) -> torch.Tensor:
    """L1 between real and fake intermediate features, averaged over layers/sub-Ds.
    Detach real features so the FM loss only flows into the generator."""
    loss = 0.0
    n = 0
    for (_, r_feats), (_, f_feats) in zip(real_outs, fake_outs):
        for r, f in zip(r_feats, f_feats):
            loss = loss + F.l1_loss(f, r.detach())
            n += 1
    return loss / max(n, 1)
