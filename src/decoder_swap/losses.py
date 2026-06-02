"""Reconstruction loss for decoder fine-tune: multi-scale mel-spec L1 + waveform L1.

DAC's real training recipe also uses adversarial losses. We skip those — for a short fine-tune that
just re-voices the decoder, mel + waveform reconstruction is the strongest signal and trains in
hours, not days.

Mel transforms use torchaudio (MPS-compatible) rather than audiotools (which wraps AudioSignal and
has its own internal STFT path that I'd rather not depend on for this experiment).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class MultiScaleMelLoss(nn.Module):
    """L1 distance between log-mel-spectrograms at several STFT scales.

    Multi-scale trick: small windows capture transients, large windows capture pitch/harmonic
    structure. Summing across scales gives a single loss well-conditioned for both.

    Forward+backward computed on CPU. MPS's STFT backward produces NaN gradients on small/zero
    bins even with `power=2.0 + clamp` (a torch/MPS bug we verified empirically). Gradient still
    flows correctly back to MPS-resident decoder params via cross-device autograd. Overhead:
    one ~1 MB MPS↔CPU copy per loss call.
    """

    DEVICE = "cpu"

    def __init__(self, sample_rate: int, scales: tuple[tuple[int, int, int], ...] | None = None):
        super().__init__()
        if scales is None:
            scales = ((2048, 512, 80), (1024, 256, 80), (512, 128, 64))
        self.mels = nn.ModuleList(
            [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    n_mels=n_mels,
                    power=2.0,
                    center=True,
                )
                for (n_fft, hop, n_mels) in scales
            ]
        )
        self._force_cpu()

    def _force_cpu(self) -> None:
        for mel in self.mels:
            mel.to(self.DEVICE)

    # Intercept .to() so external code can't accidentally move us back to MPS.
    def to(self, *args, **kwargs):  # type: ignore[override]
        return self

    def cuda(self, *args, **kwargs):  # type: ignore[override]
        return self

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """y_hat, y: (B, 1, T) audio tensors; may be on any device. Loss returns on y_hat.device."""
        orig_device = y_hat.device
        y_hat_c = y_hat.to(self.DEVICE)
        y_c = y.to(self.DEVICE)
        if y_hat_c.shape != y_c.shape:
            n = min(y_hat_c.shape[-1], y_c.shape[-1])
            y_hat_c = y_hat_c[..., :n]
            y_c = y_c[..., :n]
        loss = y_hat_c.new_zeros(())
        for mel in self.mels:
            m_hat = torch.log(torch.clamp(mel(y_hat_c.squeeze(1)), min=1e-5))
            m = torch.log(torch.clamp(mel(y_c.squeeze(1)), min=1e-5))
            loss = loss + F.l1_loss(m_hat, m)
        return (loss / len(self.mels)).to(orig_device)


def waveform_l1(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = min(y_hat.shape[-1], y.shape[-1])
    return F.l1_loss(y_hat[..., :n], y[..., :n])
