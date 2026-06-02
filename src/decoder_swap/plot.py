"""M5 figures: waveform / spectrogram / onset / RMS overlays for S1 vs S2, plus training loss curve."""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def _onset_times(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", hop_length=512)


def _rms(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(r)), sr=sr, hop_length=512)
    return t, r


def comparison_figure(s1: np.ndarray, s2: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    """2x2: (waveforms overlay) (mel-spec S1) (mel-spec S2) (RMS+onsets overlay)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    t = np.arange(len(s1)) / sr

    # 1) Waveforms overlaid
    ax = axes[0, 0]
    ax.plot(t, s1, color="C0", linewidth=0.4, alpha=0.7, label="S1 (D1)")
    ax.plot(t, s2, color="C3", linewidth=0.4, alpha=0.5, label="S2 (D2 techno-trained)")
    ax.set_title("Waveform overlay")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude"); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, t[-1])

    # 2) S1 mel-spectrogram
    ax = axes[0, 1]
    m1 = librosa.power_to_db(librosa.feature.melspectrogram(y=s1, sr=sr, n_mels=128) + 1e-10)
    librosa.display.specshow(m1, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("S1 mel-spectrogram (D1 decoder)")

    # 3) S2 mel-spectrogram
    ax = axes[1, 1]
    m2 = librosa.power_to_db(librosa.feature.melspectrogram(y=s2, sr=sr, n_mels=128) + 1e-10)
    librosa.display.specshow(m2, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("S2 mel-spectrogram (D2 decoder)")

    # 4) RMS envelopes overlaid + onset markers
    ax = axes[1, 0]
    t1, r1 = _rms(s1, sr)
    t2, r2 = _rms(s2, sr)
    ax.plot(t1, r1, color="C0", label="S1 RMS", linewidth=1.0)
    ax.plot(t2, r2, color="C3", label="S2 RMS", linewidth=1.0, alpha=0.7)
    # Onset markers (sparse — cap to first 80 from each so plot stays readable on long clips)
    o1 = _onset_times(s1, sr)[:80]
    o2 = _onset_times(s2, sr)[:80]
    for x in o1:
        ax.axvline(x, color="C0", linestyle="--", alpha=0.25, linewidth=0.6)
    for x in o2:
        ax.axvline(x, color="C3", linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_title("RMS envelopes + onset markers (first 80 onsets)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("RMS"); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, t[-1])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def training_loss_figure(losses_json: Path, out_path: Path) -> None:
    losses = json.loads(Path(losses_json).read_text())
    losses = np.array(losses, dtype=np.float64)
    finite_mask = np.isfinite(losses)
    steps = np.arange(1, len(losses) + 1)

    # Smoothed (50-step moving avg) for trend visibility.
    win = 50
    kernel = np.ones(win) / win
    safe = np.where(finite_mask, losses, np.nan)
    # Pandas-free moving avg with NaN handling: replace NaN with previous valid value first.
    safe_fill = safe.copy()
    last = np.nan
    for i in range(len(safe_fill)):
        if np.isfinite(safe_fill[i]):
            last = safe_fill[i]
        elif np.isfinite(last):
            safe_fill[i] = last
    safe_fill = np.where(np.isfinite(safe_fill), safe_fill, np.nanmedian(safe_fill))
    smooth = np.convolve(safe_fill, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(steps[finite_mask], losses[finite_mask], color="C0", alpha=0.25,
            linewidth=0.6, label="per-step loss")
    ax.plot(steps, smooth, color="C3", linewidth=1.6, label=f"{win}-step moving avg")
    ax.set_title("M3 decoder fine-tune — training loss")
    ax.set_xlabel("step"); ax.set_ylabel("mel + waveform L1")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
