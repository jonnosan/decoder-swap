"""CORPUS_NEW loader. Decodes each file once into a mono float32 numpy array, serves random crops."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


class CorpusDataset:
    """Holds one or more audio files in RAM (mono, target_sr float32) and yields random fixed-length crops.

    Not a torch IterableDataset on purpose — for an experiment with one process and small batches,
    plain numpy + a manual sampler is simpler and avoids worker/pickle complexity.
    """

    def __init__(self, paths: list[str | Path], target_sr: int, segment_samples: int, seed: int = 0):
        if not paths:
            raise ValueError("corpus is empty")
        self.target_sr = int(target_sr)
        self.segment_samples = int(segment_samples)
        self.rng = np.random.default_rng(seed)

        self.tracks: list[np.ndarray] = []
        total_samples = 0
        for p in paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(p)
            y, _ = librosa.load(str(p), sr=self.target_sr, mono=True)
            if len(y) < self.segment_samples:
                raise ValueError(f"{p} is shorter than one segment ({len(y)} < {segment_samples})")
            y = y.astype(np.float32, copy=False)
            self.tracks.append(y)
            total_samples += len(y)
        self.total_samples = total_samples
        # Sample track index weighted by length so longer files contribute proportionally.
        lens = np.array([len(t) for t in self.tracks], dtype=np.float64)
        self.track_probs = lens / lens.sum()

    def summary(self) -> str:
        mins = self.total_samples / self.target_sr / 60.0
        return (
            f"{len(self.tracks)} tracks · {mins:.1f} min total · "
            f"segment {self.segment_samples} samples ({self.segment_samples/self.target_sr:.2f} s) "
            f"@ {self.target_sr} Hz"
        )

    def random_batch(self, batch_size: int) -> torch.Tensor:
        """Return (B, 1, segment_samples) float32 CPU tensor of independent random crops."""
        out = np.empty((batch_size, 1, self.segment_samples), dtype=np.float32)
        for i in range(batch_size):
            ti = int(self.rng.choice(len(self.tracks), p=self.track_probs))
            track = self.tracks[ti]
            max_start = len(track) - self.segment_samples
            start = int(self.rng.integers(0, max_start + 1))
            out[i, 0, :] = track[start : start + self.segment_samples]
        return torch.from_numpy(out)
