"""Trainer for the parallel factorised RVQ translator (M6.A v2)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .translator_rvq import TranslatorRVQ, TranslatorRVQConfig, ar_loss


@dataclass
class TranslatorRVQTrainConfig:
    # Data
    n_codebooks: int = 9
    vocab_size: int = 1024
    frame_rate: float = 86.1328125
    batch_size: int = 8
    window_seconds: float = 3.0

    # Optimisation
    seed: int = 0
    steps: int = 1700
    lr: float = 3e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0      # default 0; flat layout needed wd>0 to fake learning,
                                    # this layout should not need it. Set explicitly if desired.
    lr_min_ratio: float = 1.0
    warmup_steps: int = 0

    # Logging / checkpointing
    log_every: int = 20
    ckpt_every: int = 500
    ckpt_dir: str = "data/checkpoints/translator/<corpus>/rvq"

    # Model
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0


@dataclass
class TranslatorRVQTrainResult:
    final_step: int
    elapsed_seconds: float
    loss_first_window: float
    loss_last_window: float
    steps_per_second: float
    ckpt_path: str
    losses: list[float]
    nan_steps: int


class FrameBatchSampler:
    """Yield (B, T_frames, K) frame windows from in-RAM token tracks.

    Identical sampling to TokenBatchSampler in train_translator.py but returns
    (B, T, K) instead of (B, K, T) since the model expects frames as the time axis.
    """

    def __init__(self, tracks: list[np.ndarray], window_frames: int, seed: int = 0):
        self.tracks = tracks
        self.window_frames = int(window_frames)
        for i, t in enumerate(tracks):
            if t.shape[-1] < self.window_frames:
                raise ValueError(
                    f"track {i} has {t.shape[-1]} frames < window {self.window_frames}"
                )
        self.rng = np.random.default_rng(seed)
        lens = np.array([t.shape[-1] for t in tracks], dtype=np.float64)
        self.track_probs = lens / lens.sum()

    def sample(self, batch_size: int) -> torch.Tensor:
        B = int(batch_size)
        n_q = self.tracks[0].shape[0]
        out = np.empty((B, self.window_frames, n_q), dtype=np.int64)
        for i in range(B):
            ti = int(self.rng.choice(len(self.tracks), p=self.track_probs))
            T = self.tracks[ti].shape[-1]
            start = int(self.rng.integers(0, T - self.window_frames + 1))
            # tracks are shape (n_q, T); we want (T, n_q) per sample
            out[i] = self.tracks[ti][:, start : start + self.window_frames].T
        return torch.from_numpy(out)


def _save_checkpoint(
    model: TranslatorRVQ,
    cfg: TranslatorRVQTrainConfig,
    completed_steps: int,
    elapsed_seconds: float,
    losses: list[float],
    ckpt_path: Path,
) -> None:
    finite = [v for v in losses if v == v]
    head = finite[: cfg.log_every] if finite else []
    tail = finite[-cfg.log_every:] if finite else []
    head_avg = (sum(head) / len(head)) if head else float("nan")
    tail_avg = (sum(tail) / len(tail)) if tail else float("nan")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "translator_config": asdict(model.cfg),
            "train_config": asdict(cfg),
            "loss_first_window": head_avg,
            "loss_last_window": tail_avg,
            "steps": completed_steps,
            "elapsed_seconds": elapsed_seconds,
        },
        ckpt_path,
    )
    (ckpt_path.parent / "translator_rvq_losses.json").write_text(json.dumps(losses))


def train_translator_rvq(
    tracks: list[np.ndarray],
    cfg: TranslatorRVQTrainConfig,
    device: str,
) -> TranslatorRVQTrainResult:
    """Train the parallel-RVQ LM on the given token tracks."""
    torch.manual_seed(cfg.seed)

    window_frames = int(round(cfg.window_seconds * cfg.frame_rate))

    tcfg = TranslatorRVQConfig(
        vocab_size=cfg.vocab_size,
        n_codebooks=cfg.n_codebooks,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=window_frames + 16,
    )
    model = TranslatorRVQ(tcfg).to(device)
    n = model.num_parameters()
    print(f"  model: {n:,} params  ({n/1e6:.2f} M)")

    sampler = FrameBatchSampler(tracks, window_frames=window_frames, seed=cfg.seed)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    def lr_for_step(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step / max(1, cfg.warmup_steps))
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.lr * (cfg.lr_min_ratio + (1.0 - cfg.lr_min_ratio) * cosine)

    random_baseline = math.log(cfg.vocab_size)
    print(f"  window: {cfg.window_seconds:.1f} s = {window_frames} frames "
          f"(seq len: {window_frames})")
    print(f"  random baseline (uniform per codebook): {random_baseline:.4f}")

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "translator_rvq.pt"
    best_ckpt_path = ckpt_dir / "translator_rvq_best.pt"

    losses: list[float] = []
    log_buf: list[float] = []
    nan_steps = 0
    t0 = time.time()
    interrupted = False
    completed_step = 0
    best_window_loss = float("inf")

    model.train()
    try:
        for step in range(1, cfg.steps + 1):
            x = sampler.sample(cfg.batch_size).to(device, non_blocking=True)
            logits = model(x)
            loss = ar_loss(logits, x)

            lv = float(loss.detach().cpu())
            losses.append(lv)
            log_buf.append(lv)

            if not torch.isfinite(loss):
                nan_steps += 1
                print(f"  step {step:>5d}: loss=NaN — skipping backward+step", flush=True)
            else:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                cur_lr = lr_for_step(step)
                for pg in optim.param_groups:
                    pg["lr"] = cur_lr
                optim.step()

            completed_step = step

            if step % cfg.log_every == 0 or step == cfg.steps:
                finite = [v for v in log_buf if v == v]
                avg = (sum(finite) / len(finite)) if finite else float("nan")
                log_buf.clear()
                elapsed = time.time() - t0
                sps = step / max(elapsed, 1e-9)
                eta_s = (cfg.steps - step) / max(sps, 1e-9)
                with torch.no_grad():
                    embed_norm = float(model.embeds[0].weight.detach().norm().cpu())
                    l0_outproj_norm = float(
                        model.encoder.layers[0].self_attn.out_proj.weight.detach().norm().cpu()
                    )
                print(
                    f"  step {step:>5d}/{cfg.steps}  loss={avg:.4f}  lr={cur_lr:.2e}  "
                    f"|embed0|={embed_norm:.2f}  |L0.outp|={l0_outproj_norm:.2f}  "
                    f"elapsed={elapsed:6.1f}s  rate={sps:.2f} steps/s  eta={eta_s:6.1f}s",
                    flush=True,
                )
                if math.isfinite(avg) and avg < best_window_loss:
                    best_window_loss = avg
                    _save_checkpoint(model, cfg, step, elapsed, losses, best_ckpt_path)
                    print(f"  [best] new best window loss {avg:.4f} -> {best_ckpt_path}",
                          flush=True)

            if step % cfg.ckpt_every == 0:
                _save_checkpoint(model, cfg, step, time.time() - t0, losses, ckpt_path)
                print(f"  [ckpt] saved at step {step} -> {ckpt_path}", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\n[interrupted by SIGINT at step {completed_step}/{cfg.steps}]  "
            f"saving final checkpoint and exiting cleanly.",
            flush=True,
        )

    elapsed = time.time() - t0
    _save_checkpoint(model, cfg, completed_step, elapsed, losses, ckpt_path)

    finite = [v for v in losses if v == v]
    head = finite[: cfg.log_every]
    tail = finite[-cfg.log_every:]
    loss_first_window = (sum(head) / len(head)) if head else float("nan")
    loss_last_window = (sum(tail) / len(tail)) if tail else float("nan")

    if interrupted:
        print(f"[ckpt] final save at step {completed_step}: {ckpt_path}", flush=True)

    return TranslatorRVQTrainResult(
        final_step=completed_step,
        elapsed_seconds=elapsed,
        loss_first_window=loss_first_window,
        loss_last_window=loss_last_window,
        steps_per_second=completed_step / max(elapsed, 1e-9),
        ckpt_path=str(ckpt_path),
        losses=losses,
        nan_steps=nan_steps,
    )
