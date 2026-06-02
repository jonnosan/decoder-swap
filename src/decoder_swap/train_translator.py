"""Train a flat-AR transformer LM on cached DAC tokens.

This is the Phase-A trainer of the token-translator build (issue #6) — a scaled-up version of
the M6.0 feasibility smoke. Same architecture (`FlatARTransformer`), bigger model defaults
(~7–19 M params depending on config), longer training (thousands of steps), checkpoint-on-tick
+ SIGINT-safe so a long MPS run keeps its work even if interrupted.

Mirrors the conventions of `train_decoder.py` (TrainConfig/TrainResult + checkpointed train_*).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .translator import FlatARTransformer, TranslatorConfig, ar_loss, flatten_codes


@dataclass
class TranslatorTrainConfig:
    # Token shape
    n_codebooks: int = 9
    vocab_size: int = 1024
    frame_rate: float = 86.1328125    # DAC 44 kHz hop=512

    # Sampling
    batch_size: int = 8
    window_seconds: float = 3.0
    seed: int = 0

    # Optimisation
    steps: int = 5000
    lr: float = 3e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 0     # linear lr ramp 0 -> lr over this many steps; 0 disables warmup

    # Logging / checkpointing
    log_every: int = 20
    ckpt_every: int = 500
    ckpt_dir: str = "data/checkpoints"

    # Model
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0


@dataclass
class TranslatorTrainResult:
    final_step: int
    elapsed_seconds: float
    loss_first_window: float
    loss_last_window: float
    steps_per_second: float
    ckpt_path: str
    losses: list[float]
    nan_steps: int


class TokenBatchSampler:
    """Yield (B, n_codebooks, window_frames) crops from in-RAM token tracks."""

    def __init__(self, tracks: list[np.ndarray], window_frames: int, seed: int = 0):
        self.tracks = tracks
        self.window_frames = int(window_frames)
        for i, t in enumerate(tracks):
            if t.shape[-1] < self.window_frames:
                raise ValueError(f"track {i} has {t.shape[-1]} frames < window {self.window_frames}")
        self.rng = np.random.default_rng(seed)
        lens = np.array([t.shape[-1] for t in tracks], dtype=np.float64)
        self.track_probs = lens / lens.sum()

    def sample(self, batch_size: int) -> torch.Tensor:
        B = int(batch_size)
        n_q = self.tracks[0].shape[0]
        out = np.empty((B, n_q, self.window_frames), dtype=np.int64)
        for i in range(B):
            ti = int(self.rng.choice(len(self.tracks), p=self.track_probs))
            T = self.tracks[ti].shape[-1]
            start = int(self.rng.integers(0, T - self.window_frames + 1))
            out[i] = self.tracks[ti][:, start : start + self.window_frames]
        return torch.from_numpy(out)


def _save_checkpoint(
    model: FlatARTransformer,
    cfg: TranslatorTrainConfig,
    completed_steps: int,
    elapsed_seconds: float,
    losses: list[float],
    ckpt_path: Path,
) -> None:
    """Save model + sidecar. Overwrites in place so the same path always holds the latest."""
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
    (ckpt_path.parent / "translator_losses.json").write_text(json.dumps(losses))


def train_translator(
    tracks: list[np.ndarray],
    cfg: TranslatorTrainConfig,
    device: str,
) -> TranslatorTrainResult:
    """Train a flat-AR transformer LM on the given token tracks. Checkpoints every cfg.ckpt_every.

    Catches KeyboardInterrupt: on SIGINT, finishes the current step, saves a final checkpoint,
    returns normally with `final_step` set to the last completed step.
    """
    torch.manual_seed(cfg.seed)

    window_frames = int(round(cfg.window_seconds * cfg.frame_rate))
    flat_len = window_frames * cfg.n_codebooks

    tcfg = TranslatorConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=flat_len + 16,
    )
    model = FlatARTransformer(tcfg).to(device)
    n_params = model.num_parameters()
    print(f"  model: {n_params:,} params  ({n_params/1e6:.2f} M)")

    sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=cfg.seed)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    def lr_for_step(step: int) -> float:
        """Linear warmup over cfg.warmup_steps, then constant. Step is 1-indexed."""
        if cfg.warmup_steps <= 0:
            return cfg.lr
        if step >= cfg.warmup_steps:
            return cfg.lr
        return cfg.lr * (step / cfg.warmup_steps)

    random_baseline = math.log(cfg.vocab_size)
    print(f"  window: {cfg.window_seconds:.1f} s = {window_frames} frames = {flat_len} flat tokens")
    print(f"  random baseline (uniform): {random_baseline:.4f}")
    if cfg.warmup_steps > 0:
        print(f"  warmup: linear lr ramp 0 -> {cfg.lr} over first {cfg.warmup_steps} steps")

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "translator_lm.pt"

    losses: list[float] = []
    log_buf: list[float] = []
    nan_steps = 0
    t0 = time.time()
    interrupted = False
    completed_step = 0

    model.train()
    try:
        for step in range(1, cfg.steps + 1):
            codes = sampler.sample(cfg.batch_size).to(device, non_blocking=True)
            flat = flatten_codes(codes)
            logits = model(flat)
            loss = ar_loss(logits, flat)

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
                # Apply warmup lr schedule by mutating the optimizer's lr in-place per step.
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
                print(
                    f"  step {step:>5d}/{cfg.steps}  loss={avg:.4f}  "
                    f"elapsed={elapsed:6.1f}s  rate={sps:.2f} steps/s  eta={eta_s:6.1f}s",
                    flush=True,
                )

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

    return TranslatorTrainResult(
        final_step=completed_step,
        elapsed_seconds=elapsed,
        loss_first_window=loss_first_window,
        loss_last_window=loss_last_window,
        steps_per_second=completed_step / max(elapsed, 1e-9),
        ckpt_path=str(ckpt_path),
        losses=losses,
        nan_steps=nan_steps,
    )


def load_translator(ckpt_path: str | Path, device: str) -> tuple[FlatARTransformer, dict]:
    """Load a translator checkpoint. Returns (model_on_device, metadata)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tcfg = TranslatorConfig(**state["translator_config"])
    model = FlatARTransformer(tcfg)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    return model, {k: v for k, v in state.items() if k != "model_state_dict"}
