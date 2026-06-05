"""Trainer for the parallel factorised RVQ translator (M6.A v2).

Optionally accepts per-frame MIDI conditioning (see midi_conditioning.py and
issue #10 Phase 1B.1). When `conds=None` the trainer is byte-identical to the
original unconditional memorize-test substrate.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .midi_conditioning import FrameConditioning, _bend_center_bin
from .translator_rvq import CondConfig, TranslatorRVQ, TranslatorRVQConfig, ar_loss


@dataclass
class TranslatorRVQTrainConfig:
    # Data
    n_codebooks: int = 9
    vocab_size: int = 1024
    frame_rate: float = 86.1328125
    batch_size: int = 8
    window_seconds: float = 3.0

    # Conditioning data alignment
    # cond_shift_frames: at input position t, feed cond[start+t+shift]. shift=1 aligns
    #   conditioning with the prediction TARGET (the next frame) rather than the current
    #   input frame — breaks the redundancy with x[t]. Default 0 = same-frame alignment.
    # cond_dropout_p: probability of zeroing the entire cond batch (per step). Forces the
    #   model to be able to predict without cond, which in turn forces it to use cond when
    #   present (classifier-free-guidance-style training).
    cond_shift_frames: int = 0
    cond_dropout_p: float = 0.0

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

    When `conds` is provided (one FrameConditioning per track, aligned with the
    corresponding token track), `sample()` also returns a parallel-window
    conditioning dict. Otherwise `cond` is None.
    """

    def __init__(
        self,
        tracks: list[np.ndarray],
        window_frames: int,
        seed: int = 0,
        conds: list[FrameConditioning] | None = None,
        cond_shift_frames: int = 0,
    ):
        self.tracks = tracks
        self.window_frames = int(window_frames)
        self.cond_shift_frames = int(cond_shift_frames)
        if self.cond_shift_frames < 0:
            raise ValueError("cond_shift_frames must be >= 0")
        if conds is None and self.cond_shift_frames != 0:
            raise ValueError("cond_shift_frames requires conds")
        # Per-track upper bound on the random start: start + W + shift <= T.
        for i, t in enumerate(tracks):
            min_len = self.window_frames + self.cond_shift_frames
            if t.shape[-1] < min_len:
                raise ValueError(
                    f"track {i} has {t.shape[-1]} frames < window+shift {min_len}"
                )
        if conds is not None:
            if len(conds) != len(tracks):
                raise ValueError(
                    f"conds len {len(conds)} != tracks len {len(tracks)}"
                )
            for i, (t, c) in enumerate(zip(tracks, conds, strict=False)):
                if c.n_frames != t.shape[-1]:
                    raise ValueError(
                        f"track {i}: conditioning has {c.n_frames} frames, "
                        f"tokens have {t.shape[-1]} — must match"
                    )
        self.conds = conds
        self.rng = np.random.default_rng(seed)
        lens = np.array([t.shape[-1] for t in tracks], dtype=np.float64)
        self.track_probs = lens / lens.sum()

    def sample(self, batch_size: int) -> torch.Tensor:
        """Backwards-compatible: returns just x (B, T, K). Drops conditioning if any.
        Use sample_pair() to also receive the parallel conditioning batch.
        """
        x, _ = self.sample_pair(batch_size)
        return x

    def sample_pair(
        self, batch_size: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        B = int(batch_size)
        n_q = self.tracks[0].shape[0]
        out = np.empty((B, self.window_frames, n_q), dtype=np.int64)
        idxs = np.empty(B, dtype=np.int64)
        starts = np.empty(B, dtype=np.int64)
        for i in range(B):
            ti = int(self.rng.choice(len(self.tracks), p=self.track_probs))
            T = self.tracks[ti].shape[-1]
            max_start = T - self.window_frames - self.cond_shift_frames + 1
            start = int(self.rng.integers(0, max_start))
            out[i] = self.tracks[ti][:, start : start + self.window_frames].T
            idxs[i] = ti
            starts[i] = start
        x = torch.from_numpy(out)
        if self.conds is None:
            return x, None
        n_p = self.conds[0].pitch_active.shape[1]
        pa = np.empty((B, self.window_frames, n_p), dtype=np.float32)
        vb = np.empty((B, self.window_frames), dtype=np.int64)
        bb = np.empty((B, self.window_frames), dtype=np.int64)
        op = np.empty((B, self.window_frames), dtype=np.int64)
        W = self.window_frames
        shift = self.cond_shift_frames
        for i in range(B):
            c = self.conds[int(idxs[i])]
            s = int(starts[i]) + shift
            pa[i] = c.pitch_active[s : s + W].astype(np.float32)
            vb[i] = c.velocity_bin[s : s + W].astype(np.int64)
            bb[i] = c.bend_bin[s : s + W].astype(np.int64)
            op[i] = c.onset_phase[s : s + W].astype(np.int64)
        cond = {
            "pitch_active": torch.from_numpy(pa),
            "velocity_bin": torch.from_numpy(vb),
            "bend_bin": torch.from_numpy(bb),
            "onset_phase": torch.from_numpy(op),
        }
        return x, cond


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
    conds: list[FrameConditioning] | None = None,
    cond_cfg: CondConfig | None = None,
) -> TranslatorRVQTrainResult:
    """Train the parallel-RVQ LM on the given token tracks.

    When `conds` is provided (parallel to `tracks`, same frame counts), the
    model is built with conditioning and trained with paired (x, cond) windows.
    `cond_cfg` is required when `conds` is provided (sets the embedding shapes).
    """
    torch.manual_seed(cfg.seed)

    window_frames = int(round(cfg.window_seconds * cfg.frame_rate))

    if (conds is None) != (cond_cfg is None):
        raise ValueError("conds and cond_cfg must be set together (or both None)")

    tcfg = TranslatorRVQConfig(
        vocab_size=cfg.vocab_size,
        n_codebooks=cfg.n_codebooks,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=window_frames + 16,
        cond=cond_cfg,
    )
    model = TranslatorRVQ(tcfg).to(device)
    n = model.num_parameters()
    print(f"  model: {n:,} params  ({n/1e6:.2f} M)"
          + (f"  (conditioning on)" if conds is not None else "  (unconditional)"))

    sampler = FrameBatchSampler(
        tracks, window_frames=window_frames, seed=cfg.seed, conds=conds,
        cond_shift_frames=cfg.cond_shift_frames,
    )
    if conds is not None:
        if cfg.cond_shift_frames:
            print(f"  cond shift: +{cfg.cond_shift_frames} frame(s) "
                  f"(input pos t conditioned on cond[start+t+{cfg.cond_shift_frames}])")
        if cfg.cond_dropout_p > 0:
            print(f"  cond dropout p={cfg.cond_dropout_p:.2f} (whole-batch)")
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

    bend_center = _bend_center_bin(cond_cfg.n_bend_bins) if cond_cfg is not None else 0
    cond_dropped_count = 0
    model.train()
    try:
        for step in range(1, cfg.steps + 1):
            x, cond = sampler.sample_pair(cfg.batch_size)
            x = x.to(device, non_blocking=True)
            if cond is not None:
                cond = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
                if cfg.cond_dropout_p > 0 and torch.rand(1).item() < cfg.cond_dropout_p:
                    cond = {
                        "pitch_active": torch.zeros_like(cond["pitch_active"]),
                        "velocity_bin": torch.zeros_like(cond["velocity_bin"]),
                        "bend_bin":     torch.full_like(cond["bend_bin"], bend_center),
                        "onset_phase":  torch.zeros_like(cond["onset_phase"]),
                    }
                    cond_dropped_count += 1
            logits = model(x, cond=cond)
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
