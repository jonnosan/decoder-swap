"""Fine-tune ONLY the decoder on CORPUS_NEW.

Pipeline per step:
  1. Sample a batch of random crops from the corpus  (B, 1, T_samp)
  2. With NO grad: preprocess + frozen encoder + frozen quantizer  -> z  (B, D, T_frames)
  3. Trainable decoder forward                                      -> y_hat
  4. Loss = mel(y_hat, y) + λ * waveform_l1(y_hat, y)
  5. backward + step

Invariants enforced:
  - encoder + quantizer require_grad=False before step 1 (freeze.py guard)
  - codebook fingerprints unchanged from start (checked every `codebook_check_every` steps)
  - if either breaks, raise — do NOT silently continue, the experiment becomes invalid
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .codec_io import Codec
from .dataset import CorpusDataset
from .freeze import freeze_for_decoder_training, remove_weight_norm_recursive
from .invariants import assert_codebook_unchanged, snapshot_codebook_fingerprints
from .losses import MultiScaleMelLoss, waveform_l1


@dataclass
class TrainConfig:
    sample_rate: int
    batch_size: int
    segment_seconds: float
    lr: float
    steps: int
    log_every: int
    codebook_check_every: int
    ckpt_every: int = 200           # save checkpoint every N steps so early-termination retains value
    waveform_weight: float = 1.0
    grad_clip: float = 5.0          # max global grad norm; protects weight_norm convs from blowing up
    ckpt_dir: str = "data/checkpoints"
    seed: int = 0


@dataclass
class TrainResult:
    final_step: int
    elapsed_seconds: float
    loss_first_window: float      # mean of first `log_every` FINITE-loss steps (after warmup)
    loss_last_window: float       # mean of last `log_every` FINITE-loss steps
    steps_per_second: float
    ckpt_path: str
    losses: list[float]           # per-step total loss (may contain NaN if a step blew up)
    nan_steps: int                # how many steps produced NaN loss (gradient skipped on those)


def _seconds_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def _save_checkpoint(
    codec: Codec,
    cfg: TrainConfig,
    completed_steps: int,
    elapsed_seconds: float,
    snapshot: list[str],
    losses: list[float],
    ckpt_path: Path,
) -> None:
    """Save D2 + sidecar JSON. Overwrites in place so M4 always finds the latest at the same path."""
    finite = [v for v in losses if v == v]
    head = finite[: cfg.log_every]
    tail = finite[-cfg.log_every:]
    head_avg = (sum(head) / len(head)) if head else float("nan")
    tail_avg = (sum(tail) / len(tail)) if tail else float("nan")
    torch.save(
        {
            "decoder_state_dict": codec.decoder.state_dict(),
            "train_config": asdict(cfg),
            "convention": asdict(codec.convention),
            "codebook_fingerprints_start": snapshot,
            "loss_first_window": head_avg,
            "loss_last_window": tail_avg,
            "steps": completed_steps,
            "elapsed_seconds": elapsed_seconds,
            "decoder_weight_norm_removed": True,
        },
        ckpt_path,
    )
    (ckpt_path.parent / "d2_losses.json").write_text(json.dumps(losses))


def train_d2(codec: Codec, corpus: CorpusDataset, cfg: TrainConfig) -> TrainResult:
    """Fine-tune codec.decoder in place. Returns metrics + checkpoint path.

    Checkpoints every `ckpt_every` steps. Catches KeyboardInterrupt — on Ctrl-C / SIGINT, finishes
    the current step, saves a final checkpoint, and returns normally with `final_step` set to the
    last completed step. So you can stop early and still keep the work.
    """
    torch.manual_seed(cfg.seed)
    device = next(codec.decoder.parameters()).device

    # Freeze front-end, double-check counts.
    freeze_for_decoder_training(codec)
    # Collapse weight_norm parametrizations in the decoder so its backward path is
    # well-behaved on MPS (see freeze.remove_weight_norm_recursive docstring).
    n_removed = remove_weight_norm_recursive(codec.decoder)
    print(f"  removed weight_norm from {n_removed} decoder submodules (MPS-safe backward)")
    codec.encoder.eval()
    codec.quantizer.eval()
    codec.decoder.train()

    # Loss + optimiser.
    mel_loss = MultiScaleMelLoss(sample_rate=cfg.sample_rate).to(device)
    optim = torch.optim.AdamW(
        [p for p in codec.decoder.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    # Frozen-codebook snapshot — checked periodically; any drift is fatal.
    snapshot = snapshot_codebook_fingerprints(codec)

    losses: list[float] = []
    t0 = time.time()
    log_buf: list[float] = []
    nan_steps = 0
    sr = cfg.sample_rate

    decoder_params = [p for p in codec.decoder.parameters() if p.requires_grad]

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "d2_decoder.pt"

    interrupted = False
    completed_step = 0
    try:
        for step in range(1, cfg.steps + 1):
            x = corpus.random_batch(cfg.batch_size).to(device, non_blocking=True)

            with torch.no_grad():
                x_pre = codec.model.preprocess(x, sr)
                z, _codes, _latents, _cm, _cb = codec.model.encode(x_pre)

            y_hat = codec.model.decode(z)
            loss = mel_loss(y_hat, x_pre) + cfg.waveform_weight * waveform_l1(y_hat, x_pre)

            lv = float(loss.detach().cpu())
            losses.append(lv)
            log_buf.append(lv)

            if not torch.isfinite(loss):
                nan_steps += 1
                print(f"  step {step:>5d}: loss={lv}  (non-finite — skipping backward+step)", flush=True)
            else:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=cfg.grad_clip)
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

            if step % cfg.codebook_check_every == 0:
                assert_codebook_unchanged(codec, snapshot)

            if step % cfg.ckpt_every == 0:
                _save_checkpoint(codec, cfg, step, time.time() - t0, snapshot, losses, ckpt_path)
                print(f"  [ckpt] saved at step {step} -> {ckpt_path}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"\n[interrupted by SIGINT at step {completed_step}/{cfg.steps}]  "
            f"saving final checkpoint and exiting cleanly.",
            flush=True,
        )

    # Final codebook check no matter what (even after interrupt).
    assert_codebook_unchanged(codec, snapshot)

    elapsed = time.time() - t0
    # Always save once more at the end (covers both the clean-completion and interrupted paths).
    _save_checkpoint(codec, cfg, completed_step, elapsed, snapshot, losses, ckpt_path)

    finite_losses = [v for v in losses if v == v]
    head = finite_losses[: cfg.log_every]
    tail = finite_losses[-cfg.log_every:]
    loss_first_window = (sum(head) / len(head)) if head else float("nan")
    loss_last_window = (sum(tail) / len(tail)) if tail else float("nan")

    if interrupted:
        print(f"[ckpt] final save at step {completed_step}: {ckpt_path}", flush=True)

    return TrainResult(
        final_step=completed_step,
        elapsed_seconds=elapsed,
        loss_first_window=loss_first_window,
        loss_last_window=loss_last_window,
        steps_per_second=completed_step / max(elapsed, 1e-9),
        ckpt_path=str(ckpt_path),
        losses=losses,
        nan_steps=nan_steps,
    )


def load_d2_into(codec: Codec, ckpt_path: str | Path) -> dict:
    """Load a D2 decoder checkpoint INTO `codec.decoder` and return the saved metadata.

    Caller is responsible for asserting codec invariants (token convention + codebook fingerprints)
    match the checkpoint's saved versions BEFORE trusting the resulting model.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    codec.decoder.load_state_dict(state["decoder_state_dict"])
    return {k: v for k, v in state.items() if k != "decoder_state_dict"}
