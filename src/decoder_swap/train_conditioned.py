"""Train the M7 jtxtok-conditioned translator (PROMPT_3 Parts B + D).

Fine-tunes a `ConditionedTranslator` initialised from the M6.A unconditional base LM
checkpoint. Implements the two non-negotiable dropouts from spec §7.1:

  - **CFG dropout**: with probability `cfg_dropout_prob` (~10–20%), the entire conditioning
    is replaced with a single PAD token. Required for classifier-free guidance at inference.
  - **MT dropout**: with probability `mt_dropout_prob`, every `MT_*` token in the
    conditioning is replaced with PAD. Teaches "MT absent ⇒ play straight; MT present
    ⇒ honour offset." (Spec §7.1, §7.2.)

Both dropouts are applied independently per training step at the BATCH level (one
decision per step, not per-sample).

Borrows the lr-warmup + checkpoint pattern from train_translator (translator.py M6.A) so a
SIGINT-safe resume strategy is consistent across milestones.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .conditioned_translator import (
    ConditionedTranslator,
    ConditionedTranslatorConfig,
    ar_loss,
)
from .jtxtok_dataset import JtxtokDacDataset
from .jtxtok_vocab import JtxtokVocab
from .translator import flatten_codes


@dataclass
class ConditionedTrainConfig:
    # Optimisation (matches M6.A's proven settings)
    steps: int = 5000
    lr: float = 1e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 50

    # Logging / checkpointing
    log_every: int = 20
    ckpt_every: int = 500
    ckpt_dir: str = "data/checkpoints/translator/<corpus>/m7"
    seed: int = 0

    # NON-NEGOTIABLE dropouts (PROMPT_3 §B3, spec §7.1)
    cfg_dropout_prob: float = 0.15        # spec recommends 10–20%
    mt_dropout_prob: float = 0.3          # MT dropout — independent of CFG dropout

    # Loaded base LM checkpoint (set externally by the script)
    base_lm_ckpt_path: str | None = None


@dataclass
class ConditionedTrainResult:
    final_step: int
    elapsed_seconds: float
    loss_first_window: float
    loss_last_window: float
    steps_per_second: float
    ckpt_path: str
    losses: list[float]
    nan_steps: int
    cfg_drop_steps: int                    # how many steps had conditioning dropped
    mt_drop_steps: int                     # how many steps had MT_* tokens dropped


def _apply_cfg_dropout(
    jtxtok_ids: torch.Tensor, jtxtok_roles: torch.Tensor, *, pad_id: int, pad_role: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the entire conditioning with a single PAD token, batch-wide.

    Returns (jtxtok_ids', jtxtok_roles') with shape (B, 1) — the decoder still receives a
    valid encoder context but it's all-padding so cross-attn's pad mask zeros it out.
    """
    B = jtxtok_ids.shape[0]
    return (
        torch.full((B, 1), pad_id, dtype=jtxtok_ids.dtype, device=jtxtok_ids.device),
        torch.full((B, 1), pad_role, dtype=jtxtok_roles.dtype, device=jtxtok_roles.device),
    )


def _apply_mt_dropout(
    jtxtok_ids: torch.Tensor, jtxtok_roles: torch.Tensor, vocab: JtxtokVocab,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace every `MT_*` token (role == "mt") with PAD, in-place on copies.

    Role-based instead of value-based so the rule applies uniformly across the MT range.
    """
    mt_role_idx = ("struct", "drum", "bass", "key", "mt", "pad").index("mt")
    pad_role_idx = ("struct", "drum", "bass", "key", "mt", "pad").index("pad")
    is_mt = (jtxtok_roles == mt_role_idx)
    new_ids = torch.where(is_mt,
                          torch.tensor(vocab.pad_id, dtype=jtxtok_ids.dtype, device=jtxtok_ids.device),
                          jtxtok_ids)
    new_roles = torch.where(is_mt,
                            torch.tensor(pad_role_idx, dtype=jtxtok_roles.dtype, device=jtxtok_roles.device),
                            jtxtok_roles)
    return new_ids, new_roles


def _lr_for_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * (step / warmup_steps)


def _save_checkpoint(
    model: ConditionedTranslator,
    cfg: ConditionedTrainConfig,
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
            "model_config": asdict(model.cfg),
            "train_config": asdict(cfg),
            "loss_first_window": head_avg,
            "loss_last_window": tail_avg,
            "steps": completed_steps,
            "elapsed_seconds": elapsed_seconds,
        },
        ckpt_path,
    )
    (ckpt_path.parent / "conditioned_losses.json").write_text(json.dumps(losses))


def train_conditioned(
    dataset: JtxtokDacDataset,
    model: ConditionedTranslator,
    vocab: JtxtokVocab,
    cfg: ConditionedTrainConfig,
    device: str,
    batch_size: int,
) -> ConditionedTrainResult:
    """Train the conditioned model on (jtxtok, DAC) pairs with CFG + MT dropouts."""
    torch.manual_seed(cfg.seed)

    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pad_role_idx = ("struct", "drum", "bass", "key", "mt", "pad").index("pad")

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "conditioned.pt"

    losses: list[float] = []
    log_buf: list[float] = []
    nan_steps = 0
    cfg_drop_steps = 0
    mt_drop_steps = 0
    t0 = time.time()
    interrupted = False
    completed_step = 0

    cfg_rng = torch.Generator(device="cpu")
    cfg_rng.manual_seed(cfg.seed + 1)

    try:
        for step in range(1, cfg.steps + 1):
            batch = dataset.sample_batch(batch_size)
            dac_codes = batch["dac_ids"].to(device, non_blocking=True)
            jtxtok_ids = batch["jtxtok_ids"].to(device, non_blocking=True)
            jtxtok_roles = batch["jtxtok_roles"].to(device, non_blocking=True)

            # CFG dropout (replace entire conditioning with PAD)
            if torch.rand((), generator=cfg_rng).item() < cfg.cfg_dropout_prob:
                jtxtok_ids, jtxtok_roles = _apply_cfg_dropout(
                    jtxtok_ids, jtxtok_roles, pad_id=vocab.pad_id, pad_role=pad_role_idx,
                )
                cfg_drop_steps += 1
            elif torch.rand((), generator=cfg_rng).item() < cfg.mt_dropout_prob:
                # MT dropout (strip MT_* tokens) — only when CFG dropout didn't fire
                jtxtok_ids, jtxtok_roles = _apply_mt_dropout(jtxtok_ids, jtxtok_roles, vocab)
                mt_drop_steps += 1

            dac_flat = flatten_codes(dac_codes)             # (B, L_dac)
            logits = model(dac_flat, jtxtok_ids, jtxtok_roles)
            loss = ar_loss(logits, dac_flat)

            lv = float(loss.detach().cpu())
            losses.append(lv)
            log_buf.append(lv)

            if not torch.isfinite(loss):
                nan_steps += 1
                print(f"  step {step:>5d}: loss=NaN — skipping", flush=True)
            else:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                cur_lr = _lr_for_step(step, cfg.lr, cfg.warmup_steps)
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
                    f"elapsed={elapsed:6.1f}s  rate={sps:.2f} steps/s  eta={eta_s:6.1f}s  "
                    f"cfg_drops={cfg_drop_steps} mt_drops={mt_drop_steps}",
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

    return ConditionedTrainResult(
        final_step=completed_step,
        elapsed_seconds=elapsed,
        loss_first_window=loss_first_window,
        loss_last_window=loss_last_window,
        steps_per_second=completed_step / max(elapsed, 1e-9),
        ckpt_path=str(ckpt_path),
        losses=losses,
        nan_steps=nan_steps,
        cfg_drop_steps=cfg_drop_steps,
        mt_drop_steps=mt_drop_steps,
    )


def load_conditioned_with_base_lm(
    cond_cfg: ConditionedTranslatorConfig,
    base_lm_ckpt_path: str | Path,
    device: str,
) -> ConditionedTranslator:
    """Build a fresh ConditionedTranslator + load M6.A base LM weights into its decoder."""
    state = torch.load(str(base_lm_ckpt_path), map_location="cpu", weights_only=False)
    base_lm_state = state["model_state_dict"]
    model = ConditionedTranslator(cond_cfg)
    info = model.load_from_unconditional(base_lm_state)
    print(f"  loaded base LM: {len(info['mapped_summary'])} mapped, "
          f"{len(info['unmapped_source_keys'])} skipped")
    return model.to(device)
