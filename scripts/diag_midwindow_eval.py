"""Evaluate the midwindow.pt ckpt (saved at step 305, where trainer recorded loss 4.6585)
on FRESH batches. If loss ~4.66 on fresh batches: model generalizes, save-timing bug.
If loss ~6.8: model output is uniform-random regardless of batch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.train_translator import TokenBatchSampler  # noqa: E402
from decoder_swap.translator import FlatARTransformer, TranslatorConfig, ar_loss, flatten_codes  # noqa: E402


def eval_ckpt(label: str, ckpt_path: Path, tracks, train_cfg, losses_recorded=None,
              expected_step=None) -> None:
    print(f"\n=== {label}: {ckpt_path.name} ===")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    tc = ckpt["translator_config"]
    print(f"  metadata: steps={ckpt['steps']} loss_last_window={ckpt['loss_last_window']:.4f}")

    cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    model = FlatARTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    window_frames = int(round(train_cfg["window_seconds"] * train_cfg["frame_rate"]))

    # 1) Check loss on the EXACT batch the trainer drew at step `expected_step`
    #    (i.e., the batch right after the save).
    if expected_step is not None and losses_recorded is not None and expected_step <= len(losses_recorded):
        sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])
        for _ in range(expected_step - 1):
            sampler.sample(train_cfg["batch_size"])
        codes = sampler.sample(train_cfg["batch_size"])
        flat = flatten_codes(codes)
        with torch.no_grad():
            loss_replay = ar_loss(model(flat), flat).item()
        trainer_recorded = losses_recorded[expected_step - 1]
        print(f"  loss on trainer's step {expected_step} batch:")
        print(f"    our:     {loss_replay:.4f}")
        print(f"    trainer: {trainer_recorded:.4f}")
    elif expected_step is not None:
        # No trainer record (step beyond training end). Just sample what the trainer
        # would have drawn next.
        sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])
        for _ in range(expected_step - 1):
            sampler.sample(train_cfg["batch_size"])
        codes = sampler.sample(train_cfg["batch_size"])
        flat = flatten_codes(codes)
        with torch.no_grad():
            loss_replay = ar_loss(model(flat), flat).item()
        print(f"  loss on trainer's step {expected_step} batch (would-be next):")
        print(f"    our:     {loss_replay:.4f}  (no trainer record)")

    # 2) Loss on FRESH batches (RNG never seen by trainer)
    fresh_losses = []
    for seed in range(5):
        s = TokenBatchSampler(tracks, window_frames=window_frames, seed=1000 + seed)
        for _ in range(5):
            codes = s.sample(train_cfg["batch_size"])
            flat = flatten_codes(codes)
            with torch.no_grad():
                fresh_losses.append(ar_loss(model(flat), flat).item())
    avg = sum(fresh_losses) / len(fresh_losses)
    print(f"  fresh-batch loss (25 batches): mean={avg:.4f} min={min(fresh_losses):.4f} max={max(fresh_losses):.4f}")


def main() -> int:
    ckpt_dir = REPO_ROOT / "data/checkpoints/translator/techno/diag_midwindow"
    losses_recorded = json.load(open(ckpt_dir / "translator_losses.json"))

    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    train_cfg = {
        "window_seconds": 3.0, "frame_rate": 86.1328125,
        "seed": 0, "batch_size": 8,
    }

    # Boundary ckpt: translator_lm.pt was last saved at the END of training (step 305).
    # Actually we want a log-boundary save (step % 20 == 0). The last log-boundary save
    # of translator_lm_best.pt was at step 260 (best window 4.8881).
    # And translator_lm.pt was saved via ckpt_every=500 - none in this run except the end.
    # So compare:
    #   translator_lm_best.pt    (log boundary, step 260)
    #   translator_lm_midwindow.pt (mid-window step 305, loss 4.66)
    #   translator_lm.pt         (end-of-training, step 305 — same as midwindow effectively)
    eval_ckpt("BEST CKPT (boundary save at step 260)",
              ckpt_dir / "translator_lm_best.pt", tracks, train_cfg, losses_recorded,
              expected_step=261)
    eval_ckpt("MID-WINDOW CKPT (step 305)",
              ckpt_dir / "translator_lm_midwindow.pt", tracks, train_cfg, losses_recorded,
              expected_step=306)  # would be 306 if trainer continued
    eval_ckpt("END CKPT (step 305)",
              ckpt_dir / "translator_lm.pt", tracks, train_cfg, losses_recorded,
              expected_step=306)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
