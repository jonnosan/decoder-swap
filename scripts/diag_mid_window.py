"""Take the saved (post-save = bad-basin) ckpt, train 5 more steps to land in the
good basin mid-log-window, then evaluate that mid-window state on FRESH batches
(different RNG than the trainer's stream)."""
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


def main() -> int:
    device = "cpu"
    base_lm_path = REPO_ROOT / "data/checkpoints/translator/techno/translator_lm_best.pt"
    ckpt = torch.load(str(base_lm_path), map_location="cpu", weights_only=False)
    tc = ckpt["translator_config"]
    train_cfg = ckpt["train_config"]
    n_saved = ckpt["steps"]
    print(f"Loaded ckpt at step {n_saved}, loss_last_window={ckpt['loss_last_window']:.4f}")

    cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    model = FlatARTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Recreate the trainer's data + optim pipeline EXACTLY.
    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    window_frames = int(round(train_cfg["window_seconds"] * train_cfg["frame_rate"]))
    sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])

    # Advance the trainer's sampler RNG to the point right AFTER step n_saved.
    print(f"Replaying sampler for {n_saved} steps to align with trainer state...")
    for _ in range(n_saved):
        sampler.sample(train_cfg["batch_size"])

    # Re-create optimizer with same config.
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # IMPORTANT: also set torch.manual_seed to match the trainer at this point.
    torch.manual_seed(train_cfg["seed"])

    # Continue training for 5 more steps. Step indices match trainer steps n_saved+1 to n_saved+5.
    losses_recorded = json.load(open(REPO_ROOT / "data/checkpoints/translator/techno/translator_losses.json"))

    print(f"\nResuming training. Comparing per-step loss to trainer's recorded losses:")
    model.train()
    for i in range(5):
        step = n_saved + i + 1
        codes = sampler.sample(train_cfg["batch_size"]).to(device)
        flat = flatten_codes(codes)
        logits = model(flat)
        loss = ar_loss(logits, flat)
        lv = float(loss.detach().cpu())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"])
        optim.step()
        recorded = losses_recorded[step - 1]
        match = "MATCH" if abs(lv - recorded) < 0.02 else f"OFF BY {lv-recorded:+.3f}"
        print(f"  step {step}: our loss={lv:.4f}  trainer recorded={recorded:.4f}  [{match}]")

    # Now the model is in the "good basin" (mid-window). Evaluate it on FRESH batches —
    # batches from a different RNG stream the trainer never saw.
    print("\n--- Eval on FRESH batches (RNG never seen by trainer) ---")
    model.eval()
    fresh_losses = []
    for seed in range(5):
        s = TokenBatchSampler(tracks, window_frames=window_frames, seed=100 + seed)
        for _ in range(5):
            codes = s.sample(train_cfg["batch_size"]).to(device)
            flat = flatten_codes(codes)
            with torch.no_grad():
                logits = model(flat)
                fresh_losses.append(ar_loss(logits, flat).item())
    avg = sum(fresh_losses) / len(fresh_losses)
    print(f"  {len(fresh_losses)} fresh batches: mean loss = {avg:.4f}")
    print(f"  min/max: {min(fresh_losses):.4f} / {max(fresh_losses):.4f}")
    print(f"  random baseline (uniform 1024): {torch.log(torch.tensor(1024.0)).item():.4f}")
    print(f"\nVerdict: model {'GENERALIZES (good basin really learned something)' if avg < 5.5 else 'DOES NOT GENERALIZE (oscillation is data-dependent, no real learning)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
