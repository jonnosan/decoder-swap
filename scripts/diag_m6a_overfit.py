"""Test whether the M6.A 'best' ckpt loss-4.67 is memorization vs broken save.

Sample with the same TokenBatchSampler the trainer used (NumPy RNG, seed=0). If we
see loss ~4.67, the model overfit specific windows. If we see ~6.80, the saved
checkpoint actually has the post-collapse weights despite the metadata claim.
"""
from __future__ import annotations

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
    print(f"ckpt: steps={ckpt['steps']}, loss_first={ckpt['loss_first_window']:.4f}, "
          f"loss_last={ckpt['loss_last_window']:.4f}")

    cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    model = FlatARTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    corpus = load_corpus("techno")
    tokens_dir = corpus.tokens_dir(codec="dac")
    npys = sorted(tokens_dir.glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    print(f"tracks loaded: {[(p.name, t.shape) for p, t in zip(npys, tracks)]}")

    window_frames = int(round(train_cfg["window_seconds"] * train_cfg["frame_rate"]))
    print(f"window_frames: {window_frames}")
    sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])

    # Replay the trainer: draw `steps` batches to advance the RNG to the trainer's state.
    # Then continue drawing to see loss at the same "epoch" as step 1820.
    print(f"\nReplaying sampler for {ckpt['steps']} steps to match trainer RNG state...")
    for _ in range(ckpt["steps"]):
        sampler.sample(train_cfg["batch_size"])

    # Now compute loss on the next 20 batches — these are the windows the trainer
    # WOULD have sampled at steps 1821..1840 (the "last window" in the log).
    losses = []
    for _ in range(20):
        codes = sampler.sample(train_cfg["batch_size"]).to(device)
        flat = flatten_codes(codes)
        with torch.no_grad():
            logits = model(flat)
            losses.append(ar_loss(logits, flat).item())
    print(f"\nLoss on next 20 trainer-RNG-aligned batches: "
          f"mean={sum(losses)/len(losses):.4f} min={min(losses):.4f} max={max(losses):.4f}")
    print(f"  samples: {[f'{x:.3f}' for x in losses[:10]]}")

    # Also: sample completely fresh (seed=0, no replay) — the trainer's loss_first_window
    # came from the first 20 batches at seed=0.
    print("\nFresh sampler at seed=0 (matches trainer's first 20 batches):")
    sampler2 = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])
    fresh_losses = []
    for _ in range(20):
        codes = sampler2.sample(train_cfg["batch_size"]).to(device)
        flat = flatten_codes(codes)
        with torch.no_grad():
            logits = model(flat)
            fresh_losses.append(ar_loss(logits, flat).item())
    print(f"  Loss: mean={sum(fresh_losses)/len(fresh_losses):.4f} "
          f"min={min(fresh_losses):.4f} max={max(fresh_losses):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
