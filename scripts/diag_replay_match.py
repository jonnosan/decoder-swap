"""Replay trainer's RNG to step N, then forward-pass the exact same batch the trainer
used at step N. The saved model's loss on that batch should match losses[N-1] from the
trainer log (4.5700 at step 1700). If not, there's a save/load bug."""
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
    print(f"ckpt: steps={ckpt['steps']}, best_window_loss={ckpt['loss_last_window']:.4f}")

    cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    model = FlatARTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    # Recreate the trainer's data pipeline EXACTLY.
    corpus = load_corpus("techno")
    tokens_dir = corpus.tokens_dir(codec="dac")
    npys = sorted(tokens_dir.glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    window_frames = int(round(train_cfg["window_seconds"] * train_cfg["frame_rate"]))
    sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])
    print(f"window_frames={window_frames}, batch={train_cfg['batch_size']}, seed={train_cfg['seed']}")

    # Read the trainer's recorded losses to compare.
    losses_recorded = json.load(open(REPO_ROOT / "data/checkpoints/translator/techno/translator_losses.json"))
    print(f"trainer recorded losses[1599]={losses_recorded[1599]:.4f}, "
          f"losses[1600]={losses_recorded[1600]:.4f}, "
          f"losses[1601]={losses_recorded[1601]:.4f}")
    print(f"trainer recorded losses[1698]={losses_recorded[1698]:.4f}, "
          f"losses[1699]={losses_recorded[1699]:.4f}")

    # The ckpt has ckpt['steps']=1600 (when best save fired). The state_dict at that
    # save was AFTER step 1600's optimizer.step(). So the saved model would produce
    # losses[1600] for the next batch the trainer drew (step 1601, which losses_recorded
    # says is 6.8214 — a SPIKE step).
    n_replay = ckpt["steps"]
    print(f"\nReplaying sampler for {n_replay} steps to reach trainer state at end of step {n_replay}...")
    for _ in range(n_replay):
        sampler.sample(train_cfg["batch_size"])
    print(f"Now sampling batch the trainer would have used at step {n_replay+1}:")
    codes = sampler.sample(train_cfg["batch_size"]).to(device)
    flat = flatten_codes(codes)

    with torch.no_grad():
        logits = model(flat)
        loss = ar_loss(logits, flat).item()
    print(f"  loaded model loss on this batch: {loss:.4f}")
    print(f"  trainer recorded loss for step {n_replay+1}: {losses_recorded[n_replay]:.4f}")
    print(f"  MATCH?  {abs(loss - losses_recorded[n_replay]) < 0.01}")

    # Also show next 5 batches
    print(f"\nNext 5 batches (steps {n_replay+2} through {n_replay+6}):")
    for i in range(5):
        codes = sampler.sample(train_cfg["batch_size"]).to(device)
        flat = flatten_codes(codes)
        with torch.no_grad():
            logits = model(flat)
            loss = ar_loss(logits, flat).item()
        recorded = losses_recorded[n_replay + i + 1] if n_replay + i + 1 < len(losses_recorded) else float("nan")
        match = "MATCH" if abs(loss - recorded) < 0.05 else "MISMATCH"
        print(f"  step {n_replay+i+2}: ckpt={loss:.4f}  trainer={recorded:.4f}  {match}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
