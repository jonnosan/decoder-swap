"""Measure M6.A unconditional model's loss across many random windows of the same dataset
the trainer drew from, to see whether the claimed loss_last_window=4.67 is real."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.jtxtok_dataset import (  # noqa: E402
    JtxtokDacDataset,
    JtxtokDacDatasetConfig,
    discover_track_pairs,
)
from decoder_swap.jtxtok_vocab import DEFAULT_VOCAB  # noqa: E402
from decoder_swap.translator import FlatARTransformer, TranslatorConfig, ar_loss, flatten_codes  # noqa: E402


def main() -> int:
    device = "cpu"
    base_lm_path = REPO_ROOT / "data/checkpoints/translator/techno/translator_lm_best.pt"
    ckpt = torch.load(str(base_lm_path), map_location="cpu", weights_only=False)
    tc = ckpt["translator_config"]
    print(f"ckpt train_config window_seconds={ckpt['train_config'].get('window_seconds')}, "
          f"batch_size={ckpt['train_config'].get('batch_size')}, "
          f"seed={ckpt['train_config'].get('seed')}")
    print(f"ckpt reported loss_first={ckpt['loss_first_window']:.4f} "
          f"loss_last={ckpt['loss_last_window']:.4f} at step {ckpt['steps']}")

    cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    model = FlatARTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    corpus = load_corpus("techno")
    pairs = discover_track_pairs(
        corpus.tokens_dir(codec="dac"), REPO_ROOT / "data/jtxtok/techno"
    )
    print(f"tracks: {[p.stem for p in pairs]}")
    print(f"DAC shapes: {[p.dac.shape for p in pairs]}")

    losses = []
    # Sample 50 batches at batch_size=8 (matches M6.A training)
    for trial in range(5):
        ds_cfg = JtxtokDacDatasetConfig(window_seconds=3.0, max_jtxtok_len=256, seed=trial)
        ds = JtxtokDacDataset(pairs, DEFAULT_VOCAB, ds_cfg)
        trial_losses = []
        for _ in range(10):
            batch = ds.sample_batch(8)
            dac_flat = flatten_codes(batch["dac_ids"]).to(device)
            with torch.no_grad():
                logits = model(dac_flat)
                loss = ar_loss(logits, dac_flat).item()
            trial_losses.append(loss)
        losses.extend(trial_losses)
        print(f"trial seed={trial}: mean={sum(trial_losses)/len(trial_losses):.4f} "
              f"min={min(trial_losses):.4f} max={max(trial_losses):.4f}")

    print(f"\n=== Across {len(losses)} batches of size 8 ===")
    print(f"mean loss: {sum(losses)/len(losses):.4f}")
    print(f"min/max:   {min(losses):.4f} / {max(losses):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
