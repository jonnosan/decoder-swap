"""Test whether torch.save mutates model parameters as a side effect."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.train_translator import TokenBatchSampler  # noqa: E402
from decoder_swap.translator import FlatARTransformer, TranslatorConfig, ar_loss, flatten_codes  # noqa: E402


def snapshot(model):
    """Return a CPU-cloned dict of param tensors."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def diff(a, b):
    """Return max abs diff across all corresponding tensors."""
    max_diff = 0.0
    which = None
    for k in a:
        d = (a[k] - b[k]).abs().max().item()
        if d > max_diff:
            max_diff = d
            which = k
    return max_diff, which


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    # Use the trainer config to build a model matching the broken run.
    cfg = TranslatorConfig(
        vocab_size=1024, d_model=256, n_layers=4, n_heads=4, d_ff=1024,
        dropout=0.0, max_seq_len=2338,
    )
    torch.manual_seed(0)
    model = FlatARTransformer(cfg).to(device)

    # Sample some batches and do a couple training steps, then save.
    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    sampler = TokenBatchSampler(tracks, window_frames=258, seed=0)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Do 50 training steps to get the model into a non-trivial state.
    model.train()
    for step in range(50):
        codes = sampler.sample(8).to(device)
        flat = flatten_codes(codes)
        logits = model(flat)
        loss = ar_loss(logits, flat)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
    print(f"after 50 steps: last loss = {loss.item():.4f}")

    # Sample one more batch to test the model's loss BEFORE the save.
    test_codes = sampler.sample(8).to(device)
    test_flat = flatten_codes(test_codes)
    with torch.no_grad():
        loss_before = ar_loss(model(test_flat), test_flat).item()
    print(f"loss BEFORE save: {loss_before:.4f}")

    # SNAPSHOT 1: before save
    snap1 = snapshot(model)

    # SAVE — this is what the trainer does.
    save_path = REPO_ROOT / "data/checkpoints/translator/techno/diag_midwindow/save_mutation_test.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"saved to {save_path.name}")

    # SNAPSHOT 2: after save
    snap2 = snapshot(model)

    # Compare snapshots
    md, key = diff(snap1, snap2)
    print(f"max abs diff of params BEFORE vs AFTER torch.save: {md:.2e}  (worst: {key})")

    # Test loss again on the same batch
    with torch.no_grad():
        loss_after = ar_loss(model(test_flat), test_flat).item()
    print(f"loss AFTER save (same batch): {loss_after:.4f}")
    print(f"loss delta: {loss_after - loss_before:+.4f}")

    # Test loss on a fresh batch
    fresh_codes = sampler.sample(8).to(device)
    fresh_flat = flatten_codes(fresh_codes)
    with torch.no_grad():
        loss_fresh = ar_loss(model(fresh_flat), fresh_flat).item()
    print(f"loss AFTER save (fresh batch): {loss_fresh:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
