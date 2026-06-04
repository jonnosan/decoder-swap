"""Compare midwindow ckpt loss on CPU vs MPS, and check if same-batch forward is
deterministic on MPS."""
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


def test_on(device: str, ckpt, tracks, batch):
    cfg = TranslatorConfig(
        vocab_size=1024, d_model=256, n_layers=4, n_heads=4, d_ff=1024,
        dropout=0.0, max_seq_len=2338,
    )
    model = FlatARTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    flat = flatten_codes(batch).to(device)
    losses = []
    with torch.no_grad():
        for _ in range(3):
            l = ar_loss(model(flat), flat).item()
            losses.append(l)
    return losses


def main() -> int:
    ckpt_path = REPO_ROOT / "data/checkpoints/translator/techno/diag_midwindow/translator_lm_midwindow.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"ckpt: steps={ckpt['steps']}  loss_last_window={ckpt['loss_last_window']:.4f}")

    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]

    # Use a fresh batch (RNG never seen by trainer).
    sampler = TokenBatchSampler(tracks, window_frames=258, seed=999)
    batch = sampler.sample(8)

    print("\n=== CPU ===")
    losses_cpu = test_on("cpu", ckpt, tracks, batch)
    print(f"  3 forwards on same batch: {[round(l,4) for l in losses_cpu]}")

    print("\n=== MPS ===")
    losses_mps = test_on("mps", ckpt, tracks, batch)
    print(f"  3 forwards on same batch: {[round(l,4) for l in losses_mps]}")

    print(f"\nCPU mean: {sum(losses_cpu)/3:.4f}")
    print(f"MPS mean: {sum(losses_mps)/3:.4f}")
    print(f"Device gap: {sum(losses_mps)/3 - sum(losses_cpu)/3:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
