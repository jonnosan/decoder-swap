"""Memorization sanity test for the parallel RVQ trainer.

Train on ONE fixed batch repeated. A working model+trainer drives loss to ~0
within a few hundred steps. If this fails, the rvq architecture/trainer has
a bug; do NOT proceed to a full corpus run."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.train_translator_rvq import FrameBatchSampler  # noqa: E402
from decoder_swap.translator_rvq import (  # noqa: E402
    TranslatorRVQ, TranslatorRVQConfig, ar_loss,
)


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    cfg = TranslatorRVQConfig(
        vocab_size=1024, n_codebooks=9,
        d_model=256, n_layers=4, n_heads=4, d_ff=1024,
        dropout=0.0, max_seq_len=300,
    )
    torch.manual_seed(0)
    model = TranslatorRVQ(cfg).to(device)
    n = model.num_parameters()
    print(f"model: {n:,} params  ({n/1e6:.2f} M)")

    # Fixed batch.
    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    sampler = FrameBatchSampler(tracks, window_frames=258, seed=0)
    fixed_batch = sampler.sample(8).to(device)  # (B=8, T=258, K=9)
    print(f"fixed batch shape: {fixed_batch.shape}")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    print(f"\n{'step':>6s}  {'train loss':>10s}  {'eval loss':>10s}  {'embed.norm':>10s}")
    model.train()
    for step in range(1, 501):
        logits = model(fixed_batch)
        loss = ar_loss(logits, fixed_batch)
        lv_train = loss.item()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                lv_eval = ar_loss(model(fixed_batch), fixed_batch).item()
            embed_norm = float(model.embeds[0].weight.norm())
            print(f"{step:>6d}  {lv_train:>10.4f}  {lv_eval:>10.4f}  {embed_norm:>10.4f}")

    if lv_train < 1.0:
        print("\nVERDICT: trainer CAN drive loss to ~0 on a single batch (model + trainer work)")
        return 0
    else:
        print("\nVERDICT: trainer CANNOT drive loss down — bug in new code, FIX BEFORE PROCEEDING")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
