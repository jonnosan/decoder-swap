"""Evaluate the RVQ best ckpt on fresh batches to check if it generalizes."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.train_translator_rvq import FrameBatchSampler  # noqa: E402
from decoder_swap.translator_rvq import (  # noqa: E402
    TranslatorRVQ, TranslatorRVQConfig, ar_loss,
)


def main() -> int:
    device = "cpu"  # deterministic
    ckpt_path = REPO_ROOT / "data/checkpoints/translator/techno/rvq/translator_rvq_best.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    tc = ckpt["translator_config"]
    print(f"ckpt: steps={ckpt['steps']}  loss_last_window={ckpt['loss_last_window']:.4f}")

    cfg = TranslatorRVQConfig(**tc)
    model = TranslatorRVQ(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]

    # Reproduce trainer's step-21 batch (the spike step right after step 20).
    sampler = FrameBatchSampler(tracks, window_frames=258, seed=0)
    for _ in range(20):
        sampler.sample(8)  # advance to step 21
    batch_21 = sampler.sample(8).to(device)
    with torch.no_grad():
        l = ar_loss(model(batch_21), batch_21).item()
    print(f"\nloss on trainer's step-21 batch (right after best save): {l:.4f}")

    # Fresh batches, RNG never seen by trainer.
    fresh = []
    for s in range(5):
        sm = FrameBatchSampler(tracks, window_frames=258, seed=1000+s)
        for _ in range(5):
            c = sm.sample(8).to(device)
            with torch.no_grad():
                fresh.append(ar_loss(model(c), c).item())
    import statistics
    print(f"fresh-batch loss (25 batches): mean={statistics.mean(fresh):.4f}  "
          f"min={min(fresh):.4f}  max={max(fresh):.4f}")
    print(f"  random baseline: 6.9315")

    if statistics.mean(fresh) < 5.5:
        print("\nVERDICT: model GENERALIZES — architecture works, need to fix training stability")
    elif statistics.mean(fresh) < 6.5:
        print("\nVERDICT: model partially generalizes")
    else:
        print("\nVERDICT: model is at random baseline — still the marginal-trick artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
