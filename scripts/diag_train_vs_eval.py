"""Test whether the M6.A model produces different output in .train() vs .eval() mode."""
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
    print(f"ckpt: steps={ckpt['steps']}, loss_last={ckpt['loss_last_window']:.4f}, "
          f"reported dropout={tc.get('dropout', 'n/a')}")

    cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    model = FlatARTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corpus = load_corpus("techno")
    tokens_dir = corpus.tokens_dir(codec="dac")
    npys = sorted(tokens_dir.glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    window_frames = int(round(train_cfg["window_seconds"] * train_cfg["frame_rate"]))
    sampler = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])

    print("\n--- Same 5 batches, .train() vs .eval() ---")
    losses_train = []
    losses_eval = []
    for i in range(5):
        codes = sampler.sample(train_cfg["batch_size"]).to(device)
        flat = flatten_codes(codes)
        model.train()
        with torch.no_grad():
            logits_t = model(flat)
            l_t = ar_loss(logits_t, flat).item()
        model.eval()
        with torch.no_grad():
            logits_e = model(flat)
            l_e = ar_loss(logits_e, flat).item()
        diff = (logits_t - logits_e).abs().max().item()
        losses_train.append(l_t)
        losses_eval.append(l_e)
        print(f"  batch {i}: train={l_t:.4f}  eval={l_e:.4f}  logits-max-diff={diff:.6f}")
    print(f"  avg train: {sum(losses_train)/len(losses_train):.4f}")
    print(f"  avg eval : {sum(losses_eval)/len(losses_eval):.4f}")

    print("\n--- Check that loss with grad enabled also matches (train mode, no grad-disable) ---")
    sampler2 = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])
    losses_train_grad = []
    model.train()
    for i in range(5):
        codes = sampler2.sample(train_cfg["batch_size"]).to(device)
        flat = flatten_codes(codes)
        logits = model(flat)         # NO torch.no_grad()
        l = ar_loss(logits, flat).item()
        losses_train_grad.append(l)
        print(f"  batch {i}: train+grad={l:.4f}")
    print(f"  avg train+grad: {sum(losses_train_grad)/len(losses_train_grad):.4f}")

    # Show some logits stats to debug
    print("\n--- Logits stats for the first batch (eval mode) ---")
    sampler3 = TokenBatchSampler(tracks, window_frames=window_frames, seed=train_cfg["seed"])
    codes = sampler3.sample(train_cfg["batch_size"]).to(device)
    flat = flatten_codes(codes)
    model.eval()
    with torch.no_grad():
        logits = model(flat)
    print(f"  logits shape: {logits.shape}")
    print(f"  logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}], mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
    # Per-position max logit
    probs = logits.softmax(-1)
    top1 = probs.max(-1).values
    print(f"  per-position p(top1): mean={top1.mean().item():.4f} median={top1.median().item():.4f}")
    # What fraction of positions have correct prediction?
    pred = logits.argmax(-1)
    correct = (pred[:, :-1] == flat[:, 1:]).float()
    print(f"  next-token accuracy: {correct.mean().item():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
