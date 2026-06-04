"""SANITY TEST: train the M6.A architecture on ONE fixed batch, repeated. A working
LM trainer should drive loss to ~0 within a few hundred steps. If ours can't, the bug
is in our trainer/model — not in the data, scale, or hyperparameters.

Logs both per-step training loss AND, every N steps, a separate "no-grad forward on
the same fixed batch" — to detect any discrepancy between the trainer's loss reading
and the model's actual forward output."""
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


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    cfg = TranslatorConfig(
        vocab_size=1024, d_model=256, n_layers=4, n_heads=4, d_ff=1024,
        dropout=0.0, max_seq_len=2338,
    )
    torch.manual_seed(0)
    model = FlatARTransformer(cfg).to(device)
    n = model.num_parameters()
    print(f"model: {n:,} params")

    # Load the corpus + draw ONE fixed batch.
    corpus = load_corpus("techno")
    npys = sorted(corpus.tokens_dir(codec="dac").glob("*.npy"))
    tracks = [np.load(p) for p in npys]
    sampler = TokenBatchSampler(tracks, window_frames=258, seed=0)
    fixed_batch = sampler.sample(8).to(device)
    fixed_flat = flatten_codes(fixed_batch)
    print(f"fixed batch shape: {fixed_flat.shape}")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    # Train on this ONE batch, repeated. Loss MUST go down monotonically.
    print(f"\n{'step':>6s}  {'train loss':>10s}  {'eval loss (no_grad)':>22s}  {'embed.norm':>10s}")
    model.train()
    for step in range(1, 501):
        logits = model(fixed_flat)
        loss = ar_loss(logits, fixed_flat)
        lv_train = loss.item()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if step % 25 == 0 or step == 1:
            # Verify the trainer's loss reading by re-running forward in no_grad mode
            # on the same exact batch IMMEDIATELY (no save/load in between).
            with torch.no_grad():
                lv_eval = ar_loss(model(fixed_flat), fixed_flat).item()
            embed_norm = float(model.embed.weight.norm())
            print(f"{step:>6d}  {lv_train:>10.4f}  {lv_eval:>22.4f}  {embed_norm:>10.4f}")

    # Final test: save & reload, confirm fresh load also gives ~same loss
    print("\n--- Save and reload test ---")
    save_path = REPO_ROOT / "data/checkpoints/translator/techno/diag_midwindow/memorize_test.pt"
    torch.save({"model_state_dict": model.state_dict(), "translator_config": {
        "vocab_size": 1024, "d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024,
        "dropout": 0.0, "max_seq_len": 2338}, "train_config": {}, "steps": 500,
        "loss_first_window": 0, "loss_last_window": lv_train, "elapsed_seconds": 0}, save_path)
    model2 = FlatARTransformer(cfg).to(device)
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()
    with torch.no_grad():
        lv_reload = ar_loss(model2(fixed_flat), fixed_flat).item()
    print(f"reloaded model loss on the same fixed batch: {lv_reload:.4f}")
    print(f"in-memory model loss on the same fixed batch: {lv_eval:.4f}")

    if lv_train < 1.0:
        print("\nVERDICT: trainer CAN drive loss to ~0 on a single batch (model + trainer work)")
    else:
        print("\nVERDICT: trainer CANNOT drive loss down even on one batch — fundamental bug")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
