"""Test the causal mask: change one token in the middle, see if logits BEFORE that
position change. If they do, future tokens are leaking into past predictions."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from decoder_swap.translator import FlatARTransformer, TranslatorConfig  # noqa: E402


def main() -> int:
    cfg = TranslatorConfig(
        vocab_size=1024, d_model=256, n_layers=4, n_heads=4, d_ff=1024,
        dropout=0.0, max_seq_len=2338,
    )
    torch.manual_seed(0)
    model = FlatARTransformer(cfg)
    model.eval()

    # Random input batch.
    L = 100
    x = torch.randint(0, 1024, (1, L))

    with torch.no_grad():
        logits1 = model(x)

    # Change the token at position L/2 = 50.
    x_mod = x.clone()
    x_mod[0, 50] = (x_mod[0, 50] + 1) % 1024

    with torch.no_grad():
        logits2 = model(x_mod)

    # Compare logits position-by-position
    diff = (logits1 - logits2).abs().max(dim=-1).values[0]
    print(f"Position-by-position max-abs-logit-diff after changing token at pos 50:")
    print(f"  pos 0..10:  {[round(d, 6) for d in diff[:11].tolist()]}")
    print(f"  pos 45..55: {[round(d, 6) for d in diff[45:56].tolist()]}")
    print(f"  pos 90..99: {[round(d, 6) for d in diff[90:100].tolist()]}")

    # Positions BEFORE pos 50 should have ZERO difference (causal mask).
    # Positions AT and AFTER pos 50 are allowed to differ.
    pre_max = diff[:50].max().item()
    post_max = diff[50:].max().item()
    print(f"\n  max diff in positions BEFORE pos 50: {pre_max:.6e}  (should be 0)")
    print(f"  max diff in positions AT/AFTER pos 50: {post_max:.6e}  (any value)")
    if pre_max > 1e-5:
        print("\n  *** CAUSAL MASK LEAK DETECTED ***")
    else:
        print("\n  Causal mask works correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
