"""Diagnose the M7.B init mismatch: at step 0, ConditionedTranslator should reproduce
FlatARTransformer's outputs (zero-init cross-attn + encoder_to_decoder + same embed/pos).

If init loss != M6.A loss, one of the components below is wrong. We print:
  - logit max-abs-diff between the two models on the same batch
  - per-component contribution: try ablating each potential difference one at a time
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from decoder_swap.conditioned_translator import (  # noqa: E402
    ConditionedTranslator,
    ConditionedTranslatorConfig,
)
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.jtxtok_dataset import (  # noqa: E402
    JtxtokDacDataset,
    JtxtokDacDatasetConfig,
    discover_track_pairs,
)
from decoder_swap.jtxtok_vocab import DEFAULT_VOCAB  # noqa: E402
from decoder_swap.train_conditioned import load_conditioned_with_base_lm  # noqa: E402
from decoder_swap.translator import FlatARTransformer, TranslatorConfig, ar_loss, flatten_codes  # noqa: E402


def main() -> int:
    device = "cpu"  # CPU keeps the numerics deterministic for diff measurement
    base_lm_path = REPO_ROOT / "data/checkpoints/translator/techno/translator_lm_best.pt"
    ckpt = torch.load(str(base_lm_path), map_location="cpu", weights_only=False)
    tc = ckpt["translator_config"]
    print(f"M6.A config from ckpt: {tc}")

    # --- Build M6.A unconditional model from ckpt ---
    uncond_cfg = TranslatorConfig(
        vocab_size=tc["vocab_size"], d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"], dropout=tc.get("dropout", 0.0),
        max_seq_len=tc["max_seq_len"],
    )
    uncond = FlatARTransformer(uncond_cfg)
    missing, unexpected = uncond.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"unconditional load: missing={missing}, unexpected={unexpected}")
    uncond.eval().to(device)

    # --- Build a paired batch ---
    corpus = load_corpus("techno")
    pairs = discover_track_pairs(
        corpus.tokens_dir(codec="dac"), REPO_ROOT / "data/jtxtok/techno"
    )
    ds_cfg = JtxtokDacDatasetConfig(window_seconds=3.0, max_jtxtok_len=256, seed=0)
    dataset = JtxtokDacDataset(pairs, DEFAULT_VOCAB, ds_cfg)
    batch = dataset.sample_batch(2)
    dac_codes = batch["dac_ids"]
    jtxtok_ids = batch["jtxtok_ids"]
    jtxtok_roles = batch["jtxtok_roles"]
    dac_flat = flatten_codes(dac_codes).to(device)
    print(f"batch: dac_flat={dac_flat.shape}, jtxtok={jtxtok_ids.shape}")

    # --- M6.A unconditional forward + loss ---
    with torch.no_grad():
        uncond_logits = uncond(dac_flat)
        uncond_loss = ar_loss(uncond_logits, dac_flat)
    print(f"\nM6.A unconditional loss: {uncond_loss.item():.4f}")

    # --- Build conditioned model and load base LM ---
    cond_cfg = ConditionedTranslatorConfig(
        dac_vocab_size=tc["vocab_size"],
        d_model=tc["d_model"], n_layers=tc["n_layers"],
        n_heads=tc["n_heads"], d_ff=tc["d_ff"],
        max_dac_seq_len=dataset.window_frames * pairs[0].dac.shape[0] + 16,
        jtxtok_vocab_size=DEFAULT_VOCAB.size, jtxtok_pad_id=DEFAULT_VOCAB.pad_id,
        enc_d_model=256, enc_n_layers=3, enc_n_heads=4, enc_d_ff=1024,
        enc_max_seq_len=272,
    )
    cond = load_conditioned_with_base_lm(cond_cfg, base_lm_path, device).eval()

    with torch.no_grad():
        cond_logits = cond(dac_flat, jtxtok_ids.to(device), jtxtok_roles.to(device))
        from decoder_swap.conditioned_translator import ar_loss as cond_ar_loss
        cond_loss = cond_ar_loss(cond_logits, dac_flat)
    print(f"\nM7.B conditioned loss (fresh init): {cond_loss.item():.4f}")
    print(f"Difference: {cond_loss.item() - uncond_loss.item():+.4f}")

    diff = (cond_logits - uncond_logits).abs()
    print(f"\nLogit max-abs-diff: {diff.max().item():.6f}")
    print(f"Logit mean-abs-diff: {diff.mean().item():.6f}")

    # --- Component diagnostics ---
    print("\n--- Per-component check ---")

    # 1. dac_embed weight equality
    eq = torch.equal(cond.dac_embed.weight, uncond.embed.weight)
    print(f"embed equal: {eq}")

    # 2. final norm equality
    eq = (torch.equal(cond.norm.weight, uncond.norm.weight)
          and torch.equal(cond.norm.bias, uncond.norm.bias))
    print(f"final norm equal: {eq}")

    # 3. pos_enc equality
    L = dac_flat.shape[1]
    eq = torch.allclose(cond.dac_pos_enc[:L], uncond.pos_enc[:L], atol=1e-6)
    print(f"pos_enc[:L] equal: {eq}")

    # 4. cross-attn out_proj is zero
    for i, block in enumerate(cond.decoder.layers):
        w = block.multihead_attn.out_proj.weight
        b = block.multihead_attn.out_proj.bias
        print(f"  layer {i}: cross-attn out_proj weight norm={w.norm().item():.3e}, "
              f"bias norm={b.norm().item() if b is not None else 'None'}")

    # 5. encoder_to_decoder
    if hasattr(cond.encoder_to_decoder, "weight"):
        w = cond.encoder_to_decoder.weight
        b = cond.encoder_to_decoder.bias
        print(f"  enc->dec: weight norm={w.norm().item():.3e}, "
              f"bias norm={b.norm().item() if b is not None else 'None'}")
    else:
        print("  enc->dec: Identity (no projection)")

    # 6. per-layer self_attn / FFN parameter equality
    print("\nPer-layer encoder→decoder copy check:")
    for i in range(uncond_cfg.n_layers):
        enc_layer = uncond.encoder.layers[i]
        dec_layer = cond.decoder.layers[i]
        checks = {
            "self_attn.in_proj_weight": (enc_layer.self_attn.in_proj_weight,
                                          dec_layer.self_attn.in_proj_weight),
            "self_attn.out_proj.weight": (enc_layer.self_attn.out_proj.weight,
                                           dec_layer.self_attn.out_proj.weight),
            "linear1.weight": (enc_layer.linear1.weight, dec_layer.linear1.weight),
            "linear2.weight": (enc_layer.linear2.weight, dec_layer.linear2.weight),
            "norm1.weight":   (enc_layer.norm1.weight, dec_layer.norm1.weight),
            "norm2(enc)->norm3(dec)": (enc_layer.norm2.weight, dec_layer.norm3.weight),
        }
        for name, (a, b_) in checks.items():
            equal = torch.equal(a, b_)
            print(f"  layer{i} {name:35s} equal={equal}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
