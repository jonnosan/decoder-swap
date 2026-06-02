"""M0: prove encoder / codebook / decoder are separately addressable and log token convention.

Run:  uv run python scripts/00_inspect_codec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/00_inspect_codec.py` without an install step.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from decoder_swap.codec_io import codebook_tensors, count_parameters, load_codec  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def fmt_int(n: int) -> str:
    return f"{n:>13,}"


def main() -> int:
    settings = load_settings()
    device = resolve_device(settings.device)
    print(f"# decoder-swap M0: codec inspection")
    print(f"config:    {settings.config_path}")
    print(f"codec:     {settings.codec_name}  model_type={settings.codec_model_type}  tag={settings.codec_model_tag}")
    print(f"device:    {device}  (torch {torch.__version__})")
    print()

    codec = load_codec(
        name=settings.codec_name,
        model_type=settings.codec_model_type,
        model_tag=settings.codec_model_tag,
        model_path=settings.codec_model_path,
        device=device,
        num_quantizers=settings.codec_num_quantizers,
    )

    conv = codec.convention
    print("## token convention")
    print(f"  sample_rate    : {conv.sample_rate} Hz")
    print(f"  hop_length     : {conv.hop_length} samples")
    print(f"  frame_rate     : {conv.frame_rate:.3f} frames/s")
    print(f"  n_codebooks    : {conv.n_codebooks}  (RVQ depth)")
    print(f"  codebook_size  : {conv.codebook_size}  (vocab per codebook)")
    print(f"  latent_dim     : {conv.latent_dim}")
    print(f"  weights_path   : {codec.extra.get('weights_path')}")
    print()

    print("## parameter breakdown (total / trainable)")
    parts = [
        ("encoder  ", codec.encoder),
        ("quantizer", codec.quantizer),
        ("decoder  ", codec.decoder),
    ]
    grand_total = 0
    grand_trainable = 0
    for label, mod in parts:
        total, trainable = count_parameters(mod)
        grand_total += total
        grand_trainable += trainable
        print(f"  {label} : total={fmt_int(total)}   trainable={fmt_int(trainable)}")
    total_full, trainable_full = count_parameters(codec.model)
    other = total_full - grand_total
    print(f"  ---------")
    print(f"  sum-of-three: total={fmt_int(grand_total)}   trainable={fmt_int(grand_trainable)}")
    print(f"  full model  : total={fmt_int(total_full)}   trainable={fmt_int(trainable_full)}  (other={other:,})")
    print()

    print("## codebook tensor shapes (RVQ stack)")
    for i, t in enumerate(codebook_tensors(codec)):
        print(f"  codebook[{i:>2}] : shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}")
    print()

    print("## module tree (top level)")
    for name, _ in codec.model.named_children():
        print(f"  - {name}")
    print()

    print("## verdict")
    has_three = bool(codec.encoder) and bool(codec.quantizer) and bool(codec.decoder)
    if has_three and grand_total > 0:
        print("  PASS: encoder / quantizer (codebook) / decoder are all separately addressable.")
        return 0
    else:
        print("  FAIL: codec does NOT cleanly split into three parts — experiment design needs rethinking.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
