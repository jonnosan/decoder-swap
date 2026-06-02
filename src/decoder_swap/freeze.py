"""Freeze codebook + encoder, keep decoder trainable.

Why freeze the WHOLE quantizer (not just the `codebook.weight` tensors)?
DAC uses factorised codebooks: each entry is 8-dim, projected to latent_dim=1024 by per-quantizer
`in_proj` / `out_proj` layers. If those projections drift during training, the encoder→discrete-token
mapping changes — meaning the "token sequence T" no longer means the same thing across D1 and D2,
and the experiment is invalid. So we freeze the entire quantizer module, not just codebook tensors.
The encoder is frozen too — we want T to be IDENTICAL across D1/D2 runs, so the only source of
variation is the decoder weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn
import torch.nn.utils as nn_utils

from .codec_io import Codec, count_parameters


def remove_weight_norm_recursive(module: nn.Module) -> int:
    """Collapse every weight_norm parametrization in `module` into a single `weight` tensor.

    Why: DAC's decoder uses `torch.nn.utils.weight_norm`, and its backward path has a known
    MPS bug (a small healthy gradient on `weight_v` can produce NaN after the optimizer step
    because of `||v||` division). Forward behaviour is unchanged by removal — we just stop
    parametrising the weight each forward call, so the buggy backward is gone.

    Returns the number of modules that had weight_norm removed.
    """
    count = 0
    for m in module.modules():
        try:
            nn_utils.remove_weight_norm(m)
            count += 1
        except ValueError:
            pass  # this module wasn't weight-normed; ignore
    return count


@dataclass
class FreezeReport:
    encoder_frozen: int
    quantizer_frozen: int
    decoder_trainable: int
    total_frozen: int
    total_trainable: int

    def as_dict(self) -> dict:
        return asdict(self)


def freeze_for_decoder_training(codec: Codec) -> FreezeReport:
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    for p in codec.quantizer.parameters():
        p.requires_grad_(False)
    for p in codec.decoder.parameters():
        p.requires_grad_(True)

    enc_total, enc_train = count_parameters(codec.encoder)
    quant_total, quant_train = count_parameters(codec.quantizer)
    dec_total, dec_train = count_parameters(codec.decoder)

    # Hard guard — if these fail the freeze didn't do what we said it did, abort loudly.
    if enc_train != 0:
        raise RuntimeError(f"encoder freeze failed: {enc_train} trainable params remain")
    if quant_train != 0:
        raise RuntimeError(f"quantizer freeze failed: {quant_train} trainable params remain")
    if dec_train != dec_total:
        raise RuntimeError(f"decoder freeze leaked: only {dec_train}/{dec_total} params trainable")

    return FreezeReport(
        encoder_frozen=enc_total,
        quantizer_frozen=quant_total,
        decoder_trainable=dec_total,
        total_frozen=enc_total + quant_total,
        total_trainable=dec_total,
    )


def print_freeze_report(rep: FreezeReport) -> None:
    def f(n: int) -> str:
        return f"{n:>13,}"
    print("## freeze report")
    print(f"  encoder   : frozen   ={f(rep.encoder_frozen)}")
    print(f"  quantizer : frozen   ={f(rep.quantizer_frozen)}")
    print(f"  decoder   : trainable={f(rep.decoder_trainable)}")
    print(f"  ---------")
    print(f"  total frozen   : {f(rep.total_frozen)}")
    print(f"  total trainable: {f(rep.total_trainable)}")
