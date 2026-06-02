"""Load a neural audio codec and expose encoder / codebook / decoder as separately addressable parts.

Only DAC (descript-audio-codec) is wired up for M0. EnCodec is a fallback option mentioned in the design
but not implemented yet — if you ever switch, this is the file to extend (keep the same Codec dataclass
interface so the rest of the pipeline doesn't care which backend is in use).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TokenConvention:
    """Everything that defines 'a token sequence' for this codec.

    D1 and D2 must agree on every field here for the experiment to be valid (invariants.py enforces).
    """
    sample_rate: int          # input/output audio sample rate (Hz)
    frame_rate: float         # token frames per second (sample_rate / hop)
    hop_length: int           # samples per token frame
    n_codebooks: int          # RVQ depth (number of residual codebooks)
    codebook_size: int        # vocab per codebook
    latent_dim: int           # continuous latent dimensionality before/after quantization


@dataclass
class Codec:
    """Bundle of (frozen-by-default) front-end + (trainable) decoder + token convention.

    `encoder` and `quantizer` carry the codebook tensors we will freeze. `decoder` is what we retrain
    for D2. `model` is the full underlying nn.Module so call sites that need the original forward path
    (e.g. DAC's own encode/decode wrappers) can use it.
    """
    name: str
    model: nn.Module
    encoder: nn.Module
    quantizer: nn.Module
    decoder: nn.Module
    convention: TokenConvention
    device: str
    extra: dict[str, Any]


def load_codec(
    name: str = "dac",
    model_type: str = "44khz",
    model_tag: str | None = None,
    model_path: str | Path | None = None,
    device: str = "cpu",
) -> Codec:
    if name != "dac":
        raise NotImplementedError(f"codec backend {name!r} not wired up yet (only 'dac' for M0)")
    return _load_dac(model_type=model_type, model_tag=model_tag, model_path=model_path, device=device)


def _load_dac(
    model_type: str,
    model_tag: str | None,
    model_path: str | Path | None,
    device: str,
) -> Codec:
    import dac
    from dac.utils import download

    if model_path is None:
        kwargs: dict[str, Any] = {"model_type": model_type}
        if model_tag is not None:
            kwargs["model_version"] = model_tag
        weights_path = download(**kwargs)
    else:
        weights_path = Path(model_path)

    model = dac.DAC.load(str(weights_path))
    model.eval()
    model.to(device)

    encoder = model.encoder
    quantizer = model.quantizer
    decoder = model.decoder

    sample_rate = int(model.sample_rate)
    hop_length = int(model.hop_length)
    frame_rate = sample_rate / hop_length
    n_codebooks = int(model.n_codebooks)
    codebook_size = int(model.codebook_size)
    latent_dim = int(model.latent_dim)

    convention = TokenConvention(
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        hop_length=hop_length,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        latent_dim=latent_dim,
    )

    return Codec(
        name="dac",
        model=model,
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        convention=convention,
        device=device,
        extra={"weights_path": str(weights_path), "model_type": model_type},
    )


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """(total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def codebook_tensors(codec: Codec) -> list[torch.Tensor]:
    """Return the codebook weight tensors in RVQ order. Used by invariants.py for byte-identity checks."""
    if codec.name != "dac":
        raise NotImplementedError(codec.name)
    out: list[torch.Tensor] = []
    for q in codec.quantizer.quantizers:
        out.append(q.codebook.weight.detach())
    return out
