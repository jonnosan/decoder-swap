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
    num_quantizers: int | None = None,
) -> Codec:
    if name == "dac":
        return _load_dac(model_type=model_type, model_tag=model_tag, model_path=model_path, device=device)
    if name == "mimi":
        return _load_mimi(model_tag=model_tag, device=device, num_quantizers=num_quantizers)
    raise NotImplementedError(f"codec backend {name!r} not wired up")


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


def _load_mimi(model_tag: str | None, device: str, num_quantizers: int | None) -> Codec:
    """Load Mimi (Kyutai). Mimi has 7 top-level modules; we group them into the standard
    encoder/quantizer/decoder triple for the experiment:
      frozen front-end : encoder + encoder_transformer + downsample + quantizer
      trainable back-end: upsample + decoder_transformer + decoder

    `num_quantizers` defaults to 8 (Mimi paper's standard low-rate config: 1 semantic + 7 acoustic).
    """
    from transformers import MimiModel

    model = MimiModel.from_pretrained(model_tag or "kyutai/mimi")
    model.eval().to(device)

    n_q = int(num_quantizers if num_quantizers is not None else 8)

    # Pack the three sub-stages of each side into ModuleLists so freeze.py + parameter counting
    # work the same way they do for DAC. The forward path doesn't go through these lists —
    # Mimi's own model.encode / model.decode methods orchestrate it.
    encoder = nn.ModuleList([model.encoder, model.encoder_transformer, model.downsample])
    decoder = nn.ModuleList([model.upsample, model.decoder_transformer, model.decoder])
    quantizer = model.quantizer

    sr = int(model.config.sampling_rate)
    fps = float(model.config.frame_rate)
    hop = max(int(round(sr / fps)), 1)
    cb_size = int(model.config.codebook_size)
    # Mimi's internal hidden dim — used here only for the convention log; not load-bearing.
    latent_dim = int(getattr(model.config, "hidden_size", 0)) or int(getattr(model.config, "audio_channels", 0))

    convention = TokenConvention(
        sample_rate=sr,
        frame_rate=fps,
        hop_length=hop,
        n_codebooks=n_q,
        codebook_size=cb_size,
        latent_dim=latent_dim,
    )

    return Codec(
        name="mimi",
        model=model,
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        convention=convention,
        device=device,
        extra={
            "weights_id": model_tag or "kyutai/mimi",
            "num_quantizers_used": n_q,
            "num_quantizers_available": int(model.config.num_codebooks),
            "num_semantic_quantizers": int(model.config.num_semantic_quantizers),
        },
    )


def encode_to_codes(codec: Codec, x: torch.Tensor) -> torch.Tensor:
    """Frozen forward: audio (B,1,T) -> discrete codes (B,n_codebooks,T_frames).

    Codec-agnostic interface for the rest of the pipeline. Always called under no_grad.
    """
    if codec.name == "dac":
        x_pre = codec.model.preprocess(x, codec.convention.sample_rate)
        _z, codes, _l, _cm, _cb = codec.model.encode(x_pre)
        return codes
    if codec.name == "mimi":
        n_q = int(codec.extra.get("num_quantizers_used", 8))
        # use_streaming=False forces a fresh KV-cache + padding state per call. Without this,
        # internal state leaks across consecutive encode() calls and every other training step's
        # forward goes NaN. Also explicit Nones on the cache args belt-and-braces.
        out = codec.model.encode(
            x,
            num_quantizers=n_q,
            encoder_past_key_values=None,
            padding_cache=None,
            use_streaming=False,
            return_dict=True,
        )
        return out.audio_codes
    raise NotImplementedError(codec.name)


def decode_from_codes(codec: Codec, codes: torch.Tensor) -> torch.Tensor:
    """Trainable forward: codes -> audio (B,1,T). Gradients flow through codec.decoder."""
    if codec.name == "dac":
        z_q, _z_p, _codes = codec.quantizer.from_codes(codes)
        return codec.model.decode(z_q)
    if codec.name == "mimi":
        out = codec.model.decode(
            codes,
            decoder_past_key_values=None,
            return_dict=True,
        )
        return out.audio_values
    raise NotImplementedError(codec.name)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """(total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def codebook_tensors(codec: Codec) -> list[torch.Tensor]:
    """Return all codebook weight tensors. Used by invariants.py for byte-identity checks.

    For Mimi we fingerprint EVERY codebook (semantic + all 31 acoustic), even ones beyond
    `num_quantizers_used` — they're frozen too, and any drift in unused codebooks would
    indicate the freeze leaked. Belt and braces.
    """
    out: list[torch.Tensor] = []
    if codec.name == "dac":
        for q in codec.quantizer.quantizers:
            out.append(q.codebook.weight.detach())
        return out
    if codec.name == "mimi":
        # Mimi codebooks are EMA-updated buffers (not parameters): `embed` is a property derived
        # from `embed_sum` / `cluster_usage`. Fingerprinting `embed` gives the right semantic check:
        # if Mimi's internal training-mode EMA updates somehow run on us, `embed` shifts and the
        # SHA catches it. The quantizer is set to .eval() before training to suppress EMA updates.
        sem = codec.quantizer.semantic_residual_vector_quantizer
        out.append(sem.layers[0].codebook.embed.detach())
        for layer in codec.quantizer.acoustic_residual_vector_quantizer.layers:
            out.append(layer.codebook.embed.detach())
        return out
    raise NotImplementedError(codec.name)
