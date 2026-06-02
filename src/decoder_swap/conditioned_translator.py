"""jtxtok-conditioned translator — M7.A (PROMPT_3 Part A).

Composes:
    JtxtokConditioningEncoder   (bidirectional encoder over jtxtok streams)
    FlatARDecoder                (causal AR decoder over DAC tokens, with cross-attention)

Inference / training input shapes:
    dac_ids        : (B, L_dac)              — flat-interleaved DAC token ids
    jtxtok_ids     : (B, L_jtxtok)           — jtxtok token ids
    jtxtok_role_ids: (B, L_jtxtok)           — per-token role tag (parallel to jtxtok_ids)

Output:
    logits         : (B, L_dac, dac_vocab)   — next-token distribution for AR loss

Key design choices (mirroring PROMPT_3 §A and the M6.0/M6.A fixes):

1. **Cross-attention inserted at every decoder block (T5-style)** rather than gated subset.
   Simplest baseline; can be revisited if compute / quality argues otherwise.

2. **Cross-attention OUT_PROJ zero-init**. With every block's `multihead_attn.out_proj`
   initialised to zero, a freshly built `ConditionedTranslator` produces *exactly the
   same outputs* as the unconditional `FlatARTransformer` it loads from — because
   cross-attn contributes a literal zero to the residual stream. This is the classic
   "drop-in residual extension" trick that lets us start from M6.A's checkpoint and
   gradually learn cross-attention without ever destabilising the base model's
   capability.

3. **Same DAC-side embedding scaling + depth-aware init as M6.A's translator.py.**
   Without them, the bigger model with conditioning would relapse into the M6.A
   regression. Keep them.

4. **Empty conditioning is a valid input** (PROMPT_3 §C "ignore" / §D "from scratch"):
   pass a length-1 jtxtok with just `BOS` (or `<PAD>`), and the encoder produces a
   trivial context that the cross-attention's pad mask zeros out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning_encoder import (
    ConditioningEncoderConfig,
    JtxtokConditioningEncoder,
)


@dataclass
class ConditionedTranslatorConfig:
    # DAC side (must match the M6.A unconditional model's config for weight-load to work)
    dac_vocab_size: int = 1024
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0
    max_dac_seq_len: int = 4096

    # jtxtok conditioning side
    jtxtok_vocab_size: int = 55
    jtxtok_pad_id: int = 0
    enc_d_model: int = 256                 # may differ from decoder d_model — projected via Linear
    enc_n_layers: int = 3
    enc_n_heads: int = 4
    enc_d_ff: int = 1024
    enc_max_seq_len: int = 1024


class ConditionedTranslator(nn.Module):
    """jtxtok-conditioned AR translator. Trained as `(jtxtok_stream) -> (DAC tokens)`."""

    def __init__(self, cfg: ConditionedTranslatorConfig):
        super().__init__()
        self.cfg = cfg

        # ---- DAC-side embedding + positional ----
        self.dac_embed = nn.Embedding(cfg.dac_vocab_size, cfg.d_model)
        self.register_buffer("dac_pos_enc", _sinusoidal_positional_encoding(
            cfg.max_dac_seq_len, cfg.d_model
        ))

        # ---- Conditioning encoder ----
        enc_cfg = ConditioningEncoderConfig(
            vocab_size=cfg.jtxtok_vocab_size,
            d_model=cfg.enc_d_model,
            n_layers=cfg.enc_n_layers,
            n_heads=cfg.enc_n_heads,
            d_ff=cfg.enc_d_ff,
            dropout=cfg.dropout,
            max_seq_len=cfg.enc_max_seq_len,
            pad_id=cfg.jtxtok_pad_id,
        )
        self.encoder = JtxtokConditioningEncoder(enc_cfg)

        # If encoder and decoder use different widths, project encoder output -> decoder width.
        if cfg.enc_d_model != cfg.d_model:
            self.encoder_to_decoder = nn.Linear(cfg.enc_d_model, cfg.d_model)
        else:
            self.encoder_to_decoder = nn.Identity()

        # ---- AR decoder with cross-attention ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        # ---- Init (matches translator.py M6.A) ----
        nn.init.normal_(self.dac_embed.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.decoder.layers:
            # Self-attention out_proj — same fix as the unconditional model.
            nn.init.normal_(block.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)
            nn.init.normal_(block.linear2.weight, mean=0.0, std=residual_std)
            if block.linear2.bias is not None:
                nn.init.zeros_(block.linear2.bias)
            # Cross-attention out_proj ZERO-INIT — the drop-in residual extension trick.
            # A freshly-built conditioned model behaves identically to the unconditional one
            # because cross-attn contributes 0 to the residual stream until learning begins.
            nn.init.zeros_(block.multihead_attn.out_proj.weight)
            if block.multihead_attn.out_proj.bias is not None:
                nn.init.zeros_(block.multihead_attn.out_proj.bias)

        # The encoder-to-decoder width projection: zero-init too if present, so empty
        # conditioning is exactly equivalent to unconditional at init time.
        if isinstance(self.encoder_to_decoder, nn.Linear):
            nn.init.zeros_(self.encoder_to_decoder.weight)
            if self.encoder_to_decoder.bias is not None:
                nn.init.zeros_(self.encoder_to_decoder.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        dac_ids: torch.Tensor,             # (B, L_dac)
        jtxtok_ids: torch.Tensor,          # (B, L_jtxtok)
        jtxtok_role_ids: torch.Tensor,     # (B, L_jtxtok)
    ) -> torch.Tensor:
        B, L_dac = dac_ids.shape
        if L_dac > self.cfg.max_dac_seq_len:
            raise ValueError(f"DAC sequence length {L_dac} exceeds max {self.cfg.max_dac_seq_len}")

        # Encoder side.
        context, ctx_pad_mask = self.encoder(jtxtok_ids, jtxtok_role_ids)
        context = self.encoder_to_decoder(context)

        # Decoder side: scaled embedding + sinusoidal pos + causal cross-attn into context.
        h = self.dac_embed(dac_ids) * math.sqrt(self.cfg.d_model)
        h = h + self.dac_pos_enc[:L_dac].unsqueeze(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L_dac, device=dac_ids.device)
        h = self.decoder(
            tgt=h,
            memory=context,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            memory_key_padding_mask=ctx_pad_mask,
        )
        h = self.norm(h)
        logits = h @ self.dac_embed.weight.t()
        return logits

    # -----------------------------------------------------------------------------------
    # Weight transfer from the unconditional M6.A model

    @torch.no_grad()
    def load_from_unconditional(self, unconditional_state_dict: dict) -> dict:
        """Copy weights from a `FlatARTransformer` checkpoint into this model's decoder.

        Mapping:
            unconditional.embed                              -> self.dac_embed
            unconditional.norm                               -> self.norm
            unconditional.encoder.layers[i].self_attn        -> self.decoder.layers[i].self_attn
            unconditional.encoder.layers[i].linear1, .2      -> self.decoder.layers[i].linear1, .2
            unconditional.encoder.layers[i].norm1            -> self.decoder.layers[i].norm1
            unconditional.encoder.layers[i].norm2            -> self.decoder.layers[i].norm3

        Cross-attention weights and the encoder/role embeddings are left at zero-init/random.

        Returns a dict summarising which keys mapped and which were skipped.
        """
        sd = unconditional_state_dict
        mapped: list[str] = []
        skipped: list[str] = []

        # Embedding + output norm
        self.dac_embed.weight.copy_(sd["embed.weight"])
        mapped.append("embed.weight -> dac_embed.weight")
        self.norm.weight.copy_(sd["norm.weight"])
        self.norm.bias.copy_(sd["norm.bias"])
        mapped.append("norm.{weight,bias} -> norm.{weight,bias}")

        # Per-block weights
        n_uncond_layers = self.cfg.n_layers
        for i in range(n_uncond_layers):
            src = f"encoder.layers.{i}"
            dst_layer = self.decoder.layers[i]

            # Self-attention (uses in_proj_weight/bias for q/k/v packed by PyTorch).
            # Same parameter names in encoder and decoder layers — direct copy.
            for k in ("self_attn.in_proj_weight", "self_attn.in_proj_bias",
                      "self_attn.out_proj.weight", "self_attn.out_proj.bias"):
                tgt_param = dict(dst_layer.named_parameters())[k]
                tgt_param.copy_(sd[f"{src}.{k}"])
            mapped.append(f"{src}.self_attn.* -> decoder.layers.{i}.self_attn.*")

            # FFN
            for k in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
                tgt_param = dict(dst_layer.named_parameters())[k]
                tgt_param.copy_(sd[f"{src}.{k}"])
            mapped.append(f"{src}.linear{{1,2}} -> decoder.layers.{i}.linear{{1,2}}")

            # Norms: encoder's norm1 -> decoder's norm1 (self-attn pre-norm); encoder's
            # norm2 -> decoder's norm3 (FFN pre-norm). decoder's norm2 is for cross-attn
            # and stays at its default identity init.
            dst_layer.norm1.weight.copy_(sd[f"{src}.norm1.weight"])
            dst_layer.norm1.bias.copy_(sd[f"{src}.norm1.bias"])
            dst_layer.norm3.weight.copy_(sd[f"{src}.norm2.weight"])
            dst_layer.norm3.bias.copy_(sd[f"{src}.norm2.bias"])
            mapped.append(f"{src}.norm1 -> decoder.layers.{i}.norm1; norm2 -> norm3")

        # Report keys in the source state dict that weren't consumed.
        for k in sd:
            if not k.startswith(("embed.", "norm.", "encoder.layers.")):
                skipped.append(k)

        return {
            "mapped_summary": mapped,
            "unmapped_source_keys": skipped,
            "note": (
                "Cross-attention, encoder, role-embed, and dac_pos_enc are NOT loaded "
                "from the unconditional checkpoint — they stay at their (zero/random) init."
            ),
        }


# ---------------------------------------------------------------------------------------
# Helpers


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def ar_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Same teacher-forced next-token CE as the unconditional model."""
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        target[:, 1:].reshape(-1).long(),
    )
