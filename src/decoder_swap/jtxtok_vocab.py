"""jtxtok v1 vocabulary for decoder-swap (M7 conditioning consumer).

Hard-codes the spec §4 token list + per-token role tags so the conditioning encoder can
factorise by voice role (M7.A Part A — see docs/prompts/PROMPT_3_decoder_swap.md and
docs/JTXTOK_SPEC.md).

CRITICAL: this MUST agree byte-for-byte with both producers — `jtxtok_extractor.tokens.vocabulary()`
and the `jtx → jtxtok` emitter in jamtronix. Any drift silently corrupts training. The
canonical source is docs/JTXTOK_SPEC.md (which mirrors the jamtronix repo); this file is the
ML-side embedding-table builder, not a re-derivation of the spec.

The "configured set" mirrors the extractor's default contract:
  drum_classes=5 (kick/snare/hat/ohat/clap), bass=onset (BASS_ON), velocity=off, key=on.
Different configurations produce different vocabularies. Both producers + this consumer
must use the same configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Roles correspond to voice-filter axis at generation time (PROMPT_3 §C). The conditioning
# encoder learns a separate role-embedding per role; inference can drop all tokens of a
# given role from the encoder input to do voice filtering.
Role = Literal["struct", "drum", "bass", "key", "mt", "pad"]

# Special pad token id (added by us, not in the spec) — used to right-pad batches in the
# conditioning encoder when sequences have different lengths.
PAD_TOKEN = "<PAD>"

# Spec §4.1 structural tokens.
_STRUCT_TOKENS = ["BOS", "EOS", "BAR"] + [f"POS_{i}" for i in range(16)]

# Spec §2.1 micro-timing tokens. Vocabulary always includes them (consumers must embed them
# because jtx emits them at inference); the extractor never produces them.
_MT_TOKENS = [f"MT_{n:+d}" for n in range(-8, 9)]

# Spec §4.2 drums — 5-class v1 default. Extended (18-class) is optional; both producers
# must agree on whichever is configured.
_DRUM_5 = ["DRUM_KICK", "DRUM_SNARE", "DRUM_HAT", "DRUM_OHAT", "DRUM_CLAP"]
_DRUM_EXTENDED = _DRUM_5 + ["DRUM_TOM", "DRUM_RIDE", "DRUM_CRASH", "DRUM_PERC"]

# Spec §4.2 velocity (optional). Both producers must match.
_VEL_TOKENS = ["VEL_LO", "VEL_MED", "VEL_HI"]

# Spec §4.3 bass.
_BASS_ON = ["BASS_ON"]
_BASS_PITCH = [f"BASS_P_{pc}" for pc in range(12)]

# Spec §4.4 key.
_KEY_TOKENS = [f"KEY_{pc}" for pc in range(12)]


BassMode = Literal["none", "onset", "pitch"]


@dataclass(frozen=True)
class JtxtokVocab:
    """Hashed vocabulary for one configuration of the spec.

    The constructor's args MUST match the producers' configuration (drum_classes,
    bass_mode, velocity_enabled, key_enabled). A mismatch silently corrupts training.

    Attributes:
        tokens: ordered list of vocabulary tokens (pad token is id 0).
        token_to_id: lookup dict.
        role_of: per-token role tag (parallel to `tokens`).
        size: convenience for `len(tokens)`.
    """
    tokens: tuple[str, ...]
    token_to_id: dict[str, int]
    role_of: tuple[Role, ...]
    drum_classes: int
    bass_mode: BassMode
    velocity_enabled: bool
    key_enabled: bool

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    def encode(self, token_stream: list[str]) -> list[int]:
        """Token list -> id list. Unknown tokens raise (they're a contract violation)."""
        return [self.token_to_id[t] for t in token_stream]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] for i in ids]

    def role_ids(self) -> tuple[int, ...]:
        """Numeric role tags parallel to `tokens`, for use as the role embedding's input."""
        role_to_idx = {r: i for i, r in enumerate(("struct", "drum", "bass", "key", "mt", "pad"))}
        return tuple(role_to_idx[r] for r in self.role_of)

    def role_mask(self, keep_roles: set[Role]) -> tuple[bool, ...]:
        """Boolean mask (parallel to `tokens`) where True = role IS in keep_roles.

        Used by voice filtering at generation time (PROMPT_3 §C, Axis 1). A token whose
        role is dropped is filtered out of the conditioning encoder's input — pad takes
        its place. `struct` and `pad` tokens are always kept (they carry no role-conditional
        information; BAR/POS/BOS/EOS are structural scaffolding).
        """
        always_keep = {"struct", "pad"}
        return tuple(r in always_keep or r in keep_roles for r in self.role_of)


def build_vocab(
    *,
    drum_classes: int = 5,
    bass_mode: BassMode = "onset",
    velocity_enabled: bool = False,
    key_enabled: bool = True,
) -> JtxtokVocab:
    """Build a JtxtokVocab for the given config. Defaults = extractor contract defaults."""
    if drum_classes not in (5, 18):
        raise ValueError(f"drum_classes must be 5 or 18, got {drum_classes}")

    tokens: list[str] = [PAD_TOKEN]              # id 0 is pad
    roles: list[Role] = ["pad"]

    for t in _STRUCT_TOKENS:
        tokens.append(t)
        roles.append("struct")

    for t in _MT_TOKENS:
        tokens.append(t)
        roles.append("mt")

    drum_set = _DRUM_5 if drum_classes == 5 else _DRUM_EXTENDED
    for t in drum_set:
        tokens.append(t)
        roles.append("drum")

    if velocity_enabled:
        for t in _VEL_TOKENS:
            tokens.append(t)
            roles.append("drum")  # velocity always modifies a drum token

    if bass_mode == "onset":
        for t in _BASS_ON:
            tokens.append(t)
            roles.append("bass")
    elif bass_mode == "pitch":
        for t in _BASS_ON:
            tokens.append(t)
            roles.append("bass")
        for t in _BASS_PITCH:
            tokens.append(t)
            roles.append("bass")
    # bass_mode == "none": no BASS_* tokens

    if key_enabled:
        for t in _KEY_TOKENS:
            tokens.append(t)
            roles.append("key")

    token_to_id = {t: i for i, t in enumerate(tokens)}
    return JtxtokVocab(
        tokens=tuple(tokens),
        token_to_id=token_to_id,
        role_of=tuple(roles),
        drum_classes=drum_classes,
        bass_mode=bass_mode,
        velocity_enabled=velocity_enabled,
        key_enabled=key_enabled,
    )


# Standard "extractor contract default" vocab — the one M7 trains against by default.
DEFAULT_VOCAB = build_vocab()
