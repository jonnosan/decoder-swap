"""Assert D1 and D2 differ in NOTHING except decoder weights.

Two flavours of check:

1. Pairwise (`assert_codec_invariants_match`) — compare two loaded Codecs (e.g. D1 vs D2 at M4).
   Verifies token convention AND byte-identical codebooks. M4 must call this before producing
   S1/S2 — if it fails, the experiment is invalid and we abort.

2. Snapshot (`snapshot_codebook_fingerprints` + `assert_codebook_unchanged`) — periodic check
   during decoder fine-tuning (M3) that the frozen codebook hasn't accidentally drifted.

Failures raise InvariantViolation. They are not warnings. They are not silenced.
"""
from __future__ import annotations

import hashlib

from .codec_io import Codec, codebook_tensors


class InvariantViolation(AssertionError):
    """Raised when the D1/D2 invariant is broken. Never catch this except in a deliberate test."""


def codebook_fingerprints(codec: Codec) -> list[str]:
    """SHA256 hex digest of each codebook tensor's raw bytes (cpu, contiguous, float32).

    Identical lists ⇒ byte-identical codebooks. We hash rather than compare raw bytes so
    the result is small enough to log/snapshot/print.
    """
    out: list[str] = []
    for t in codebook_tensors(codec):
        t_cpu = t.detach().to("cpu").contiguous().float()
        out.append(hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest())
    return out


def snapshot_codebook_fingerprints(codec: Codec) -> list[str]:
    """Take a snapshot of the codebook state, e.g. at start of training, to compare against later."""
    return codebook_fingerprints(codec)


def assert_token_convention_matches(a: Codec, b: Codec) -> None:
    if a.convention != b.convention:
        raise InvariantViolation(
            "token convention mismatch between codecs:\n"
            f"  a: {a.convention}\n"
            f"  b: {b.convention}"
        )


def assert_codebooks_byte_identical(a: Codec, b: Codec) -> None:
    fa = codebook_fingerprints(a)
    fb = codebook_fingerprints(b)
    if len(fa) != len(fb):
        raise InvariantViolation(
            f"codebook count mismatch: a has {len(fa)} codebooks, b has {len(fb)}"
        )
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(fa, fb)) if x != y]
    if diffs:
        lines = [f"  codebook[{i}]: a={x[:16]}…  b={y[:16]}…" for i, x, y in diffs]
        raise InvariantViolation(
            f"codebooks differ on {len(diffs)} of {len(fa)} entries:\n" + "\n".join(lines)
        )


def assert_codebook_unchanged(codec: Codec, snapshot: list[str]) -> None:
    """During training: assert the frozen codebook has not drifted from `snapshot`."""
    current = codebook_fingerprints(codec)
    if len(current) != len(snapshot):
        raise InvariantViolation(
            f"codebook count changed: was {len(snapshot)}, now {len(current)}"
        )
    diffs = [(i, c, s) for i, (c, s) in enumerate(zip(current, snapshot)) if c != s]
    if diffs:
        lines = [f"  codebook[{i}]: snapshot={s[:16]}…  now={c[:16]}…" for i, c, s in diffs]
        raise InvariantViolation(
            f"FROZEN codebook drifted during training on {len(diffs)} of {len(current)} entries:\n"
            + "\n".join(lines)
        )


def assert_codec_invariants_match(a: Codec, b: Codec) -> None:
    """The one M4 must call before producing S1 and S2."""
    assert_token_convention_matches(a, b)
    assert_codebooks_byte_identical(a, b)
