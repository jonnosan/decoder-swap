"""M2: freeze codebook + encoder, prove invariants catch byte-level codebook changes.

Three checks:
  1. Freeze report — encoder + quantizer frozen, decoder trainable, counts match M0 breakdown.
  2. Self-match — load the codec twice, both pass the invariant (token convention + byte-identical codebooks).
  3. Negative test — deliberately mutate one entry of codebook[0], assert the invariant FAILS LOUDLY.

If (3) silently passes, the invariant is useless and the whole experiment is unsafe to continue.

Run:  uv run python scripts/02_freeze_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from decoder_swap.codec_io import load_codec  # noqa: E402
from decoder_swap.freeze import freeze_for_decoder_training, print_freeze_report  # noqa: E402
from decoder_swap.invariants import (  # noqa: E402
    InvariantViolation,
    assert_codebook_unchanged,
    assert_codec_invariants_match,
    codebook_fingerprints,
    snapshot_codebook_fingerprints,
)
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def main() -> int:
    settings = load_settings()
    device = resolve_device(settings.device)
    print("# decoder-swap M2: freeze + invariants")
    print(f"device: {device}")

    def fresh_codec():
        return load_codec(
            name=settings.codec_name,
            model_type=settings.codec_model_type,
            model_tag=settings.codec_model_tag,
            model_path=settings.codec_model_path,
            device=device,
        )

    # ---------- (1) freeze ----------
    codec_a = fresh_codec()
    report = freeze_for_decoder_training(codec_a)
    print_freeze_report(report)
    print()

    # ---------- (2) self-match ----------
    print("## self-match (load codec twice, expect IDENTICAL token convention + codebooks)")
    codec_b = fresh_codec()
    fps_a = codebook_fingerprints(codec_a)
    print(f"  9 codebook fingerprints (first 16 hex chars each):")
    for i, h in enumerate(fps_a):
        print(f"    codebook[{i}] = {h[:16]}…")
    try:
        assert_codec_invariants_match(codec_a, codec_b)
        print("  PASS: token convention matches + all 9 codebooks byte-identical.")
    except InvariantViolation as e:
        print(f"  UNEXPECTED FAIL — two fresh loads should be identical:\n{e}")
        return 1
    print()

    # ---------- (3) deliberate mutation — must be caught ----------
    print("## negative test (deliberately mutate codebook[0][0,0] += 1e-6, expect LOUD FAIL)")
    snapshot = snapshot_codebook_fingerprints(codec_a)
    with torch.no_grad():
        # Even the tiniest perturbation must be detected. Pick a single float, bump it by 1e-6.
        if codec_a.name == "dac":
            cb0 = codec_a.quantizer.quantizers[0].codebook.weight
        elif codec_a.name == "mimi":
            cb0 = codec_a.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed
        else:
            raise NotImplementedError(codec_a.name)
        before = cb0[0, 0].item()
        cb0[0, 0] = cb0[0, 0] + 1e-6
        after = cb0[0, 0].item()
    print(f"  mutated codebook[0][0,0]: {before:+.10f} -> {after:+.10f}  (delta = {after-before:+.2e})")

    caught = False
    try:
        assert_codebook_unchanged(codec_a, snapshot)
    except InvariantViolation as e:
        caught = True
        # Show a snippet so it's visible the assertion produced useful detail.
        msg = str(e).splitlines()
        for line in msg[:3]:
            print(f"  ↳ {line}")

    if not caught:
        print("  CRITICAL FAIL: a 1e-6 codebook mutation went UNDETECTED.")
        print("  The whole experiment is unsafe to continue — the invariant is broken.")
        return 2
    print("  PASS: 1e-6 single-element mutation was detected.")

    # Also verify the pairwise check trips, not just the snapshot one.
    try:
        assert_codec_invariants_match(codec_a, codec_b)
        print("  CRITICAL FAIL: pairwise invariant did NOT trip on mutated codec_a.")
        return 3
    except InvariantViolation:
        print("  PASS: pairwise invariant also trips when comparing mutated codec_a vs fresh codec_b.")

    print()
    print("## verdict")
    print("  Freeze configured correctly + invariants detect codebook drift at single-element scale.")
    print("  Safe to proceed to M3 (decoder fine-tune).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
