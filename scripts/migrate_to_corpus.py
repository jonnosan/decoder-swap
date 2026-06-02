"""One-shot migration into the per-corpus folder layout.

Moves the historical single-corpus artifacts into the new `<artifact>/<corpus>/` layout:

  data/tokens_dac/*.npy                    -> data/tokens_dac/<corpus>/
  data/checkpoints/translator/translator_lm.pt -> data/checkpoints/translator/<corpus>/
  data/checkpoints/d2_decoder.pt           -> data/checkpoints/d2/<corpus>/

Idempotent: if a destination already has files, the script refuses rather than overwriting
(so accidental re-runs across corpora don't silently merge data). Re-run with --force to
override that safety check.

Run:
  uv run python scripts/migrate_to_corpus.py techno          # migrate Vytis assets under techno/
  uv run python scripts/migrate_to_corpus.py techno --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.corpus import load_corpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="corpus name to migrate to (must have corpora/<name>.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="show what would move; don't move")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if destination already contains files")
    return ap.parse_args()


def _list_pre(p: Path, glob: str = "*") -> list[Path]:
    return sorted(p.glob(glob)) if p.exists() else []


def move(src: Path, dst_dir: Path, dry: bool) -> bool:
    """Move src into dst_dir/<src.name>. Returns True if move was planned/executed."""
    dst = dst_dir / src.name
    if dst.exists():
        print(f"  [skip] {src.name} — destination already exists: {dst}")
        return False
    if dry:
        print(f"  [dry]  {src} -> {dst}")
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  moved  {src.name} -> {dst}")
    return True


def main() -> int:
    args = parse_args()
    corpus = load_corpus(args.corpus)
    print(f"# migrate to corpus '{corpus.name}'  {'(dry run)' if args.dry_run else ''}")

    # 1. DAC tokens: data/tokens_dac/*.npy -> data/tokens_dac/<corpus>/
    src_tokens_dir = REPO_ROOT / "data/tokens_dac"
    dst_tokens_dir = corpus.tokens_dir(codec="dac")
    legacy_token_files = [p for p in _list_pre(src_tokens_dir, "*.npy") if p.parent == src_tokens_dir]
    if legacy_token_files:
        existing = _list_pre(dst_tokens_dir, "*.npy")
        if existing and not args.force:
            print(f"REFUSING: destination {dst_tokens_dir} already has {len(existing)} .npy file(s). "
                  f"Use --force to override.")
            return 2
        print(f"DAC tokens: {len(legacy_token_files)} -> {dst_tokens_dir}")
        for p in legacy_token_files:
            move(p, dst_tokens_dir, args.dry_run)
    else:
        print(f"DAC tokens: no legacy .npy in {src_tokens_dir} (already migrated or never created)")

    # 2. Translator checkpoint: data/checkpoints/translator/translator_lm.pt
    #    -> data/checkpoints/translator/<corpus>/
    src_translator_dir = REPO_ROOT / "data/checkpoints/translator"
    dst_translator_dir = corpus.translator_ckpt_dir()
    legacy_translator = [
        p for p in _list_pre(src_translator_dir, "*.pt") + _list_pre(src_translator_dir, "*.json")
        if p.parent == src_translator_dir
    ]
    if legacy_translator:
        existing = _list_pre(dst_translator_dir, "*")
        if existing and not args.force:
            print(f"REFUSING: destination {dst_translator_dir} already has files. Use --force.")
            return 2
        print(f"translator ckpt: {len(legacy_translator)} file(s) -> {dst_translator_dir}")
        for p in legacy_translator:
            move(p, dst_translator_dir, args.dry_run)
    else:
        print(f"translator ckpt: no legacy file(s) at top of {src_translator_dir}")

    # 3. Decoder-swap D2 (M3): data/checkpoints/d2_decoder.pt + sidecar -> data/checkpoints/d2/<corpus>/
    src_ck = REPO_ROOT / "data/checkpoints"
    dst_d2_dir = corpus.d2_ckpt_dir()
    legacy_d2 = [src_ck / "d2_decoder.pt", src_ck / "d2_losses.json"]
    legacy_d2 = [p for p in legacy_d2 if p.exists()]
    if legacy_d2:
        existing = _list_pre(dst_d2_dir, "*")
        if existing and not args.force:
            print(f"REFUSING: destination {dst_d2_dir} already has files. Use --force.")
            return 2
        print(f"D2 (M3) ckpt: {len(legacy_d2)} file(s) -> {dst_d2_dir}")
        for p in legacy_d2:
            move(p, dst_d2_dir, args.dry_run)
    else:
        print(f"D2 (M3) ckpt: no legacy d2_decoder.pt at top of {src_ck}")

    print()
    print("done" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
