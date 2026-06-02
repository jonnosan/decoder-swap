"""Per-corpus config loader and path conventions for the multi-style pipeline.

Each corpus (techno, vaporwave, …) has its own `corpora/<name>.yaml` describing source audio
paths + held-out clips, and gets its own folder for cached tokens, checkpoints, and results.

The jtxtok contract (`docs/JTXTOK_SPEC.md`) is corpus-agnostic by design — same vocabulary
across all corpora — so a "techno model" and a "vaporwave model" are the same architecture
with different weights loaded from different per-corpus checkpoint folders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPORA_DIR = REPO_ROOT / "corpora"


@dataclass
class Corpus:
    """One genre/style's corpus configuration. Loaded from corpora/<name>.yaml."""
    name: str
    description: str
    audio_paths: list[str]
    heldout: list[str] = field(default_factory=list)
    config_path: Path | None = None

    def tokens_dir(self, codec: str = "dac", repo_root: Path | None = None) -> Path:
        """Where this corpus's cached codec tokens live."""
        root = repo_root or REPO_ROOT
        return root / f"data/tokens_{codec}" / self.name

    def translator_ckpt_dir(self, repo_root: Path | None = None) -> Path:
        """Translator (M6.A base LM + M7 conditioned) checkpoints for this corpus."""
        root = repo_root or REPO_ROOT
        return root / "data/checkpoints/translator" / self.name

    def d2_ckpt_dir(self, repo_root: Path | None = None) -> Path:
        """Decoder-swap M3 D2 decoder checkpoint dir for this corpus (historical)."""
        root = repo_root or REPO_ROOT
        return root / "data/checkpoints/d2" / self.name

    def results_dir(self, label: str, repo_root: Path | None = None) -> Path:
        """Standard results-dir convention: results/<label>_<corpus>/."""
        root = repo_root or REPO_ROOT
        return root / "results" / f"{label}_{self.name}"


def load_corpus(name: str, corpora_dir: Path | None = None) -> Corpus:
    """Load a corpus by name from corpora/<name>.yaml. Raises if not found."""
    cdir = corpora_dir or DEFAULT_CORPORA_DIR
    path = cdir / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in cdir.glob("*.yaml")) if cdir.exists() else []
        raise FileNotFoundError(
            f"corpus config not found: {path}. "
            f"Available: {available or '(none — corpora/ is empty)'}"
        )
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Corpus(
        name=raw["name"],
        description=raw.get("description", ""),
        audio_paths=list(raw["audio_paths"]),
        heldout=list(raw.get("heldout") or []),
        config_path=path,
    )


def list_corpora(corpora_dir: Path | None = None) -> list[str]:
    """Names of all corpora with a config file in corpora/."""
    cdir = corpora_dir or DEFAULT_CORPORA_DIR
    if not cdir.exists():
        return []
    return sorted(p.stem for p in cdir.glob("*.yaml"))
