"""Load config.yaml and resolve runtime device."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass
class Settings:
    raw: dict[str, Any]
    config_path: Path

    @property
    def codec_name(self) -> str:
        return self.raw["codec"]["name"]

    @property
    def codec_model_type(self) -> str:
        return self.raw["codec"]["model_type"]

    @property
    def codec_model_tag(self) -> str | None:
        return self.raw["codec"].get("model_tag")

    @property
    def codec_model_path(self) -> str | None:
        return self.raw["codec"].get("model_path")

    @property
    def device(self) -> str:
        return self.raw.get("device", "auto")


def load_settings(path: Path | str | None = None) -> Settings:
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open() as f:
        raw = yaml.safe_load(f)
    return Settings(raw=raw, config_path=p)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
