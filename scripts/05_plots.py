"""M5: generate the figures for the writeup. Reads the wavs in results/m4_compare* and the
training-loss JSON, writes PNGs alongside.

Run: uv run python scripts/05_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import soundfile as sf  # noqa: E402

from decoder_swap.plot import comparison_figure, training_loss_figure  # noqa: E402


def main() -> int:
    pairs = [
        (REPO_ROOT / "results" / "m4_compare", "Blue Kentucky Girl (country) — DAC D1 vs D2"),
        (REPO_ROOT / "results" / "m4_compare_rock", "AC/DC — Back In Black (rock) — DAC D1 vs D2"),
        (REPO_ROOT / "results" / "m4_compare_mimi", "Blue Kentucky Girl — Mimi D1 vs D2 (1.1 kbps, 12.5 fps)"),
    ]
    for d, title in pairs:
        s1, sr1 = sf.read(d / "S1.wav")
        s2, sr2 = sf.read(d / "S2.wav")
        assert sr1 == sr2
        out = d / "comparison.png"
        comparison_figure(s1, s2, sr1, title=title, out_path=out)
        print(f"wrote {out}")

    losses_json = REPO_ROOT / "data" / "checkpoints" / "d2_losses.json"
    if losses_json.exists():
        out = REPO_ROOT / "results" / "m3_training_loss.png"
        training_loss_figure(losses_json, out)
        print(f"wrote {out}")
    else:
        print(f"skipping loss curve — {losses_json} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
