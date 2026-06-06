"""Pick a representative sample of auto-extracted tracks for user listening.

Strategy:
  - Top 3 by score (best successes)
  - Bottom 3 by score (worst extractions — most likely problems)
  - Middle 2 (median quality)

Stages those into results/auto_extract/listen_samples/ with rendered audio
and a markdown index.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    summary_path = REPO_ROOT / "results" / "auto_extract_summary.json"
    if not summary_path.exists():
        print("missing summary", file=sys.stderr)
        return 1
    results = json.loads(summary_path.read_text())
    succeeded = sorted([r for r in results if "score" in r], key=lambda r: r["score"])
    if not succeeded:
        print("no successes", file=sys.stderr)
        return 1

    # Pick representative samples
    picks = []
    # Top 3
    picks.extend([("top", r) for r in succeeded[-3:][::-1]])
    # Bottom 3
    picks.extend([("bot", r) for r in succeeded[:3]])
    # Middle 2 (around the median)
    mid_idx = len(succeeded) // 2
    picks.extend([("mid", r) for r in succeeded[max(0, mid_idx-1):mid_idx+1]])

    # Dedupe (top can overlap with bot if very few tracks)
    seen_slugs = set()
    unique_picks = []
    for label, r in picks:
        if r["slug"] in seen_slugs:
            continue
        seen_slugs.add(r["slug"])
        unique_picks.append((label, r))

    out_dir = REPO_ROOT / "results" / "auto_extract" / "listen_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio files + build index
    index = []
    index.append("# Listen samples — auto-extract review")
    index.append("")
    index.append("Suggested listening: spot-check the top samples confirm quality, then check bottom samples to see what failure modes look like.")
    index.append("")
    index.append("| Tier | Score | High-act | BPM | Pre-merge | Offset | Slug |")
    index.append("|---|---:|---:|---:|---:|---:|---|")
    for label, r in unique_picks:
        slug = r["slug"]
        src_auto = REPO_ROOT / "results" / "auto_extract" / f"{slug}_auto30.wav"
        src_orig = REPO_ROOT / "results" / "auto_extract" / f"{slug}_original30.wav"
        for src in (src_auto, src_orig):
            if src.exists():
                shutil.copy(src, out_dir / src.name)
        score = r["score"]
        stats = r.get("stats", {})
        index.append(f"| {label} | {score:.3f} | {stats.get('n_high_activity', 0)} | {r.get('bpm', 0):.2f} | {r.get('pre_merge_ms', '?')} | {r.get('anchor_offset_steps', '?'):+d} | {slug} |")

    index.append("")
    index.append("## Files in this folder")
    index.append("")
    for label, r in unique_picks:
        index.append(f"- `{r['slug']}_original30.wav` + `{r['slug']}_auto30.wav` — {label}, score {r['score']:.3f}")

    (out_dir / "INDEX.md").write_text("\n".join(index))
    print(f"Staged {len(unique_picks)} sample tracks in {out_dir}")
    print(f"Index: {out_dir / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
