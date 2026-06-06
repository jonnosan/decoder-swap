"""Generate a human-readable summary of auto-extract results.

Reads results/auto_extract_summary.json and produces a markdown table sorted by
score. Highlights low-scoring tracks that may need manual review.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    summary_path = REPO_ROOT / "results" / "auto_extract_summary.json"
    if not summary_path.exists():
        print(f"missing: {summary_path}", file=sys.stderr)
        return 1
    results = json.loads(summary_path.read_text())

    succeeded = [r for r in results if "score" in r]
    failed = [r for r in results if "error" in r]
    succeeded.sort(key=lambda r: -r.get("score", 0))

    out = []
    out.append(f"# Auto-extract results — {len(results)} tracks")
    out.append("")
    out.append(f"- **Succeeded**: {len(succeeded)} / {len(results)}")
    out.append(f"- **Failed**: {len(failed)}")
    if succeeded:
        scores = [r["score"] for r in succeeded]
        out.append(f"- **Score range**: min={min(scores):.3f}, median={sorted(scores)[len(scores)//2]:.3f}, max={max(scores):.3f}")
        # Tiers
        high = [r for r in succeeded if r["score"] >= 0.5]
        mid = [r for r in succeeded if 0.3 <= r["score"] < 0.5]
        low = [r for r in succeeded if r["score"] < 0.3]
        out.append(f"- **High score (≥0.5)**: {len(high)} tracks")
        out.append(f"- **Mid score (0.3-0.5)**: {len(mid)} tracks")
        out.append(f"- **Low score (<0.3)**: {len(low)} tracks  — likely auto-sweep failures, need manual review")
    out.append("")
    out.append("## Per-track results (sorted by score)")
    out.append("")
    out.append("| Score | High-act | BPM | Anchor offset | Pre-merge | Region | Slug |")
    out.append("|---:|---:|---:|---:|---:|---|---|")
    for r in succeeded:
        score = r["score"]
        stats = r.get("stats", {})
        high_act = stats.get("n_high_activity", 0)
        bpm = r.get("bpm", 0)
        offset = r.get("anchor_offset_steps", "?")
        premerge = r.get("pre_merge_ms", "?")
        region_start = r.get("region_start", 0)
        slug = r.get("slug", "?")
        out.append(f"| {score:.3f} | {high_act} | {bpm:.2f} | {offset:+d} | {premerge} | t={region_start:.1f}s | {slug} |")

    if failed:
        out.append("")
        out.append("## Failed tracks")
        out.append("")
        for r in failed:
            out.append(f"- **{r.get('slug', '?')}**: {r.get('error', 'unknown')}")

    out.append("")
    out.append("## Key observations")
    out.append("")
    # Param distribution analysis
    if succeeded:
        n_premerge_0 = sum(1 for r in succeeded if r.get("pre_merge_ms") == 0)
        n_premerge_60 = sum(1 for r in succeeded if r.get("pre_merge_ms") == 60)
        out.append(f"- **pre_merge=0** picked for {n_premerge_0} tracks, **pre_merge=60** for {n_premerge_60}")
        from collections import Counter
        offsets = Counter(r.get("anchor_offset_steps", 0) for r in succeeded)
        out.append(f"- **Anchor offset distribution**: {dict(offsets)}")
        bpm_diags_x = [r.get("bpm_diag", {}).get("bpm_cross") for r in succeeded]
        bpm_diags_l = [r.get("bpm_diag", {}).get("bpm_librosa") for r in succeeded]
        n_cross_used = sum(1 for r in succeeded
                          if r.get("bpm_diag", {}).get("bpm_cross")
                          and abs(r.get("bpm", 0) - r["bpm_diag"]["bpm_cross"]) < 0.5)
        out.append(f"- **BPM from cross-corr**: {n_cross_used} tracks, librosa fallback: {len(succeeded) - n_cross_used}")

    out_path = REPO_ROOT / "results" / "auto_extract_summary.md"
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path}")
    print()
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
