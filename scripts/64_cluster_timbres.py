"""Cluster the per-note feature vectors into N timbre IDs (k-means).

After 61_extract_per_note_data.py has produced features.npy for a dataset, this
script standardises the features and runs k-means to assign each note a
`timbre_id` in [0, N). The result is saved as timbre_ids.npy alongside the
other per-note arrays, plus cluster_centers.npy for later inspection.

The semantic meaning of each cluster emerges from the data; we attach human-
readable names after listening to representative notes per cluster.

Run:
  .venv/bin/python scripts/64_cluster_timbres.py --dataset vytis_vol1_v2 --n-timbres 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "per_note"))
    ap.add_argument("--n-timbres", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-reps", type=int, default=3,
                    help="number of representative note indices to record per cluster")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.root) / args.dataset
    if not data_dir.exists():
        print(f"ERROR: dataset dir not found: {data_dir}")
        return 1

    features = np.load(data_dir / "features.npy")     # (N, F)
    meta = json.loads((data_dir / "meta.json").read_text())
    feature_names = meta.get("feature_names", [f"f{i}" for i in range(features.shape[1])])
    N, F = features.shape
    print(f"# cluster timbres · dataset={args.dataset}  notes={N}  features={F}")

    # Standardise features (zero-mean, unit-variance per column).
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)

    print(f"  k-means: n_clusters={args.n_timbres}  seed={args.seed}")
    km = KMeans(n_clusters=args.n_timbres, random_state=args.seed, n_init=10)
    ids = km.fit_predict(Xs).astype(np.int8)
    centers_std = km.cluster_centers_                  # (K, F) in standardised space
    centers_raw = scaler.inverse_transform(centers_std)  # in original feature units

    # Per-cluster summary
    counts = Counter(int(x) for x in ids)
    print(f"\n  cluster sizes:")
    for c in sorted(counts):
        print(f"    cluster {c:>2}  {counts[c]:>5d} notes  ({100*counts[c]/N:5.1f}%)")

    print(f"\n  cluster centroids (raw feature values):")
    header = "    " + "cluster".ljust(8) + "  ".join(f"{n[:18]:>18s}" for n in feature_names)
    print(header)
    for c in range(args.n_timbres):
        row = f"    {c:<8d}" + "  ".join(f"{centers_raw[c, i]:>18.4f}" for i in range(F))
        print(row)

    # Pick representative note indices per cluster (closest to the centroid in std space).
    reps: dict[int, list[int]] = {}
    for c in range(args.n_timbres):
        idx_c = np.where(ids == c)[0]
        if len(idx_c) == 0:
            reps[c] = []
            continue
        dists = np.linalg.norm(Xs[idx_c] - centers_std[c], axis=1)
        order = np.argsort(dists)[: args.n_reps]
        reps[c] = [int(idx_c[i]) for i in order]
        # Annotate the rep notes with their per-note metadata if available
        meta_notes = meta.get("notes", [])
        if meta_notes:
            for ri in reps[c]:
                if 0 <= ri < len(meta_notes):
                    n_info = meta_notes[ri]
                    print(f"    rep[cluster={c}] idx={ri:>5d}  "
                          f"song={n_info.get('song','?'):<40s} "
                          f"t={n_info.get('song_start_s', 0):>7.1f}s  "
                          f"pitch={n_info.get('pitch','?'):>3}  "
                          f"vel={n_info.get('velocity','?'):>3}")

    np.save(data_dir / "timbre_ids.npy", ids)
    np.save(data_dir / "cluster_centers_raw.npy", centers_raw.astype(np.float32))
    np.save(data_dir / "cluster_centers_std.npy", centers_std.astype(np.float32))
    np.save(data_dir / "scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(data_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))

    # Append cluster info to meta.json so downstream code finds it.
    meta["n_timbres"] = args.n_timbres
    meta["timbre_seed"] = args.seed
    meta["timbre_representatives"] = reps
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  saved:")
    print(f"    timbre_ids.npy         {ids.shape}  dtype {ids.dtype}")
    print(f"    cluster_centers_raw    {centers_raw.shape}")
    print(f"    scaler_{{mean,scale}}    {scaler.mean_.shape}")
    print(f"  meta.json updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
