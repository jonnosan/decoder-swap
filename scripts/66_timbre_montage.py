"""Per-timbre listening montage from the ORIGINAL bass.wav slices.

For each k-means timbre cluster, this script picks N notes nearest to the
cluster centroid (in standardised feature space), slices each note's audio
from the source bass.wav at the timestamp recorded in meta.json, and stitches
them into one playable file with short gaps.

Useful diagnostic: confirms whether each cluster is actually representing
bass content vs Demucs noise / silence / leaked drum hits. The model can only
be as good as what's in each cluster.

Run:
  .venv/bin/python scripts/66_timbre_montage.py --dataset vytis_vol1_v2 --n-per-cluster 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "per_note"))
    ap.add_argument("--song-root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-per-cluster", type=int, default=10,
                    help="number of representative notes to include per cluster")
    ap.add_argument("--clip-ms", type=float, default=700.0,
                    help="audio length per note in the montage (ms)")
    ap.add_argument("--gap-ms", type=float, default=250.0,
                    help="silent gap between successive notes (ms)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.root) / args.dataset
    if not data_dir.exists():
        print(f"ERROR: dataset not found at {data_dir}")
        return 1
    meta = json.loads((data_dir / "meta.json").read_text())
    n_timbres = int(meta.get("n_timbres", 0))
    if n_timbres == 0:
        print(f"ERROR: dataset has no timbre_ids — run scripts/64_cluster_timbres.py first")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "results" / "timbre_montage" / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(data_dir / "features.npy")
    timbre_ids = np.load(data_dir / "timbre_ids.npy")
    centers_std = np.load(data_dir / "cluster_centers_std.npy")
    scaler_mean = np.load(data_dir / "scaler_mean.npy")
    scaler_scale = np.load(data_dir / "scaler_scale.npy")
    # Standardise features to find nearest-to-centroid notes per cluster.
    Xs = (features - scaler_mean) / scaler_scale

    notes_meta = meta.get("notes", [])
    if not notes_meta or "song_start_s" not in notes_meta[0]:
        print("ERROR: meta.json notes lack song_start_s — re-extract with the updated 61 script.")
        return 1
    n_notes = len(notes_meta)
    print(f"# timbre montage  dataset={args.dataset}  n_notes={n_notes}  n_timbres={n_timbres}")

    # Cache loaded songs (bass.wav) since multiple notes share a song.
    loaded_songs: dict[str, tuple[np.ndarray, int]] = {}

    def load_song(slug: str) -> tuple[np.ndarray, int]:
        if slug not in loaded_songs:
            bp = Path(args.song_root) / slug / "stems" / "bass.wav"
            audio, sr = sf.read(bp, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
            loaded_songs[slug] = (audio, sr)
            print(f"  loaded {bp}  ({len(audio)/sr:.1f}s)")
        return loaded_songs[slug]

    for c in range(n_timbres):
        idx_c = np.where(timbre_ids == c)[0]
        if len(idx_c) == 0:
            print(f"  [cluster {c}] empty")
            continue
        # Pick N nearest to centroid
        dists = np.linalg.norm(Xs[idx_c] - centers_std[c], axis=1)
        order = np.argsort(dists)[: args.n_per_cluster]
        chosen = idx_c[order]

        # Build a montage
        sample_rate: int | None = None
        clip_samples: int | None = None
        gap_samples: int | None = None
        chunks: list[np.ndarray] = []
        annotations: list[dict] = []
        for ni in chosen:
            info = notes_meta[int(ni)]
            slug = info["song"]
            start_s = float(info["song_start_s"])
            audio, sr = load_song(slug)
            if sample_rate is None:
                sample_rate = sr
                clip_samples = int(sr * args.clip_ms / 1000.0)
                gap_samples = int(sr * args.gap_ms / 1000.0)
            s0 = int(round(start_s * sr))
            s1 = min(len(audio), s0 + clip_samples)
            clip = audio[s0:s1]
            if len(clip) < clip_samples:
                clip = np.pad(clip, (0, clip_samples - len(clip)))
            chunks.append(clip)
            chunks.append(np.zeros(gap_samples, dtype=np.float32))
            annotations.append({
                "note_idx": int(ni),
                "song": slug,
                "song_start_s": start_s,
                "pitch": info.get("pitch"),
                "velocity": info.get("velocity"),
                "duration_s": info.get("duration_s"),
                "quality_rms": info.get("quality_rms"),
            })
        montage = np.concatenate(chunks).astype(np.float32)
        montage = np.clip(montage, -1.0, 1.0)
        out_wav = out_dir / f"montage_timbre{c}.wav"
        sf.write(str(out_wav), montage, sample_rate or 44100)
        print(f"  [cluster {c}]  {len(idx_c):>5d} total · "
              f"montage of {len(chosen)} notes -> {out_wav.name}  "
              f"({len(montage)/(sample_rate or 44100):.1f}s)")
        (out_dir / f"montage_timbre{c}.json").write_text(json.dumps(annotations, indent=2))

    print(f"\n# done. files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
