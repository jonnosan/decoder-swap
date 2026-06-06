"""Score a bass stem's "bass-pipeline-friendliness" using a simple audio heuristic.

Heuristic: score = onset_density × env_peakiness
  - onset_density   = number of attack events per second in 500-3000 Hz bandpass
  - env_peakiness   = std/mean of the onset_strength envelope in same band
  Product captures both "has rhythmic events" AND "events are distinct from noise floor".

Empirical thresholds (from 19-track validation set 2026-06-05):
  >= 3.0  → auto-include  (5/5 confirmed-good tracks in sample)
  1.0-3.0 → manual review (borderline; capture drum-bleed bass + mediocre extractions)
  <  1.0  → auto-exclude  (drones, pads, silent stems)

User preference (2026-06-05): precision-leaning gate — use threshold 3.0, accept that
some good-but-drum-bleed tracks (e.g. Mayday Bolland) will be excluded.

Run:
  .venv/bin/python scripts/70_score_bass_quality.py \\
    data/song_test/dmxkrew_101_tonight/stems/bass.wav
  .venv/bin/python scripts/70_score_bass_quality.py \\
    --threshold 3.0 \\
    data/song_test/*/stems/bass.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def score(path: Path) -> tuple[float, float, float]:
    """Return (onset_density, env_peakiness, score)."""
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(-1)
    sos = butter(4, [500 / (sr / 2), 3000 / (sr / 2)], "band", output="sos")
    yb = sosfiltfilt(sos, x)
    on = librosa.onset.onset_detect(
        y=yb, sr=sr, hop_length=512, backtrack=True,
        pre_max=4, post_max=4, pre_avg=20, post_avg=20,
        delta=0.2, wait=int(0.10 * sr / 512),
    )
    dur_s = len(x) / sr
    onset_density = len(on) / dur_s
    env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=512)
    env_peakiness = float(np.std(env) / (np.mean(env) + 1e-9))
    return onset_density, env_peakiness, onset_density * env_peakiness


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="bass.wav paths to score")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="score below this = excluded; default 3.0 (precision-leaning)")
    args = ap.parse_args()

    print(f"{'path':70s}  {'on/s':>5}  {'envP':>5}  {'score':>6}  verdict")
    print("-" * 100)
    rows = []
    for p in args.paths:
        p = Path(p)
        try:
            od, ep, sc = score(p)
        except Exception as e:
            print(f"{p.name}: ERROR {e}", file=sys.stderr)
            continue
        rows.append((p, od, ep, sc))
    rows.sort(key=lambda r: -r[3])
    n_inc = n_exc = 0
    for p, od, ep, sc in rows:
        verdict = "INCLUDE" if sc >= args.threshold else "exclude"
        if sc >= args.threshold:
            n_inc += 1
        else:
            n_exc += 1
        # Show only the immediate parent dir + filename for brevity
        display = str(p.parent.parent.name) if p.parent.parent.name else str(p)
        print(f"{display:70s}  {od:5.2f}  {ep:5.2f}  {sc:6.2f}  {verdict}")
    print(f"\n{n_inc} included / {n_exc} excluded (threshold {args.threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
