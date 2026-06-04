"""Score every MIDI note in the bass stem and emit *retrigger-clean* sample candidates.

The naive picker in script 34 (longest sustained pYIN run) chose a 3.15s nasty
outro sustain on Beltram. This script does two things better:

  (1) Ranks candidates by a composite of bass-band energy ratio, pitch stability,
      duration sweet-spot, onset clarity, edge-guard, and pitch-near-mode.
  (2) Polishes each saved sample so it sounds clean when retriggered:
        - snap start to nearest zero-crossing within ±5 ms of MIDI onset
        - snap end to nearest zero-crossing within ±10 ms of target end
        - fade-in 3 ms / fade-out 30 ms (avoid retrigger clicks)
      A candidate with no clear onset at the MIDI start (oc < 0.05) is filtered
      out — those tend to be pYIN mid-sustain segmentation errors.

Outputs:
  data/song_test/<slug>/bass_samples/
    candidate_NN__score=X.XX__pitch=NN__start=NNN.Ns.wav   (top-N polished audio)
    candidates.json                                          (full ranked metadata)

Run:
  .venv/bin/python scripts/37_extract_bass_exemplars.py --slug beltram_machine
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results" / "stems_v1"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--max-sample-len-s", type=float, default=1.5)
    ap.add_argument("--min-sample-len-s", type=float, default=0.4)
    ap.add_argument("--bass-band-lo", type=float, default=40.0)
    ap.add_argument("--bass-band-hi", type=float, default=300.0)
    ap.add_argument("--edge-guard-s", type=float, default=10.0)
    ap.add_argument("--onset-clarity-min", type=float, default=0.05,
                    help="filter out candidates with onset-clarity below this (mid-sustain artifacts)")
    ap.add_argument("--zc-search-start-ms", type=float, default=5.0,
                    help="snap sample START to nearest zero-cross within ±this many ms of MIDI onset")
    ap.add_argument("--zc-search-end-ms", type=float, default=10.0,
                    help="snap sample END to nearest zero-cross within ±this many ms of target end")
    ap.add_argument("--fade-in-ms", type=float, default=3.0)
    ap.add_argument("--fade-out-ms", type=float, default=30.0)
    return ap.parse_args()


def bass_band_energy_ratio(seg: np.ndarray, sr: int, lo: float, hi: float,
                           n_fft: int = 4096) -> float:
    spec = np.abs(np.fft.rfft(seg, n=max(n_fft, len(seg)))) ** 2
    freqs = np.fft.rfftfreq(max(n_fft, len(seg)), d=1.0 / sr)
    in_band = float(np.sum(spec[(freqs >= lo) & (freqs <= hi)]))
    total = float(np.sum(spec) + 1e-12)
    return in_band / total


def pitch_stability(seg: np.ndarray, sr: int, fmin: float = 40.0, fmax: float = 400.0) -> float:
    if len(seg) < 4096:
        return 0.0
    try:
        f0, voiced, _ = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=sr,
                                     frame_length=2048, hop_length=256, fill_na=np.nan)
    except Exception:
        return 0.0
    mask = np.isfinite(f0) & (voiced if voiced is not None else np.ones_like(f0, dtype=bool))
    f0v = f0[mask]
    if len(f0v) < 4:
        return 0.0
    rel_std = float(np.std(f0v) / (np.mean(f0v) + 1e-9))
    return float(1.0 / (1.0 + 10.0 * rel_std))


def duration_score(dur_s: float, lo: float = 0.6, hi: float = 1.5) -> float:
    if dur_s <= lo / 2 or dur_s >= hi * 2:
        return 0.0
    if dur_s < lo:
        return (dur_s - lo / 2) / (lo / 2)
    if dur_s <= hi:
        return 1.0
    return (hi * 2 - dur_s) / hi


def nearest_zero_crossing(audio: np.ndarray, target_idx: int, search_radius: int) -> int:
    """Return the index of the zero-crossing closest to target_idx within ±search_radius samples.
    If no crossing found in window, return target_idx unchanged.
    """
    lo = max(0, target_idx - search_radius)
    hi = min(len(audio) - 1, target_idx + search_radius)
    if hi - lo < 2:
        return target_idx
    region = audio[lo:hi + 1]
    # Crossings are where the sign changes between consecutive samples
    sgn = np.sign(region)
    # Treat sample==0 as positive (avoids triple-counting through zero)
    sgn[sgn == 0] = 1
    crossings = np.where(np.diff(sgn) != 0)[0]  # indices into region [0, len-1)
    if len(crossings) == 0:
        return target_idx
    abs_idxs = lo + crossings
    return int(abs_idxs[np.argmin(np.abs(abs_idxs - target_idx))])


def apply_fades(seg: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    out = seg.astype(np.float32, copy=True)
    fin = int(fade_in_ms / 1000.0 * sr)
    fout = int(fade_out_ms / 1000.0 * sr)
    if fin > 0 and fin < len(out):
        out[:fin] *= np.linspace(0.0, 1.0, fin, dtype=np.float32)
    if fout > 0 and fout < len(out):
        out[-fout:] *= np.linspace(1.0, 0.0, fout, dtype=np.float32)
    return out


def polish_sample(audio: np.ndarray, sr: int, raw_start: int, raw_end: int,
                  zc_start_ms: float, zc_end_ms: float,
                  fade_in_ms: float, fade_out_ms: float) -> tuple[np.ndarray, int, int]:
    """Snap boundaries to nearest zero-crossings, apply fades. Returns (audio, snapped_start, snapped_end)."""
    zc_start_r = max(1, int(zc_start_ms / 1000.0 * sr))
    zc_end_r = max(1, int(zc_end_ms / 1000.0 * sr))
    start = nearest_zero_crossing(audio, raw_start, zc_start_r)
    end = nearest_zero_crossing(audio, raw_end, zc_end_r)
    if end <= start:
        end = min(len(audio), start + max(1, raw_end - raw_start))
    seg = audio[start:end].copy()
    seg = apply_fades(seg, sr, fade_in_ms, fade_out_ms)
    return seg, start, end


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    stem_path = song_dir / "stems" / f"{args.stem_name}.wav"
    midi_json = song_dir / "semantic" / f"{args.stem_name}.json"
    out_dir = song_dir / "bass_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_root) / args.slug / "bass_samples"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"# bass exemplar ranking · slug={args.slug}")
    print(f"input: {stem_path}, {midi_json}")
    print(f"output: {out_dir}  (mirrored to {results_dir})")

    y_st, sr = sf.read(stem_path, dtype="float32", always_2d=True)
    y_mono = y_st.mean(axis=1).astype(np.float32) if y_st.shape[1] > 1 else y_st[:, 0]
    track_dur = len(y_mono) / sr
    notes = json.loads(midi_json.read_text())
    print(f"  track:  {track_dur:.1f}s · sr={sr}")
    print(f"  notes:  {len(notes)} total")

    pitch_counts = Counter(int(n["pitch"]) for n in notes)
    mode_pitch = pitch_counts.most_common(1)[0][0]
    print(f"  modal pitch: MIDI {mode_pitch} ({librosa.midi_to_note(mode_pitch)})"
          f"  (count={pitch_counts[mode_pitch]} / {len(notes)})")

    print("  computing onset envelope ...")
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=512)
    onset_max = float(np.max(onset_env) + 1e-9)

    candidates: list[dict] = []
    filtered_no_onset = 0
    for idx, note in enumerate(notes):
        dur_full = float(note["end_s"]) - float(note["start_s"])
        if dur_full < args.min_sample_len_s:
            continue
        seg_len_s = min(dur_full, args.max_sample_len_s)
        raw_start = int(float(note["start_s"]) * sr)
        raw_end = min(raw_start + int(seg_len_s * sr), len(y_mono))
        if raw_end - raw_start < int(args.min_sample_len_s * sr):
            continue

        # Hard gate: there must be a real onset at the MIDI start
        onset_frame = min(int(raw_start / 512), len(onset_env) - 1)
        oc_local = float(np.max(onset_env[max(0, onset_frame - 5):onset_frame + 6]))
        oc = oc_local / onset_max
        if oc < args.onset_clarity_min:
            filtered_no_onset += 1
            continue

        seg_raw = y_mono[raw_start:raw_end]
        mid_s = (float(note["start_s"]) + float(note["end_s"])) / 2.0
        in_edge_guard = (mid_s < args.edge_guard_s
                         or mid_s > track_dur - args.edge_guard_s)

        bbr = bass_band_energy_ratio(seg_raw, sr, args.bass_band_lo, args.bass_band_hi)
        ps = pitch_stability(seg_raw, sr)
        ds = duration_score(seg_len_s)
        nae = 0.0 if in_edge_guard else 1.0
        nop = max(0.0, 1.0 - abs(int(note["pitch"]) - mode_pitch) / 24.0)

        composite = (
            0.35 * bbr + 0.20 * ps + 0.15 * ds
            + 0.10 * oc + 0.05 * nae + 0.15 * nop
        )

        candidates.append({
            "idx": idx,
            "start_s": float(note["start_s"]),
            "end_s": float(note["end_s"]),
            "pitch": int(note["pitch"]),
            "pitch_name": str(librosa.midi_to_note(int(note["pitch"]))),
            "velocity": int(note["velocity"]),
            "raw_sample_len_s": float(seg_len_s),
            "scores": {
                "bass_band_ratio": float(bbr),
                "pitch_stability": float(ps),
                "duration_score": float(ds),
                "onset_clarity": float(oc),
                "not_at_edges": float(nae),
                "not_outlier_pitch": float(nop),
            },
            "composite": float(composite),
        })

    candidates.sort(key=lambda c: c["composite"], reverse=True)
    print(f"  filtered {filtered_no_onset} notes with no onset at MIDI start")
    print(f"  scored {len(candidates)} candidates")

    print()
    print(f"## top {args.top_n} candidates (polished: zc + fades)")
    print(f"{'#':>3}  {'score':>5}  {'pitch':>6}  {'start_s':>8}  "
          f"{'len_s':>6}  {'bbr':>5}  {'ps':>5}  {'ds':>5}  {'oc':>5}  {'nop':>5}")
    saved: list[dict] = []
    for rank, c in enumerate(candidates[: args.top_n], start=1):
        raw_start = int(c["start_s"] * sr)
        raw_end = min(raw_start + int(c["raw_sample_len_s"] * sr), len(y_mono))
        polished, sn_start, sn_end = polish_sample(
            y_mono, sr, raw_start, raw_end,
            args.zc_search_start_ms, args.zc_search_end_ms,
            args.fade_in_ms, args.fade_out_ms,
        )
        polished_len_s = len(polished) / sr
        fname = (f"candidate_{rank:02d}__score={c['composite']:.3f}"
                 f"__pitch={c['pitch_name']}__start={c['start_s']:.1f}s.wav")
        sf.write(out_dir / fname, polished, sr, subtype="FLOAT")
        sf.write(results_dir / fname, polished, sr, subtype="FLOAT")
        s = c["scores"]
        print(f"{rank:>3}  {c['composite']:>5.3f}  {c['pitch_name']:>6}  "
              f"{c['start_s']:>8.2f}  {polished_len_s:>6.2f}  "
              f"{s['bass_band_ratio']:>5.2f}  {s['pitch_stability']:>5.2f}  "
              f"{s['duration_score']:>5.2f}  {s['onset_clarity']:>5.2f}  "
              f"{s['not_outlier_pitch']:>5.2f}")
        c_out = dict(c)
        c_out["polished_sample"] = {
            "file": fname,
            "start_sample": int(sn_start),
            "end_sample": int(sn_end),
            "polished_len_s": float(polished_len_s),
            "fade_in_ms": float(args.fade_in_ms),
            "fade_out_ms": float(args.fade_out_ms),
        }
        saved.append(c_out)

    (out_dir / "candidates.json").write_text(json.dumps(saved + candidates[args.top_n:], indent=2))
    (results_dir / "candidates.json").write_text(json.dumps(saved + candidates[args.top_n:], indent=2))
    print()
    print(f"  audition top-{args.top_n} from: {results_dir}")
    print(f"  full metadata:   {results_dir / 'candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
