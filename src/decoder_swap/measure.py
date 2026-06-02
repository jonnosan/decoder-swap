"""§3 metrics for the M4 hypothesis test, plus the verdict block.

Two metric families:

A) Structure-preservation (hypothesis predicts S1 and S2 should AGREE):
   - onset/transient timing  : matched-onset F1 within 50 ms tolerance + mean abs time difference
   - energy-envelope         : Pearson correlation of RMS-over-time
   - tempo                   : librosa.beat tempo estimate for each, report both + delta

B) Realisation-change (hypothesis predicts S1 and S2 should DIFFER):
   - timbre                  : MFCC L1 distance, log-mel L1 distance (dB), spectral-centroid stats
   - pitch                   : chroma cosine similarity (partial preservation expected)

Also reports "reconstruction quality" for each decoder relative to the original input — the
forgetting probe the user actually cares about: input vs S1 (D1 baseline) and input vs S2
(if D2 has forgotten country, this number is HIGH).
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


# ---------- low-level helpers ----------

def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    return librosa.power_to_db(m + 1e-10)


def _mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)


def _chroma(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)


# ---------- A) structure-preservation ----------

def onset_agreement(s1: np.ndarray, s2: np.ndarray, sr: int, tolerance_s: float = 0.050) -> dict:
    o1 = librosa.onset.onset_detect(y=s1, sr=sr, units="time", hop_length=512)
    o2 = librosa.onset.onset_detect(y=s2, sr=sr, units="time", hop_length=512)

    # Greedy nearest-neighbour matching within tolerance. For each s1 onset, claim the closest
    # unclaimed s2 onset if within tolerance. Simple and good enough for a metric.
    used2: set[int] = set()
    matches: list[tuple[float, float]] = []
    for t1 in o1:
        best = None
        for j, t2 in enumerate(o2):
            if j in used2:
                continue
            d = abs(float(t2) - float(t1))
            if d <= tolerance_s and (best is None or d < best[0]):
                best = (d, j, float(t2))
        if best is not None:
            _, j, t2 = best
            used2.add(j)
            matches.append((float(t1), t2))

    n1, n2, nm = len(o1), len(o2), len(matches)
    precision = (nm / n2) if n2 else 0.0
    recall = (nm / n1) if n1 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if matches:
        diffs_ms = [abs(a - b) * 1000.0 for a, b in matches]
        mean_abs_ms = float(np.mean(diffs_ms))
    else:
        mean_abs_ms = float("nan")
    return {
        "n_onsets_s1": int(n1),
        "n_onsets_s2": int(n2),
        "n_matched_within_50ms": int(nm),
        "mean_abs_diff_ms": mean_abs_ms,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tolerance_ms": tolerance_s * 1000.0,
    }


def rms_envelope_correlation(s1: np.ndarray, s2: np.ndarray, sr: int) -> dict:
    r1 = librosa.feature.rms(y=s1, frame_length=2048, hop_length=512)[0]
    r2 = librosa.feature.rms(y=s2, frame_length=2048, hop_length=512)[0]
    r1, r2 = _align(r1, r2)
    if len(r1) < 2:
        return {"correlation": float("nan"), "n_frames": int(len(r1))}
    return {
        "correlation": float(np.corrcoef(r1, r2)[0, 1]),
        "n_frames": int(len(r1)),
        "s1_mean_rms": float(np.mean(r1)),
        "s2_mean_rms": float(np.mean(r2)),
    }


def tempo_agreement(s1: np.ndarray, s2: np.ndarray, sr: int) -> dict:
    t1, _ = librosa.beat.beat_track(y=s1, sr=sr)
    t2, _ = librosa.beat.beat_track(y=s2, sr=sr)
    t1 = float(np.atleast_1d(t1)[0])
    t2 = float(np.atleast_1d(t2)[0])
    return {"tempo_s1_bpm": t1, "tempo_s2_bpm": t2, "abs_delta_bpm": abs(t1 - t2)}


# ---------- B) realisation-change ----------

def mfcc_distance(a: np.ndarray, b: np.ndarray, sr: int, n_mfcc: int = 20) -> float:
    ma = _mfcc(a, sr, n_mfcc=n_mfcc)
    mb = _mfcc(b, sr, n_mfcc=n_mfcc)
    n = min(ma.shape[1], mb.shape[1])
    return float(np.mean(np.abs(ma[:, :n] - mb[:, :n])))


def mel_db_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    la = _mel_db(a, sr)
    lb = _mel_db(b, sr)
    n = min(la.shape[1], lb.shape[1])
    return float(np.mean(np.abs(la[:, :n] - lb[:, :n])))


def spectral_centroid_summary(s1: np.ndarray, s2: np.ndarray, sr: int) -> dict:
    sc1 = librosa.feature.spectral_centroid(y=s1, sr=sr, hop_length=512)[0]
    sc2 = librosa.feature.spectral_centroid(y=s2, sr=sr, hop_length=512)[0]
    sc1, sc2 = _align(sc1, sc2)
    corr = float(np.corrcoef(sc1, sc2)[0, 1]) if len(sc1) >= 2 else float("nan")
    return {
        "correlation": corr,
        "s1_mean_hz": float(np.mean(sc1)),
        "s2_mean_hz": float(np.mean(sc2)),
        "s1_std_hz": float(np.std(sc1)),
        "s2_std_hz": float(np.std(sc2)),
    }


def chroma_cosine_similarity(s1: np.ndarray, s2: np.ndarray, sr: int) -> float:
    c1 = _chroma(s1, sr)
    c2 = _chroma(s2, sr)
    n = min(c1.shape[1], c2.shape[1])
    c1, c2 = c1[:, :n], c2[:, :n]
    num = (c1 * c2).sum(axis=0)
    den = np.linalg.norm(c1, axis=0) * np.linalg.norm(c2, axis=0) + 1e-12
    return float(np.mean(num / den))


# ---------- aggregate + verdict ----------

@dataclass
class Verdict:
    clause: str
    status: str       # "supported" | "not supported" | "mixed"
    headline_metric: str
    headline_value: float
    note: str


def compute_metrics(y_input: np.ndarray, s1: np.ndarray, s2: np.ndarray, sr: int) -> dict:
    """Compute every metric. Returns a JSON-serialisable dict."""
    return {
        "sample_rate": int(sr),
        "duration_seconds": float(len(y_input) / sr),
        "structure_preservation": {
            "onset_agreement_s1_vs_s2": onset_agreement(s1, s2, sr),
            "rms_envelope_correlation_s1_vs_s2": rms_envelope_correlation(s1, s2, sr),
            "tempo_agreement_s1_vs_s2": tempo_agreement(s1, s2, sr),
        },
        "realisation_change": {
            "mfcc_distance_s1_vs_s2": mfcc_distance(s1, s2, sr),
            "mel_db_distance_s1_vs_s2": mel_db_distance(s1, s2, sr),
            "spectral_centroid_summary_s1_vs_s2": spectral_centroid_summary(s1, s2, sr),
            "chroma_cosine_similarity_s1_vs_s2": chroma_cosine_similarity(s1, s2, sr),
        },
        "forgetting_probe": {
            # How close does each decoder reconstruct the original country audio?
            # If D2 has forgotten country, mel_db_distance(input, S2) >> (input, S1).
            "mel_db_distance_input_vs_s1": mel_db_distance(y_input, s1, sr),
            "mel_db_distance_input_vs_s2": mel_db_distance(y_input, s2, sr),
            "mfcc_distance_input_vs_s1": mfcc_distance(y_input, s1, sr),
            "mfcc_distance_input_vs_s2": mfcc_distance(y_input, s2, sr),
        },
    }


def derive_verdict(metrics: dict) -> list[Verdict]:
    """Plain-English judgement per hypothesis clause. Thresholds are heuristic, called out as such."""
    sp = metrics["structure_preservation"]
    rc = metrics["realisation_change"]
    fp = metrics["forgetting_probe"]
    out: list[Verdict] = []

    # Clause 1: transient/onset timing preserved
    f1 = sp["onset_agreement_s1_vs_s2"]["f1"]
    if f1 >= 0.6:
        s = "supported"
    elif f1 < 0.3:
        s = "not supported"
    else:
        s = "mixed"
    out.append(Verdict(
        clause="onset/transient timing preserved between S1 and S2",
        status=s, headline_metric="onset F1 (50 ms tolerance)", headline_value=float(f1),
        note=f"S1 onsets={sp['onset_agreement_s1_vs_s2']['n_onsets_s1']}, "
             f"S2 onsets={sp['onset_agreement_s1_vs_s2']['n_onsets_s2']}, "
             f"mean abs diff = {sp['onset_agreement_s1_vs_s2']['mean_abs_diff_ms']:.1f} ms",
    ))

    # Clause 2: energy envelope preserved
    corr = sp["rms_envelope_correlation_s1_vs_s2"]["correlation"]
    if corr >= 0.7:
        s = "supported"
    elif corr < 0.3:
        s = "not supported"
    else:
        s = "mixed"
    out.append(Verdict(
        clause="energy envelope preserved between S1 and S2",
        status=s, headline_metric="RMS Pearson r", headline_value=float(corr),
        note="time-aligned RMS envelopes correlate over the whole clip",
    ))

    # Clause 3: timbre changed (substantial = supported)
    mel_db = rc["mel_db_distance_s1_vs_s2"]
    # Calibration: identical signals = 0 dB. 0–3 dB ≈ "perceptually similar". 3–8 dB ≈ "noticeably
    # different". 8 dB+ ≈ "clearly different timbral world". These are heuristics, not standards.
    if mel_db >= 8.0:
        s = "supported"
    elif mel_db < 3.0:
        s = "not supported"
    else:
        s = "mixed"
    out.append(Verdict(
        clause="timbre clearly changed between S1 and S2",
        status=s, headline_metric="log-mel L1 distance (dB)", headline_value=float(mel_db),
        note=f"MFCC L1 = {rc['mfcc_distance_s1_vs_s2']:.2f}; "
             f"spectral centroid r = {rc['spectral_centroid_summary_s1_vs_s2']['correlation']:.3f}; "
             f"S1 centroid mean = {rc['spectral_centroid_summary_s1_vs_s2']['s1_mean_hz']:.0f} Hz; "
             f"S2 centroid mean = {rc['spectral_centroid_summary_s1_vs_s2']['s2_mean_hz']:.0f} Hz",
    ))

    # Clause 4: pitch partially preserved (chroma cos sim — high = perfectly preserved, low = lost)
    cc = rc["chroma_cosine_similarity_s1_vs_s2"]
    if cc >= 0.85:
        s = "supported (pitch largely preserved)"
    elif cc >= 0.5:
        s = "supported (pitch partially preserved — hypothesis predicted drift)"
    else:
        s = "mixed — significant pitch drift"
    out.append(Verdict(
        clause="pitch partially preserved (with room for drift)",
        status=s, headline_metric="chroma cosine similarity", headline_value=float(cc),
        note="1.0 = identical pitch content; 0.0 = unrelated",
    ))

    # Bonus: forgetting probe — D2's faithfulness to country input vs D1's
    d_in_s1 = fp["mel_db_distance_input_vs_s1"]
    d_in_s2 = fp["mel_db_distance_input_vs_s2"]
    delta = d_in_s2 - d_in_s1
    if delta >= 3.0:
        s = "supported — D2 reconstructs country MUCH less faithfully than D1 (forgetting)"
    elif delta < 0.5:
        s = "not supported — D2 still reconstructs country about as faithfully as D1 (no forgetting)"
    else:
        s = "mixed"
    out.append(Verdict(
        clause="(forgetting probe) D2 reconstructs the held-out clip less faithfully than D1",
        status=s, headline_metric="Δ log-mel distance (S2 - S1) vs input, dB",
        headline_value=float(delta),
        note=f"input↔S1 = {d_in_s1:.2f} dB (D1 baseline); input↔S2 = {d_in_s2:.2f} dB (D2)",
    ))

    return out
