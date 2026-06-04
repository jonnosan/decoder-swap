"""Step 3 of the stems pivot: compare the four reference audios and report metrics.

Four things to compare:
  (R0) original                              — ground truth
  (R1) sum-of-uncompressed-stems             — separation-only floor (Demucs cost)
  (R2) sum-of-DAC-roundtripped-stems         — THE thing we're testing
  (R3) full-mix DAC roundtrip                — codec-only baseline (no separation)

Metrics vs R0: Mel L1, log-spectral distance, SI-SDR. All computed on a mono downmix
(L+R averaged) for simplicity; stereo audio is preserved in the listening-test WAVs.

Run:
  .venv/bin/python scripts/32_compare_stems.py --slug beltram_machine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results" / "stems_v1"))
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    return ap.parse_args()


def load_stereo(path: Path, expected_sr: int | None = None) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if expected_sr is not None and sr != expected_sr:
        raise SystemExit(f"sr mismatch reading {path}: {sr} != {expected_sr}")
    return y, sr


def align_lengths(arrs: list[np.ndarray]) -> list[np.ndarray]:
    """Trim all stereo arrays (shape [N, 2]) to the shortest length so sums are sample-aligned."""
    n = min(a.shape[0] for a in arrs)
    return [a[:n] for a in arrs]


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    return y.mean(axis=1).astype(np.float32, copy=False)


def mel_l1(ref: np.ndarray, est: np.ndarray, sr: int, n_mels: int, n_fft: int, hop: int) -> float:
    import librosa
    ref_m = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    est_m = librosa.feature.melspectrogram(y=est, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    ref_db = librosa.power_to_db(ref_m, ref=1.0, top_db=80.0)
    est_db = librosa.power_to_db(est_m, ref=1.0, top_db=80.0)
    return float(np.mean(np.abs(ref_db - est_db)))


def log_spectral_distance(ref: np.ndarray, est: np.ndarray, n_fft: int, hop: int) -> float:
    """Mean LSD in dB: per-frame RMS of 10*log10(|R|^2/|E|^2) across freq, then mean across time."""
    import librosa
    R = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) ** 2
    E = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop)) ** 2
    eps = 1e-10
    diff_db = 10.0 * np.log10((R + eps) / (E + eps))
    per_frame = np.sqrt(np.mean(diff_db ** 2, axis=0))
    return float(np.mean(per_frame))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Higher is better."""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    s_target = alpha * ref
    e_noise = est - s_target
    return float(10.0 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-12) + 1e-12))


def main() -> int:
    args = parse_args()
    song_dir = Path(args.root) / args.slug
    if not song_dir.exists():
        raise SystemExit(f"no song dir at {song_dir}")
    results_dir = Path(args.results_root) / args.slug
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"# stems v1 comparison · slug={args.slug}")
    print(f"input:   {song_dir}")
    print(f"results: {results_dir}")

    # Load everything as stereo float32, align lengths
    orig, sr = load_stereo(song_dir / "original.wav")
    stem_names = ["drums", "bass", "other", "vocals"]
    stems_raw = [load_stereo(song_dir / "stems" / f"{n}.wav", expected_sr=sr)[0] for n in stem_names]
    stems_dac = [load_stereo(song_dir / "stems_dac" / f"{n}.wav", expected_sr=sr)[0] for n in stem_names]
    full_dac, _ = load_stereo(song_dir / "full_dac" / "full.wav", expected_sr=sr)

    all_arrays = [orig] + stems_raw + stems_dac + [full_dac]
    all_arrays = align_lengths(all_arrays)
    orig = all_arrays[0]
    stems_raw = all_arrays[1:5]
    stems_dac = all_arrays[5:9]
    full_dac = all_arrays[9]
    n_samples = orig.shape[0]
    print(f"  sample rate:  {sr} Hz")
    print(f"  aligned len:  {n_samples} samples ({n_samples/sr/60:.2f} min)")

    # Build the four reference audios
    sum_stems_raw = np.sum(np.stack(stems_raw, axis=0), axis=0)
    sum_stems_dac = np.sum(np.stack(stems_dac, axis=0), axis=0)

    refs = {
        "R0_original": orig,
        "R1_sum_uncompressed_stems": sum_stems_raw,
        "R2_sum_dac_stems": sum_stems_dac,
        "R3_full_dac_baseline": full_dac,
    }

    # Save listening-test copies (preserve stereo)
    print()
    print("## listening-test files")
    for name, audio in refs.items():
        out = results_dir / f"{name}.wav"
        sf.write(out, audio, sr, subtype="FLOAT")
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        print(f"  {out.name}  RMS={rms:.4f}  peak={peak:.4f}")

    # Compute metrics (mono downmix)
    print()
    print("## metrics (mono downmix; reference = R0_original)")
    print(f"{'name':<32}  {'mel_L1 (dB)':>12}  {'LSD (dB)':>10}  {'SI-SDR (dB)':>12}")
    ref_mono = to_mono(orig)
    rows = []
    for name, audio in refs.items():
        if name == "R0_original":
            continue
        est_mono = to_mono(audio)
        m = mel_l1(ref_mono, est_mono, sr, args.n_mels, args.n_fft, args.hop)
        l = log_spectral_distance(ref_mono, est_mono, args.n_fft, args.hop)
        s = si_sdr(ref_mono, est_mono)
        rows.append((name, m, l, s))
        print(f"  {name:<30}  {m:>12.3f}  {l:>10.3f}  {s:>12.3f}")

    # Stem-level diagnostic: how much energy does each stem have in the raw separation?
    print()
    print("## per-stem energy (uncompressed Demucs output)")
    for name, s in zip(stem_names, stems_raw, strict=False):
        rms = float(np.sqrt(np.mean(s ** 2)))
        peak = float(np.max(np.abs(s)))
        frac = float(np.mean(s ** 2)) / float(np.mean(orig ** 2) + 1e-12)
        print(f"  {name:<8} RMS={rms:.4f}  peak={peak:.4f}  energy_frac_of_orig={frac:.3f}")

    print()
    print("## interpretation key")
    print("  R1 vs R0  →  fidelity cost of Demucs separation alone")
    print("  R2 vs R1  →  fidelity cost of DAC roundtrip on top of separation")
    print("  R2 vs R3  →  the headline experimental question (does decomposition help?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
