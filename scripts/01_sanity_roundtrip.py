"""M1: encode→quantize→decode a clip with the ORIGINAL codec, save it, report similarity.

This is the "before we touch anything, does the codec work end-to-end?" check.
Pass criterion: mel-spectrogram cosine similarity ≥ 0.90 between input and round-trip.

Run:
  uv run python scripts/01_sanity_roundtrip.py                 # uses librosa's trumpet example
  uv run python scripts/01_sanity_roundtrip.py --input clip.wav
  uv run python scripts/01_sanity_roundtrip.py --input clip.wav --seconds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from decoder_swap.codec_io import load_codec  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def load_audio_mono(path: str | Path, target_sr: int, max_seconds: float | None) -> np.ndarray:
    """Load and return mono float32 at target_sr. Truncated to max_seconds if set."""
    duration = max_seconds if max_seconds is not None else None
    y, _ = librosa.load(str(path), sr=target_sr, mono=True, duration=duration)
    return y.astype(np.float32)


def mel_cosine_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Cosine similarity of log-mel spectrograms. Robust to phase / small time shifts."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    mb = librosa.feature.melspectrogram(y=b, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    la = librosa.power_to_db(ma + 1e-10).flatten()
    lb = librosa.power_to_db(mb + 1e-10).flatten()
    num = float(np.dot(la, lb))
    den = float(np.linalg.norm(la) * np.linalg.norm(lb)) + 1e-12
    return num / den


def waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant SDR. Higher is better. ~25 dB is excellent for a perceptual codec."""
    n = min(len(reference), len(estimate))
    s, s_hat = reference[:n], estimate[:n]
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    alpha = float(np.dot(s_hat, s) / (np.dot(s, s) + 1e-12))
    target = alpha * s
    noise = s_hat - target
    return 10.0 * float(np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="audio file path; defaults to librosa example 'trumpet'")
    ap.add_argument("--seconds", type=float, default=6.0, help="max seconds to keep from the input")
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "m1_sanity"),
        help="where to write input_resampled.wav and roundtrip.wav",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    device = resolve_device(settings.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# decoder-swap M1: sanity round-trip")
    print(f"device: {device}")

    codec = load_codec(
        name=settings.codec_name,
        model_type=settings.codec_model_type,
        model_tag=settings.codec_model_tag,
        model_path=settings.codec_model_path,
        device=device,
    )
    sr = codec.convention.sample_rate

    src_path = args.input or librosa.example("trumpet")
    print(f"input:  {src_path}")
    y = load_audio_mono(src_path, target_sr=sr, max_seconds=args.seconds)
    print(f"loaded: {len(y)} samples @ {sr} Hz  ({len(y)/sr:.2f} s, mono float32)")

    # (B=1, C=1, T) on device
    x = torch.from_numpy(y)[None, None, :].to(device)

    # DAC's preprocess pads to a multiple of hop_length and normalises if needed.
    with torch.no_grad():
        x_pre = codec.model.preprocess(x, sr)
        z, codes, _latents, _cm, _cb = codec.model.encode(x_pre)
        y_hat_t = codec.model.decode(z)

    print(f"codes shape: {tuple(codes.shape)}  (B, n_codebooks, T_frames)")
    print(f"  expected n_codebooks={codec.convention.n_codebooks}, "
          f"T_frames≈{int(np.ceil(x_pre.shape[-1] / codec.convention.hop_length))}")
    print(f"z shape    : {tuple(z.shape)}  (B, latent_dim, T_frames)")
    print(f"audio out  : {tuple(y_hat_t.shape)}")

    y_hat = y_hat_t.squeeze().detach().cpu().float().numpy()
    # Trim padded tail back to original length so metrics line up sample-for-sample.
    y_hat = y_hat[: len(y)]

    # Save both for listening.
    src_out = out_dir / "input_resampled.wav"
    recon_out = out_dir / "roundtrip.wav"
    sf.write(src_out, y, sr, subtype="PCM_16")
    sf.write(recon_out, y_hat, sr, subtype="PCM_16")
    print(f"wrote: {src_out}")
    print(f"wrote: {recon_out}")

    # Metrics.
    wav_corr = waveform_correlation(y, y_hat)
    mel_cos = mel_cosine_similarity(y, y_hat, sr=sr)
    sisdr = si_sdr_db(y, y_hat)

    print()
    print("## similarity (input vs round-trip)")
    print(f"  waveform Pearson r  : {wav_corr:+.4f}     (modest is OK — perceptual codec, not bit-exact)")
    print(f"  mel-spec cosine sim : {mel_cos:+.4f}     (target ≥ 0.90)")
    print(f"  SI-SDR              : {sisdr:+.2f} dB    (≥ 15 dB is solid for a perceptual codec at 8 kbps)")
    print()
    if mel_cos >= 0.90:
        print("  PASS: round-trip is recognisable; encode→quantize→decode works end-to-end.")
        return 0
    print("  FAIL: round-trip is too far from input — investigate before proceeding.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
