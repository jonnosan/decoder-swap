"""Mimi probe: does the round-trip baseline sound acceptable on MUSIC at 8 codebooks (1.1 kbps)?

Mimi (kyutai/mimi) was trained primarily for speech. Before committing M3-style training time on
Mimi we need to know that the *original* round-trip already gives usable music output — else the
experiment can't separate \"decoder change\" from \"codec inadequate for music.\"

Run: uv run python scripts/06_mimi_roundtrip_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import MimiModel  # noqa: E402

from decoder_swap.settings import resolve_device  # noqa: E402


def main() -> int:
    device = resolve_device("auto")
    print(f"# Mimi music round-trip probe — device={device}")

    print("loading kyutai/mimi …")
    t0 = time.time()
    model = MimiModel.from_pretrained("kyutai/mimi")
    model.eval().to(device)
    print(f"loaded in {time.time() - t0:.1f}s")

    sr = model.config.sampling_rate
    fps = model.config.frame_rate
    cb_size = model.config.codebook_size
    print(f"  sampling_rate={sr} Hz  frame_rate={fps} fps  codebook_size={cb_size}")
    print(f"  max codebooks={model.config.num_codebooks}  semantic={model.config.num_semantic_quantizers}")

    # Use 8 codebooks — Mimi paper's standard low-rate configuration.
    n_q = 8
    bits_per_frame = n_q * int(np.log2(cb_size))
    bitrate = bits_per_frame * fps
    print(f"  using {n_q} codebooks ⇒ {bits_per_frame} bits/frame  bitrate ≈ {bitrate:.0f} bps")
    print()

    # Use the same Blue Kentucky Girl clip we've been working with; first 30 s is enough.
    src = "/Users/jonno/src/decoder_swap/results/m1_sanity/Blue Kentucky Girl.mp3"
    y, _ = librosa.load(src, sr=sr, mono=True, duration=30.0)
    y = y.astype(np.float32, copy=False)
    print(f"input    : {len(y)} samples ({len(y)/sr:.2f} s @ {sr} Hz mono)")

    # (B=1, C=1, T)
    x = torch.from_numpy(y)[None, None, :].to(device)
    with torch.no_grad():
        enc = model.encode(x, num_quantizers=n_q, return_dict=True)
        codes = enc.audio_codes  # (B, n_q, T_frames)
        dec = model.decode(codes, return_dict=True)
        y_hat_t = dec.audio_values  # (B, 1, T_samples)

    print(f"codes    : shape={tuple(codes.shape)}  (B, n_codebooks, T_frames)")
    print(f"           expected T_frames ≈ {int(np.ceil(len(y) * fps / sr))}")
    print(f"audio_out: shape={tuple(y_hat_t.shape)}")

    y_hat = y_hat_t.squeeze().detach().cpu().float().numpy()
    y_hat = y_hat[: len(y)]

    out_dir = REPO_ROOT / "results" / "mimi_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_out = out_dir / "input_24k.wav"
    rt_out = out_dir / "roundtrip_mimi_8cb.wav"
    sf.write(in_out, y, sr, subtype="PCM_16")
    sf.write(rt_out, y_hat, sr, subtype="PCM_16")
    print()
    print(f"wrote: {in_out}")
    print(f"wrote: {rt_out}")

    # Mel cosine similarity (same metric as M1 sanity).
    ma = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    mb = librosa.feature.melspectrogram(y=y_hat, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    la = librosa.power_to_db(ma + 1e-10).flatten()
    lb = librosa.power_to_db(mb + 1e-10).flatten()
    mel_cos = float(np.dot(la, lb) / (np.linalg.norm(la) * np.linalg.norm(lb) + 1e-12))

    n = min(len(y), len(y_hat))
    wav_corr = float(np.corrcoef(y[:n], y_hat[:n])[0, 1])
    print()
    print(f"## round-trip similarity (input vs Mimi-roundtrip)")
    print(f"  mel-spec cosine sim : {mel_cos:+.4f}     (was 0.9971 for DAC on full song)")
    print(f"  waveform Pearson r  : {wav_corr:+.4f}     (was 0.892 for DAC; expect lower here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
