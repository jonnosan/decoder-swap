"""Extract per-note (DAC tokens, conditioning) windows from a corpus.

Per the user's redirect 2026-06-04: train the bass synth model one note at a
time, not over whole basslines. Each training sample is ONE note — eliminating
the within-song AR x→x shortcut by construction.

Output (under data/per_note/<dataset_name>/):
  tokens.npy       (N_notes, W, K)  int16    DAC tokens for each note window
  pitch_active.npy (N_notes, W, P)  uint8    multi-hot per-frame pitch
  velocity_bin.npy (N_notes, W)     int8
  bend_bin.npy     (N_notes, W)     int8
  onset_phase.npy  (N_notes, W)     int8
  meta.json                                    note metadata, dataset description
  silence_frame.npy (K,) int16                  the "silence" DAC token (used as BOS at inference)

W = window_frames + 1 (frame 0 is the silence bootstrap, frames 1..W-1 are the note).

Run:
  .venv/bin/python scripts/61_extract_per_note_data.py \\
    --dataset mayday_corpus \\
    --slugs mayday_d1t01_westbam_the_mayday_anthem mayday_d1t02_beltram_machine ... \\
    --window-frames 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import encode_to_codes, load_codec  # noqa: E402
from decoder_swap.midi_conditioning import (  # noqa: E402
    FrameCondConfig,
    _bend_center_bin,
    build_from_notes,
)
from decoder_swap.settings import resolve_device  # noqa: E402

DAC_FPS = 86.1328125

# Audio-feature names (in the order saved to features.npy). Kept in sync with
# compute_note_features().
FEATURE_NAMES = [
    "audio_rms",                # mean RMS of the note's audio (used for quality + feature)
    "spectral_centroid_mean",   # brightness
    "spectral_centroid_std",    # brightness modulation (sweep)
    "spectral_flatness_mean",   # pure-tone vs noisy
    "attack_time_ms",           # 10%→90% envelope time
    "sustain_ratio",            # sustain RMS / peak RMS (sustained vs pluck)
    "hf_energy_ratio",          # energy above bass band / total (distortion proxy)
]
N_FEATURES = len(FEATURE_NAMES)


def compute_note_features(
    audio: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256,
) -> np.ndarray:
    """Compute a small feature vector over one note's audio segment.

    audio: 1D float32 mono samples for the note window (typically 25,600 samples / 580 ms).
    Returns: shape (N_FEATURES,) float32.
    """
    import librosa
    out = np.zeros(N_FEATURES, dtype=np.float32)
    if len(audio) < n_fft:
        return out
    a = audio.astype(np.float32, copy=False)
    out[0] = float(np.sqrt(np.mean(a ** 2) + 1e-12))           # audio_rms
    S = np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop, center=True))
    if S.size == 0 or S.sum() < 1e-9:
        return out
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    out[1] = float(centroid.mean())
    out[2] = float(centroid.std())
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    out[3] = float(flatness.mean())
    # Attack time: 10% → 90% of peak amplitude envelope (using a short-window RMS).
    win = max(1, int(0.005 * sr))                              # 5 ms windows
    env = np.sqrt(np.convolve(a ** 2, np.ones(win) / win, mode="same") + 1e-12)
    if env.max() > 0:
        peak = env.max()
        peak_idx = int(np.argmax(env))
        # Find first crossing of 10% and 90% BEFORE the peak
        thr_lo = 0.10 * peak
        thr_hi = 0.90 * peak
        pre = env[: peak_idx + 1]
        i_lo = int(np.argmax(pre >= thr_lo)) if (pre >= thr_lo).any() else 0
        i_hi = int(np.argmax(pre >= thr_hi)) if (pre >= thr_hi).any() else peak_idx
        out[4] = float(max(0, i_hi - i_lo) / sr * 1000.0)       # attack_time_ms
        # Sustain ratio: mean RMS over [peak..peak+50ms] divided by peak.
        sustain_end = min(len(env), peak_idx + int(0.05 * sr))
        out[5] = float(env[peak_idx:sustain_end].mean() / (peak + 1e-12))
    # HF energy ratio: bins above ~500 Hz / total.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hf_mask = freqs >= 500.0
    total = float(S.sum())
    hf = float(S[hf_mask].sum())
    out[6] = hf / (total + 1e-9)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="name for the output dir under data/per_note/")
    ap.add_argument("--slugs", nargs="+", required=True,
                    help="song slugs (under data/song_test/) to include")
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "per_note"))
    ap.add_argument("--window-frames", type=int, default=50,
                    help="number of NOTE frames per sample (silence is +1 BOS)")
    ap.add_argument("--pitch-lo", type=int, default=24)
    ap.add_argument("--pitch-hi", type=int, default=50)
    ap.add_argument("--min-audio-rms", type=float, default=0.005,
                    help="drop notes whose first-100ms audio RMS is below this. "
                         "Filters silence + Demucs noise-floor artefacts.")
    ap.add_argument("--feature-attack-window-ms", type=float, default=100.0,
                    help="window length (ms) at the note onset used for the quality-filter RMS "
                         "(features themselves use the whole note window)")
    return ap.parse_args()


def get_silence_dac_frame(codec, device) -> np.ndarray:
    """Encode 0.5s of digital silence and return the first DAC token frame, shape (K,) int16."""
    sr = codec.convention.sample_rate
    silence = torch.zeros(1, 1, int(0.5 * sr), device=device)
    with torch.no_grad():
        codes = encode_to_codes(codec, silence)
    # Take the middle frame (avoid boundary effects)
    mid = codes.shape[-1] // 2
    return codes[0, :, mid].cpu().numpy().astype(np.int16)


def main() -> int:
    args = parse_args()
    device = resolve_device("auto")
    print(f"# per-note extract · dataset={args.dataset} · device={device}")

    cond_cfg = FrameCondConfig(pitch_lo=args.pitch_lo, pitch_hi=args.pitch_hi)
    W_note = int(args.window_frames)
    W_total = W_note + 1   # +1 for silence-BOS at frame 0
    bend_center = _bend_center_bin(cond_cfg.n_bend_bins)
    n_pitches = cond_cfg.n_pitches

    print(f"  window: {W_note} note frames + 1 silence-BOS = {W_total} total frames")

    # Load codec once (for silence frame extraction)
    print("loading DAC codec ...")
    codec = load_codec(name="dac", model_type="44khz", device=device)
    silence_frame = get_silence_dac_frame(codec, device)
    print(f"  silence DAC frame: {silence_frame.tolist()}")

    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tokens: list[np.ndarray] = []
    all_pitch_active: list[np.ndarray] = []
    all_velocity_bin: list[np.ndarray] = []
    all_bend_bin: list[np.ndarray] = []
    all_onset_phase: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    note_meta: list[dict] = []
    n_dropped_low_rms = 0

    import soundfile as sf

    t0 = time.time()
    for slug in args.slugs:
        sd = Path(args.root) / slug
        tokens_path = sd / "stems_dac_tokens" / f"{args.stem_name}.npy"
        midi_path = sd / "semantic" / f"{args.stem_name}.json"
        audio_path = sd / "stems" / f"{args.stem_name}.wav"
        if not tokens_path.exists() or not midi_path.exists():
            print(f"  WARN: missing data for {slug} — skipping")
            continue
        if not audio_path.exists():
            print(f"  WARN: missing audio at {audio_path} — skipping")
            continue
        codes = np.load(tokens_path)   # (K, T_full)
        T_full = codes.shape[-1]
        K = codes.shape[0]
        with open(midi_path) as f:
            notes = json.load(f)
        # Load the bass audio mono for quality filter + feature extraction.
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        n_quality_window = int(args.feature_attack_window_ms / 1000.0 * sr)
        n_full_window = int(round(W_note / DAC_FPS * sr))   # samples per 50-frame window
        print(f"  [{slug}]  tokens {codes.shape}  notes {len(notes)}  "
              f"audio {len(audio)/sr:.1f}s @ {sr}Hz")

        n_added_song = 0
        n_dropped_song = 0
        for note in notes:
            p = int(note["pitch"])
            if p < cond_cfg.pitch_lo or p > cond_cfg.pitch_hi:
                continue
            start_frame = int(np.floor(float(note["start_s"]) * DAC_FPS))
            if start_frame < 0 or start_frame >= T_full:
                continue
            end_frame = min(start_frame + W_note, T_full)
            n_actual = end_frame - start_frame
            if n_actual <= 0:
                continue

            # Slice the note's audio for quality filter + features.
            start_sample = int(round(float(note["start_s"]) * sr))
            note_audio = audio[start_sample : start_sample + n_full_window]
            if len(note_audio) < n_full_window:
                # zero-pad short notes near end-of-file so RMS comparison is fair
                note_audio = np.pad(note_audio, (0, n_full_window - len(note_audio)))
            quality_rms = float(np.sqrt(np.mean(
                note_audio[:n_quality_window] ** 2
            ) + 1e-12))
            if quality_rms < args.min_audio_rms:
                n_dropped_song += 1
                n_dropped_low_rms += 1
                continue

            features = compute_note_features(note_audio, sr)

            # Tokens for this note window (W_note frames; pad with silence if too short)
            note_tokens = np.tile(silence_frame[:, None], (1, W_note))   # (K, W_note)
            note_tokens[:, :n_actual] = codes[:, start_frame:end_frame]

            # Build conditioning for ONLY this note over W_note frames, treating
            # frame 0 (note onset) as the first frame.
            single_note_cond = build_from_notes(
                [{
                    "pitch": p,
                    "velocity": int(note.get("velocity", 96)),
                    "start_s": 0.0,
                    "end_s": (float(note["end_s"]) - float(note["start_s"])),
                    "pitch_bends": note.get("pitch_bends") or [],
                }],
                fps=DAC_FPS, n_frames=W_note, cfg=cond_cfg,
            )

            # x layout: x[0] = silence-BOS; x[1..W_note] = the note's audio tokens.
            tokens_w = np.concatenate([
                silence_frame[:, None],
                note_tokens,
            ], axis=1).T.astype(np.int16)   # (W_total, K)

            # cond layout (target-aligned per cond_shift=1 logic): cond[t] describes
            # what x[t+1] should contain. So cond[0] = note frame-0 cond (telling the
            # model: "predict the note's first frame next"), cond[1] = note frame-1
            # cond, ..., cond[W_note-1] = note frame-(W_note-1) cond. cond[W_note] is
            # appended as a duplicate of the last frame (it's never consumed since
            # we never predict beyond x[W_note]).
            pa = np.concatenate([
                single_note_cond.pitch_active.astype(np.uint8),
                single_note_cond.pitch_active[-1:].astype(np.uint8),
            ], axis=0)   # (W_total, n_pitches)

            vb = np.concatenate([
                single_note_cond.velocity_bin.astype(np.int8),
                single_note_cond.velocity_bin[-1:].astype(np.int8),
            ], axis=0)

            bb = np.concatenate([
                single_note_cond.bend_bin.astype(np.int8),
                single_note_cond.bend_bin[-1:].astype(np.int8),
            ], axis=0)

            op = np.concatenate([
                single_note_cond.onset_phase.astype(np.int8),
                single_note_cond.onset_phase[-1:].astype(np.int8),
            ], axis=0)

            all_tokens.append(tokens_w)
            all_pitch_active.append(pa)
            all_velocity_bin.append(vb)
            all_bend_bin.append(bb)
            all_onset_phase.append(op)
            all_features.append(features)
            note_meta.append({
                "song": slug,
                "pitch": p,
                "velocity": int(note.get("velocity", 96)),
                "duration_s": float(note["end_s"]) - float(note["start_s"]),
                "duration_frames": min(W_note, int(np.ceil((float(note["end_s"]) - float(note["start_s"])) * DAC_FPS))),
                "had_bend": bool(note.get("pitch_bends")),
            })
            note_meta[-1].update({
                "song_start_s": float(note["start_s"]),
                "quality_rms": quality_rms,
            })
            n_added_song += 1
        print(f"    added {n_added_song} notes  (dropped {n_dropped_song} below "
              f"min-audio-rms {args.min_audio_rms})")

    n = len(all_tokens)
    print(f"\n# total notes extracted: {n}  (dropped {n_dropped_low_rms} total "
          f"low-RMS notes)  ({time.time()-t0:.1f}s)")

    tokens_arr   = np.stack(all_tokens, axis=0)
    pa_arr       = np.stack(all_pitch_active, axis=0)
    vb_arr       = np.stack(all_velocity_bin, axis=0)
    bb_arr       = np.stack(all_bend_bin, axis=0)
    op_arr       = np.stack(all_onset_phase, axis=0)
    features_arr = np.stack(all_features, axis=0).astype(np.float32)

    np.save(out_dir / "tokens.npy", tokens_arr)
    np.save(out_dir / "pitch_active.npy", pa_arr)
    np.save(out_dir / "velocity_bin.npy", vb_arr)
    np.save(out_dir / "bend_bin.npy", bb_arr)
    np.save(out_dir / "onset_phase.npy", op_arr)
    np.save(out_dir / "features.npy", features_arr)
    np.save(out_dir / "silence_frame.npy", silence_frame)
    (out_dir / "meta.json").write_text(json.dumps({
        "dataset": args.dataset,
        "n_notes": n,
        "window_frames_note": W_note,
        "window_frames_total": W_total,
        "n_codebooks": int(tokens_arr.shape[-1]),
        "vocab_size": int(codec.convention.codebook_size),
        "frame_rate": DAC_FPS,
        "cond_cfg": cond_cfg.__dict__,
        "slugs": args.slugs,
        "stem": args.stem_name,
        "min_audio_rms": args.min_audio_rms,
        "n_dropped_low_rms": int(n_dropped_low_rms),
        "feature_names": FEATURE_NAMES,
        "notes": note_meta,
    }, indent=2))

    print(f"  shapes:")
    print(f"    tokens.npy        {tokens_arr.shape} dtype {tokens_arr.dtype}")
    print(f"    pitch_active.npy  {pa_arr.shape} dtype {pa_arr.dtype}")
    print(f"    velocity_bin.npy  {vb_arr.shape}")
    print(f"    bend_bin.npy      {bb_arr.shape}")
    print(f"    onset_phase.npy   {op_arr.shape}")
    print(f"    features.npy      {features_arr.shape} dtype {features_arr.dtype}")
    print(f"  feature stats (mean / median / max):")
    for i, name in enumerate(FEATURE_NAMES):
        col = features_arr[:, i]
        print(f"    {name:24s}  {col.mean():>10.4f}  {np.median(col):>10.4f}  {col.max():>10.4f}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
