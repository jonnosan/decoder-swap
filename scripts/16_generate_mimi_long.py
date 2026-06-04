"""Generate 20-second files for listening comparison.

Produces five files in results/mimi_gen_long/ all starting at the same position
in Vytis Vol. 3 (the same prompt position as the 8s comparison):

  prompt_orig_20s.wav  — 20s of original audio (no codec round-trip)
  prompt_mimi_20s.wav  — 20s of original audio AFTER Mimi encode→decode
  gen_greedy_20s.wav   — 2s prompt + 18s model-generated (greedy)
  gen_t08_20s.wav      —                                 (temperature 0.8)
  gen_t10_20s.wav      —                                 (temperature 1.0)

The model's context window is 116 frames (~9.3s). Long generation uses a
sliding window: each new frame is predicted given the last 116 frames.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import decode_from_codes, encode_to_codes, load_codec  # noqa: E402
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "mimi_gen_long"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = REPO_ROOT / "data/checkpoints/translator/techno/rvq_mimi/translator_rvq_best.pt"

# Same prompt position as the previous 8s run (track 1, frame 36538).
PROMPT_TRACK_IDX = 1            # Vytis Vol. 3
PROMPT_START_FRAME = 36538
PROMPT_FRAMES = 25              # 2 seconds of prompt
TOTAL_FRAMES = 250              # 20 seconds total
TEMPERATURES = [0.0, 0.8, 1.0]


def slide_generate(model, prompt, total_frames, temperature, max_ctx, n_codebooks):
    """Append one frame at a time. Each forward uses the last max_ctx frames only."""
    ctx = prompt.clone()                              # (1, T, K) on device
    n_to_gen = total_frames - ctx.shape[1]
    with torch.no_grad():
        for _ in range(n_to_gen):
            window = ctx[:, -max_ctx:, :]             # most recent max_ctx frames
            logits = model(window)                    # (1, T, K, V)
            next_logits = logits[:, -1, :, :]         # (1, K, V)
            if temperature == 0.0:
                next_tokens = next_logits.argmax(dim=-1)   # (1, K)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_tokens = torch.stack([
                    torch.multinomial(probs[0, k], 1).squeeze(-1)
                    for k in range(n_codebooks)
                ]).unsqueeze(0)
            ctx = torch.cat([ctx, next_tokens.unsqueeze(1)], dim=1)
    return ctx


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    print("loading trained LM...")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = TranslatorRVQConfig(**ckpt["translator_config"])
    model = TranslatorRVQ(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    max_ctx = cfg.max_seq_len
    print(f"  ckpt: steps={ckpt['steps']}  max_seq_len={max_ctx}  cb={cfg.n_codebooks}")

    print("loading Mimi codec...")
    codec = load_codec(name="mimi", model_tag=None, device=device,
                       num_quantizers=cfg.n_codebooks)
    sr = codec.convention.sample_rate
    hop = codec.convention.hop_length
    print(f"  Mimi @ {sr} Hz, {codec.convention.frame_rate} fps, hop {hop}")

    # Locate the source MP3 + offset.
    corpus = load_corpus("techno")
    audio_paths = corpus.audio_paths
    src_mp3 = audio_paths[PROMPT_TRACK_IDX]
    start_sample = PROMPT_START_FRAME * hop
    total_samples = TOTAL_FRAMES * hop
    print(f"source: {src_mp3}")
    print(f"  frames [{PROMPT_START_FRAME}, {PROMPT_START_FRAME + TOTAL_FRAMES})  "
          f"= samples [{start_sample}, {start_sample + total_samples})  "
          f"= {total_samples/sr:.1f}s")

    # 1) ORIGINAL 20-second audio (no codec round-trip).
    print("\n[1/3] original audio (no codec)...")
    y_orig, _ = librosa.load(str(src_mp3), sr=sr, mono=True,
                              offset=start_sample / sr,
                              duration=total_samples / sr)
    sf.write(str(OUT_DIR / "prompt_orig_20s.wav"),
             np.clip(y_orig, -1.0, 1.0), sr)
    print(f"  -> prompt_orig_20s.wav  ({len(y_orig)/sr:.1f}s)")

    # 2) MIMI ROUND-TRIP — encode the 20s of original, then decode.
    print("\n[2/3] Mimi round-trip ...")
    x = torch.from_numpy(y_orig).float().to(device).view(1, 1, -1)
    with torch.no_grad():
        rt_codes = encode_to_codes(codec, x)                       # (1, K, T_frames)
        rt_audio = decode_from_codes(codec, rt_codes.long())       # (1, 1, T)
    rt_np = np.clip(rt_audio[0, 0].cpu().numpy(), -1.0, 1.0)
    sf.write(str(OUT_DIR / "prompt_mimi_20s.wav"), rt_np, sr)
    print(f"  -> prompt_mimi_20s.wav  ({len(rt_np)/sr:.1f}s)")

    # 3) GENERATIONS — 2s prompt + 18s model-generated, three sampling temps.
    # Build the 25-frame prompt from cached Mimi tokens for this track.
    print("\n[3/3] generating from trained LM (sliding window) ...")
    tokens_path = corpus.tokens_dir(codec="mimi") / (Path(src_mp3).stem + ".npy")
    track_codes = np.load(tokens_path)  # (K, T_frames)
    prompt_np = track_codes[:, PROMPT_START_FRAME : PROMPT_START_FRAME + PROMPT_FRAMES]
    prompt_t = torch.from_numpy(prompt_np.T).long().unsqueeze(0).to(device)  # (1, 25, K)

    for temp in TEMPERATURES:
        suffix = "greedy" if temp == 0.0 else f"t{int(temp*10):02d}"
        print(f"  temperature={temp} ({suffix})")
        t0 = time.time()
        full = slide_generate(model, prompt_t, TOTAL_FRAMES, temp, max_ctx, cfg.n_codebooks)
        dt = time.time() - t0
        codes_for_decode = full[0].T.unsqueeze(0).long()             # (1, K, T)
        with torch.no_grad():
            audio = decode_from_codes(codec, codes_for_decode)
        audio_np = np.clip(audio[0, 0].cpu().numpy(), -1.0, 1.0)
        out_path = OUT_DIR / f"gen_{suffix}_20s.wav"
        sf.write(str(out_path), audio_np, sr)
        print(f"    -> {out_path.name}  ({len(audio_np)/sr:.1f}s, gen time {dt:.1f}s)")

    print(f"\ndone. files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
