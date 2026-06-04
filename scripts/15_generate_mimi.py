"""Generate audio from the trained Mimi RVQ model and decode it with Mimi.

Bypasses loss measurement entirely — produces actual audio so we can listen
and judge whether the model has learned anything useful, regardless of what
the loss numbers say.

Run:
  uv run python scripts/15_generate_mimi.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import load_codec, decode_from_codes  # noqa: E402
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.train_translator_rvq import FrameBatchSampler  # noqa: E402
from decoder_swap.translator_rvq import TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "mimi_gen"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = REPO_ROOT / "data/checkpoints/translator/techno/rvq_mimi/translator_rvq_best.pt"

PROMPT_FRAMES = 25      # 2 seconds of real audio as prompt
GEN_FRAMES = 100        # generate 8 seconds total
TEMPERATURES = [0.0, 0.8, 1.0]  # 0.0 = greedy


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    print("loading trained LM...")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = TranslatorRVQConfig(**ckpt["translator_config"])
    model = TranslatorRVQ(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  ckpt: steps={ckpt['steps']}  loss_last={ckpt['loss_last_window']:.4f}")
    print(f"  arch: vocab={cfg.vocab_size} cb={cfg.n_codebooks} d={cfg.d_model} L={cfg.n_layers}")

    print("loading Mimi codec for decoding...")
    codec = load_codec(name="mimi", model_tag=None, device=device,
                       num_quantizers=cfg.n_codebooks)
    sr = codec.convention.sample_rate
    print(f"  Mimi @ {sr} Hz, {codec.convention.frame_rate} fps")

    # Pull a real 2s prompt from the corpus to seed generation
    corpus = load_corpus("techno")
    tracks = [np.load(p) for p in sorted(corpus.tokens_dir(codec="mimi").glob("*.npy"))]
    sampler = FrameBatchSampler(tracks, window_frames=PROMPT_FRAMES, seed=42)
    prompt_batch = sampler.sample(1).to(device)  # (1, PROMPT_FRAMES, K)
    print(f"prompt shape: {prompt_batch.shape}")

    for temp in TEMPERATURES:
        print(f"\n--- generating @ temperature={temp} ---")
        ctx = prompt_batch.clone()
        t0 = time.time()
        with torch.no_grad():
            for step in range(GEN_FRAMES - PROMPT_FRAMES):
                logits = model(ctx)  # (1, T, K, V)
                next_logits = logits[:, -1, :, :]  # (1, K, V)
                if temp == 0.0:
                    next_tokens = next_logits.argmax(dim=-1)  # (1, K)
                else:
                    probs = torch.softmax(next_logits / temp, dim=-1)
                    next_tokens = torch.stack([
                        torch.multinomial(probs[0, k], 1).squeeze(-1)
                        for k in range(cfg.n_codebooks)
                    ]).unsqueeze(0)
                ctx = torch.cat([ctx, next_tokens.unsqueeze(1)], dim=1)
        dt = time.time() - t0
        print(f"  generated {ctx.shape[1]} frames in {dt:.1f}s")

        # Decode to audio. codec expects shape (B, n_q, T_frames)
        codes_for_decode = ctx[0].T.unsqueeze(0)  # (1, K, T_frames)
        with torch.no_grad():
            audio = decode_from_codes(codec, codes_for_decode.long())  # (1, 1, T)
        audio_np = audio[0, 0].cpu().numpy()
        # Normalise then save
        audio_np = np.clip(audio_np, -1.0, 1.0)
        suffix = "greedy" if temp == 0.0 else f"t{int(temp*10):02d}"
        out_path = OUT_DIR / f"gen_{suffix}.wav"
        sf.write(str(out_path), audio_np, sr)
        print(f"  -> {out_path}  ({len(audio_np)/sr:.1f}s)")

    # Also save the prompt as audio for comparison.
    print("\n--- saving prompt for reference ---")
    prompt_codes = prompt_batch[0].T.unsqueeze(0)
    with torch.no_grad():
        prompt_audio = decode_from_codes(codec, prompt_codes.long())
    prompt_np = np.clip(prompt_audio[0, 0].cpu().numpy(), -1.0, 1.0)
    sf.write(str(OUT_DIR / "prompt.wav"), prompt_np, sr)
    print(f"  -> {OUT_DIR/'prompt.wav'}  ({len(prompt_np)/sr:.1f}s)")

    print(f"\ndone. open {OUT_DIR} to listen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
