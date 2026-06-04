"""Try top-k / top-p sampling variants and longer prompt to fix the failure
modes we heard:
  - greedy → static after 2s
  - temp=1.0 → coherent fragments but jumps between songs

Variants generated (all 20 s, same prompt position as 16_):
  gen_topk50_t08.wav     top-k=50, temp=0.80, 2s prompt
  gen_topp95_t085.wav    top-p=0.95, temp=0.85, 2s prompt
  gen_topp95_t085_5s.wav top-p=0.95, temp=0.85, 5s prompt   (more anchor context)
  gen_topp90_t075.wav    top-p=0.90, temp=0.75, 2s prompt   (more conservative)
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

from decoder_swap.codec_io import decode_from_codes, load_codec  # noqa: E402
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "mimi_gen_long"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = REPO_ROOT / "data/checkpoints/translator/techno/rvq_mimi/translator_rvq_best.pt"

PROMPT_TRACK_IDX = 1
PROMPT_START_FRAME = 36538
TOTAL_FRAMES = 250                   # 20 s

# (suffix, prompt_frames, top_k, top_p, temperature)
VARIANTS = [
    ("topk50_t08",     25, 50,  None, 0.80),
    ("topp95_t085",    25, None, 0.95, 0.85),
    ("topp95_t085_5s", 62, None, 0.95, 0.85),  # 5 s prompt instead of 2 s
    ("topp90_t075",    25, None, 0.90, 0.75),
]


def sample_token(logits: torch.Tensor, temperature: float,
                 top_k: int | None, top_p: float | None) -> torch.Tensor:
    """logits: (K, V) for one frame. Returns (K,) token ids."""
    K, V = logits.shape
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        topk_vals, _ = logits.topk(top_k, dim=-1)
        thresh = topk_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        # Mask tokens beyond the nucleus (keep the first token even if it exceeds p).
        mask = cum > top_p
        mask[:, 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")),
                                    sorted_logits)
        # Scatter back
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.stack([torch.multinomial(probs[k], 1).squeeze(-1) for k in range(K)])


def slide_generate(model, prompt, total_frames, *, temperature, top_k, top_p, max_ctx):
    ctx = prompt.clone()
    with torch.no_grad():
        for _ in range(total_frames - ctx.shape[1]):
            window = ctx[:, -max_ctx:, :]
            logits = model(window)[:, -1, :, :]   # (1, K, V)
            next_tokens = sample_token(logits[0], temperature, top_k, top_p)
            ctx = torch.cat([ctx, next_tokens.view(1, 1, -1)], dim=1)
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

    print("loading Mimi codec...")
    codec = load_codec(name="mimi", model_tag=None, device=device,
                       num_quantizers=cfg.n_codebooks)
    sr = codec.convention.sample_rate

    corpus = load_corpus("techno")
    audio_paths = corpus.audio_paths
    src_mp3 = audio_paths[PROMPT_TRACK_IDX]
    tokens_path = corpus.tokens_dir(codec="mimi") / (Path(src_mp3).stem + ".npy")
    track_codes = np.load(tokens_path)
    print(f"source: {Path(src_mp3).name}")

    torch.manual_seed(0)
    for suffix, n_prompt, top_k, top_p, temp in VARIANTS:
        prompt_np = track_codes[:, PROMPT_START_FRAME : PROMPT_START_FRAME + n_prompt]
        prompt_t = torch.from_numpy(prompt_np.T).long().unsqueeze(0).to(device)
        print(f"\n[{suffix}] prompt={n_prompt}f ({n_prompt/12.5:.1f}s)  "
              f"top_k={top_k}  top_p={top_p}  temp={temp}")
        t0 = time.time()
        full = slide_generate(
            model, prompt_t, TOTAL_FRAMES,
            temperature=temp, top_k=top_k, top_p=top_p, max_ctx=max_ctx,
        )
        codes_for_decode = full[0].T.unsqueeze(0).long()
        with torch.no_grad():
            audio = decode_from_codes(codec, codes_for_decode)
        audio_np = np.clip(audio[0, 0].cpu().numpy(), -1.0, 1.0)
        out_path = OUT_DIR / f"gen_{suffix}.wav"
        sf.write(str(out_path), audio_np, sr)
        print(f"  -> {out_path.name}  ({len(audio_np)/sr:.1f}s, gen {time.time()-t0:.1f}s)")

    print(f"\ndone. files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
