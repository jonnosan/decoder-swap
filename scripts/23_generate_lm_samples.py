"""Generate ~20 min of LM-produced audio for the GAN fixer's unpaired-fake set.

For each starting position in the corpus, run the trained Mimi RVQ LM with a
short prompt and generate 30 seconds of audio, varying the sampling temperature
to get diverse failure modes. Decode through Mimi to get the audio the fixer
will see at inference time. Strip the prompt portion so we only train on the
LM's actual outputs.

Output: data/fixer/vytis/lm_samples.npy  (float16, single contiguous waveform)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import decode_from_codes, load_codec  # noqa: E402
from decoder_swap.corpus import load_corpus  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

OUT_PATH = REPO_ROOT / "data/fixer/vytis/lm_samples.npy"
LM_CKPT = REPO_ROOT / "data/checkpoints/translator/techno/rvq_mimi/translator_rvq_best.pt"

PROMPT_FRAMES = 25            # 2 s of real audio
GEN_FRAMES = 375              # +30 s LM-generated
TOTAL_FRAMES = PROMPT_FRAMES + GEN_FRAMES

# 40 starts spread across the corpus × 3 sampling configs = 40 × 30 s = 20 min.
N_STARTS = 40
SAMPLING = [
    dict(temperature=0.8,  top_p=0.95),
    dict(temperature=0.85, top_p=0.95),
    dict(temperature=1.0,  top_p=0.95),
]


def sample_token(logits, temperature, top_p):
    K, V = logits.shape
    logits = logits / temperature
    if top_p is not None and 0.0 < top_p < 1.0:
        sl, si = logits.sort(dim=-1, descending=True)
        cum = sl.softmax(-1).cumsum(-1)
        mask = cum > top_p
        mask[:, 0] = False
        sl = torch.where(mask, torch.full_like(sl, float("-inf")), sl)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, si, sl)
    probs = torch.softmax(logits, dim=-1)
    return torch.stack([torch.multinomial(probs[k], 1).squeeze(-1) for k in range(K)])


def slide_generate(model, prompt, total, *, temperature, top_p, max_ctx, n_codebooks):
    ctx = prompt.clone()
    with torch.no_grad():
        for _ in range(total - ctx.shape[1]):
            window = ctx[:, -max_ctx:, :]
            logits = model(window)[:, -1, :, :]
            tokens = sample_token(logits[0], temperature, top_p)
            ctx = torch.cat([ctx, tokens.view(1, 1, -1)], dim=1)
    return ctx


def main() -> int:
    device = resolve_device("auto")
    print(f"device: {device}")

    # Load LM
    ckpt = torch.load(str(LM_CKPT), map_location="cpu", weights_only=False)
    cfg = TranslatorRVQConfig(**ckpt["translator_config"])
    model = TranslatorRVQ(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    max_ctx = cfg.max_seq_len
    n_cb = cfg.n_codebooks
    print(f"LM loaded: max_ctx={max_ctx}  cb={n_cb}")

    # Load Mimi for decoding
    codec = load_codec(name="mimi", device=device, num_quantizers=n_cb)
    sr = codec.convention.sample_rate
    hop = codec.convention.hop_length  # 1920 samples per frame at 12.5 fps / 24kHz
    print(f"Mimi @ {sr} Hz, hop {hop}")

    # Load all Mimi token tracks (these are our prompt source)
    corpus = load_corpus("techno")
    tokens_dir = corpus.tokens_dir(codec="mimi")
    tracks = [np.load(p) for p in sorted(tokens_dir.glob("*.npy"))]
    track_probs = np.array([t.shape[-1] for t in tracks], dtype=np.float64)
    track_probs = track_probs / track_probs.sum()
    rng = np.random.default_rng(0)

    audio_chunks = []
    t0 = time.time()
    plan = []
    for i in range(N_STARTS):
        ti = int(rng.choice(len(tracks), p=track_probs))
        T = tracks[ti].shape[-1]
        start = int(rng.integers(0, T - TOTAL_FRAMES))
        plan.append((ti, start))

    for i, (ti, start) in enumerate(plan):
        config = SAMPLING[i % len(SAMPLING)]
        # Build prompt from real Mimi tokens at this position.
        prompt_np = tracks[ti][:, start : start + PROMPT_FRAMES]
        prompt_t = torch.from_numpy(prompt_np.T).long().unsqueeze(0).to(device)

        full = slide_generate(
            model, prompt_t, TOTAL_FRAMES,
            temperature=config["temperature"], top_p=config["top_p"],
            max_ctx=max_ctx, n_codebooks=n_cb,
        )
        # Drop the prompt frames: we only want the LM-generated portion.
        gen_tokens = full[:, PROMPT_FRAMES:, :]  # (1, GEN_FRAMES, K)
        codes_for_decode = gen_tokens[0].T.unsqueeze(0).long()  # (1, K, T_frames)
        with torch.no_grad():
            audio = decode_from_codes(codec, codes_for_decode)
        audio_np = audio[0, 0].cpu().numpy().astype(np.float32)
        audio_chunks.append(audio_np)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_STARTS - i - 1)
        print(f"  [{i+1:2d}/{N_STARTS}]  track={ti} start={start:>6d}  "
              f"temp={config['temperature']} top_p={config['top_p']}  "
              f"len={len(audio_np)/sr:.1f}s  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
              flush=True)

    full_audio = np.concatenate(audio_chunks).astype(np.float16)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, full_audio)
    print(f"\nwrote {OUT_PATH}  ({len(full_audio)/sr/60:.1f} min @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
