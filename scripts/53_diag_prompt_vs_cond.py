"""Diagnostic: does the trained model use MIDI conditioning, or only the prompt?

Method:
  - Prompt = first N frames of segment A (e.g. t=60s..)
  - Cond   = MIDI conditioning for segment B (e.g. t=30s..)
  - Generate
  - Compare resulting audio against ref_A (DAC of segment A) and ref_B (DAC of segment B).

If the model is cond-driven:  output matches ref_B
If the model is prompt-driven: output matches ref_A

Use with:
  .venv/bin/python scripts/53_diag_prompt_vs_cond.py --slug beltram_machine \
      --ckpt data/checkpoints/bass_translator/beltram_machine_5k/translator_rvq_best.pt \
      --prompt-start-s 60 --cond-start-s 30 --seconds 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import decode_from_codes, load_codec  # noqa: E402
from decoder_swap.midi_conditioning import FrameCondConfig, build_from_json  # noqa: E402
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import CondConfig, TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

DAC_FPS = 86.1328125


def reconstruct_translator_config(d: dict) -> TranslatorRVQConfig:
    d = dict(d)
    if isinstance(d.get("cond"), dict):
        d["cond"] = CondConfig(**d["cond"])
    return TranslatorRVQConfig(**d)


def slice_cond_batch(cond, start, end, device):
    return {
        "pitch_active": torch.from_numpy(cond["pitch_active"][start:end].astype(np.float32)).unsqueeze(0).to(device),
        "velocity_bin": torch.from_numpy(cond["velocity_bin"][start:end].astype(np.int64)).unsqueeze(0).to(device),
        "bend_bin":     torch.from_numpy(cond["bend_bin"][start:end].astype(np.int64)).unsqueeze(0).to(device),
        "onset_phase":  torch.from_numpy(cond["onset_phase"][start:end].astype(np.int64)).unsqueeze(0).to(device),
    }


@torch.no_grad()
def slide_generate(model, prompt, cond_full, total_frames, max_ctx, device):
    ctx = prompt.clone()
    K = ctx.shape[-1]
    while ctx.shape[1] < total_frames:
        t_pred = ctx.shape[1]
        ws = max(0, t_pred - max_ctx)
        wx = ctx[:, ws:t_pred, :]
        cw = slice_cond_batch(cond_full, ws, t_pred, device)
        logits = model(wx, cond=cw)
        next_tok = logits[:, -1, :, :].argmax(dim=-1).unsqueeze(1)   # (1, 1, K)
        ctx = torch.cat([ctx, next_tok], dim=1)
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="default song for both prompt and cond unless overridden")
    ap.add_argument("--prompt-slug", default=None,
                    help="override the song the PROMPT comes from (default: --slug)")
    ap.add_argument("--cond-slug", default=None,
                    help="override the song the COND comes from (default: --slug)")
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-start-s", type=float, required=True)
    ap.add_argument("--cond-start-s", type=float, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--prompt-frames", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    prompt_slug = args.prompt_slug or args.slug
    cond_slug = args.cond_slug or args.slug

    device = resolve_device("auto")
    print(f"# prompt-vs-cond diagnostic · slug={args.slug} · device={device}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = reconstruct_translator_config(ckpt["translator_config"])
    model = TranslatorRVQ(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    max_ctx = cfg.max_seq_len
    print(f"  ckpt steps={ckpt['steps']}  max_seq_len={max_ctx}  conditioned={cfg.cond is not None}")

    def load(slug):
        sd = Path(args.root) / slug
        codes = np.load(sd / "stems_dac_tokens" / f"{args.stem_name}.npy")
        T = int(codes.shape[-1])
        cond_full = build_from_json(
            sd / "semantic" / f"{args.stem_name}.json",
            fps=DAC_FPS, n_frames=T, cfg=FrameCondConfig(),
        )
        return codes, cond_full

    codes_p, _              = load(prompt_slug)
    _,        cond_full_c   = load(cond_slug)
    codes_a,  _             = load(prompt_slug)   # ref A = prompt source
    codes_b,  _             = load(cond_slug)     # ref B = cond source

    n_frames = int(round(args.seconds * DAC_FPS))
    fa = int(round(args.prompt_start_s * DAC_FPS))      # segment A: PROMPT
    fb = int(round(args.cond_start_s * DAC_FPS))        # segment B: COND
    cond_shift = int(ckpt.get("train_config", {}).get("cond_shift_frames", 0))
    print(f"  prompt from [{prompt_slug}] t={args.prompt_start_s:.1f}s (frame {fa})")
    print(f"  cond   from [{cond_slug}]   t={args.cond_start_s:.1f}s (frame {fb})  shift=+{cond_shift}")
    print(f"  duration {args.seconds:.1f}s ({n_frames} frames)")

    # 1) Generate with mixed prompt (A) and cond (B)
    prompt_np = codes_p[:, fa : fa + args.prompt_frames]
    prompt = torch.from_numpy(prompt_np.T.astype(np.int64)).unsqueeze(0).to(device)
    cs = fb + cond_shift
    cond_b = {
        "pitch_active": cond_full_c.pitch_active[cs : cs + n_frames],
        "velocity_bin": cond_full_c.velocity_bin[cs : cs + n_frames],
        "bend_bin":     cond_full_c.bend_bin[cs : cs + n_frames],
        "onset_phase":  cond_full_c.onset_phase[cs : cs + n_frames],
    }
    full = slide_generate(model, prompt, cond_b, n_frames, max_ctx, device)

    codec = load_codec(name="dac", model_type="44khz", device=device)
    sr = codec.convention.sample_rate
    with torch.no_grad():
        audio = decode_from_codes(codec, full[0].T.unsqueeze(0).long())
        ref_a_codes = torch.from_numpy(codes_a[:, fa : fa + n_frames].astype(np.int64)).unsqueeze(0).to(device)
        ref_b_codes = torch.from_numpy(codes_b[:, fb : fb + n_frames].astype(np.int64)).unsqueeze(0).to(device)
        ref_a = decode_from_codes(codec, ref_a_codes)
        ref_b = decode_from_codes(codec, ref_b_codes)
    gen_np = np.clip(audio[0, 0].cpu().numpy(), -1.0, 1.0)
    ref_a_np = np.clip(ref_a[0, 0].cpu().numpy(), -1.0, 1.0)
    ref_b_np = np.clip(ref_b[0, 0].cpu().numpy(), -1.0, 1.0)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "bass_translator" / args.slug / "diag_prompt_vs_cond"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"promptA-{prompt_slug}@{int(args.prompt_start_s)}_condB-{cond_slug}@{int(args.cond_start_s)}"
    sf.write(str(out_dir / f"gen_{tag}.wav"), gen_np, sr)
    sf.write(str(out_dir / f"refA_{prompt_slug}@{int(args.prompt_start_s)}s.wav"), ref_a_np, sr)
    sf.write(str(out_dir / f"refB_{cond_slug}@{int(args.cond_start_s)}s.wav"), ref_b_np, sr)

    import librosa
    def mel(y):
        S = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512))
        return librosa.power_to_db(librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=64, fmin=20, fmax=4000))
    Mg, Ma, Mb = mel(gen_np), mel(ref_a_np), mel(ref_b_np)
    print(f"\n  Mel L1(gen vs refA={prompt_slug}@{int(args.prompt_start_s)}s)  : {np.mean(np.abs(Mg-Ma)):5.2f} dB  (prompt source)")
    print(f"  Mel L1(gen vs refB={cond_slug}@{int(args.cond_start_s)}s)  : {np.mean(np.abs(Mg-Mb)):5.2f} dB  (cond source)")
    print(f"  Mel L1(refA vs refB)             : {np.mean(np.abs(Ma-Mb)):5.2f} dB  (baseline difference)")
    print()
    print(f"  Interpretation:")
    print(f"    if gen≈refA: model is prompt-driven (ignoring cond)")
    print(f"    if gen≈refB: model is cond-driven (overrides prompt)")
    print(f"\n  files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
