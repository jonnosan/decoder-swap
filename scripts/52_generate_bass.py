"""Generate bass audio from MIDI through the trained bass DAC-codec-LM (issue #10 1B.1).

Pipeline:
  1. Load trained checkpoint
  2. Load MIDI JSON for the same song (or a different MIDI for mutation)
  3. Build frame-aligned conditioning for the target span at 86.13 fps
  4. Bootstrap with a few frames of training DAC tokens
  5. AR-generate frame-by-frame, conditioning on the future MIDI at each step
  6. Decode DAC tokens → audio, save WAV

For Phase 1B.1 acceptance the "fed training MIDI" case should reproduce audio
that sounds like the training bass (modulo codec loss). Also generates an
unconditional comparison (zeroed conditioning) to verify the conditioning
actually influences output.

Run:
  .venv/bin/python scripts/52_generate_bass.py --slug beltram_machine
  .venv/bin/python scripts/52_generate_bass.py --slug beltram_machine --seconds 30 --start-s 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decoder_swap.codec_io import decode_from_codes, load_codec  # noqa: E402
from decoder_swap.midi_conditioning import (  # noqa: E402
    FrameCondConfig,
    build_from_json,
)
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import CondConfig, TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

DAC_FPS = 86.1328125


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    ap.add_argument("--stem-name", default="bass")
    ap.add_argument("--ckpt", default=None,
                    help="default: data/checkpoints/bass_translator/<slug>/translator_rvq_best.pt")
    ap.add_argument("--midi-json", default=None,
                    help="default: data/song_test/<slug>/semantic/<stem>.json")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/bass_translator/<slug>/gen/")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="where in the MIDI to start (seconds)")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="duration to generate")
    ap.add_argument("--prompt-frames", type=int, default=8,
                    help="number of training DAC frames to bootstrap with")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (0 = greedy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also-unconditional", action="store_true", default=True,
                    help="also generate with zeroed conditioning as a sanity check")
    return ap.parse_args()


def reconstruct_translator_config(ckpt_cfg: dict) -> TranslatorRVQConfig:
    """The saved cfg may have `cond` as a nested dict (from asdict). Restore CondConfig."""
    ckpt_cfg = dict(ckpt_cfg)
    cond = ckpt_cfg.get("cond")
    if isinstance(cond, dict):
        ckpt_cfg["cond"] = CondConfig(**cond)
    return TranslatorRVQConfig(**ckpt_cfg)


def slice_cond_batch(
    pitch_active: np.ndarray, velocity_bin: np.ndarray,
    bend_bin: np.ndarray, onset_phase: np.ndarray,
    start: int, end: int, device: str,
) -> dict[str, torch.Tensor]:
    pa = torch.from_numpy(pitch_active[start:end].astype(np.float32)).unsqueeze(0).to(device)
    vb = torch.from_numpy(velocity_bin[start:end].astype(np.int64)).unsqueeze(0).to(device)
    bb = torch.from_numpy(bend_bin[start:end].astype(np.int64)).unsqueeze(0).to(device)
    op = torch.from_numpy(onset_phase[start:end].astype(np.int64)).unsqueeze(0).to(device)
    return {
        "pitch_active": pa,
        "velocity_bin": vb,
        "bend_bin": bb,
        "onset_phase": op,
    }


@torch.no_grad()
def slide_generate_conditioned(
    model: TranslatorRVQ,
    prompt: torch.Tensor,           # (1, T_prompt, K)
    cond_full: dict[str, np.ndarray],   # full target-span conditioning, np
    total_frames: int,
    temperature: float,
    max_ctx: int,
    device: str,
) -> torch.Tensor:
    """AR-generate frames [T_prompt, total_frames). cond_full must cover [0, total_frames)."""
    ctx = prompt.clone()
    K = ctx.shape[-1]
    n_to_gen = total_frames - ctx.shape[1]
    for step in range(n_to_gen):
        t_pred = ctx.shape[1]   # we want logits for frame t_pred-1 to sample frame t_pred
        window_start = max(0, t_pred - max_ctx)
        window_x = ctx[:, window_start:t_pred, :]
        cond_window = slice_cond_batch(
            cond_full["pitch_active"], cond_full["velocity_bin"],
            cond_full["bend_bin"], cond_full["onset_phase"],
            window_start, t_pred, device,
        )
        logits = model(window_x, cond=cond_window)        # (1, W, K, V)
        next_logits = logits[:, -1, :, :]                  # (1, K, V)
        if temperature == 0.0:
            next_tokens = next_logits.argmax(dim=-1)        # (1, K)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tokens = torch.stack([
                torch.multinomial(probs[0, k], 1).squeeze(-1) for k in range(K)
            ]).unsqueeze(0)
        ctx = torch.cat([ctx, next_tokens.unsqueeze(1)], dim=1)
    return ctx


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device("auto")
    print(f"# bass generation · slug={args.slug} · device={device}")

    song_dir = Path(args.root) / args.slug
    ckpt_path = (Path(args.ckpt) if args.ckpt
                 else REPO_ROOT / "data" / "checkpoints" / "bass_translator" / args.slug
                      / "translator_rvq_best.pt")
    midi_json = (Path(args.midi_json) if args.midi_json
                 else song_dir / "semantic" / f"{args.stem_name}.json")
    out_dir = (Path(args.out_dir) if args.out_dir
               else REPO_ROOT / "results" / "bass_translator" / args.slug / "gen")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = song_dir / "stems_dac_tokens" / f"{args.stem_name}.npy"

    print(f"  ckpt:  {ckpt_path}")
    print(f"  midi:  {midi_json}")
    print(f"  out:   {out_dir}")

    # Load model
    print("\nloading trained LM ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = reconstruct_translator_config(ckpt["translator_config"])
    model = TranslatorRVQ(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    max_ctx = cfg.max_seq_len
    print(f"  steps={ckpt['steps']}  max_seq_len={max_ctx}  cb={cfg.n_codebooks}  "
          f"d_model={cfg.d_model}  conditioned={cfg.cond is not None}")
    if cfg.cond is not None:
        cc = cfg.cond
        print(f"  cond: pitches={cc.n_pitches} vel={cc.n_velocity_bins} "
              f"bend={cc.n_bend_bins} onset={cc.n_onset_phases}")

    # Load codec
    print("\nloading DAC codec ...")
    codec = load_codec(name="dac", model_type="44khz", device=device)
    sr = codec.convention.sample_rate
    print(f"  DAC @ {sr} Hz, {codec.convention.frame_rate:.2f} fps, hop {codec.convention.hop_length}")

    # Load training DAC tokens (for prompt) — find total available frames
    codes = np.load(tokens_path)
    T_total = int(codes.shape[-1])
    start_frame = int(round(args.start_s * DAC_FPS))
    n_frames = int(round(args.seconds * DAC_FPS))
    if start_frame + n_frames > T_total:
        n_frames = T_total - start_frame
        print(f"  WARN: clipped duration to {n_frames/DAC_FPS:.1f}s (track only "
              f"{T_total/DAC_FPS:.1f}s)")
    print(f"\ntarget: frames [{start_frame}, {start_frame+n_frames}) "
          f"= {n_frames/DAC_FPS:.1f}s starting at {args.start_s:.1f}s")

    # Read cond_shift_frames from saved train_config — same alignment as training.
    cond_shift = int(ckpt.get("train_config", {}).get("cond_shift_frames", 0))
    if cond_shift:
        print(f"  cond shift: +{cond_shift} frame(s) (target-aligned, from train_config)")

    # Build full conditioning for the target span — note: build over the full track
    # then slice, so timing is consistent with how training built it.
    cond_cfg_for_build = FrameCondConfig()  # defaults match script 51's defaults
    cond_full_track = build_from_json(midi_json, fps=DAC_FPS, n_frames=T_total, cfg=cond_cfg_for_build)
    # Apply cond shift: at input position t (frame start_frame+t), feed cond[start_frame+t+shift].
    cs = start_frame + cond_shift
    ce = cs + n_frames
    if ce > T_total:
        # Trim and right-pad with silence cond (rare for full-song spans).
        valid = T_total - cs
        cond_slice = {
            "pitch_active": np.zeros((n_frames, cond_full_track.pitch_active.shape[1]), dtype=np.uint8),
            "velocity_bin": np.zeros((n_frames,), dtype=np.int8),
            "bend_bin":     np.full((n_frames,), cond_full_track.bend_bin[0], dtype=np.int8),
            "onset_phase":  np.zeros((n_frames,), dtype=np.int8),
        }
        # _bend_center_bin: borrow from one of the existing silent positions if any, else compute
        from decoder_swap.midi_conditioning import _bend_center_bin
        cond_slice["bend_bin"][:] = _bend_center_bin(cond_full_track.cfg.n_bend_bins)
        cond_slice["pitch_active"][:valid] = cond_full_track.pitch_active[cs:cs+valid]
        cond_slice["velocity_bin"][:valid] = cond_full_track.velocity_bin[cs:cs+valid]
        cond_slice["bend_bin"][:valid]     = cond_full_track.bend_bin[cs:cs+valid]
        cond_slice["onset_phase"][:valid]  = cond_full_track.onset_phase[cs:cs+valid]
    else:
        cond_slice = {
            "pitch_active": cond_full_track.pitch_active[cs:ce],
            "velocity_bin": cond_full_track.velocity_bin[cs:ce],
            "bend_bin":     cond_full_track.bend_bin[cs:ce],
            "onset_phase":  cond_full_track.onset_phase[cs:ce],
        }
    n_active = int((cond_slice["pitch_active"].sum(axis=1) > 0).sum())
    print(f"  conditioning: {n_active}/{n_frames} active frames ({100*n_active/max(1,n_frames):.0f}%)")

    # Prompt: first args.prompt_frames frames of training DAC tokens for the same span
    prompt_np = codes[:, start_frame : start_frame + args.prompt_frames]      # (K, T_prompt)
    prompt = torch.from_numpy(prompt_np.T.astype(np.int64)).unsqueeze(0).to(device)  # (1, T_prompt, K)

    # 1) GENERATE with training MIDI conditioning
    print(f"\n[A] generating WITH training MIDI conditioning (temp={args.temperature}) ...")
    t0 = time.time()
    full = slide_generate_conditioned(
        model, prompt, cond_slice, n_frames, args.temperature, max_ctx, device,
    )
    dt = time.time() - t0
    print(f"    generation: {dt:.1f}s  ({n_frames/dt:.1f} frames/s)")

    # Decode to audio
    codes_for_decode = full[0].T.unsqueeze(0).long()       # (1, K, T)
    with torch.no_grad():
        audio = decode_from_codes(codec, codes_for_decode)
    audio_np = np.clip(audio[0, 0].cpu().numpy(), -1.0, 1.0)
    suffix = "greedy" if args.temperature == 0 else f"t{int(args.temperature*100):03d}"
    out_path = out_dir / f"bass_cond_{suffix}_{args.start_s:.0f}s_{args.seconds:.0f}s.wav"
    sf.write(str(out_path), audio_np, sr)
    print(f"    -> {out_path.name}  ({len(audio_np)/sr:.1f}s)")

    # Also: reference — decode the ORIGINAL DAC tokens for the same span
    ref_codes = torch.from_numpy(codes[:, start_frame : start_frame + n_frames].astype(np.int64))
    ref_codes = ref_codes.unsqueeze(0).to(device)
    with torch.no_grad():
        ref_audio = decode_from_codes(codec, ref_codes)
    ref_np = np.clip(ref_audio[0, 0].cpu().numpy(), -1.0, 1.0)
    ref_path = out_dir / f"bass_ref_dac_{args.start_s:.0f}s_{args.seconds:.0f}s.wav"
    sf.write(str(ref_path), ref_np, sr)
    print(f"    -> {ref_path.name}  (DAC round-trip of training bass — anchor)")

    # 2) GENERATE with zeroed conditioning (control)
    if args.also_unconditional and cfg.cond is not None:
        print(f"\n[B] generating with ZEROED conditioning (control) ...")
        # "Silent everywhere": no pitches active, zero velocity (the silent bin),
        # bend bin = center, onset_phase = SILENT (0).
        from decoder_swap.midi_conditioning import _bend_center_bin
        bend_center = _bend_center_bin(cfg.cond.n_bend_bins)
        zero_cond = {
            "pitch_active": np.zeros_like(cond_slice["pitch_active"]),
            "velocity_bin": np.zeros_like(cond_slice["velocity_bin"]),
            "bend_bin":     np.full_like(cond_slice["bend_bin"], bend_center),
            "onset_phase":  np.zeros_like(cond_slice["onset_phase"]),
        }
        t0 = time.time()
        full_z = slide_generate_conditioned(
            model, prompt, zero_cond, n_frames, args.temperature, max_ctx, device,
        )
        dt = time.time() - t0
        print(f"    generation: {dt:.1f}s")
        codes_for_decode_z = full_z[0].T.unsqueeze(0).long()
        with torch.no_grad():
            audio_z = decode_from_codes(codec, codes_for_decode_z)
        audio_z_np = np.clip(audio_z[0, 0].cpu().numpy(), -1.0, 1.0)
        out_path_z = out_dir / f"bass_zerocond_{suffix}_{args.start_s:.0f}s_{args.seconds:.0f}s.wav"
        sf.write(str(out_path_z), audio_z_np, sr)
        print(f"    -> {out_path_z.name}")

    print(f"\ndone. files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
