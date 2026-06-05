"""Synthesize a bassline from MIDI using the trained per-note model.

For each note in the target MIDI:
  1. Build conditioning for the 50-frame note window (cond[0] is silent-BOS)
  2. AR-generate 50 frames given the silence-BOS + conditioning
  3. Decode to audio
  4. Place at the note's start_s in the output buffer (overlap-add for tails)

Run:
  .venv/bin/python scripts/63_generate_per_note.py \\
    --ckpt data/checkpoints/per_note/mayday_corpus/per_note_best.pt \\
    --target-midi data/song_test/mayday_d1t02_beltram_machine/semantic/bass.json \\
    --duration-s 30 \\
    --out results/per_note/mayday_corpus/gen_beltram_midi.wav
"""
from __future__ import annotations

import argparse
import json
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
    _bend_center_bin,
    build_from_notes,
)
from decoder_swap.settings import resolve_device  # noqa: E402
from decoder_swap.translator_rvq import CondConfig, TranslatorRVQ, TranslatorRVQConfig  # noqa: E402

DAC_FPS = 86.1328125


def reconstruct_translator_config(d: dict) -> TranslatorRVQConfig:
    d = dict(d)
    if isinstance(d.get("cond"), dict):
        d["cond"] = CondConfig(**d["cond"])
    return TranslatorRVQConfig(**d)


@torch.no_grad()
def generate_note_tokens(
    model: TranslatorRVQ,
    silence_frame: torch.Tensor,        # (K,) long, on device
    cond_frames: dict[str, torch.Tensor],   # each (1, W_total, *) on device. timbre_id is (1,)
    W_note: int,
    device: str,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Run AR generation for one note: start with silence-BOS, generate W_note frames.

    Returns (W_note, K) of generated DAC tokens.
    """
    K = silence_frame.shape[0]
    W_total = W_note + 1
    # ctx starts with just the silence frame
    ctx = silence_frame.view(1, 1, K).to(torch.long)   # (1, 1, K)
    # Per-frame keys we slice by t_pred each step; timbre_id is per-note so unchanged.
    per_frame_keys = {"pitch_active", "velocity_bin", "bend_bin", "onset_phase"}
    for _ in range(W_note):
        t_pred = ctx.shape[1]
        cw = {}
        for k, v in cond_frames.items():
            cw[k] = v[:, :t_pred].clone() if k in per_frame_keys else v
        logits = model(ctx, cond=cw)
        next_logits = logits[:, -1, :, :]   # (1, K, V)
        if temperature == 0.0:
            next_tokens = next_logits.argmax(dim=-1)   # (1, K)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tokens = torch.stack([
                torch.multinomial(probs[0, k], 1).squeeze(-1) for k in range(K)
            ]).unsqueeze(0)
        ctx = torch.cat([ctx, next_tokens.unsqueeze(1)], dim=1)
    return ctx[0, 1:]   # drop the silence-BOS frame, return (W_note, K)


def build_note_cond(
    note: dict,
    cond_cfg: FrameCondConfig,
    W_note: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-frame conditioning arrays for one note over W_note frames."""
    p = int(note["pitch"])
    note_dict = {
        "pitch": p,
        "velocity": int(note.get("velocity", 96)),
        "start_s": 0.0,
        "end_s": (float(note["end_s"]) - float(note["start_s"])),
        "pitch_bends": note.get("pitch_bends") or [],
    }
    snc = build_from_notes([note_dict], fps=DAC_FPS, n_frames=W_note, cfg=cond_cfg)
    # Target-aligned cond (same layout as training): cond[t] describes target x[t+1].
    # cond[0] = note frame-0 cond (tells the model what to play first).
    pa = np.concatenate([
        snc.pitch_active.astype(np.float32),
        snc.pitch_active[-1:].astype(np.float32),
    ], axis=0)
    vb = np.concatenate([
        snc.velocity_bin.astype(np.int64),
        snc.velocity_bin[-1:].astype(np.int64),
    ], axis=0)
    bb = np.concatenate([
        snc.bend_bin.astype(np.int64),
        snc.bend_bin[-1:].astype(np.int64),
    ], axis=0)
    op = np.concatenate([
        snc.onset_phase.astype(np.int64),
        snc.onset_phase[-1:].astype(np.int64),
    ], axis=0)
    return pa, vb, bb, op


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target-midi", required=True,
                    help="path to bass.json describing the target bassline")
    ap.add_argument("--duration-s", type=float, default=20.0,
                    help="how many seconds of MIDI to render")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-notes", type=int, default=1000)
    ap.add_argument("--timbre-id", type=int, default=None,
                    help="if the model was trained with timbre cond, this picks which timbre "
                         "to render. Default: 0 (first cluster) if model has timbres, else unused.")
    args = ap.parse_args()

    device = resolve_device("auto")
    print(f"# per-note synth · device={device}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  midi: {args.target_midi}")
    print(f"  out:  {args.out}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = reconstruct_translator_config(ckpt["translator_config"])
    model = TranslatorRVQ(cfg); model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  model: steps={ckpt.get('steps', '?')}  d_model={cfg.d_model}  layers={cfg.n_layers}")

    meta = ckpt.get("meta", {})
    cc = meta.get("cond_cfg", {})
    cond_cfg = FrameCondConfig(
        pitch_lo=int(cc.get("pitch_lo", 24)),
        pitch_hi=int(cc.get("pitch_hi", 50)),
        n_velocity_bins=int(cc.get("n_velocity_bins", 8)),
        n_bend_bins=int(cc.get("n_bend_bins", 16)),
        bend_range_cents=float(cc.get("bend_range_cents", 200.0)),
        onset_window_frames=int(cc.get("onset_window_frames", 3)),
        release_window_frames=int(cc.get("release_window_frames", 2)),
    )
    W_note = int(meta.get("window_frames_note", 50))
    print(f"  note window: {W_note} frames ({W_note/DAC_FPS*1000:.0f} ms)")

    # Resolve --timbre-id given the model's config.
    n_timbres_in_model = (cfg.cond.n_timbres if (cfg.cond is not None) else 0)
    timbre_id_used: int | None = None
    if n_timbres_in_model > 0:
        timbre_id_used = 0 if args.timbre_id is None else int(args.timbre_id)
        if timbre_id_used < 0 or timbre_id_used >= n_timbres_in_model:
            print(f"ERROR: --timbre-id {timbre_id_used} out of range "
                  f"[0, {n_timbres_in_model}).")
            return 1
        print(f"  timbre: id={timbre_id_used} (of {n_timbres_in_model} clusters)")
    elif args.timbre_id is not None:
        print(f"  WARN: --timbre-id {args.timbre_id} ignored — model has no timbre cond.")

    # Load silence_frame from the checkpoint's dataset (saved at extraction time).
    dataset = meta.get("dataset", "mayday_corpus")
    silence_path = REPO_ROOT / "data" / "per_note" / dataset / "silence_frame.npy"
    if not silence_path.exists():
        print(f"ERROR: silence_frame.npy not found at {silence_path}")
        return 1
    silence_np = np.load(silence_path).astype(np.int64)
    silence_t = torch.from_numpy(silence_np).to(device)

    print("loading DAC ...")
    codec = load_codec(name="dac", model_type="44khz", device=device)
    sr = codec.convention.sample_rate
    hop = codec.convention.hop_length

    with open(args.target_midi) as f:
        all_notes = json.load(f)
    notes = [n for n in all_notes
             if float(n["start_s"]) >= args.start_s
             and float(n["start_s"]) < args.start_s + args.duration_s
             and cond_cfg.pitch_lo <= int(n["pitch"]) <= cond_cfg.pitch_hi]
    notes = notes[: args.max_notes]
    print(f"  notes in window: {len(notes)} (of {len(all_notes)} total)")

    n_samples = int(round(args.duration_s * sr))
    output_buffer = np.zeros(n_samples, dtype=np.float32)
    weight_buffer = np.zeros(n_samples, dtype=np.float32)   # for overlap-add normalization

    t0 = time.time()
    for i, note in enumerate(notes):
        if i % 50 == 0:
            print(f"  [{i:>4}/{len(notes)}] elapsed={time.time()-t0:.1f}s", flush=True)
        # Build per-note cond on CPU (small), then move to device
        pa, vb, bb, op = build_note_cond(note, cond_cfg, W_note)
        cond_frames = {
            "pitch_active": torch.from_numpy(pa).unsqueeze(0).to(device),   # (1, W_total, P)
            "velocity_bin": torch.from_numpy(vb).unsqueeze(0).to(device),
            "bend_bin":     torch.from_numpy(bb).unsqueeze(0).to(device),
            "onset_phase":  torch.from_numpy(op).unsqueeze(0).to(device),
        }
        if timbre_id_used is not None:
            cond_frames["timbre_id"] = torch.tensor([timbre_id_used], dtype=torch.long, device=device)
        tokens = generate_note_tokens(
            model, silence_t, cond_frames, W_note, device, temperature=args.temperature,
        )   # (W_note, K)

        # Decode this note's tokens
        with torch.no_grad():
            audio = decode_from_codes(codec, tokens.T.unsqueeze(0).long())
        audio_np = audio[0, 0].cpu().numpy().astype(np.float32)
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Place at note.start_s in the output buffer (offset relative to args.start_s)
        rel_start_s = float(note["start_s"]) - args.start_s
        start_sample = int(round(rel_start_s * sr))
        if start_sample < 0:
            start_sample = 0
        end_sample = min(n_samples, start_sample + len(audio_np))
        if end_sample <= start_sample:
            continue
        n_to_write = end_sample - start_sample
        # Triangular envelope to smooth boundaries (avoid clicks at note ends)
        env = np.ones(n_to_write, dtype=np.float32)
        fade_n = min(256, n_to_write // 4)
        if fade_n > 0:
            env[:fade_n] = np.linspace(0, 1, fade_n)
            env[-fade_n:] = np.linspace(1, 0, fade_n)
        output_buffer[start_sample:end_sample] += audio_np[:n_to_write] * env
        weight_buffer[start_sample:end_sample] += env

    # Normalize overlapping regions (avoid double-volume where notes overlap)
    weight_buffer = np.maximum(weight_buffer, 1e-3)
    output_buffer = output_buffer / weight_buffer
    # But where there was no signal, weight was 1e-3; output stays 0 (already 0/1e-3=0)

    output_buffer = np.clip(output_buffer, -1.0, 1.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), output_buffer, sr)
    print(f"\n  rendered {len(notes)} notes in {time.time()-t0:.1f}s -> {out_path}")
    print(f"  output: {len(output_buffer)/sr:.1f}s @ {sr} Hz, RMS {np.sqrt(np.mean(output_buffer**2)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
