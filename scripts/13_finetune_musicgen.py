"""LoRA fine-tune MusicGen-small on the Vytis chunks.

Uses HuggingFace transformers + PEFT to fine-tune only LoRA adapters on the
MusicGen decoder, keeping the rest of the model frozen. Trains with next-token
loss on audio codes produced by the model's frozen audio encoder.

Run:
  uv run python scripts/13_finetune_musicgen.py --max-steps 5      # smoke test
  uv run python scripts/13_finetune_musicgen.py --max-steps 2000   # full
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/musicgen/vytis"
OUT_DIR = REPO_ROOT / "data/checkpoints/musicgen/vytis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="facebook/musicgen-small")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}, torch {torch.__version__}")

    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"loading {args.model_name}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model_name)
    print(f"loaded in {time.time()-t0:.1f}s  sr={model.config.audio_encoder.sampling_rate}")

    # Inspect decoder modules to figure out LoRA target names.
    target_names = set()
    for name, _ in model.decoder.named_modules():
        # MusicGen decoder uses q_proj, k_proj, v_proj, out_proj for attention.
        for t in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if name.endswith(t):
                target_names.add(t)
    target_modules = sorted(target_names)
    print(f"LoRA target modules: {target_modules}")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules, lora_dropout=0.05, bias="none",
    )

    # Apply LoRA to the decoder only (audio encoder and text encoder stay frozen).
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    # Freeze everything not LoRA.
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    model.to(device)

    # Load metadata + shuffle.
    meta_path = DATA_DIR / "metadata.jsonl"
    rows = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    print(f"dataset: {len(rows)} chunks")

    def sample_batch(bs: int):
        items = random.sample(rows, bs)
        audios = [np.load(r["path"]) for r in items]
        texts = [r["text"] for r in items]
        # Pad/truncate to fixed length (10s @ 32kHz).
        L = 32000 * 10
        audios = [a[:L] if len(a) >= L else np.pad(a, (0, L-len(a))) for a in audios]
        audio_t = torch.from_numpy(np.stack(audios)).float()  # (B, T)
        return audio_t, texts

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Audio encoder normalises the audio internally. processor handles it.
    losses = []
    t0 = time.time()
    accum = 0
    optim.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        audio_t, texts = sample_batch(args.batch_size)

        # Encode text + audio to model input format.
        inputs = processor(
            audio=audio_t.numpy(), sampling_rate=32000,
            text=texts, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # MusicGen forward needs:
        #   input_ids: text token ids (from processor)
        #   attention_mask: text mask
        #   labels: encoded audio tokens (the processor produces input_values for audio,
        #     which the model encodes internally with audio_encoder; labels are derived).
        # In HF MusicgenForConditionalGeneration.forward(), passing input_values causes
        # the model to encode the audio internally and use the result as labels.
        outputs = model(**inputs, return_dict=True)
        loss = outputs.loss
        if loss is None:
            print(f"step {step}: model returned no loss; dumping outputs keys: {list(outputs.keys())}")
            return 1
        (loss / args.grad_accum).backward()
        accum += 1

        if accum == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)
            accum = 0

        lv = float(loss.detach().cpu())
        losses.append(lv)

        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
            dt = time.time() - t0
            print(f"step {step}/{args.max_steps}  loss={avg:.4f}  "
                  f"elapsed={dt:6.1f}s  rate={step/max(dt,1e-9):.2f} steps/s",
                  flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            save_path = OUT_DIR / f"lora_step{step}"
            model.decoder.save_pretrained(str(save_path))
            print(f"  [save] LoRA -> {save_path}", flush=True)

    losses_path = OUT_DIR / "train_losses.json"
    losses_path.write_text(json.dumps(losses))
    print(f"\ndone. losses -> {losses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
