"""Diagnose where the decoder fine-tune NaN comes from.

Runs a few steps with autograd.detect_anomaly to pinpoint the offending op, and prints
each loss component plus pre-clip gradient norm so we can see whether the issue is:
  - mel loss NaN (forward)
  - exploding gradient (backward, pre-clip)
  - MPS-specific weight_norm bug (try CPU comparison via DSWAP_DEVICE=cpu)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from decoder_swap.codec_io import load_codec  # noqa: E402
from decoder_swap.dataset import CorpusDataset  # noqa: E402
from decoder_swap.freeze import freeze_for_decoder_training, remove_weight_norm_recursive  # noqa: E402
from decoder_swap.losses import MultiScaleMelLoss, waveform_l1  # noqa: E402
from decoder_swap.settings import load_settings, resolve_device  # noqa: E402


def main() -> int:
    settings = load_settings()
    device = os.environ.get("DSWAP_DEVICE") or resolve_device(settings.device)
    print(f"# M3 NaN diagnostic — device={device}")

    codec = load_codec(
        name=settings.codec_name,
        model_type=settings.codec_model_type,
        model_tag=settings.codec_model_tag,
        model_path=settings.codec_model_path,
        device=device,
    )
    freeze_for_decoder_training(codec)
    n_removed = remove_weight_norm_recursive(codec.decoder)
    print(f"removed weight_norm from {n_removed} decoder submodules")
    codec.decoder.train()

    sr = codec.convention.sample_rate
    seg_samples = int(round(1.5 * sr))
    corpus = CorpusDataset(
        paths=settings.raw["corpora"]["new"],
        target_sr=sr,
        segment_samples=seg_samples,
        seed=0,
    )
    mel_loss_fn = MultiScaleMelLoss(sample_rate=sr).to(device)
    opt_kind = os.environ.get("DSWAP_OPT", "adamw").lower()
    if opt_kind == "sgd":
        optim = torch.optim.SGD(
            [p for p in codec.decoder.parameters() if p.requires_grad],
            lr=1e-5,
        )
    else:
        optim = torch.optim.AdamW(
            [p for p in codec.decoder.parameters() if p.requires_grad],
            lr=1e-5,
        )
    print(f"optimiser = {type(optim).__name__}")
    decoder_params = [p for p in codec.decoder.parameters() if p.requires_grad]

    torch.autograd.set_detect_anomaly(True)

    for step in range(1, 11):
        x = corpus.random_batch(2).to(device)
        with torch.no_grad():
            x_pre = codec.model.preprocess(x, sr)
            z, _codes, _l, _cm, _cb = codec.model.encode(x_pre)

        # Forward — log decoder output stats before loss.
        y_hat = codec.model.decode(z)
        y_finite = torch.isfinite(y_hat).all().item()
        y_max = y_hat.detach().abs().max().item() if y_finite else float("nan")

        mel_l = mel_loss_fn(y_hat, x_pre)
        wav_l = waveform_l1(y_hat, x_pre)
        loss = mel_l + wav_l

        print(f"step {step:>2d}: y_finite={y_finite}  |y_hat|_max={y_max:.4f}  "
              f"mel={float(mel_l):.4f}  wav={float(wav_l):.4f}  total={float(loss):.4f}")

        if not torch.isfinite(loss):
            print("  loss is non-finite — stopping diagnostic.")
            return 0

        optim.zero_grad(set_to_none=True)
        loss.backward()
        # Pre-clip global grad norm.
        total_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in decoder_params if p.grad is not None))
        print(f"        grad_norm_pre_clip = {float(total_norm):.4f}")

        torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=5.0)
        optim.step()

        # Detect post-step param NaN.
        bad = []
        for name, p in codec.decoder.named_parameters():
            if not torch.isfinite(p).all():
                bad.append(name)
        if bad:
            print(f"  POISONED params after step (first 5): {bad[:5]}")
            return 1

    print("\nNo NaN observed in 10 steps with batch=2, lr=1e-5, grad_clip=5.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
