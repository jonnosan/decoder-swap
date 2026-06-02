# decoder-swap

A small, self-contained research experiment. **Does the structural information carried by a neural
audio codec's token sequence survive a re-voiced decoder?**

## Hypothesis

A neural audio codec splits into:

- **encoder**  : audio → continuous latent
- **codebook** : latent → discrete tokens (RVQ)
- **decoder**  : tokens → audio

A token sequence **T** carries temporal/structural information (what happens when). The **decoder
weights** carry acoustic realisation (what it sounds like).

If we **freeze** encoder + codebook and **retrain only the decoder** on a different audio corpus,
then run the SAME T through both:

- the original decoder D1 → S1
- the retrained decoder D2 → S2

we expect:

1. transient/onset timing roughly **preserved** between S1 and S2 (structure lives in T + frozen codebook),
2. timbre / spectral character clearly **changed** (realisation lives in the retrained decoder),
3. pitch somewhat preserved but free to drift.

The deliverable is **evidence** (audio + plots + numbers), not an app. A clean refutation is as
valuable as a confirmation.

## Hard invariant

D1 and D2 must differ in **nothing** except decoder weights — same codebook (byte-for-byte), same
encoder, same vocab size, same RVQ depth, same frame rate, same latent dim, same sample rate. If
any of those differ, the experiment is invalid. `src/decoder_swap/invariants.py` enforces this with
SHA-256 fingerprints of every codebook tensor, checked **before** D1/D2 produce outputs.

A negative test (deliberately bumping one codebook entry by 1e-6) confirms the invariant catches
single-element drift — see `scripts/02_freeze_check.py`.

## Status

All milestones complete (2026-06-02):

- [x] **M0** — skeleton + codec verification
- [x] **M1** — sanity round-trip
- [x] **M2** — freeze + invariants
- [x] **M3** — decoder fine-tune on CORPUS_NEW
- [x] **M4** — comparison (S1 vs S2)
- [x] **M5** — plots + writeup

## Setup

```bash
cd decoder_swap
uv sync                                        # creates .venv and installs deps
uv run python scripts/00_inspect_codec.py      # M0: prove the codec splits into 3 freezable parts
uv run python scripts/01_sanity_roundtrip.py   # M1: end-to-end round-trip works
uv run python scripts/02_freeze_check.py       # M2: invariant + negative-test
uv run python scripts/03_train_d2.py --steps 2500   # M3: ~110 min on M4 Pro MPS
uv run python scripts/04_compare.py            # M4: produces S1.wav + S2.wav + metrics.json
uv run python scripts/05_plots.py              # M5: figures
```

The first run of any script that touches the codec downloads the DAC 44 kHz weights to
`~/.cache/descript/` (~300 MB).

## Methodology

### Codec
[Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (DAC) 44 kHz model.
Token convention used throughout: sample_rate 44100 Hz, hop_length 512 samples,
frame_rate 86.13 fps, **9 RVQ codebooks × 1024 entries**, latent_dim 1024.

### CORPUS_NEW
Two deep-techno DJ sets — Vytis "Greatest Hits" Vol. 1 (122 min) + Vol. 3 (75 min) —
**197 min total**. Loaded to RAM as mono float32 (~2 GB) and sampled as random 1.5 s crops.

### Held-out clips (NOT in CORPUS_NEW)
- *Blue Kentucky Girl* (Patty Loveless cover, country, 3 min 25 s)
- *Back In Black* (AC/DC, rock, 4 min 16 s)

### Training (M3)
- Freeze encoder + entire quantizer (not just `codebook.weight` — DAC's factorised codebooks
  also have per-quantizer `in_proj`/`out_proj` layers that must stay frozen, else the
  encoder→discrete-token mapping changes and the experiment is invalid).
- Decoder weights remain trainable; before training, every `weight_norm` parametrization in the
  decoder is collapsed (see "Real bugs hit along the way" below).
- Loss: **multi-scale log-mel L1** (windows 2048/1024/512, n_mels 80/80/64) **+ waveform L1**.
- Optimiser: AdamW lr=1e-4, grad-clip global norm 5.0.
- 2500 steps × batch 4 × 1.5 s crops = **2.5 h of audio (with replacement) seen by gradient**.
- Wall clock: **108 min on M4 Pro MPS**, 0.38 steps/s.
- Codebook SHA-256 fingerprints asserted byte-identical every 25 steps (100 checks, all passed).
- Periodic checkpoint every 200 steps (so SIGINT keeps the work).

### Loss trajectory

![training loss](results/m3_training_loss.png)

Loss descended cleanly from a first-window average of 1.02 to a last-window average of 0.55
(−45.7%). After step ~2200 the descent flattened and the per-window noise envelope widened —
characteristic of having reached the gradient-signal floor for this objective on this corpus.

### Evaluation (M4)
For each held-out clip:
1. Load fresh codecs **codec_d1** and **codec_d2**; load D2's saved decoder weights into codec_d2.
2. `assert_codec_invariants_match(codec_d1, codec_d2)` — abort if codebook fingerprints differ.
3. Process audio in 30-second chunks (M4 Pro MPS budget is ~30 GB; full 4-min songs encoded in
   one pass blow it):
   - frozen `encode(x)` once → tokens **T** and quantized latent **z**
   - `decoder_d1(z)` → S1 chunk
   - `decoder_d2(z)` → S2 chunk
4. Concatenate chunks; save `S1.wav` and `S2.wav`; compute structure + realisation + forgetting
   metrics; emit verdict block.

## Results

### Headline numbers

|  | **Blue Kentucky Girl** (country) | **Back In Black** (rock) | hypothesis says |
|---|---:|---:|---|
| Onset F1 (50 ms tolerance) | 0.886 | 0.874 | HIGH ✓ |
| Mean abs onset diff | 2.2 ms | 1.5 ms | tiny ✓ |
| RMS envelope Pearson r | 0.946 | 0.973 | HIGH ✓ |
| log-mel L1 distance (dB) | 2.68 | 2.63 | HIGH ✗ |
| MFCC L1 distance | 5.47 | 5.10 | HIGH partial |
| Spectral centroid Pearson r | 0.685 | 0.799 | LOW-ish ✓ partial |
| S1 → S2 centroid mean shift (Hz) | 2624 → 2131 | 3083 → 2735 | shifted ✓ |
| Chroma cosine similarity | 0.978 | 0.953 | partial preservation ✓ |
| Forgetting probe Δ (dB) | +0.96 | +1.06 | HIGH ✗ |

### Figures

![country comparison](results/m4_compare/comparison.png)

![rock comparison](results/m4_compare_rock/comparison.png)

(Open `results/m4_compare/{input,S1,S2}.wav` and `results/m4_compare_rock/{input,S1,S2}.wav` to
listen.)

### What the metrics say

- **Hypothesis clauses 1 & 2 (structure preserved): strongly supported.** Onsets land within
  1.5–2.2 ms across S1 and S2 (well under perceptual transient sensitivity ~10 ms). RMS envelope
  correlation 0.95+. Identical token grids produce structurally near-identical audio across two
  totally different decoders. The "structural information lives in T" half of the hypothesis
  holds cleanly.
- **Hypothesis clause 3 (timbre clearly changed): NOT supported at heuristic threshold.** Mel-dB
  distance is 2.6 dB on both clips — below the 3 dB "noticeably different" bar. There is a real
  shift (centroid drops ~500 Hz; MFCC L1 ≈ 5; spectral centroid correlation around 0.7–0.8) but
  it's a tonal *nudge*, not a transplant into a different timbral world.
- **Hypothesis clause 4 (pitch partially preserved): supported, in the "almost unchanged"
  direction.** Chroma cosine 0.95–0.98 — pitch class content is preserved very strongly. We
  expected drift; we got barely any.
- **Forgetting probe: weak.** D2 reconstructs each held-out input only ~1 dB worse than D1.
  D2 hasn't *forgotten* country/rock so much as tinted it.

### Perceptual finding (the most informative result)

> "Both S2s sound to me like the corresponding S1 played through a ring modulator effect."
> — user, listening test

This is exactly the perceptual signature the numbers describe:

| ring-mod behaviour | what the metrics show |
|---|---|
| envelope preserved | RMS Pearson r ≈ 0.95 |
| onsets preserved | onset F1 ≈ 0.88, mean diff < 2.2 ms |
| pitch content preserved | chroma cosine ≈ 0.95+ |
| spectral character shifted, slightly metallic | centroid down ~500 Hz, mel-dB ≈ 2.6 |
| but NOT replaced with a different musical world | mel-dB 2.6 (below "different timbral world") |

A ring modulator multiplies a signal by a carrier — it preserves envelope while replacing
spectral content with sum-and-difference sidebands. The perceptual mapping is sharp: D2 is
imposing a roughly-fixed spectral fingerprint onto whatever it decodes, exactly as a carrier
modulates a signal.

### Why we got "ring-mod" and not "deep techno"

D2 was trained on **multi-scale log-mel L1 + waveform L1 only** — no adversarial discriminator.
What does that loss actually reward? *"Make the average spectral content of your output match the
average spectral content of techno."* The cheapest path to reducing that loss across thousands of
diverse techno crops is to learn **a characteristic spectral envelope to impose**, rather than to
learn the much harder skill of *producing convincing techno music*.

That imposed envelope manifests as a ring-mod-like overlay when applied to country/rock tokens:
structure passes through; spectral fingerprint gets layered on top.

This is a well-known limitation of mel-loss-only neural codec fine-tunes — you get *spectral
palette transfer*, not *musical re-composition*. The full DAC training recipe adds adversarial
discriminators specifically to push past this; they're what convinces the decoder to produce
*realistic-sounding* outputs rather than mel-bin-matching outputs.

### Caveat — most of the hypothesis was on easy ground

The token grid runs at **86.13 fps × 9 codebooks × 10 bits = 7.75 kbps**, with **90 bits per
~11.6 ms frame** to specify "what spectrum is present right now." A learned codebook with 90
bits/frame doesn't just specify envelope and timing — it specifies pitch class, rough chord
identity, and timbral category. Mapping that to the four hypothesis clauses:

| hypothesis clause | what actually carries it | was the decoder really tested? |
|---|---|---|
| 1. onset timing preserved | 86 Hz frame rate (far above need ≈ 50 Hz) | **no, given by token rate** |
| 2. RMS envelope preserved | per-frame energy info | **no, given by token rate** |
| 4. pitch preserved | per-frame 90 bits of spectral info | **largely given by token rate** |
| 3. realisation changes | decoder synthesis choices | **yes — this was the actual test** |

The clean confirmations of clauses 1, 2, and 4 are therefore weaker evidence for the
"structure-lives-in-T" hypothesis than they look — the token grid pretty much *guarantees* those
findings regardless of what the decoder does. The one genuine test was clause 3, the realisation
change, and that's where the partial confirmation + the ring-modulator perceptual signature
sits.

To actually test clauses 1, 2, and 4 you'd need a token grid sparse enough that preservation
isn't automatic — DAC's lower-bitrate variants (24 kHz @ 8 kbps, 16 kHz @ 6 kbps with 12 narrow
codebooks) are the natural place to look. Tracked as
[issue #5](https://github.com/jonnosan/decoder-swap/issues/5).

### Verdict

The hypothesis is **partially supported**:
- The *structure-lives-in-T* half is **strongly confirmed** by every structural metric.
- The *realisation-lives-in-decoder* half is **partially confirmed** — there IS a measurable
  realisation change, and it has a specific perceptual signature (ring-mod-like overlay), but it
  is not a *genre transplant*. The decoder learned **spectral fingerprint transfer**, not
  **timbral synthesis**.

A clean refutation, or a clean partial confirmation, is as valuable as a clean full confirmation
— per the design brief. We have the latter.

## Mimi follow-up (same session)

After the DAC run we re-ran the whole pipeline on **[Mimi](https://huggingface.co/kyutai/mimi)**
(Kyutai) as a second codec — same M3 → M4 recipe, completely different bottleneck:

| | DAC 44 kHz | Mimi |
|---|---:|---:|
| Audio rate | 44 100 Hz | 24 000 Hz |
| Frame rate | 86.13 fps | **12.5 fps** (≈7× lower) |
| Codebooks used | 9 | 8 (1 semantic + 7 acoustic) |
| Total bitrate | 7.75 kbps | **1.1 kbps** (≈7× lower) |
| M3 wall clock | 108 min | **9.3 min** |
| M3 loss improvement | −45.7 % | −30.2 % |
| M3 NaN-step rate | 0 % | **48 %** (caught + skipped) |

**Mimi-specific integration notes worth recording:**
- 7 top-level submodules instead of 3. Logical grouping for the freeze:
  *frozen front-end* = `encoder + encoder_transformer + downsample + quantizer` (39.4 M);
  *trainable* = `upsample + decoder_transformer + decoder` (39.9 M).
- Codebooks are **EMA buffers** (`.embed`), not `nn.Parameter` — different fingerprint path
  in `codebook_tensors()`.
- `weight_norm` not used in Mimi (0 modules removed).
- **Segment length must be a multiple of `hop_length` (1920 samples).** At `segment_seconds=1.5`
  (= 18.75 frames at 24 kHz) Mimi's internal padding produced NaN on **every even step**, deterministically.
  `segment_seconds=2.0` (= 25 frames exact) dropped the smoke-test NaN rate from 47 % → 3 %.
  Snapping `segment_samples` to `floor(samples / hop) * hop` in `CorpusDataset` would be the
  bulletproof fix.
- Even with the fix, **NaN rate climbed back to ~48 % at scale** as the decoder weights drifted —
  effective training was ~1 300 real gradient updates, not 2 500. Loss still descended cleanly,
  but a real fix needs deeper diagnosis (most likely a Mimi transformer attention path that's
  sensitive to specific normalisation states the optimizer drifts into).

### Mimi M4 results vs DAC M4 (same Blue Kentucky Girl held-out clip)

| metric | DAC | Mimi | direction |
|---|---:|---:|---|
| Onset F1 | 0.886 | 0.812 | mostly preserved |
| RMS envelope r | 0.946 | 0.972 | essentially identical |
| **log-mel L1 distance (dB)** | **2.68** | **4.19** | larger spectral shift ✓ |
| MFCC L1 | 5.47 | 7.23 | larger |
| Spectral centroid r | 0.685 | 0.927 | tighter centroid tracking |
| Chroma cosine | **0.978** | **0.897** | **much more pitch drift** |
| Forgetting Δ (input↔S2 − input↔S1, dB) | +0.96 | **−1.17** | **D2 reconstructs input *better* than D1** |

### Perception vs metrics — the most informative finding of the session

User's listening report on Mimi S2:
> "S1 sounds closer to the original than S2. S2 sounds like S1 via an extreme ring modulator."

Mapped against the metrics:
- The mel-dB *forgetting probe* says D2 is **closer** to the input than D1 — directly contradicting
  the user's perception that D2 is the more artificial-sounding one.
- Only **chroma cosine** (which dropped from 0.978 → 0.897) tracked the perceptual change.

This reveals a real limitation of the experiment's headline metric: **mel-dB distance is
insensitive to ring-modulator-style distortion**. Ring-mod preserves a lot of *aggregate*
log-mel energy distribution while sounding dramatically more artificial — so D2's output can
look numerically closer in mel space than D1's while being audibly further from real audio.
Any future re-voicing experiment should treat mel-dB as ONE signal and combine it with chroma
and adversarial-discriminator-style realism scores. Tracked as
[issue #6](https://github.com/jonnosan/decoder-swap/issues/6) (filed during wrap-up).

### What this tells us about the mechanism (refined)

The DAC result was "subtle ring-mod overlay." The Mimi result is "extreme ring-mod overlay."
Same character, sharper signal. Mechanistic story:

- mel-loss-only fine-tuning teaches the decoder to **impose a characteristic spectral fingerprint**.
- The **less freedom** the decoder has in its token grid (sparser tokens, fewer bits/frame), the
  **more pronounced** the fingerprint becomes when applied to OOD inputs.
- DAC at 7.75 kbps gave the decoder enough room to render naturally → subtle ring-mod.
- Mimi at 1.1 kbps gave it much less room → extreme ring-mod.

Also worth noting on the *direction* of forgetting:
- **DAC D1** was pretrained on broad music + speech, so techno-specialisation *narrowed* its capability
  on country.
- **Mimi D1** was pretrained primarily on speech, so techno-specialisation actually *generalised* it
  toward music in general — D2 reconstructs country better than D1 does.

So "fine-tuning a decoder narrows it toward the new corpus" is too simple. The right statement is:
**the decoder moves toward its training data**. Whether that means "more specialised" or
"more general" depends entirely on where the decoder started.

## Real bugs hit along the way

These cost time to find — recording so we don't pay them again.

1. **MPS `weight_norm` backward bug.** DAC's decoder uses `torch.nn.utils.weight_norm` on every
   conv. On MPS, the backward through `||v||` division produces NaN in `weight_v` even when the
   forward loss is finite and the pre-clip gradient norm is small. **Fix:** call
   `torch.nn.utils.remove_weight_norm` on every weight-normed submodule of the decoder before
   training. Forward behaviour is unchanged (the parametrization gets collapsed into a single
   `weight` tensor); the buggy backward path is gone. The saved D2 checkpoint records
   `decoder_weight_norm_removed: True` so M4 mirrors the removal before loading state_dict.
2. **MPS STFT backward NaN.** `torchaudio.transforms.MelSpectrogram` on MPS produces NaN
   gradients via `LogBackward0` on small/near-zero mel bins, even with `power=2.0 + clamp(min=1e-5)`
   in the forward path (clamp passes NaN through). The bug is in the STFT backward on MPS, not
   the loss formulation. **Fix:** keep `MultiScaleMelLoss` permanently on CPU; gradient flows
   correctly back to the MPS-resident decoder via cross-device autograd. Overhead is one ~1 MB
   MPS↔CPU copy per loss call.
3. **MPS OOM on 4+ minute decodes.** Decoding the full AC/DC clip in a single pass blew the M4
   Pro's 30 GB MPS budget (D1 + D2 + decoder activations). **Fix:** chunk audio into 30 s
   segments, decode independently per chunk, `torch.mps.empty_cache()` between. Per-chunk
   artifacts at boundaries are identical in S1 and S2 (shared encoder, identical tokens) so they
   don't bias the comparison.

## What to try next

### The actual goal (clarified mid-session)

> "I want to find a way that lets us 'play a country song in techno sounds' in a way that the
> intermediate processing (i.e. tokens from encoder to decoder) is observable."

Both halves of that constraint matter. End-to-end models like MusicGen / Stable Audio / Riffusion
could produce *something that sounds like techno performing the song*, but they don't preserve
the clean `audio → tokens → audio` shape we had — MusicGen-Melody uses chromagrams not tokens as
its structural conditioning, so the "what the system thinks this song is" representation gets lost.

### Recommended direction: token translation

The decoder-swap design has a hard ceiling for genre transplant — the encoder bakes the input
into the tokens too completely, and re-training the decoder can only impose a spectral fingerprint
on top of that. The natural extension that **keeps observability AND can achieve real genre
transplant** is to add a learned **token translator** in the middle:

```
country audio → ENCODER (frozen) → T_country (observable)
                                   ↓
                            TRANSLATOR (the new piece)
                                   ↓
                                T_techno (observable — directly comparable to T_country!)
                                   ↓
                                DECODER (original D1) → techno-style audio
```

Both `T_country` and `T_techno` are inspectable. You can ask which codebook entries the
translator changed, whether rhythm stayed, whether harmony shifted — *the experiment becomes
more interpretable, not less*.

A first-attempt minimum-viable version is a small autoregressive transformer (~10-30 M params)
trained on the technical corpus's tokens, then used at inference with input tokens as
cross-attention context to steer generation. Tracked as
[issue #6](https://github.com/jonnosan/decoder-swap/issues/6) — the new top-priority next step.

### Deprioritised: adversarial decoder ([epic #3](https://github.com/jonnosan/decoder-swap/issues/3))

Originally the next step after M5. Still scientifically interesting — would *reduce* the ring-mod
fingerprint by training the decoder to fool a discriminator. But: it can't achieve real genre
transplant either, because the structural ceiling is the same. Better as a comparison data point
*after* the translator path produces results, not as the primary next step.

## Codec licence

DAC ships under MIT, but the **pretrained weights** have their own terms — check
[descript-audio-codec releases](https://github.com/descriptinc/descript-audio-codec/releases) if
you intend to use outputs commercially. This is a research experiment; treat outputs accordingly.

## Layout

```
src/decoder_swap/
  codec_io.py       # load DAC, expose encoder/codebook/decoder
  freeze.py         # freeze front-end; remove_weight_norm helper
  invariants.py     # SHA-256 codebook fingerprints + pairwise/snapshot assertions
  losses.py         # MultiScaleMelLoss (CPU) + waveform L1
  dataset.py        # CorpusDataset — in-RAM random crops
  train_decoder.py  # M3 fine-tune with periodic checkpointing + SIGINT-safe save
  run_experiment.py # M4 D1/D2 comparison (chunked)
  measure.py        # §3 metrics + plain-English verdict block
  plot.py           # M5 comparison + loss-curve figures
scripts/
  00_inspect_codec.py    # M0
  01_sanity_roundtrip.py # M1
  02_freeze_check.py     # M2 (incl. negative test)
  03_train_d2.py         # M3 — overridable steps/lr/batch via CLI
  03b_diag_nan.py        # the script that found the MPS NaN bugs (kept for posterity)
  04_compare.py          # M4 — overridable input + out-dir
  05_plots.py            # M5
results/
  m1_sanity/             # round-trip wavs
  m3_training_loss.png
  m4_compare/            # input.wav, S1.wav, S2.wav, metrics.json, comparison.png (country)
  m4_compare_rock/       # same, for AC/DC
data/                    # gitignored — corpus paths in config.yaml, checkpoints, intermediates
```
