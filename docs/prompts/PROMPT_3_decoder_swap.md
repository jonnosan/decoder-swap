# Claude Code prompt — extend decoder-swap to ingest `jtxtok`

> Read `JTXTOK_SPEC.md` in full before starting (esp. §7). Build this LAST — only after the `audio→jtxtok` extractor's output has been validated standalone (spec §8). This extends the existing decoder-swap AR transformer; it does NOT require restarting the current unconditional base run — that run is the base model this builds on.

## Context

decoder-swap currently trains an autoregressive transformer over DAC tokens of a techno corpus (the unconditional base model — see repo issue #6). This task adds **jtxtok conditioning**: the model learns to render a jtxtok structural skeleton into corpus-style DAC tokens, and at generation time can be driven by jtxtok from either producer (extractor or jtx).

The model is **producer-agnostic** — it must not care whether a jtxtok stream came from the extractor (training) or jtx (inference). It consumes the v1 vocabulary, nothing more.

## Part A — Conditioning architecture

- Add a **conditioning encoder** over the jtxtok vocabulary (spec §4) feeding the AR transformer via **cross-attention**. Embed the jtxtok tokens (small, fixed vocab — far smaller than the DAC side).
- Keep jtxtok conditioning **factorised by voice role** (drums / bass / key are distinguishable), so generation-time voice filtering (Part C) can select role subsets.
- **RVQ target layout:** the DAC/target side's flat-vs-factorised layout is decoder-swap's existing choice — inspect the current run and match it. It is independent of jtxtok conditioning (spec §7.1). Confirm which the base run uses before wiring the output head.

## Part B — Training (spec §7.1, §7.2)

Pairs: `(jtxtok conditioning) → (DAC tokens)`.

### B1. Corpus pairs (the bulk)
- conditioning = the **extractor's** jtxtok for each corpus clip; target = that clip's DAC tokens.
- The corpus stream carries **no `MT_*`** (extractor omits it).

### B2. Synthetic micro-timing pairs (spec §7.2) — the SOLE `MT_*` teacher
- A **minority** of training pairs are synthetic: jtx→fluidsynth **drum-only** sweeps.
- **Controlled sweeps:** same drum pattern rendered at several `MT_*` values (e.g. `-6,-3,0,+3,+6`), **same neutral percussion patch**, so timbre is held constant and only timing varies — isolating MT as the sole explanatory variable.
- These pairs always carry `MT_*`; the corpus pairs never do.
- Keep the synthetic share small — the corpus dominates and owns timbre/output distribution; synthetic owns MT only.

### B3. Dropout (both NON-NEGOTIABLE)
- **CFG / condition dropout:** randomly drop the *entire* conditioning (→ null/empty) ~10–20% of steps. Without this, classifier-free guidance and "ignore conditioning" / from-scratch modes cannot be retrofitted without retraining.
- **`MT_*` dropout:** independently, randomly strip `MT_*` modifiers on a fraction of steps (applies to the synthetic stream, since only it carries MT). Teaches "MT absent ⇒ play straight; MT present ⇒ honour offset", and prevents the model coupling MT-presence to synthetic timbre.

## Part C — Generation (spec §7.3): two orthogonal controls

Expose these as independent knobs — do NOT hard-code the named modes as separate code paths; they are presets in this 2-D space.

- **Axis 1 — voice filter (content):** select which role-prefixed tokens are passed to the conditioning encoder (e.g. `DRUM_*` + `BASS_*` only, dropping `KEY_*`/melody). Applied at decoder-swap's input.
- **Axis 2 — CFG scale (adherence):** continuous guidance dial from ignore → loose → faithful, using the condition-dropout-trained model.

Named modes as presets:

| Mode | Axis 1 (voices fed) | Axis 2 (CFG) |
|---|---|---|
| Faithful render | all in-contract | high |
| Rhythm not melody | `DRUM_` (+`BASS_`) only | high on what's fed |
| Ignore / free | none, or all | ~0 |
| From scratch | none + `BOS` seed | unconditional |

- **From-scratch generation** = empty conditioning + `BOS` start token. Same code path as "ignore". Ensure an empty-conditioning run is valid (this also delivers the standalone unconditional generation discussed separately).
- Inputs at generation time: jtxtok stream (from jtx or extractor, or empty) + voice-filter selection + CFG scale + sampling controls (temperature/top-k/top-p) + RNG seed + length. Decode sampled DAC tokens → audio via the existing DAC decoder path.

## Part D — start token

- If the base run was trained WITHOUT a `BOS`, add a reserved `BOS` id and do a short fine-tune from the existing checkpoint (one new embedding row; no full retrain) so from-scratch generation works. If `BOS` is already present, reuse it.
- Alternatively, continuation-from-snippet (seed with a short real token prefix) works with no `BOS` — support it regardless as a generation mode.

## Verification

- **Conditioning sanity:** feed a known drum pattern as jtxtok with high CFG → generated audio's drum hits should land on the conditioned grid positions.
- **MT sanity:** feed the same pattern with vs without `MT_*` (swing) → audible groove difference on drums. (Expect this to be the weakest-learned dimension; spec §7.2 domain-gap note.)
- **Axis independence:** verify voice filter and CFG scale move independently (e.g. drums-only at high CFG ignores conditioned melody but still grooves).
- **From-scratch:** empty conditioning + `BOS` + seed → coherent corpus-style audio; different seeds → different outputs.
- **Corpus discrimination:** the original goal — confirm that conditioning/generation reflects the trained corpus's character.

## Deliverables

1. Conditioning encoder + cross-attention integration on the existing AR transformer.
2. Training path: corpus pairs (B1) + synthetic MT sweeps (B2) + both dropouts (B3).
3. Generation path: the two orthogonal controls (C), from-scratch (C/D), producer-agnostic jtxtok ingestion.
4. A note recording the confirmed RVQ target layout and whether a `BOS` fine-tune was needed.
