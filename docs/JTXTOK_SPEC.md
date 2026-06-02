# jtxtok v1 — token format specification

> **Canonical source:** <https://github.com/jonnosan/jamtronix/blob/main/docs/JTXTOK_SPEC.md>.
> This file is a verbatim copy in the decoder-swap repo for offline readability and to make the
> contract explicit alongside the consuming code. If this copy and the canonical version
> disagree, the canonical version wins — open a PR to resync.

**Status:** v1 draft. This document is the *contract* shared by three components:

1. **jtx emitter** — produces jtxtok from jtx's internal event sequence (inference-time conditioning source).
2. **audio→jtxtok extractor** — produces jtxtok from a corpus audio file (training-time conditioning source).
3. **decoder-swap ingestion** — consumes jtxtok as cross-attention conditioning for training and generation.

Both producers (1, 2) MUST emit byte-identical vocabulary. The consumer (3) is producer-agnostic: it cannot tell, and must not care, whether a jtxtok stream came from jtx or the extractor. Any drift between producers silently corrupts training/inference alignment.

**Version this file.** The conditioning encoder bakes in the vocabulary. Adding/removing/renaming a token is a breaking change requiring a `version` bump and (at minimum) an embedding-table change in decoder-swap.

---

## 1. Design principles

- **Rhythm-and-grid first, harmony as a thin hint.** The vocabulary is scoped to what is *reliably extractable from real techno audio*, because the training-side producer is the extractor, not jtx. Drums + metric grid extract reliably; full chord/quality does not. So v1 carries precise rhythm + coarse key only.
- **Explicit metric grid.** Bar and position are first-class tokens (REMI's one load-bearing idea), not inferred from time-deltas. Grid = **16th notes, 16 positions per bar**.
- **Factorised by voice role.** Tokens are tagged by role so decoder-swap can filter (which voices to feed) and weight (how strongly to obey) per role. This is the lever for the three conditioning modes.
- **In-contract vs jtx-internal.** Anything the extractor cannot produce from audio is OUT of contract and MUST NOT appear in a jtxtok stream used for ML. jtx MAY emit such detail to a separate debug/sidecar stream.

---

## 2. Grid and timing

- **Resolution:** 16 positions per bar (16th-note grid). `POS_0` … `POS_15`.
- **Meter:** 4/4 assumed for v1. (A `METER_*` token is reserved for future use; v1 producers emit 4/4 only.)
- **Bar delimiter:** `BAR` token marks the start of each bar. Position tokens are bar-relative.
- **Quantisation:** all events snap to the nearest 16th. jtx events are expected to be 16th-aligned already (bar-by-bar regen). The extractor quantises detected onsets to the grid after beat/downbeat tracking.
- **Fine-event escape hatch:** for the rare sub-16th event (ratchet/flam/roll), an optional modifier `FINE_<n>` MAY follow an event token, where `<n>` ∈ {2,3,4} means "subdivide this 16th into n and place on the first such sub-step." v1 producers MAY ignore this entirely; the extractor is NOT required to produce it. Reserved so the grid need not be doubled.

### 2.1 Micro-timing (swing / groove) — placement vs feel

The 16th grid carries *where an event sits structurally*; it does NOT capture swing or groove. Those are represented by a separate, optional **micro-timing offset** so that structure stays clean and short while feel is expressed continuously rather than quantised into a finer grid. (Finer grids would bloat sequences AND be unextractable from real audio — the wrong fix.)

- **`MT_<n>`** — optional per-event modifier giving a signed micro-timing deviation *within the event's 16th cell*. `<n>` ∈ −8..+8 in fine units, where ±8 = ±half a 16th cell (i.e. the offset never crosses into a neighbouring grid position). Positive = late ("behind the beat"), negative = early.
  - Example: `POS_4 DRUM_SNARE MT_+3` — snare on grid position 4, pushed late by 3/8 of a cell.
  - Swing = positive `MT_*` on the off-grid 16ths (the even/odd subdivisions you choose to swing). Humanise = small random offsets.
- **MT comes from jtx and synthetic drum sweeps only — NEVER from the extractor.** Two producers touch `MT_*`:
  - **jtx** KNOWS its micro-timing exactly (it generates the groove) and emits `MT_*` fully, at inference.
  - **synthetic drum sweeps** (§7.2) are the sole source of `MT_*` *supervision* during training — clean, ground-truth, drum-only.
  - The **audio→jtxtok extractor emits NO `MT_*` at all.** Every event it produces sits exactly on the 16th grid. Onset-deviation measurement on dense techno is too noisy to be a useful teacher, and mixing noisy extractor MT with clean synthetic MT would *contaminate* the clean signal (the model would learn that MT values are unreliable indicators of timing, partly undoing the sweeps). So the corpus stream carries placement only; micro-timing is owned outright by the synthetic sweeps.
- **Why this is clean, not a compromise:** one teacher ⇒ `MT_<n>` has exactly one learned meaning. The corpus teaches structure, drum placement, coarse harmony, and timbre; the synthetic sweeps teach what an offset does to a percussive onset. decoder-swap reconciles them via **`MT_*` dropout during training** (see §7.1): the entire corpus stream has no MT, so the model learns "MT absent ⇒ play straight; MT present ⇒ honour the offset." This is exactly the inference behaviour wanted — jtx supplies MT live and the model grooves; nothing supplies MT and it plays straight. Same optionality+dropout pattern as CFG and harmony.
- **Competence is drum-centric.** The contract PERMITS `MT_*` on any event (jtx may emit it on bass), and the dropout-trained model won't choke on it — but the learned *competence* is on drums, because (a) percussive transients are where micro-timing is acoustically legible, (b) synthetic→corpus transfer is most trustworthy for percussion (a kick transient is similar across soundbanks), and (c) groove in techno lives in the drums anyway. Do NOT expect swung pads/bass to render convincingly in v1.
- **Fine-unit range** (−8..+8) is an open item. There is no extractor MT threshold to decide — the extractor omits MT entirely.

---

## 3. Voice roles

| Role prefix | Meaning | jtx emits | Extractor produces | Conditioning intent |
|---|---|---|---|---|
| `DRUM_` | percussion hits, per-instrument | yes | yes (ADT) | strong / faithful |
| `BASS_` | bass onsets (+ optional pitch) | yes | approx (sep + pitch) | medium, noisy |
| `KEY_`  | coarse key/root hint | yes | approx (chroma/key est.) | weak hint |

Out-of-contract (jtx-internal only, NEVER in ML stream): algorithm ID, modulator/CC curves, follower-derivation metadata, poly/pad voicing detail, lead/melody note content.

> Rationale: the extractor cannot recover these from audio, so the model could never learn to condition on them. Melody/pad content is deliberately excluded — it is the part the transformer is meant to *invent*, and the part audio extraction handles worst. The exclusion and the aesthetic intent point the same way.

---

## 4. Token vocabulary (v1)

### 4.1 Structural
- `BOS` — beginning of sequence (enables from-scratch generation; see §7).
- `EOS` — end of sequence.
- `BAR` — bar delimiter.
- `POS_0` … `POS_15` — position within bar (16th grid).
- `MT_-8` … `MT_+8` — optional per-event micro-timing offset within the cell (swing/groove); see §2.1. Follows the event token it modifies.

### 4.2 Drums
Per-instrument onset tokens. **Required v1 set (5-class):**
- `DRUM_KICK`
- `DRUM_SNARE`
- `DRUM_HAT` (closed hat)
- `DRUM_OHAT` (open hat)
- `DRUM_CLAP`

Open-hat and clap are required, not optional, because the offbeat open-hat and backbeat clap are core techno groove signifiers. They are within reliable ADT range (ADTOF / 18-class Vogl CRNN cover them), but the two distinctions most likely to smear are **closed-vs-open hat** and **clap-vs-snare** (acoustically similar) — validate these specifically (§8).

Further extended set (OPTIONAL; include only if model + corpus support reliably). Producers MUST agree on whichever subset is enabled:
- `DRUM_TOM`, `DRUM_RIDE`, `DRUM_CRASH`, `DRUM_PERC`

Velocity (OPTIONAL): a coarse `VEL_LO|MED|HI` modifier MAY follow a drum token. Extractor support is optional; if either producer omits velocity, BOTH must omit it (contract symmetry).

### 4.3 Bass
- `BASS_ON` — bass onset, pitch unspecified.
- `BASS_P_<pc>` — bass onset with pitch class, `<pc>` ∈ 0..11 (C=0). Use ONLY if the extractor's bass pitch tracking is trusted on your corpus; otherwise emit `BASS_ON` from both producers. (Contract symmetry: pick one and configure both producers identically.)

### 4.4 Harmony hint (coarse, key/root only)
- `KEY_<pc>` — current tonal centre, `<pc>` ∈ 0..11 (C=0). **No quality** (no major/minor) in v1 — root only, the reliable extraction target.
- Emitted at most once per bar (on the `BAR` token, before position-0 events), or only when it changes (producers MUST agree which; recommended: emit on change, carry forward otherwise).

---

## 5. Stream ordering (canonical form)

Within each bar, tokens are ordered:

```
BAR [KEY_<pc>]
  POS_<p> <event> [<event> ...]   # events sharing a position grouped under one POS token
  POS_<q> <event> ...
  ...
```

- Positions ascending; only positions with events are emitted.
- Multiple simultaneous events at one position follow a single `POS_<p>`, ordered: drums, then bass, (key handled at bar level).
- Whole stream: `BOS BAR ... BAR ... EOS`.

Example (one bar, four-on-floor kick, snare on 2 & 4, closed hats on 8ths, A-minor centre):
```
BOS
BAR KEY_9
  POS_0  DRUM_KICK DRUM_HAT
  POS_2  DRUM_HAT
  POS_4  DRUM_KICK DRUM_SNARE DRUM_HAT
  POS_6  DRUM_HAT
  POS_8  DRUM_KICK DRUM_HAT
  POS_10 DRUM_HAT
  POS_12 DRUM_KICK DRUM_SNARE DRUM_HAT
  POS_14 DRUM_HAT
EOS
```
(`KEY_9` = A. Note: root only — minor/major not encoded in v1.)

Same bar with swung off-beat hats (the even-16th hats pushed late via micro-timing), as jtx would emit:
```
BAR KEY_9
  POS_0  DRUM_KICK DRUM_HAT
  POS_2  DRUM_HAT MT_+4
  POS_4  DRUM_KICK DRUM_SNARE DRUM_HAT
  POS_6  DRUM_HAT MT_+4
  POS_8  DRUM_KICK DRUM_HAT
  POS_10 DRUM_HAT MT_+4
  POS_12 DRUM_KICK DRUM_SNARE DRUM_HAT
  POS_14 DRUM_HAT MT_+4
```
(The extractor would emit the straight version above — it never produces `MT_*`. The swung version is jtx-only, or comes from synthetic drum sweeps in training. The model learns to honour `MT_*` when present.)

---

## 6. The two producers — conformance notes

### 6.1 jtx emitter (prompt 1)
- Serialise jtx's internal event sequence directly to §4 tokens. No MIDI round-trip on the ML path (MIDI export remains the separate DAW path).
- Emit the FULL tagged stream (all in-contract roles). Do NOT implement mode/filtering logic here — voice selection happens in decoder-swap.
- Confirm jtx events are 16th-aligned; if finer timing exists and matters, use `FINE_*`, else quantise.
- jtx-internal detail (algorithm, CC, voicings) → separate sidecar, never in the jtxtok ML stream.

### 6.2 audio→jtxtok extractor (prompt 2)
Pipeline, each stage quantised to the §2 grid:
1. **Beat/downbeat grid** — `madmom` or BeatNet → defines `BAR` / `POS_*`. (Techno four-on-floor: reliable.)
2. **Drums** — ADT model, **5-class v1: kick/snare/hat/open-hat/clap** → `DRUM_*`. Use a model that covers open-hat and clap (ADTOF, or 18-class Vogl CRNN restricted to these 5). Quantise onsets to nearest 16th. Class set is a config flag but v1 default is 5-class (MUST match jtx).
3. **Bass (optional)** — Demucs bass stem → monophonic pitch → `BASS_ON` / `BASS_P_*`. Treat as noisy.
4. **Key (coarse)** — chroma / key estimation → `KEY_<pc>`, per bar or on-change.
- **Emits NO `MT_*`.** All onsets land exactly on the 16th grid. Micro-timing is owned by jtx (inference) and synthetic drum sweeps (training, §7.2) — never measured from corpus audio. This deletes the riskiest sub-task (onset-deviation measurement on dense techno) from the extractor entirely.
- Accept the caveats: rhythm/grid reliable; bass approximate; key coarse; quality/melody/algorithm/micro-timing NOT produced.
- **Validate standalone before decoder-swap depends on it** (see §8).

---

## 7. decoder-swap ingestion (prompt 3) — behavioural contract

### 7.1 Training
- For each corpus clip: conditioning = extractor's jtxtok; target = clip's DAC tokens. Train the cross-attention conditioning path.
- **Condition-dropout for CFG (NON-NEGOTIABLE):** randomly drop the entire conditioning (replace with empty/null) ~10–20% of steps during training. Without this, classifier-free guidance and the "ignore conditioning" mode are impossible to retrofit without retraining.
- **Micro-timing dropout:** independently of CFG dropout, randomly strip `MT_*` modifiers from the conditioning on a fraction of steps. Because the corpus stream NEVER carries `MT_*` (the extractor omits it) and the synthetic drum sweeps (§7.2) ALWAYS carry it, the model learns "`MT_*` absent ⇒ render straight; `MT_*` present ⇒ honour the offset." Dropout on the synthetic stream prevents the model from coupling MT-presence to synthetic timbre. This is what lets jtx drive groove at inference (MT present) while the all-MT-free corpus plays straight.
- RVQ output layout (flat/interleaved vs factorised) is decoder-swap's own choice on the *target/DAC* side and is independent of jtxtok. jtxtok conditioning is always factorised-by-role. Confirm the run's existing layout when implementing; it does not affect this spec.

### 7.2 Micro-timing supervision via synthetic drum sweeps

`MT_*` supervision comes exclusively from synthetic jtx→fluidsynth pairs, scoped tightly so they teach *timing* and not *timbre*:

- **Drums only.** Percussive transients are where micro-timing is acoustically legible, and where synthetic→corpus transfer is most trustworthy (a kick transient is similar across soundbanks; a synth pad is not). No pads/bass/melody in the synthetic set.
- **Controlled sweeps.** Render the SAME drum pattern at multiple `MT_*` values (e.g. `MT_-6,-3,0,+3,+6`), same neutral percussion patch. Holding everything else fixed and varying only the offset isolates MT as the sole explanatory variable — the constant timbre carries no useful gradient, so the model can only learn from the timing. This is a *better* MT signal than any real-audio extraction could give, because real audio never provides the same groove at known-varied offsets.
- **Neutral percussion patch.** As techno-neutral as available, to minimise timbre leakage.
- **Minority share.** Synthetic pairs are a small fraction of training; the real corpus dominates and owns timbre and the output distribution. Synthetic owns MT only.
- **Domain gap (named risk):** the model learns MT on synthetic-timbre drums and must transfer to corpus-timbre drums. Timing transfers across timbre well for percussion, so this is the favourable case — but it is why synthetic stays minority and drum-scoped, never a timbre source.

> This supersedes the earlier "fallback augmentation" idea: synthetic MT supervision is now the *planned, primary* MT teacher, not a rescue. The general-purpose objection to fluidsynth in training (it teaches GM timbre) does not apply here because the set is drum-only, timbre-isolated by sweep design, and minority-share.

### 7.3 Generation — two orthogonal controls
- **Axis 1 — voice filter (content):** select which role-prefixed tokens are passed (e.g. `DRUM_*` + `BASS_*` only). Applied at decoder-swap's input, NOT in jtx.
- **Axis 2 — CFG scale (adherence):** continuous dial from ignore → loose → faithful, from the dropout-trained model.

Named modes are presets in this 2-D space:

| Mode | Axis 1 (voices) | Axis 2 (CFG) |
|---|---|---|
| Faithful render | all in-contract | high |
| Rhythm not melody | `DRUM_` (+`BASS_`) only | high on what's fed |
| Ignore / free | none, or all | ~0 |
| From scratch | none + `BOS` seed | n/a (unconditional) |

- **From-scratch** = empty conditioning + `BOS` start token. Same code path as "ignore". Ensure empty-condition runs are valid.
- Producer-agnostic: generation accepts jtxtok from jtx (live) or extractor (re-render) identically.

---

## 8. Standalone extractor validation (do before prompt 3)

Minimal gate before decoder-swap is wired to depend on extraction quality:
1. Run the extractor on a handful of corpus clips.
2. Eyeball/listen: do all five extracted classes (`DRUM_KICK/SNARE/HAT/OHAT/CLAP`) and the bar grid match what you hear? (Click-track the quantised onsets back over the audio, one click pitch per class.)
3. **Check the smear-prone pairs specifically:** closed-hat vs open-hat, and clap vs snare. These are the acoustically similar distinctions ADT most often confuses. If either pair is unreliable on your corpus, decide per-pair whether to merge (e.g. fold open-hat back into hat) — and if you merge, jtx must merge identically.
4. Expected outcome: kick/snare/grid clean (proceed); hat/ohat and clap/snare separation acceptable (or merged); bass noisy (acceptable); key coarse (acceptable). If core drums/grid are poor, STOP and fix extraction before training — a bad result downstream would otherwise be ambiguous (model vs data).

---

## 9. Build sequencing

1. **This spec** (done first; cited by all prompts).
2. **Prompt 2 (extractor)** — riskiest, longest pole. Build + validate standalone (§8).
3. **Prompt 1 (jtx emit)** — independent, low-risk; needed only for *inference*, not the first training run.
4. **Prompt 3 (decoder-swap ingest)** — last, once extractor output is trusted. Must include §7.1 condition-dropout.

The current unconditional decoder-swap run is the base model prompt 3 extends; nothing here requires restarting it.

---

## Open items to resolve before freezing v1
- [ ] Confirm decoder-swap RVQ target layout (flat vs factorised) — affects prompt 3 only, not this spec.
- [ ] Drum class set: **5-class (kick/snare/hat/ohat/clap) is the v1 default** — confirm the ADT model separates hat/ohat and clap/snare on the corpus (§8); fall back to merging a pair only if validation fails. Both producers must match the final set.
- [ ] Decide bass: `BASS_ON` only vs `BASS_P_*` — depends on extractor pitch reliability; both producers must match.
- [ ] Decide velocity: include `VEL_*` or omit — both producers must match.
- [ ] Confirm jtx internal timing is 16th-aligned.
- [ ] Confirm jtx exposes micro-timing offsets (for `MT_*` emission) — almost certainly yes if groove is generated, but verify it is queryable per-event.
- [ ] Decide `MT_*` fine-unit range (default −8..+8 = ±half a cell).
- [ ] Decide synthetic drum-sweep parameters (§7.2): which `MT_*` values to sweep, neutral percussion patch choice, synthetic share of training mix.
- [ ] ~~Extractor `MT_*` threshold~~ — RESOLVED: extractor emits no `MT_*`. Micro-timing is jtx + synthetic sweeps only.
