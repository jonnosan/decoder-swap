# Claude Code prompt — build the `audio→jtxtok` extractor (new tool)

> Build this FIRST. It is the gating-risk component (extraction quality determines whether the whole pipeline is viable) and it must be validated standalone before decoder-swap depends on it. Read `JTXTOK_SPEC.md` in full before starting — it is the contract this tool must conform to byte-for-byte.

## Goal

Create a standalone command-line tool that takes a techno audio file (mp3/wav) and emits a **jtxtok v1** token stream conforming to `JTXTOK_SPEC.md`. This is the *training-time* conditioning producer: its output will be paired with each corpus clip's DAC tokens to train decoder-swap's conditioning path.

## Hard contract constraints (from JTXTOK_SPEC.md — do not deviate)

- Emit the **exact** v1 vocabulary: `BOS`, `EOS`, `BAR`, `POS_0`…`POS_15`, `DRUM_*`, `BASS_*`, `KEY_<pc>`. Same token spellings as the jtx emitter will use — any drift silently corrupts training.
- **Grid = 16th notes, 16 positions/bar.** All onsets quantise to the grid.
- **Emit NO `MT_*` whatsoever.** Micro-timing is owned by jtx + synthetic sweeps only (spec §2.1, §6.2). Every onset lands exactly on its grid position. Do NOT implement onset-deviation measurement — it is deliberately excluded as too noisy.
- **No out-of-contract tokens:** no algorithm IDs, no CC/modulators, no chord *quality* (root only), no melody/pad note content.
- Canonical ordering per §5: `BOS BAR [KEY_<pc>] POS_<p> <events...> ... EOS`. Within a position, order drums then bass. Key emitted at bar level, on-change (carry forward otherwise).

## Pipeline (spec §6.2)

Implement as composable stages, each quantising to the 16th grid:

1. **Beat/downbeat grid** — use `madmom` (DBNDownBeatTrackingProcessor) or BeatNet. Output: bar boundaries + a 16th-note grid per bar. This defines `BAR` and the `POS_*` lattice. (Techno four-on-floor: expect this to be reliable.)
2. **Drums** — automatic drum transcription. **v1 default: 5-class — kick/snare/hat(closed)/open-hat/clap** → `DRUM_KICK`/`DRUM_SNARE`/`DRUM_HAT`/`DRUM_OHAT`/`DRUM_CLAP`. Use a pretrained ADT model that covers open-hat and clap (ADTOF, or the 18-class Vogl CRNN restricted to these 5 — NOT the 3-class model, which lacks ohat/clap). Quantise detected onsets to nearest 16th. Class set is a config flag (5-class default; can extend or fall back to merge a pair if validation fails — MUST match jtx).
3. **Bass (optional, config-gated)** — Demucs to isolate bass stem → monophonic pitch detection → `BASS_ON` (or `BASS_P_<pc>` if pitch trusted). Default to `BASS_ON` only unless `--bass-pitch` is set. Treat as noisy.
4. **Key (coarse)** — chroma-based key/root estimation → `KEY_<pc>` (0..11, C=0). **Root only, no quality.** Per bar or on-change.

## Config flags (must be explicit so both producers can be matched)

- `--drum-classes {5,18}` (default 5: kick/snare/hat/ohat/clap)
- `--bass {none,onset,pitch}` (default onset)
- `--velocity {off,on}` (default off) — if on, emit coarse `VEL_LO|MED|HI` after drum tokens
- `--key {off,on}` (default on)
- These mirror the spec's "both producers must match" open items. Document the chosen defaults prominently — they become the contract jtx must follow.

## Output format

- One jtxtok stream per input file, written as a `.jtxtok` text file: whitespace-separated tokens, newline per bar for readability (whitespace is not semantically significant — the consumer tokenises on the vocabulary).
- Also emit a small JSON sidecar with: detected tempo, bar count, grid confidence per bar (from the beat tracker), and which config flags were used. This sidecar is for debugging/validation only, NOT part of the jtxtok stream.

## Standalone validation harness (spec §8 — REQUIRED, build alongside)

This is not optional polish — it is how we decide whether the pipeline is good enough to proceed to decoder-swap:

- A `validate` subcommand that takes an audio file, runs extraction, and produces a **click-track overlay**: render the quantised `DRUM_*` onsets as clicks (different pitch per drum class, all 5) mixed over the original audio, so a human can listen and hear whether the extracted classes and bar grid match.
- **Specifically surface the smear-prone pairs:** report closed-hat vs open-hat and clap vs snare separately, since those are the distinctions ADT most often confuses. If a pair is unreliable, the config supports merging it (and jtx must merge identically).
- Print a summary: tempo, bars, per-class onset counts (all 5), mean beat-tracker confidence.
- Run on a handful of corpus clips and EXPECT: kick/snare/grid clean (proceed); hat/ohat and clap/snare separation acceptable or merged; bass noisy (acceptable); key coarse (acceptable). If core drums/grid are poor, report clearly so we stop and fix extraction before training.

## Engineering notes

- Target the local M4 Pro. `madmom`, `librosa`, Demucs all run on it. Pin versions; note any that need `pip install --break-system-packages` equivalents in the project's setup.
- Keep stages decoupled (each takes audio + grid, returns tokens) so individual stages can be swapped/disabled and so the ADT model can be upgraded later (e.g. the MERT-conditioned Noise-to-Notes approach) without touching the rest.
- Deterministic output for a given input + config (fix any model seeds), so validation is reproducible.

## Deliverables

1. The extractor tool with the pipeline above.
2. The `validate` click-overlay harness.
3. A short README documenting the chosen contract defaults (drum classes, bass mode, velocity, key) — these defaults are what jtx (prompt 1) must match.
4. Do NOT wire this into decoder-swap. It is standalone until its output is validated.
