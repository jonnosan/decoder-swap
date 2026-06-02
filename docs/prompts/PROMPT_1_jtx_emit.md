# Claude Code prompt — extend jtx to emit `jtxtok`

> Read `JTXTOK_SPEC.md` in full before starting — it is the contract this emitter must conform to byte-for-byte. This emitter and the `audio→jtxtok` extractor (separate tool) MUST produce identical vocabulary; mismatches silently corrupt training/inference. Where this prompt and the extractor's documented defaults disagree, the extractor's defaults win (it is the constrained producer) — match them.

## Context

jtx already builds a song internally as a sequence of structured events ("snare hit", "pad chord Am") and only lowers to MIDI as a final export step. This task adds a NEW output path that serialises that internal event sequence directly to jtxtok v1 — **not** via MIDI. The existing MIDI export stays untouched as the DAW path.

jtx is the *inference-time* conditioning source: at generation time it drives decoder-swap. It is NOT used for training (the extractor handles that). So this emitter's job is faithful, complete serialisation of what jtx knows.

## What to build

A jtxtok emitter that walks jtx's internal event sequence and outputs v1 tokens per `JTXTOK_SPEC.md`.

### Tokens to emit (spec §4)

- Structural: `BOS` … `EOS`, `BAR`, `POS_0`…`POS_15` (16th grid, 16/bar).
- Drums: **5-class v1 — `DRUM_KICK`/`DRUM_SNARE`/`DRUM_HAT`(closed)/`DRUM_OHAT`(open)/`DRUM_CLAP`**. Map jtx's percussion voices to these five. jtx should have distinct internal voices for open-hat and clap — if it collapses clap into snare or open-hat into hat internally, decide whether to add the voice or map it (and match whatever the extractor settled on in validation). Match the extractor's final class set exactly.
- Bass: `BASS_ON` (or `BASS_P_<pc>` if the extractor's contract uses pitch — match it).
- Harmony: `KEY_<pc>` (root only, 0..11, C=0), bar-level, on-change.
- **`MT_<n>`** (micro-timing, −8..+8): **jtx emits this fully** — it is the authoritative micro-timing source (spec §2.1). Read jtx's per-event timing offset within each 16th cell and emit the corresponding `MT_*`. Map jtx's internal offset units to the −8..+8 range (±8 = ±half a cell). This is the whole point of having jtx drive groove at inference, so do not skip it.

### Ordering (spec §5)

`BOS BAR [KEY_<pc>] POS_<p> <events...> POS_<q> ... BAR ... EOS`. Within a position: drums, then bass. `MT_*` immediately follows the event token it modifies (e.g. `DRUM_HAT MT_+4`).

### Hard constraints

- **No MIDI round-trip.** Read jtx's internal events directly.
- **Emit the FULL tagged stream** — all in-contract roles, every voice. Do NOT implement any mode/voice-filtering/conditioning-strength logic here. Voice selection and CFG are decoder-swap's job (spec §7.3). jtx always emits everything.
- **In-contract only** on the jtxtok stream. Out-of-contract jtx-internal detail (algorithm ID, modulator/CC curves, follower metadata, poly/pad voicing, lead/melody note content — spec §3) MUST NOT appear. If you want to preserve it for debugging, write it to a SEPARATE sidecar stream, never into the `.jtxtok`.
- **Same token spellings** as the extractor. Treat the extractor's README defaults as the source of truth for class set / bass mode / velocity / key.

### Quantisation

- Confirm jtx's internal events are 16th-aligned (they should be, given bar-by-bar regen). Snap positions to the 16th grid.
- Micro-timing within the cell goes to `MT_*`, NOT to grid position — placement and feel are separate (spec §2.1).
- Sub-16th structural events (ratchets/flams), if jtx produces them, use the `FINE_<n>` escape token (spec §2); otherwise quantise.

## Output

- `.jtxtok` text file per song: whitespace-separated tokens, newline per bar.
- Reproducible: same jtx `(song, seed)` → identical jtxtok (jtx is already deterministic — preserve that).

## Verification

- Round-trip sanity: emit jtxtok for a few known jtx songs, manually check a bar against jtx's debug log (which already reads like "bar 1 beat 1 kick drum") — the tokens should match the log's structure.
- Cross-producer check: pick a jtx song, render it (jtx's existing audio path), run that audio through the extractor, and compare the two jtxtok streams. They will NOT match on `MT_*` (extractor emits none) or exact drum detail (extraction is lossy), but the BAR/POS grid and gross drum pattern should align. Large structural disagreement signals a vocabulary/grid mismatch to fix.

## Deliverables

1. The jtxtok emitter on jtx's internal event path.
2. The optional jtx-internal sidecar (out-of-contract detail), clearly separate.
3. A note confirming which extractor contract defaults were matched (drum classes, bass mode, velocity, key) and how jtx's timing units map to `MT_*`'s −8..+8.
