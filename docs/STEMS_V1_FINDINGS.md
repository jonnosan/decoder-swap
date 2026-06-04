# Stems pivot — v1 empirical findings (2026-06-04)

Single-day session that motivated and validated the stems-pivot architecture. All on one
song: *Beltram – Machine* (instrumental techno, 3 min 57 s) from the Mayday compilation.

## Step 1: per-stem DAC roundtrip vs full-mix DAC

**Question:** does decomposing audio into Demucs stems and re-summing add fidelity cost
worth caring about, compared to leaving the full mix intact?

**Setup:** Demucs htdemucs → 4 stems → DAC 44 kHz / 9 cb roundtrip each stem (stereo L/R
independently to preserve image) → sum, plus a full-mix DAC roundtrip baseline. Three
reference points compared to the original on mono downmix:

| Reference | Mel L1 (dB) | LSD (dB) | SI-SDR (dB) |
|---|---:|---:|---:|
| R1 sum(uncompressed stems) — *separation-only floor* | 0.181 | 12.6 | **36.4** |
| R2 sum(DAC-roundtripped stems) — *the test* | 2.760 | 13.7 | 1.28 |
| R3 full-mix DAC roundtrip — *codec-only baseline* | 2.623 | 9.0 | 1.30 |

**Findings:**

- Demucs separation cost is negligible (R1 vs R0).
- Per-stem and full-mix DAC roundtrips are tied on Mel L1 and SI-SDR. LSD favors R3 but is
  fragile in low-energy bins.
- SI-SDR ~1 dB on both DAC paths is the usual DAC phase-distortion penalty, not a
  perceptual judgement.

**Conclusion:** Foundation is solid for adding the semantic-token layer. Going per-stem
costs nothing relative to the equivalent full-mix codec roundtrip.

## Step 2: bass → MIDI → resynth → recombine (v1, pYIN)

**Setup:** pYIN on `bass.wav` → 513 quantised-pitch note events → pitch-shift one exemplar
(longest sustained note) to each note pitch, time-stretch to note duration, sum.

**User listening verdict:** the **exemplar pitch-shift** path sounded "awful" (the auto-
selected exemplar was a 3.15 s outro sustain at D♯1, requiring pitch-shifts of >30 semitones
for many notes). The **synth path** (saw + ADSR + LP at 1.2 kHz) sounded "pretty good"
both in isolation and in the mix.

**Architectural implications:**

- **Exemplar pitch-shift is the wrong fit for bass.** At sub-bass frequencies (~40 Hz) the
  audio content is rumble; there's no harmonic content to carry across pitches via
  pitch-shifting. No amount of sample-quality polish recovers this.
- **Generic synth voice is a viable placeholder for any pitched-monophonic stem** while the
  full codec-LM is built. Bass timbre is genre-conventional enough in techno that saw +
  ADSR + LP slots in without offending.

## Step 3: pYIN is wrong for polyphonic bass

**Diagnostic:** user noticed the resynth was monotone where the original "pumps" with
octave / fifth jumps on roughly every 4th note. Diagnostic script ran a CQT analysis on a
10 s window and showed:

```
C1   ████████████████████  ← root, present everywhere
C♯1  ███████████████████   ← (smearing of C1 in the CQT)
C2   ███████████████       ← THE PUMP — pYIN missed entirely
```

pYIN reported all 33 notes in the window at MIDI 27 (D♯1) — wrong fundamental *and*
missing the octave-up entirely. pYIN is single-pitch and biased toward the lowest
fundamental, so for any bassline that's "drone + rhythmic higher voice" it collapses to
the drone.

**Resolution:** swap pYIN for **basic-pitch** (Spotify's polyphonic audio→MIDI). On
Beltram bass with pitch range clamped to MIDI 24..50 (D3 cap):

- 373 notes instead of 513
- Modal pitch **C2** (212 notes) — pYIN was wrong about the root by ~3 semitones
- **D♯2** (98 notes) — the "pump" is actually a **minor third** above C2, not a fifth or
  octave as guessed by ear
- G2 (15 notes), C1 (15), D♯1 (19), F1 (11) — small ornaments and the sub-bass drone

Engineering notes:
- basic-pitch 0.3 calls `scipy.signal.gaussian` which was removed in scipy 1.13+; shimmed
  with `scipy.signal.windows.gaussian`.
- Default TF backend fails to load under TF 2.16 (Keras 3 incompatibility); switched to
  ONNX backend via `FilenameSuffix.onnx` in the build path.
- basic-pitch also computes per-note pitchbend curves which were initially discarded;
  later captured (see Step 5).

## Step 4: pipeline driver — repeatable end-to-end

Single-command driver (`scripts/40_run_pipeline.py --in <audio> --slug <name>`) that runs:

```
30 separate_stems        Demucs htdemucs       ~30 s
33 bass_to_midi          basic-pitch + bend    ~5 s
37 extract_bass_exemplars score + polish        ~1 s
34 midi_to_bass_exemplar  exemplar resynth      ~2 s  (A/B baseline)
36 midi_to_bass_synth     saw synth + bend      <1 s
35 reassemble_swap_bass   sum stems             <1 s
```

Idempotent: each stage skips if its output exists. Re-runs after a code change touch only
affected stages.

## Step 5: pitchbend capture

basic-pitch returns per-note pitch-bend curves as integer bin offsets (1 bin = 1/3
semitone = ~33.33 cents) at ~86 fps. These were initially discarded; wired into the JSON
output as `pitch_bends: [{time_s, cents}]` per note, and into the `.mid` file as standard
MIDI `PitchBend` events (±2 semitone range).

Synth (script 36) updated to use a phase-accumulating sawtooth oscillator with
per-sample instantaneous frequency derived from the bend curve, so the synth's pitch
drifts smoothly along the captured curve instead of sitting on a fixed pitch.

Beltram bass pitchbend stats: 373/373 notes have bend; max abs deviation 167 cents; abs
median ~0 cents; abs 90th percentile ~33 cents. So most of the bend is subtle (within
⅓ semitone) with occasional larger excursions — consistent with TB-303-style portamento /
LFO modulation in the original synth.

## Step 6: Option C — basic-pitch on `other.wav`

**Question:** does the MIDI approach generalise from bass to `other` (the synth-heavy
stem), or does it fail in a way that justifies committing to the codec-LM build?

**Result:** with pitch range opened to MIDI 24..96 (C1..C7), basic-pitch produced:

- **1102 notes** in 4 min (3× the bass count)
- Spread across **5½ octaves**
- Top 4 pitches were **A♯ at every octave** (A♯4: 230, A♯2: 111, A♯5: 101, A♯6: 86)
- Median note duration 128 ms (shorter / choppier than bass)
- 1102/1102 notes with pitchbend
- Max abs bend 133 cents

**Interpretation, with user correction:**
- The user clarified `other` here is *not* sustained pads — it's high-energy
  early-90s acid with at least two distinct synths, one of which sounds like stacked
  slightly-detuned sawtooths.
- 4.6 notes/sec is plausibly correct for 16th-note acid sequences at ~130-140 BPM × 2
  synth voices.
- A♯ at multiple octaves is partly real (two synths at different registers playing the
  same key class) and partly basic-pitch's octave-hallucination from rich saw harmonics.
- "Pitchbend on every note" is partly real (filter sweep interpreted as continuous bend)
  and partly artefact.

**Conclusion:** MIDI is the wrong representation for `other` *regardless* of whether the
content is sustained-pad or rhythmic-acid. The fix isn't to detect "is it pads or acid";
it's that codec-LM captures whatever timbral specificity is present (supersaw stacks,
303-style resonance sweeps, FM bells) without trying to symbolise it. The codec already
knows what those sound like in audio; conditioning info tells it when to produce that
texture, not how.

This also reinforces the architectural call for **bass to go DAC-codec-LM** (not just the
saw-synth baseline) — acid-era bass is often TB-303 lineage, which a generic saw+LP can't
capture.

## Findings consolidated into the architecture

1. **Demucs separation is free.** Per-stem extraction is the right starting point.
2. **MIDI is for note-like content** (drums hit-positions, bass note events with pitchbend).
   It's the wrong tool for texture-heavy content.
3. **Codec-LM is for textural/timbral content.** Per-corpus DAC-codec-LMs scoped to one
   stem at a time are what was previously failing in the M6.A full-mix approach.
4. **Per-stem-per-corpus matches the user's quality target** ("fits in to the corpus"
   generation, not faithful reproduction).
5. **basic-pitch with ONNX backend + scipy shim** is the polyphonic audio→MIDI tool.
6. **Pitchbend curves are free with basic-pitch** and meaningfully change the synth
   character once propagated through to the resynth.

These findings are the empirical basis for the architectural commitments in
`docs/STEMS_ARCHITECTURE.md`.

## On-disk artefacts from this session

- Scripts: `30_separate_stems.py`, `31_dac_roundtrip_stems.py`, `32_compare_stems.py`,
  `33_bass_to_midi.py`, `34_midi_to_bass_exemplar.py`, `35_reassemble_swap_bass.py`,
  `36_midi_to_bass_synth.py`, `37_extract_bass_exemplars.py`, `40_run_pipeline.py`,
  `diag_bass_pitches.py`
- Stems: `data/song_test/beltram_machine/stems/{drums,bass,other,vocals}.wav`
- DAC roundtrips: `data/song_test/beltram_machine/stems_dac/`,
  `data/song_test/beltram_machine/full_dac/`
- MIDI: `data/song_test/beltram_machine/semantic/bass.{json,mid}`,
  `data/song_test/beltram_machine/semantic/other.{json,mid}`
- Bass resynth variants: `stems_resynth/bass.wav` (exemplar), `stems_synth/bass.wav` (synth)
- Listening A/B set: `results/stems_v1/beltram_machine/swap_bass/`,
  `results/stems_v1/beltram_machine/bass_samples/`
- New deps added to `pyproject.toml`: `demucs>=4.0`, `pretty_midi>=0.2`, `basic-pitch>=0.3`
