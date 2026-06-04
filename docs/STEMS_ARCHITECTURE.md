# Stems pivot — architecture design

**Status:** active, current direction as of 2026-06-04.
**Supersedes:** the M6.A / jtxtok-conditioning line of work (see `README.md` § Pivot history).

## The problem with the previous approach

The codec-token-LM pipeline asked one model to handle drums + bass + harmony + atmosphere
*jointly*. That single channel had to be:

- **Predictable enough** for an autoregressive LM to model
- **Faithful enough** that the LM's output, when decoded back through the codec, sounded real

These two pressures pulled in opposite directions. The LM's outputs were OOD for the codec
decoder (hum/glitch artefacts the fixer post-net only partially recovered). Mimi at low
bitrate sounded scratchy on music to begin with. Modelling the joint distribution of music
tokens turned out too hard at the small model + small corpus scale we had.

## The new framing

Split the problem along source-separation lines *before* choosing a representation, then
let each stem use the tool that actually fits its content type.

```
   audio
     │
     │  (1) Demucs htdemucs → 4 stems
     ▼
   stems: drums / bass / other / vocals
     │
     │  (2) per-stem extraction (semantic) + per-stem resynthesis (acoustic)
     ▼
   reassembly: sum of per-stem resyntheses
```

Empirical floor (validated 2026-06-04 on Beltram – Machine):
- Sum-of-uncompressed-stems vs original: Mel L1 0.18 dB, SI-SDR 36 dB → Demucs separation
  is effectively free.
- Per-stem DAC roundtrip sum vs full-mix DAC roundtrip: Mel L1 2.76 vs 2.62 dB → no penalty
  for going per-stem.

So the foundation is solid; the question is how to handle each stem.

## Per-stem representation mapping

| Stem | Content type | Representation | Resynthesis | Status |
|---|---|---|---|---|
| **drums** | Impulsive transient + reverb / processing tail. In techno specifically: heavy comp, reverb, sidechain, delay on hats. Per-hit extraction is hard because retriggering would stack reverb tails non-physically. | Onset MIDI + drum-type tags (kick / snare / hat / perc). **No** per-hit sample extraction. | Canned MIDI synth — fluidsynth + GM percussion soundfont. Doesn't preserve corpus drum character, but sidesteps the wet-tail problem entirely. Bar-at-a-time sampling is an alternative path (preserves groove character) but is parked for now. | M2 (issue #11) |
| **bass** | Pitched, often near-monophonic but with octave doubling / pump intervals. In acid-era techno: TB-303-style filtered/resonant content, fast 16th-note sequences. | Polyphonic MIDI (basic-pitch) with **per-note pitchbend curves** (basic-pitch already computes these; previously discarded). | Two paths: (1) **synth baseline** — saw + ADSR + LP filter, phase-accumulating saw honors pitchbend, used as A/B reference; (2) **target — per-corpus DAC-codec-LM** conditioned on MIDI+bend (+ exemplar). | M1 (issue #10) |
| **other** | Textural, polyphonic, sound-design-heavy. In acid-era techno: stacked detuned saws, supersaws, FM bells, filter sweeps. **Not** ambient pads — high-energy rhythmic synth content. | Codec tokens (DAC) + simple conditioning (per-frame loudness, key class, rhythm density). MIDI is the **wrong** representation: basic-pitch on `other.wav` produces ~1100 notes spread across 5½ octaves, shredding one or two synth voices into a polyphonic-octave pile-up. | Per-corpus DAC-codec-LM (same architecture as bass but with different conditioning) | Future (after M1) |
| **vocals** | (Skipped for instrumental corpora) | — | — | N/A |

## Codec choice — DAC

DAC 44 kHz / 9 cb at 86 fps is the codec for both bass and `other` codec-LM work. Reasons:

- **Music-trained** (Mimi is speech-trained; Mimi-at-low-bitrate sounded scratchy on music
  in the previous line of work).
- **Higher temporal resolution** (86 fps vs Mimi's 12.5 fps) preserves the spectral
  evolution needed for filter sweeps, attack character, sidechain pumping.
- **Already wired up in this repo** — `codec_io.py`, `scripts/07_cache_translator_tokens.py`,
  parallel-RVQ LM scaffolding.

## Why MIDI for some stems but not others

It's about whether the content is **note-shaped**:

- A bass line is a sequence of discrete pitched events with clear onsets → MIDI is the
  natural representation, and a generative model can fill in *how* each note sounds given
  the corpus distribution.
- A synth pad with a 4-bar filter sweep is *not* a sequence of notes — it's one continuous
  spectral morph. Trying to encode it as MIDI shreds it into ~5 notes/sec of chips that
  don't capture the actual morphing texture.

The honest question per stem is "is the content note-like or texture-like" and codec-LM is
the only universal answer because codec tokens encode whatever-is-there without
assumptions about events.

## How "continuous modulation" arises in the codec-LM path

A clarification because this comes up: in the codec-LM bass path, things like filter
sweeps don't come from MIDI CC. They emerge because:

1. Codec tokens at 86 fps encode the audio's instantaneous spectrum frame by frame.
2. A filter sweep is just a sequence of frames whose spectra evolve smoothly — nothing
   special about it from the codec's perspective.
3. During training, the LM sees pairs like (MIDI: "C2 attack") + (codec tokens: spectrum
   that starts bright, sweeps down over ~15 frames, lands resonant). It learns "in this
   corpus, C2 attacks are typically followed by this kind of spectral trajectory."
4. At inference, given new MIDI, the model generates codec frames that include those
   learned trajectories — interpolating between seen examples rather than copying them.

So the architecture is **MIDI = what plays when, codec-LM = corpus-typical "how it sounds"
filled in by the learned distribution**. The model is a per-corpus, MIDI-conditioned,
neural synth voice. Mutation knobs (transpose, re-time, swap exemplar) live in the MIDI
domain; timbral fidelity to the corpus comes from the LM.

If we later want explicit user control over modulation, MIDI CC can be added as additional
conditioning. For the initial product target (corpus-style generation), the model learning
modulation from the data is exactly what we want.

## Corpus framing

The user's eventual target is a "mini corpus" of 1–100 songs, single-style, no vocals.
Examples: the Mayday compilation, a Vytis DJ set. The narrow-style assumption is
load-bearing:

- A small generative model can fit a narrow distribution with much less data than a
  general-music model would need.
- The "fit in to the corpus" quality target (per memory file
  `feedback_jamtronix_quality_target.md`) is well-defined when the corpus is stylistically
  coherent.
- More data isn't strictly better — diversity in source is a *negative* for corpus-fit
  (per memory file `feedback_corpus_fit_goal.md`).

Per-corpus YAML configs (`corpora/<name>.yaml`) already exist for the techno corpus and
the convention scales to additional style-coherent corpora as added.

## What's parked but available to reuse for 1B

- `codec_io.py` — DAC loading + encode/decode
- `scripts/07_cache_translator_tokens.py` — corpus → DAC token cache
- `src/decoder_swap/translator_rvq.py` — parallel-codebook AR scaffolding
- `src/decoder_swap/fixer.py` + GAN variants — post-net if codec-LM artefacts intrude
- Per-corpus YAML convention in `corpora/`

The build for M1 is largely a re-target of M6.A scaffolding (bass-only + MIDI conditioning
instead of full-mix + jtxtok conditioning), not a from-scratch infrastructure project.
