# Build prompts — jamtronix pipeline

The three Claude Code prompts that define the components of the
[jamtronix](https://github.com/jonnosan/jamtronix) pipeline. Copied here from the planning
artefacts so the consuming repo carries its own contract.

| Prompt | Component | Repo | Status |
|---|---|---|---|
| [PROMPT_1_jtx_emit.md](PROMPT_1_jtx_emit.md) | `jtx → jtxtok` emitter (inference-time conditioning source) | jtx | not built |
| [PROMPT_2_extractor.md](PROMPT_2_extractor.md) | `audio → jtxtok` extractor (training-time conditioning source) | separate tool, gating-risk | not built |
| [PROMPT_3_decoder_swap.md](PROMPT_3_decoder_swap.md) | jtxtok-conditioned generation (this repo) | decoder-swap | tracked in issue #8 |

The shared contract is [`../JTXTOK_SPEC.md`](../JTXTOK_SPEC.md). Both producers (PROMPT_1, PROMPT_2)
must emit byte-identical vocabulary; the consumer (PROMPT_3) is producer-agnostic.

**Build order** (per spec §9): PROMPT_2 first (riskiest, standalone-validated), then PROMPT_1
(low-risk, needed for inference), then PROMPT_3 (depends on PROMPT_2 output being trusted; needs
PROMPT_1 only for the inference path and for synthetic MT sweeps in training).

The current decoder-swap unconditional base run is the model PROMPT_3 extends; it does not
require restarting when PROMPT_3 begins.
