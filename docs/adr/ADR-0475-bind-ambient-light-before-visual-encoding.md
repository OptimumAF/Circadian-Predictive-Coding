# ADR-0475: Bind ambient light before accepting visual observations

## Context

Full R4.5 explicitly requires lighting drift alongside camera, object and bounded
dynamics changes. Camera/mass interfaces exist. The fixed mass MLP ablation found
no aggregate B-label benefit over equal A-only continuation, so another mass
optimization variant does not address the missing visual domain axis.

The pinned SDK exposes scene ambient light as three float32 values, a public
setter and `get_obs()` without a control step. Reset already returns a rendered
observation. Changing light afterward cannot relabel those old pixels as changed.

## Decision

Add one immutable visual TRAIN17 half-ambient declaration. Compose it at the
existing episode boundary with a focused infra helper for native read/write and
refresh accounting. Read the actual scene after explicit reset, set its exact
representable float32 half once, and verify it. Discard the reset packet and pay
for one explicit observation refresh before the old encoder accepts any pixels.

Capture expected measurements and IO identities before external callbacks and
recheck them afterward. Verify light before controls, post-control encoding and
episode export. Refuse changed bindings or measurements permanently. Keep owned
original readback and constant-size counters; failed construction retains its
original error, partial request and paid work even if cleanup also fails.

Defaults do not resolve a scene, read/set light or refresh observations. Do not
combine this fixture implicitly with camera/mass interventions. The frozen RGB
recipe stays unchanged; the distinct context names the environmental intervention.

## Alternatives

- Relabel reset pixels: invalid because they preceded the intervention.
- Override the simulator task or reset lighting loader: larger factory surface
  than the public setter/refresh contract needed here.
- Add another adaptation runner: unnecessary; existing episode and stream ports
  remain the integration path after this prerequisite is actually verified.

## Consequences

One extra render is explicit. Existing visual work counts encoding attempts and
owned frame copies; lighting work separately counts discarded reset observation,
refresh, reads and writes. Constructor/reset/renderer/evaluation/cleanup overhead
and hidden physics remain unmeasured, not zero. Only ambient light is deliberately
set; directional-light or physical invariance is not claimed from intent.

Synthetic gates prove boundary behavior. A separately frozen native check must
observe the API and pixels before native support is accepted. Learning, native
return, recurrence curves, constraint satisfaction, neural generalization and
PC/CPC benefit remain separate open requirements.

## Integration follow-up, 2026-10-19

Carry the same typed context, verified original readback and paid work through
existing teacher and frozen-rollout interfaces. Default CLI fields stay unchanged;
Python result dataclass appends three None fields. Successful qualification checks
complete visual/controller/light lifecycles; partial failures retain valid paid
costs, including coherent frame exports, without qualifying for success.

Final metadata checks follow all owner/getter callbacks. Compare original
measurements and cumulative counters with owned port storage through one pure
check, avoiding a new live scene read or callback cycle. Shared visual-work
validation removes duplicate checks and prevents false rendering costs. This is
a concrete existing-boundary requirement, with no new orchestration framework.
411 focused platform cases establish scoped behavior; no new native/training
experiment or learning benefit follows from this integration acceptance.


## Follow-up: saved lighting input in the existing stream (2026-10-19)

The existing CLI now explicitly admits qualified visual TRAIN17 lighting receipts.
One bounded owned payload supplies both released pairs and original acquisition
metadata; current live TaskTeacherResult qualification remains authoritative.
Complete serialized context fields are required before Python constructors, since
dataclass defaults cannot establish omitted evidence. The same small field check
also closes that demonstrated gap in mass inputs. Defaults/None fields remain
compatible; mixed/heldout data refuses before fitting. Relabelling B as seed19 or
adding another runner was rejected because either loses identity or duplicates the
existing runtime. Input correctness grants no new learning/native authority; jointly
freeze treatment and equal-exposure A-only control before either comparative fit.
