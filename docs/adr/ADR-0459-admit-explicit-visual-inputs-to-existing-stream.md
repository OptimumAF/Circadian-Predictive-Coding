# ADR-0459: Admit explicit visual inputs to the existing delayed stream

## Context

The accepted public RGB teacher emits 789D observations and its qualified
receipt is 1,390,451 bytes. The existing stream CLI reader requires 42D and caps
inputs at 1 MiB, so these labels cannot reach the already generic actor/shadow
runtime. That runtime already supports the dimension and owns complete state.

## Decision

Extend the existing CLI reader with an explicit visual mode. Require both inputs
to match the complete frozen RGB recipe, exact JSON field and element types,
789D spec and TRAIN17/19 identities, plus ordinary complete-episode criteria.
Refuse duplicate visual JSON fields before overwrite and impossible pixel-range
values. Keep physical public vectors in their original units. Allow at most
4 MiB per selected visual file, preserving the state path's 1 MiB cap and legacy
contract. Report selected mode, recipe and input cap only for the visual route.

Why this: measured receipt size demonstrates the required cap change; reusing
the reviewed native teacher output bound keeps it finite. Explicit selection
prevents silently mixing differently defined observations or enlarging all reads.
Keep IO/format validation at the CLI boundary, with no core/app/runtime rewrite.

## Alternatives

Broadly increasing the state cap or inferring visual mode from shape would hide
selection and weaken the existing contract. A new stream runner would duplicate
working label-arrival, budget and state-ownership logic. A physical-provenance
framework is unnecessary for this declared local receipt boundary.

## Consequences

Released visual pairs can use the ordinary one-packet, tick2 private update and
frozen serving path. Synthetic fixture replay demonstrates IO/runtime behavior
only; it does not qualify new native trajectories or useful visual adaptation.
Existing capture costs are prior evidence rather than current rendering work.
Native acquisition/retention/recurrence still needs a separate small frozen
protocol. No new dependency, encoder, learner, promotion or comparative claim.
