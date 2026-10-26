# ADR-0328: Preserve reserved roles in complete episode probes

## Context

The isolated fresh-task probe has correct binary fork/budget isolation, but its
binary scorer cannot consume recurrent five-action episodes. Complete episode
records originally allow only benchmark train/final-test roles; converting those
records into probes would blur the scientific data boundary.

## Decision

Explicitly add reserved probe_train/probe_test record roles. The existing probe
requires exact batch/record role, context, arrays, IDs and class-count agreement,
one count across the full envelope and complete metadata on all packets or none.
Use one ordered complete episode per native call. No automatic benchmark-role
conversion, final-data release or remembered-cue injection.

Dispatch K>=3 to existing categorical CE/argmax metrics; preserve original K2
scalar binary metrics and reports. Keep exacttrain replay encoding, benchmark
ingress and native train_episode gates intact. The existing disposable probe app
alone delegates validated reserved packet arrays to its owned learner fork.

Report owned episode IDs/transition counts with explicit record/Python/native
memory exclusions. Reuse original arrays/opaque-state detachment, permanent
charges, checkpoint/timing and partial-failure rules. Add no runner or authority.

## Alternatives

- Re-tagging benchmark final data as probe_test would weaken evaluation isolation.
- Flattening independent episodes would erase recurrent reset boundaries.
- Inferring binary/categorical semantics from prediction shape would silently
  change the frozen metric. K2 remains explicitly scalar binary.

## Consequences

The same isolated probe now measures categorical complete episodes, including a
native recurrent integration. Role and class refusals remain early and explicit.
Logical array bytes exclude retained record/native copies; future experiments
must account for full process work/memory too. Component correctness is not a
scientific learnability or navigation benefit; a separate frozen protocol remains
required, and all prior negative results are preserved.
