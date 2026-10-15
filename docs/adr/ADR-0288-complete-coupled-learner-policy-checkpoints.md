# ADR-0288: Pair complete native checkpoints with learner policy

## Context

Complete native CoupledSnapshot files preserve PC/CPC model state, while the
owned CoupledLearner keeps update rates, inference cap/rate and guarded policy
separately. Native-only restoration therefore needs externally fixed settings
and cannot by itself represent a complete reusable learner checkpoint.

## Decision

Add explicit core policy/checkpoint records, a model-free detached policy getter,
and a separate bounded canonical envelope that embeds the existing native body.
Subtract exact metadata overhead before native codec work. Decode policy and
complete native bytes before explicit model reconstruction and the normal owned
learner constructor. Preserve the existing native snapshot and learner APIs.

Why this: composition reuses the established array codec and file-store boundary,
while retaining the settings required for the next update. No second byte budget
or implicit recovery authority is created. Guarded null-versus-policy semantics,
CPC chemistry and native array bytes remain explicit.

## Alternatives

Changing CoupledSnapshot would broaden the native model contract and affect
existing callers. Leaving settings in caller code would keep checkpoints
incomplete. Generic pickle/object reconstruction would bypass explicit bounded
schemas. Silently converting a lossy manual integer rate would change policy;
the restore factory instead refuses it before constructing a model.

## Consequences

No new dependency, learner algorithm, seed selection or scientific metric.
Capture remains a trusted quiescent operation; encoded bindings are consistency
checks, not authority. Full managed/runtime/lifecycle/actor recovery remains a
subsequent integration boundary. Fixed PC/neutral/active continuation under both
existing inference modes validates this implementation without benefit claims.
