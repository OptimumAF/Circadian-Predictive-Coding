# ADR-0581: Reuse native component sleep under original consolidation authority

## Context

The finite action schema exposes sleep allowance but lacks an executor. NumPy
CircadianLearner already supports complete native component-sleep state through
ActorShadowRuntime.consolidate. TorchFeatureSnapshot deliberately describes a
fixed no-sleep model, so relaxing its clock/lineage checks would change its meaning.

## Decision

Add an optional native component-sleep preparation port to the existing NumPy
adapter, restricted to fixed width and disabled replay/split/prune. It prepares a
detached full native state using original homeostasis/reset logic. The existing
runtime owns the original consolidation quota, restore, commit, receipt and
wall/RSS checks. The existing resource wrapper accepts only a declared one-event
sleep action and enforces its original finite envelope and serving priority.

The unit is one admitted event attempt, distinct from optimizer updates. Separate
counters preserve native preparation and publication uncertainty; failed work
stays paid. Original consolidation metadata pins prior receipts/IDs/versions and
quota before more work. It stores no full-model success chain. Lost postcommit
metadata reads stop future work rather than inferring a successful publication.

Why this: the compatible native state and commit boundary already exist. A small
optional port enables the declared learner effect without a new runner, model
schema, budget or arbitrary controller callback.

## Alternatives and consequences

- Mutate Torch's no-sleep policy/clock schema: rejected; existing invariants and
  accepted snapshots keep their original meaning.
- Build another sleep/checkpoint framework: unnecessary; original native/runtime
  paths already stage and publish complete state.
- Allow replay/structural changes or mixed actions now: deferred; those require
  explicit exposure/capacity/ownership contracts and original budget accounting.

Scoped tests and one tiny native wake/sleep/wake case establish correctness, not
task adaptation or comparative benefit. Source/procedure/results self-review is
not independent acceptance. Full empirical/deployment/release requirements stay
open. Evidence: issue65 and
`artifacts/runs/bounded-component-sleep-20261023/RESULTS.md`.
