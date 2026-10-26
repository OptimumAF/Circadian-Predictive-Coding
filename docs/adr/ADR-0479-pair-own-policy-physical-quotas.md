# ADR-0479: Pair recurrent policies at equal physical exposure

## Context

R4.3 requires changed controls, own-policy trajectories and equal interaction
allowances. The existing reward and frozen recurrent quota routes already cover
each arm. The older paired BC/RL mode has a different warm-start contract and
refuses action remapping. Legally forced corridor actions can succeed without
learning, so frozen initialization is a necessary control for attribution.

## Decision

Compose the existing `_run_arm` in a small quota comparison adapter. Frozen and
trained arms have distinct owners, samplers and original budgets. Exact complete
native initialized state must match before the trained arm opens a runtime or
collects work. A private optional initial-state argument permits quota deferral;
old calls keep their previous behavior.

Both arms finish their own 96 TRAIN interactions before this pair obtains final
packets or executes readouts. Compare every context/cue at physical boundaries
0/24/48/72/96 while retaining actual optimizer indices. Keep at most two paid row
seals and eight shared final packets; check preceding owners/evidence before more
work and after resource callbacks. Failures keep costs and partial reports.

## Alternatives

Common teacher trajectories would not measure different on-policy behavior.
Reusing the failed supervisor qualifier as an accepted control would change its
status without evidence. A new learner, environment or orchestration framework
is unnecessary. Updating the existing fork implementation merely to lower the
measured cost would add an unrelated state-management change to this comparison.

## Consequences

Equal physical exposure does not imply equal updates, resets or computation.
Existing recurrent forks initialize before restoring, so count those Gaussian
draws and allocations as well as source initializers. Raw categorical CE is a
diagnostic distinct from conditional live success. Fixed catalogue instances are
not independent random seeds; no universal, CPC or causal advantage follows.

## Timing follow-up after the incomplete paired study

The first actual pair exhausted its shared90-second body during evaluation.
Add bounded phase/guard wall counters to the existing adapter using a separate
performance clock. Keep telemetry outside paid rows and publish it after work.
Nested timings overlap and include measurement overhead in the original budget;
no observer may extend that budget. Deterministic-clock tests verify exceptions,
invalid intervals and unreached phases. This justifies instrumentation, not a
study retry, a changed metric or relaxed evidence checks.

## Prospective wall policy amendment

The original90s paired attempt was incomplete; initialized fork/snapshot timing
did not establish a dominant cost. Why this: expose a bounded prospective cap
(default90/max180) instead of silently changing a default or renewing an existing
session. Capture one deadline before both arms; retain identical quotas, metrics,
final coverage and all failed attempts. Any numerical resource retry must be
explicitly amended, frozen and reviewed, with cumulative costs and no claim of
independent replication. A larger cap alone neither guarantees useful behavior
nor explains the original deadline failure.

## Recorded presentation follow-up, 2026-10-20

Decision: reuse the existing standalone inspector and inject its episode
projector into a pure paired-source binder. Separate quota ordinals bind missing
raw physical labels without rewriting evidence. This avoids a parallel inspector
or a circular dispatcher dependency. Preserve raw/conditioned probabilities and
map only applied controls, since observations use spatial directions. Costs and
comparisons remain recorded presentation, never training or selection input.
Full topology/lineage/sleep integration remains open; no scientific claim changes.
[Evidence and resource exception](../../artifacts/runs/paired-inspector-20261020/RESULTS.md).

## Selected saved native state, 2026-10-20

Bind each observation by its original physical ordinal, then validate native
updates: frozen allfive indices are0, so native-index lookup is ambiguous.
Validate actual arrays/counters before display without model reconstruction.
Preserve legacy partial unbound rawobservations but withhold their stagebinding;
clear old fields on missing selection. Array shapes are not lineage/sleep events.
[Source, executed checks, failed launcher and bounded repair](../../artifacts/runs/paired-state-view-20261020/RESULTS.md).
