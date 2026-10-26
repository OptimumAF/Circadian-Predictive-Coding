# ADR-0501: Reusable simultaneous readiness corrections

## Context

The accepted fixed A15+B50+curatedA50 native correction recipe solves both known
TRAIN contexts. The matched old-information control recovers A and loses B.
The production single-batch correction API excludes curated saved labels; separate
reference replay has different update/data semantics. Useful behavior currently
depends on historical artifact-local runtime code.

## Decision

Add a narrow preparation module and separate runtime adapter using the existing
ActorShadowRuntime, RecordedContinuousHead and original-MSE native kernel. Admit
the three exact raw TRAIN families, release both corrections before projection,
preserve per-row raw/model lineage/clocks, and apply eight equal-row updates.
Return the existing detached-candidate result only on complete success; retain
bounded paid prefixes and failed work. Keep legacy APIs and qualification rules.
Core and app remain independent of outer adapters. No new IO/dependency/framework.

## Alternatives

- Import artifact helpers in application code: couples supported functionality
  to historical directories, mutable stage paths and experiment side effects.
- Coerce mixed provisional data into a qualified episode: loses authority and
  clock distinctions.
- Use separate reference replay: changes the experimentally established recipe.
- Extend the large single-batch adapter with unrelated branches: broadens the
  patch and obscures its current contracts.

## Consequences

The three-source correction step becomes callable through ordinary production
imports. The later explicit pre-curation extension below supplies the preceding
A+B stage; both-stage supported stream integration remains unfinished.
The first API is intentionally specific to the existing raw42/executedTS12
Panda TRAIN17/19 setup. More families/tasks require explicit typed admission and
tests. Synthetic software evidence does not re-establish native equivalence,
generalization, reliability, safety or PC/CPC advantage; those remain separate
scientific requirements with prospectively frozen budgets.

## Explicit pre-curation mode, 2026-10-21

The same required positional curated argument now accepts explicit None for the
preceding A+B stage. Why this: the accepted successful recipe includes both
eight-update stages; starting directly with three sources changes that recipe.
Every supplied curated batch retains its original arrival gate before projection.
No absent-C rows, predictions or clocks are manufactured. Both stages use the
same existing private runtime and native kernel. Supported stream integration
and exact numerical/action equivalence remain open after scoped software checks.

## Supported full stream composition, 2026-10-21

Add joint_readiness_stream as a fixed composition at the existing detached B
checkpoint observer. Why this: qualified A/B and saved provisional corrections
can now execute the complete accepted recipe without artifact-local runtime code.
Both release gates are admitted before base initialization, with explicit ordered
ticks. Two original eight-update scopes retain separate budgets/reports; base
A/B/A continues unchanged. A complete base and both branches are necessary to
release the adapted candidate. Duplicate preflight preparation is reported rather
than hidden. No new runner, persistence, promotion or model-family mechanism.

The prospective native-free comparison covers every preserved numerical result
and saved35command trace. Original historical matrices are unavailable, so it
does not claim independent bytewise matrix equality or a new physics run. Further
unseen-reset control/reliability evidence remains separate and must be budgeted.

## Reserved guard scoring boundary, 2026-10-21

Extend only the existing outer frozen-rollout and app history boundary with an
explicit guard option. Why this: native ports already reserve31/37, while the
outer path assumed TRAIN. Bind original role/context/reset/episode identity
through close; forbid correction/teacher/intervention modes before preparation.
Legacy TRAIN defaults and binding/report schema remain. A new runner or role
conversion would duplicate orchestration or grant unintended training authority.

Scoped four-family fake-port software checks pass independently on both
platforms. The native question remains incomplete after preserved construction
failure and one environment repair timeout:31 is a complete negative;37 unknown.
No reliable control, CPC or release claim follows. No further native retry is
included in this decision; next work is TRAIN-only support diagnostics.

## Explicit TRAIN support diagnostic, 2026-10-21

Add a separate read-only summary consuming existing JointReadinessData. Why this:
conditioning/residual diagnostics already exist; marginal recorded feature and
history support was missing. Making this an explicit call exposes its validation/
projection cost and preserves existing training/preparation reports. Recompute
from raw42/actualfloat32 commands, reset history per source, never inspect targets
or clocks, and retain original TRAIN structural bindings. This does not grant
qualification or release authority. No new orchestration, fit or dependency.

34 focused cases per platform and one frozen115-row description independently
accepted. Stage imbalance/absent curated-settle cell are descriptive TRAIN facts,
not causal attribution of guard failure or proof of a better method. Next proposed
sampling control changes empirical weighting explicitly under a separate frozen
protocol; per-batch MSE/default behavior and all original evidence stay intact.

## Recorded-stage control and negative result, 2026-10-21

Context: explicit TRAIN diagnostics measured18/12/5/66/14 stage rows. Decision:
add optional deterministic equal-stage sampling through a pure original-index
planner and existing runtime; retain default equal-row recipe. Why this: test a
conventional measured weighting hypothesis without changing loss, labels or
orchestration. All rows appear across eight batches; original causal history
stays attached to each row. Planning/gather and changed source/context exposure
are visible. Broadly changing default loss or filtering rows was unnecessary.

Consequences:90 focused checks/platform and final statics independently accepted.
The frozen control worsened all three final TRAIN losses, retained native A17
success but lost B19. Both original same-seed observations matched. Preserve the
option as experimental and keep the simpler unbalanced default. No quota/seed/
dose tuning or guard retry. Independent acceptance concerns scoped correctness
and negative findings, not physical fresh-task learning, reliability or CPC.
Next outcome is a native endpoint for the existing private fresh-task probe;
current probe has imitation readouts but no before/after observer.
[TRAIN result](../../artifacts/runs/stage-balanced-corrections-20261021/RESULTS.md) ·
[Native result](../../artifacts/runs/stage-balanced-known-context-20261021/RESULTS.md).

## Private fresh-probe physical endpoint seam, 2026-10-21

Context: existing corrected-source private SGD probe improves reserved-reset
imitation but has no physical before/after evidence. Decision: borrow its existing
readout synchronously through an explicit None-returning observer; reuse ordinary
frozen serving with narrow reserved probe_test24/history admission. Why this:
port already supports24; another runner or exported adapted checkpoint would add
unnecessary authority and orchestration. Original source/runtime/paid report and
budget witnesses are pinned before opaque observations and proved from native
fields after the last getters/clocks. Failures preserve prior paid work and stop.
Default schema/delegation unchanged; ambiguous zero-velocity pair refuses observer.

Consequences: borrowed readout must not be retained/trained/exported/promoted;
result contains no learner. Readouts/costs are caller-owned and separate from
probe imitation work. No labels enter the observer or test training path; final
remains unavailable.180 scoped checks/platform and final statics independently
accepted. A separately frozen fixed native24 before/after question is still needed
before any physical usefulness claim. Old native guard37 allowance remains closed.
[API](../continuous-probe-endpoints.md) ·
[Software evidence](../../artifacts/runs/continuous-probe-endpoints-20261021/RESULTS.md).

The separate fixed native24 question has now completed and is independently
accepted: both before/after fail50controls; imitation gains reproduce exactly,
all physical gains zero. Preserve this negative result and unchanged recipe.
Next measure physical old-A/B retention after the same update, using the existing
observer and a separate frozen two-scene protocol; no new method/seed selection
from probe-test outcomes. Full learning/reliability/CPC goals remain open.
[Native evidence](../../artifacts/runs/native-fresh-probe-endpoints-20261021/RESULTS.md).
