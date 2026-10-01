# ADR-0158: Validate all final views before fixed endpoint evaluation

Date: 2026-10-01
Status: Accepted for P6.7c1b implementation; actual scored execution gated.

## Context

P6.7c1a binds both complete training results and the unchanged analysis
declaration. Its state proof has no file/source/resource authority. Existing
final release returns a separate role view, leaving original roles sealed.
The independent endpoint contract requires A-after-A, A-after-B and B-after-B
for every original cell. Original models all threshold probabilities at 0.5.
No reserved confirmation final value has been opened.

## Decision

Use a small pure core contract for released final arrays, ordered IDs, the
original role digest and typed endpoint counts/failures. App orchestrates
global state gates, final-role pairing, the fixed three evaluations, explicit
null cells and full post-evaluation checks. A new infra adapter composes the
unchanged final release and existing model prediction. No old source changes.

Before any release, verify the complete reproduced training state. Release
both phases for every held row, bind original final IDs/counts/phase/seed and
independently recompute the original ASCII-ID/little-endian-float64/shape hash.
Equal phase/seed source identities (including gating/replay reuse) must have
equal final signatures. Recheck all released views and training state before
the first evaluation. Keep original roles separate and sealed.

Evaluate every scheduled endpoint, in frozen family/seed/arm/endpoint order,
using its declared held checkpoint and final phase. Preserve each exact
correct count and example count. Accuracy is correct/count under the unchanged
binary threshold; no clipping, rounding, model/seed/metric selection or tuning.

Declare numerical prediction failures prospectively: nonfinite probabilities
or `FloatingPointError` produce a typed failure/null endpoint with a stable
code and optional exception type. All three calls are still attempted. A
cell with any failed endpoint retains all its raw endpoints/role identity,
an explicit failure and null aggregate accuracy. Other exceptions, invalid
metadata/results/counts/probability shape/range, callback/resource failures
and any state/content drift abort the gate; no partial scientific success.
Do not retry, drop a cell, substitute an accuracy or reduce the scope.

Globally verify training state and all original/released roles again after
evaluation before returning a result. App records its local release/call/
example/failure totals; the later process boundary must independently observe
actual calls/work and establish all source/request/file/resource provenance.
Checkpoint callbacks allow that boundary to stop at fixed barriers/calls.

## Alternatives

- Score each role immediately after release: a late role or state error could
  occur after earlier accuracies without a global released-view gate.
- Broadly catch exceptions as failed accuracy: could mask broken contracts,
  changed models or resource limits; keep those as gate failures.
- Return only three floats: loses exact discrete counts and individual
  numerical failure evidence needed for full-scope analysis and readback.

## Consequences

No new scientific setting, threshold, metric, seed, budget or baseline change.
Source identities are frozen before any fabricated scored fixture. Tests use
the original development training with raising original-final/outer/reserved
sentinels, fabricated final arrays/counts, and explicitly labeled whole-scope
delegation spies. They prove correctness, not actual reserved execution or
resource use. C1c's complete source/request/process/artifact gate remains
required before c1/c2; P6.11b still owns actual all-seed reports.
