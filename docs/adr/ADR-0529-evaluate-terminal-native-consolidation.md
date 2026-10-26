# ADR-0529: Evaluate terminal native consolidation in the existing stream

## Context

The owned two-state operator passes scoped state/math/software checks. Its task
effect remains unknown. The accepted warm A6/B6 control retains A and fails B;
the earlier after-A reset acquires B but loses retention and anchor tolerance.
Both adverse results remain fixed evidence.

## Decision

Add default-off consolidation rank to the existing optimizer-control runner.
Bind accepted initial/A6 weights, capture one reference, continue unchanged warm
B6, bind accepted after-B weights, then apply fixed half/half complete-operator
consolidation at rank8 with one terminal reset and six original readouts.
Keep the current scalar head and all original metrics, directions and criteria.
The fixed-hash reference reader exposes after-B weights only on explicit request.
Keep receipts/costs before callbacks and preserve paid failures.

Why this: the warm endpoint matches TRAIN and optimizer history up to the
intervention. A terminal reset with no later TRAIN preserves inference weights.
The previous after-A reset arm cannot isolate this terminal operator's effect.

## Alternatives

- Another owner/capture increment: does not answer task usefulness.
- After-A consolidation followed by reset B6: changes the tested question and
  needs a separate prospective protocol.
- Rank/weight/dose search: exceeds this one-case allowance and risks selecting
  a favorable development endpoint.

## Consequences

No new framework, dependency or core/app interface. Existing defaults remain.
The primary control is historical and capture/algebra/proof work is additional;
equal TRAIN exposure is not equal total compute. The current operator already
contains A history, so these are correlated complete operators, not independent
task deltas. Allocated rank8 does not establish numerical rank or memory saving.
Inherited temperature, no return phase and one development case limit inference.
Software acceptance establishes correctness at its scope; task and full R5.4
rank/calibration/retention acceptance require separate evidence.

[Protocol and scoped evidence](../../artifacts/runs/native-consolidation-stream-20261022/PROTOCOL.md).
