# ADR-0494: Explicit binary gripper execution

## Context

The bounded nearest controller solves known A/B contexts using the same fixed
taskspace12 information, while recorded neural controls remain unsuccessful.
The existing BP binary loss still produces soft commands. Native soft targets
are valid, so endpoint execution is a separate hypothesis rather than an inferred
fix. Changing the training objective would also break the common four-family
comparison contract.

## Decision

Add one optional fixed recipe at the existing frozen serving boundary. Apply
the sign rule to float64 gripper predictions before native rounding, opening on
zero and preserving XYZ. Keep original predictions, selected commands and actual
returns distinct in bounded receipts. Use actual native commands for history.
Retain paid returns before later validation, refuse corrupted prior evidence and
run final checks after observers. Keep existing soft defaults and training losses.

## Alternatives

- Retain soft serving: remains the default and accepted control.
- Change BP's loss: already supported separately, without PC/CPC parity.
- Fit a decoder threshold from evaluations: excluded from this fixed condition.
- Teacher takeover: answers a different question about autonomous control.

## Consequences

The option adds no learner, runner, simulator controller or dependency. Its
receipt storage is bounded by the existing 50-control horizon. Validation and
copy costs are additional to prediction costs.

Scoped software checks pass on both platforms. The first native condition is a
complete negative: BP predicted an open gripper throughout. It does not justify
threshold tuning, a claimed CPC advantage or a broad four-family experiment.
Further fitting diagnosis must keep old results and evaluation isolation intact.
