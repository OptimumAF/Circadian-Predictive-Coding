# Phase 6 continual outcome and cost contract

## Primary outcome fields

For every seed, method, arm, and phase, keep the full task-accuracy matrix:
`A_after_A`, `A_after_B`, and `B_after_B` in the current two-task stream.
The two **primary** measures are:

1. `final_mean_task_accuracy = (A_after_B + B_after_B) / 2`. Each task has
   equal weight regardless of its number of examples; every reported score
   must state whether it came from outer-selection development or a sealed
   final role.
2. `signed_forgetting_A = A_after_A - A_after_B`. Positive means A
   deteriorated; negative means A improved after B. Always publish both
   A accuracies beside it. A lower forgetting number caused only by worse
   `A_after_A` is not a retention benefit.

The existing two-task `balanced_score` is numerically identical to final
mean task accuracy when it is computed from the same A/B post-B roles. Keep
it only as a compatibility/consistency field, never as an additional
independent endpoint or a substitute for the explicit accuracy matrix.
If a legacy report's roles or weights differ, label it separately rather
than treating the fields as equivalent.

An optional retention ratio is `A_after_B / A_after_A` only when
`A_after_A > 0`; at zero denominator it is **undefined/null**, not zero,
infinity, or an invented favorable value. Ratios above one are allowed and
mean improvement. Signed forgetting remains the primary retention field.
All accuracies must be finite in `[0, 1]` and derived values must be
recomputed from stored accuracies. No metric or denominator rule may be
changed after viewing development or final outcomes.

## Secondary trajectory and uncertainty

Adaptation speed and accuracy over the stream are secondary. A study may
report them only at prospectively named wake checkpoints with a declared
role and evaluation cost. Do not choose the most favorable checkpoint
afterward. For A→B→A or longer streams, extend the task-accuracy matrix and
predeclare the stage/task averaging rule before data are opened; do not
silently reuse the two-task formula.

Pair methods and factor arms within each source seed. Publish each seed,
paired difference, observed mean and dispersion, and uncertainty interval
when the planned sample size supports one. The seed/run is the replication
unit. A three-seed pilot is feasibility and direction evidence only; it is
not a confirmatory significance test. Report every predeclared cell and
failure, not just the best arm. P6.11 will set specific intervals and
multiple-contrast interpretation for the larger matrix.

## Cost vector and evaluation seal

Each factor study must predeclare and record: wake optimizer updates and
row presentations; latent inference loops and example-iterations; offered,
selected, **applied**, and rejected replay rows/updates; retained replay
examples and array bytes; sleep attempts/acceptances/rollbacks and guard
evaluations; initial, final, and peak trainable parameters and width; and
wall-time and peak process memory with measurement scope. Mark a field
`unmeasured` with a reason rather than inventing it. Distinguish equal
work/capacity controls from intentionally unequal treatment cost.
Accuracy and forgetting must be shown alongside this vector, not collapsed
into one composite winner. P6.10 remains open until all required costs are
measured in the relevant studies.

Use arrived train rows for wake/replay and the inner guard for any rollback.
Use outer selection for development decisions only. Freeze settings,
stopping, contrasts, and independent confirmation seeds before opening
final roles. Release final roles only after all relevant arms and seeds
pass the global train-only gate. Development-only pilots leave final roles
unopened; their scores must not be relabeled as confirmation.

`src/core/continual_metrics.py` implements the two-task arithmetic as a
pure validated value object. The P6.3 gating pilot predates this general
contract but used the same explicit formulas; its null result is unchanged.
The fixed v9–v14 outcomes are historical inputs to audit and design, not
prospectively selected by this later contract.
