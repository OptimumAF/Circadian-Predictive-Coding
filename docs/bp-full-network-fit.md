# Explicit full-network TRAIN fit control

The existing `NumpyContinuousBCLearner` offers one opt-in generalized
Gauss–Newton (GGN) update across all four MLP parameter blocks. Ordinary SGD and
the separate head-only MSE operation keep their existing behavior and limits.
This conventional BP control is not a circadian mechanism or an automatic
training/promotion policy.

```python
from src.core.learner_ports import TrainingUpdateFailed

# learner must already own its original TRAIN policy, state and work allowance.
try:
    result = learner.train_full_gauss_newton_episode(train_episode)
except TrainingUpdateFailed as failure:
    record_paid_failure(failure.definition, failure.work, str(failure))
else:
    record_update(result.diagnostic, result.loss_after, result.work)
```

`train_full_gauss_newton_batch(features, targets, row_weights=None)` is the
corresponding raw-array boundary. The caller owns TRAIN release authority; the
episode entry checks exact TRAIN episode role/spec. Neither entry admits a
PROBE_TRAIN policy, converts roles, fits preprocessing, selects examples, accesses
evaluation, starts a runtime or promotes a serving actor. Frozen observation
standardization applies once. Optional weights must be finite and strictly
positive, and do not replace row-exposure counts.

Each call allows 1..220 rows, at most 64 inputs/hidden units and 1,092 total
parameters. Parameter order is C-flattened W1, b1, W2, b2. The pure implementation
is `src/core/continuous_full_gauss_newton.py`; it knows arrays and the action spec,
not learner ownership, roles or IO. The adapter owns state validation and commit.

The objective is the existing normalized active-axis half-MSE, optionally with
the existing gripper logit2z cross entropy and row weights. Constant axes have
zero curvature. It solves the damped full-parameter GGN system with damping
1e-3/rcond1e-12, caps the step norm at 1 and evaluates exactly one trial using
the original loss kernel. No line search, target clipping or metric change.
A wrong saturated gripper logit retains its original gradient even when its
represented curvature rounds to zero.

Only a strict finite decrease publishes a new owned snapshot. Publication adds
one completed update and the input row count, and explicitly resets **all**
momentum slots to zero: a curvature step is not an SGD velocity. Parameter,
optimizer, policy, normalization and counter owners/values are checked around
numerical calls and before commit. Rejection/failure raises `TrainingUpdateFailed`
with paid attempts/returns and publishes nothing. A corruption is refused, not
silently repaired. A preexisting fork/actor retains its original state.

One full returned proposal pays three forwards and two backwards, including an
unused trial derivative. Why this: reusing the unchanged objective kernel keeps
one authority for weighting, physical bounds and mixed loss. Work reports kernel
requests/returns, Jacobian/curvature cells, Gram and solve dimensions, selected
dense MACs and selected numeric payload bytes. Solver arithmetic/workspace,
elementwise work, additional validation/copies and Python/allocator/RSS costs are
additional; `solver_work_known` is false. For D12/H64/A4 at220 rows, the Jacobian
alone is7,687,680B and Gram9,539,712B, excluding temporaries and solver workspace.
Caller experiment budgets/clocks/receipts remain separate and are not renewed.

[Scoped numerical/state checks](reports/full-network-fit-control-20261101.md)
qualify the software. A later [fixed H64 experiment](reports/full-network-native-fit-20261101.md)
accepted16updates and solved known PickCube17/19/20 in22/20/15controls; Push17
still failed at50controls. Weighted original TRAIN loss fell .06508→.001799.
This is useful control on those known resets, with extra optimizer computation;
reliability, generalization, retention/recovery, CPC benefit and release readiness
remain open. Default SGD/head behavior is unchanged.

[Earlier failed attempts](reports/full-network-native-control-20261101.md) and
all costs remain preserved. The amended protocol retained the one raw TRAIN
target discrepancy whose numerical float32 targets agreed, without changing
labels/weights/loss or retrospectively completing the old failed gates.

The existing [disposable probe](continuous-fresh-task-probes.md) now offers an
explicit `optimizer="full-gauss-newton"` option. Its role-bound adapter calls
the same inner owned commit with separate reserved training authority; this
public API remains TRAIN-only. Default SGD is unchanged. Software composition
is qualified; no new native acquisition/retention or scientific benefit follows.

A [fixed16-update private native case](reports/private-full-fit-native-20261101.md)
subsequently completed with excellent training fit but failures on both fresh
resets and all3old tasks. Its untouched source remains useful on the old tasks;
the adapted private child was discarded. Strong stored-trajectory fit alone did
not yield successful control or retention, and no additional optimizer dose is
authorized by that negative result.
