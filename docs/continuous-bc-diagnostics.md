# TRAIN action residual diagnostics

The separate [TRAIN conditioning API](continuous-bc-conditioning.md) adds bounded
activation and actual-gradient summaries. This residual API and schema remain
unchanged; both preserve the learner and refuse non-TRAIN data.

`NumpyContinuousBCLearner.diagnose_training_episode(episode, stages)` performs
one prediction over a complete TRAIN episode. It returns immutable summaries
without changing parameters, momentum, policy, or training counters. A TRAIN
policy, matching TRAIN episode/spec, and valid stage labels are required before
prediction. `probe_train`, `probe_test`, and all other non-TRAIN roles are refused.

## Responsibilities and boundaries

```text
src/core/continuous_bc_diagnostics.py    Pure bounded action summaries
src/adapters/numpy_continuous_bc.py     TRAIN gate and one existing predict call
tests/test_continuous_bc_diagnostics.py Independent numeric and boundary tests
```

The pure core takes float64 predictions, float64 targets, the declared action
specification, and one stage label per row. It has no learner, role conversion,
simulator, RNG, filesystem, or evaluation-selection responsibility. The adapter
uses the existing episode extraction and prediction interfaces. No new runner,
dependency, snapshot format, or training objective is introduced.

## Metrics

For each nonconstant axis, normalize each physical action by the declared center
and half-width, then subtract **predicted minus target**. Separate normalization
avoids overflow from subtracting opposite large physical endpoints. Reports give
signed mean error, MAE, RMSE, half-MSE, and maximum absolute error for every axis,
along with its index, units, and active flag. Duplicate unit names remain distinct
through axis indices.

Overall and stage half-MSE average over rows and active axes. Constant axes must
equal their fixed bound; their metrics are zero and they are excluded from the
objective denominator. An entirely constant action specification has objective
zero. Stage order follows first appearance, including noncontiguous occurrences;
every group retains its actual row indices and count. A row-weighted combination
of stage objectives reconstructs the overall objective. An unweighted average of
stage means generally does not.

The definition uses physical predictions. At extremely narrow bounds, physical
output rounding may differ from the learner's latent tanh objective. For the
accepted robot bounds `[-1, 1]`, the saved-state objective matches the accepted
training value within the prospectively frozen `1e-12` tolerance.

## Usage and costs

```python
# learner owns the matching policy; episode is a qualified TRAIN record.
# stages is an exact tuple of one label per actual transition.
report = learner.diagnose_training_episode(episode, stages)
print(report.residuals.overall.normalized_half_mse)
for stage in report.residuals.stages:
    print(stage.label, stage.row_indices, stage.summary.normalized_half_mse)
print(report.prediction_calls, report.prediction_rows, report.prediction_bytes)
```

Bounds are 1..256 rows, 1..16 axes, at most 16 distinct labels, and at most 32
UTF-8 bytes per nonempty native string label, without edge whitespace. Outputs
retain scalar summaries and row identities, not an unlimited record history or
the raw residual matrix. Prediction bytes count the float64 prediction payload;
they exclude snapshots, validation copies, activations, summaries, and process
memory. Callers account separately for serialized reports and other memory.

The accepted saved-state diagnosis used one forward over 29 TRAIN rows, 116
values / 928 raw prediction bytes, and a 10,885-byte JSON report. It performed no
optimizer update, simulator call, or probe-file read. Full state remained exact.

## Observed TRAIN limitation

The existing 64-update BC state has overall half-MSE `0.062273216206435826`.
Gripper error contributes 75.10% of total squared action error. Stage half-MSE:

| Stage | Rows | Half-MSE | Gripper MAE |
|---|---:|---:|---:|
| align | 8 | 0.057720 | 0.498041 |
| descend | 8 | 0.044215 | 0.479022 |
| grasp | 6 | 0.109540 | 0.892934 |
| lift | 7 | 0.047601 | 0.328978 |

All axes and stages remain in the saved report. These TRAIN observations do not
establish why the frozen probe failed, learned task success, retention, recovery,
or a circadian advantage. The prior untrained and trained probe negatives remain
unchanged. No more model training or simulator work was admitted in this increment.

## Fixed TRAIN continuation executed

A separately frozen continuation from64 completed192additional updates on the
same29unique TRAIN examples, retaining all192 returned losses and three complete
diagnostics. Defaults and objective stayed fixed. No model/simulator retry or
best-checkpoint selection was used. Independent review accepted source and actual
curves, work/state and final eight-array payload.

| Total updates | TRAIN half-MSE | Gripper MAE | Vertical-action MAE |
|---|---:|---:|---:|
| 64 | 0.062273 | 0.533688 | 0.222286 |
| 128 | 0.042838 | 0.377534 | 0.232527 |
| 256 | 0.032012 | 0.254991 | 0.230701 |

More optimization reduced squared loss on this one demonstration, while vertical
MAE remained worse than64. The original owner stayed exact while its fork trained;
each diagnostic preserved all state. Added training exposure5568rows; diagnostic
exposure87rows/3forwards, cumulative raw prediction bytes2784; report67907B,
final-only state50102fileB/48192rawB. No probe, final or simulator work. These
curves do not establish autonomous robot skill or data coverage/generalization.
[Frozen protocol, exact commands and independent readback](../artifacts/runs/continuous-bc-train-continuation-20261019/RESULTS.md).

The frozen policy is now bound to the existing rollout entry point, preserving
its zero-action default and prediction/control/state/role checks. One reviewed
seed17 TRAIN execution of final256 completed50steps with no task success; offline
fit still does not establish robot skill. [Runtime API, negative evidence and next
TRAIN input-conditioning question](frozen-bc-robot-rollout.md). No training during
that rollout, seed23 probe rerun, best-checkpoint selection, or final-data access.

[Exact commands, scoped checks, costs, and independent review](../artifacts/runs/continuous-bc-residuals-20261019/RESULTS.md).
