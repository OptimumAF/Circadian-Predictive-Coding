# Observe an existing CLM training gradient

Native CLM learners can return the scalar global L2 gradient norm already computed
by their existing clipping operation. This is an opt-in diagnostic for a committed
update, not a measure of plasticity, task success or retention. It changes no loss,
optimizer, schedule, clipping limit, data role or update allowance. No dependencies,
environment variables or CLI flags were added.

Use the explicit native port with an already admitted frozen batch and targets:

```python
result = learner.train_batch_observed(batch, targets)
loss = result.diagnostic                     # Original TrainingDiagnostic
observation = result.observation
print(observation.pre_clip_l2_norm)
```

`CLMHeadLearner`, `CLMCatalogLearner`, `CLMStateOnlyCatalogLearner`, `CLMHeadProbe`
and `CLMCatalogProbe` expose this port. Probe features still require their original
`probe_train` role. Disposable probes preserve the source and cannot install their
updates into it. Direct learners also accept `train_batch(...,
record_gradient_norm=True)`; the default remains the original loss diagnostic.
The flag must be an exact boolean and is checked before paid native TRAIN work.

`CLMObservedUpdate` contains only the loss diagnostic and immutable
`CLMGradientObservation`. Its fields are:

| Field | Meaning |
| --- | --- |
| `committed_update` | Native completed-update count for this returned update |
| `training_rows` | Number of query rows in this update |
| `gradient_tensor_count` | Number of optimized parameter tensors with gradients |
| `pre_clip_l2_norm` | Existing `clip_grad_norm_` returned scalar, before clipping |
| `clip_limit` | Original fixed limit, `1.0` |
| `definition` | `clm_optimizer_pre_clip_global_l2_v1` |

For state-only updates the norm covers only optimized state-head parameters;
the action head and scale remain fixed. A tensor count is not a parameter-element
count. The norm is unscaled, not divided by row or parameter count. Its scale
depends on the objective, parameter scope, reduction and batch. It is not directly
comparable across unlike configurations or a learning benefit by itself. A finite
zero is an observed result; a missing or lost return is unknown, never zero.

Why this: reuse the existing clipping return to avoid another norm reduction,
forward or backward pass. Converting the scalar and constructing/reporting records
still has overhead. No native model observation history, live gradient/tensor,
optimizer reference or callback is retained. Default snapshot, physical-work and
`TrainingDiagnostic` schemas stay unchanged.

The existing workflow helper can record the same observations:

```python
from src.app.clm_workflow_training import fit_catalog_workflow

record = fit_catalog_workflow(
    learner, batch, targets, report, passes=8, record_gradient_norm=True,
)
observations = record["gradient_observations"]
```

The original fresh 8..32-pass authority, twelve current-policy TRAIN rows, delayed
labels, inbox, receipts, clocks and budget remain required. Each scalar row is
bound to its native update and original loss receipt; the list is bounded by the
declared pass count. Only the opt-in report adds `gradient_observation_definition`
and `gradient_observations`. The ordinary report is unchanged. The workflow holds
one last native scalar return and bounded report/owner/proof data, not an unlimited
success-record chain.

Changing or replacing a previous observation, its returned owner, fixed features,
budget or original evidence refuses more work. Save failures stop the run and keep
the trusted paid prefix. Failed/unreturned native work keeps its native charges;
it cannot fabricate a norm, renew the allowance or be retried on that owner.
Trusted local Python guards are not hostile-process containment.

No final evaluation, promotion, tuning, replay selection or adaptation controller
consumes this value. No probe labels enter the source learner. A future experiment
must freeze its own question, observations, data, dose, metrics and resource limits
before execution; this API does not authorize replaying spent experiments.

Structure: `src/core/clm_training_observation.py` owns scalar declarations,
`src/adapters/clm_head_learner.py` uses the original native clip return and existing
family/probe dispatch, and `src/app/clm_workflow_training.py` binds returned rows to
the existing inbox. Core imports no Torch or adapter; app imports no native backend.
Extend the existing ports and frozen protocols rather than adding a new runner.

Focused tests require a prospectively declared Tiny-model allowance. The bounded
Windows/Linux commands, failures, costs and self-review are recorded in
[the qualification report](reports/clm-gradient-observation-20261025.md) and
[issue #156](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/156).
They establish scoped software correctness only; no new pretrained experiment,
gradient diagnosis of an old run, independent acceptance, CPC advantage or release
readiness is claimed.
