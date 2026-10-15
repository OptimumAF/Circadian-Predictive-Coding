# Matched raw-coordinate BP baseline

Run the fixed local matching example:

```powershell
.venv/Scripts/python.exe -m scripts.run_scaled_matched_backprop --describe
.venv/Scripts/python.exe -m scripts.run_scaled_matched_backprop
```

This supplies a necessary baseline for [nonlinear coupled scaling](coupled-pc-scaling.md).
It verifies matching configuration, initialization, representation, weight/bias
rate factors and training exposure. It does not compare generalization, compute
efficiency or width stability. No heldout/final data or parameter tuning is used.

## Objective, coordinates and update

BP receives the same `CoupledScalingState` as PC, restricted to
`config.circadian is None` and `wake_state is None`. The forward map is identical:

```text
A_l = a_l*raw_W_l; d_l = c*raw_b_l
h_0 = X; h_l = tanh(h_(l-1) @ A_l + d_l)
logits = h_H @ A_out + d_out
loss_BP = mean(logit_BCE(logits,y))
```

Every hidden activation is a function of the parameters. BP differentiates
through all of them using the existing `evaluate_matched_backprop` kernel. PC's
held-latent partials are a different derivative: at feedforward initialization,
hidden PC residuals are zero, while the BP task derivative through hidden layers
can be nonzero. The diagnostic IDs preserve this distinction:

- BP: `numpy_matched_binary_bce_preupdate_v1`, preupdate feedforward task loss.
- PC: `numpy_coupled_binary_energy_v1`, preupdate relaxed total energy.

The same raw-coordinate chain rule and optimizer factors apply:

```text
grad_raw_W = a_l*grad_effective_W; grad_raw_b = c*grad_effective_b
raw_W_new = raw_W - base_rate*R*a_l*grad_effective_W
raw_b_new = raw_b - base_rate*R*c*grad_effective_b
effective rate factors = R*a_l² (weights), R*c² (biases)
```

The new core reuses the existing scaling helpers; it introduces no second
initialization or rate rule. Separate raw-gradient evaluation may refuse an
unrepresentable derivative. Steps combine effective gradients and the complete
rate/scale product before rounding, allowing a representable update even when a
raw partial overflows. Candidate raw/effective state, feedforward loss and
gradients validate before return. Source state is unchanged on failure.

Precision beta is retained as exact config metadata for matching, strict restore
and future paired protocols. It does not enter BP's feedforward likelihood or
gradients. No latent inference, chemical traces, gating, momentum or adaptive
optimizer is added. CPC state is explicitly refused.

Why this: the prior effective uniform-rate BP baseline does not share the new
per-layer effective rates. Matching initialization alone would leave an optimizer
confound. BP and PC retain different training objectives/procedures and work;
neither matched factors nor equal exposure promises identical updates or compute.

## Modules and ports

```text
src/core/scaled_matched_backprop.py              full BP partials and pure raw step
src/adapters/scaled_matched_backprop_learner.py   fixed rate, raw state and owned forks
scripts/run_scaled_matched_backprop.py           fixed three-scheme matching example
tests/test_scaled_matched_backprop.py            independent objective/update controls
tests/test_scaled_matched_backprop_learner.py    ownership/restore/runtime controls
tests/test_scaled_matched_backprop_cli.py        actual matching/native/fork example
```

```python
from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner

state = initialize_coupled_scaling(
    CoupledScalingConfig("mean_field", 2, 3, 2, gamma0=1.5, precisions=(0.7, 1.3)), 23)
learner = ScaledMatchedBackpropLearner(state, learning_rate=0.01)
snapshot = learner.snapshot_state()
fork = learner.fork()
fork.restore_state(snapshot)
# train_batch(features, targets): arrived training labels only.
# predict(features): target-free, no inference or parameter mutation.
```

Snapshot/restore/fork preserve complete owned raw parameters/config and fixed
base rate. Restore requires the exact config; effective-only parameters or native
BP snapshots cannot substitute. The shared raw state describes coordinates, not
the update algorithm: adapter class selects BP versus PC and owns its fixed
training settings. There is no automatic disk decoder, clock/budget restoration
or authority transfer. Existing native/scaling/runtime defaults are unchanged.
The shared16-layer/1M parameter bound applies before allocation.

## Example and focused checks

Three schemes at seed23,D2,N3,H2,beta(.7,1.3),MFgamma1.5 share exact initial
raw/effective state and predictions. Main BP and PC learners receive two identical
training batches/base rate; PC uses two inference steps. A separate BP fork
repeats the second wake to verify complete continuation. The extra fork work is
reported rather than counted as equal-compute evidence. Fan-in also runs two
native BP updates from the same effective initial state. Numerical native
agreement is fixed prospectively at rtol1e-13/atol1e-15, with actual differences
and bitwise status recorded; exact original/fork checks remain required.

Focused pytest scope is the three new test modules above. Use Ruff check/format
and mypy (`--follow-imports=silent --no-incremental --no-sqlite-cache
--cache-dir=NUL --platform win32` and Linux) for those tests and the three new
source modules. Exact executed commands, results, counts, skipped gates and saved
states are under `artifacts/runs/scaled-matched-backprop-20261015/RESULTS.md`.

Accepted R1.8b1 evidence:62 focused cases/0fail/error/skip, six-file Ruff/format/
Windows+Linux mypy and independent design/source/evidence review pass. Saved CLI
81,871B retains all three schemes; initial BP/PC state/predictions, BP forks and
the fan-in native control are exact in this example. Joint60rawBP attempts/13rawPC/
10nativeBP/393objective attempts including scalar finite differences stay within
scope; unused36/35/14/1655 closed. Wider/scientific gates were not run.

R1.8b measurement and parent R1.8 remain open. Next add an isolated measurement
runner using these owned BP/PC/CPC ports, explicit roles and delayed-label data;
verify admission/partial-failure/final-scoring isolation before a frozen small
pilot. Report actual work and negative/inconclusive outcomes. Do not select
baselines, seeds or metrics to favor CPC.
