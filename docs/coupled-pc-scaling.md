# Opt-in raw/effective scaling for coupled tanh/BCE PC

The existing coupled energy supplies a complete local nonlinear objective. This
module adds explicit raw coordinates, initialization and update factors for PC,
neutral CPC and active CPC. It does not establish width invariance, convergence,
efficiency or comparative learning benefit. The bias, precision and energy choices
below are local extensions; the accepted linear reference does not prove them.

## Equations and choices

For input width D, common hidden width N and H hidden layers, let the effective
weights and biases be `A_l=a_l*W_l` and `d_l=c*b_l`. Raw `W,b` are trainable.
The input and native hidden latents z keep their original coordinates:

```text
prediction_l = tanh(z_(l-1) @ A_l + d_l), z_0 = X
logits = z_H @ A_out + d_out
E = mean(logit_BCE(logits,y))
    + sum_l beta_l * ||z_l - prediction_l||² / (2*batch_size)
```

No output latent, extra gamma factor, width normalization of the energy or
learnable precision is added. Beta remains the declared native precision tuple
in[1e-6,1e6]. The hidden penalty sums over units, so unchanged beta is not a claim
that energy distributions or inference dynamics stay constant across widths.

| Scheme | Raw weight std | Weight forward factors a | Raw SGD multiplier R | Bias factor c |
|---|---|---|---|---|
| legacy_gaussian | .5 | all1 | 1 | 1 |
| fan_in | first1/sqrtD, remaining1/sqrtN | all1 | 1 | 1 |
| mean_field | all1 | first1/sqrtD, middle1/sqrtN, output1/(gamma0*N) | gamma0²*N | 1/sqrt(stored R) |

Biases initialize at zero. All schemes draw the same standard-normal weight
arrays in the same order for a shape/seed, then apply their raw standard
deviations. The weight choices reuse [the linear scaling controls](linear-pc-scaling.md).
The legacy name identifies that initialization distribution, not the behavior of
an older nonlinear learner. The fan-in raw/effective update is the existing
native coupled update, with full state equivalence checked separately.

Stable scalar/array multiplication and the native G*gate-then-rate order may
differ by rounding. The CLI freezes full native state/energy agreement at
rtol1e-13/atol1e-15 before fixtures, records maximum differences and bitwise status,
and keeps config/policy/count, PC/neutral parity and fork continuation exact.

Why this bias choice: holding effective bias learning rates approximately at the
declared base rate prevents the raw multiplier alone from making them grow with
width. It is an explicit local algorithm choice, not a result attributed to the
linear paper. The implementation computes c from the stored representable R.
Ideal algebra gives `R*c²=1`; finite rounding, especially near subnormal R, need
not give exactly1. Report the actual stable computed effective bias factor.

## Derivatives and updates

At fixed latent coordinates, with all other arguments fixed:

```text
grad_raw_W_l E = a_l * grad_A_l E
grad_raw_b_l E = c   * grad_d_l E
grad_z E       = native effective-coordinate latent partial

W_new = W - learning_rate * R * a_l * effective_weight_partial
b_new = b - learning_rate * R * c   * effective_bias_partial

effective weight learning-rate factor = R*a_l²
effective bias learning-rate factor   = R*c²
```

These are partial derivatives of the declared energy at held latents. They do
not differentiate through inference or optimize an analytic hidden equilibrium.
`evaluate_coupled_scaling` returns the native evaluation and true ungated raw
partials. It explicitly refuses derivatives outside finite float64 range.

Steps use native effective partials directly in a stable product. A raw partial
may overflow although its product with the raw rate factor is representable;
requiring raw-gradient diagnostics would wrongly reject that update. Scalar
factors and broadcast CPC gates enter mantissa/exponent multiplication before
final rounding. This also avoids first rounding a subnormal gradient times a
small gate to zero or forming an overflowing ungated delta before gating it.

## CPC and inference

PC has no wake state. CPC retains its complete native fixed chemistry policy,
chemical vector, plasticity vector and wake count. Prospective chemistry uses
the final inferred last-hidden activity, exactly as in native CPC. Its prospective
gates multiply last-hidden incoming weight columns/bias entries and output weight
rows; output bias is ungated. This is an update preconditioner. The reported
mathematical gradients do not differentiate through gates or chemistry.

The default inference loop is the existing fixed-step loop in effective
coordinates. An explicit `CoupledInferencePolicy` uses the existing guarded
path; neither choice changes latent coordinate scaling or thresholds implicitly.
The candidate's held-latent energy and all raw/effective/wake fields validate
before a pure replacement is returned. Failure leaves the source state unchanged.

## Modules and usage

```text
src/core/coupled_pc_scaling.py           config, owned raw state, map/partials/step
src/adapters/coupled_scaling_learner.py  fixed native ports, raw snapshots and forks
scripts/run_coupled_pc_scaling.py       fixed equation/native-control example
tests/test_coupled_pc_scaling.py        independent objective/derivative/update controls
tests/test_coupled_scaling_learner.py   owned ports, restore and continuation controls
tests/test_coupled_pc_scaling_cli.py    actual bounded CLI outcomes
```

The new domain is uniform hidden widths,1..16 hidden layers and at most1M total
parameter entries including biases. Config/state validate before allocation and
own all mutable arrays and policy records. Existing native classes/adapters,
topology state and checkpoint formats are unchanged.

```python
from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
from src.adapters.coupled_scaling_learner import CoupledScalingLearner

config = CoupledScalingConfig("mean_field", 2, 3, 2, gamma0=1.5,
                              precisions=(0.7, 1.3))
learner = CoupledScalingLearner(initialize_coupled_scaling(config, 23),
                               learning_rate=0.01, inference_steps=2,
                               inference_learning_rate=0.05)
# learner.train_batch(features, targets) uses arrived training labels only.
# learner.predict(features) receives no labels and performs no latent inference.
snapshot = learner.snapshot_state()
fork = learner.fork()
fork.restore_state(snapshot)
```

Raw snapshots retain the exact scaling/precision/CPC config. Restore requires
that same config. Effective parameters alone cannot replace a raw snapshot:
they omit raw coordinate/update semantics. The adapter owns fixed training
settings separately; snapshots do not restore budgets, clocks or authority.
There is no momentum/adaptive optimizer state.

## Fixed engineering example and validation

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_pc_scaling --describe
.venv/Scripts/python.exe -m scripts.run_coupled_pc_scaling
```

The full command records three schemes x PC/neutral/active CPC at one frozen
shape/seed, all raw/effective initial/final states and actual factors. Each runs
one wake then independent original/fork continuation; fan-in also runs matched
two-wake native controls. These are training-only equation checks, not a scaling
grid or heldout comparison. No settings are selected from these outcomes.

Focused tests are the three modules listed above. Use Ruff check/format and
scoped mypy (`--follow-imports=silent --no-incremental --no-sqlite-cache
--cache-dir=NUL --platform win32` and Linux) on those tests plus the three new
source modules. Exact executed commands/results/budgets/skips are recorded under
`artifacts/runs/coupled-scaling-mapping-20261015/RESULTS.md`.

Accepted R1.8a evidence:90 distinct focused cases/0fail/error/skip, six-file
Ruff/format/Windows+Linux mypy and independent design/source/outcome review pass.
Saved CLI180,756B retains all methods; every fork and PC/neutral raw comparison
is exact. Active fan-in native max state difference6.776263578034403e-21 and energy
difference0 meet the frozen numerical agreement. Joint98raw attempts/16native
updates/1guardedinfer/763energy+objective evaluations, unused30/48/127/3333 closed.

R1.8 remains open for a prospectively bounded isolated measurement and assessment
of this explicit mapping. A matched raw-coordinate BP control with the same
initialization and bias/weight update factors must be implemented/verified first;
the existing uniform effective-rate BP baseline alone does not match the new
per-layer rates. R1.6 broader structural/scaling interactions are still
unfinished. Add another scaling choice only with explicit objective/latent/bias/
precision/gating equations, independent derivatives and complete continuation
controls before measurements. Preserve existing negative studies and defaults.
