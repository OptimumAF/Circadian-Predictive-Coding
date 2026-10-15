# Coupled binary predictive coding, version 1

This opt-in NumPy learner implements objective
`numpy_coupled_binary_energy_v1`. It is separate from the historical PC and CPC
learners and their training diagnostics. It is not a reproduction of a published
scaling method or evidence of a learning advantage.

## Energy, shapes and boundary conditions

Use row batches, batch size `B > 0`, input `z_0 = X` of shape `(B, D)`, and one
or more hidden layers. Hidden latent `z_l` has shape `(B, H_l)`, weight `W_l`
shape `(H_(l-1), H_l)`, and bias `b_l` shape `(1, H_l)`. Output weight `V` has
shape `(H_L, 1)` and output bias `c` shape `(1, 1)`.

```text
p_l = tanh(z_(l-1) W_l + b_l)
r_l = z_l - p_l
o   = z_L V + c
E   = (1/B) sum_i [logaddexp(0, o_i) - y_i o_i]
      + sum_l beta_l sum_(i,j) r_l[i,j]^2 / (2B)
```

Input `X` is fixed. Training targets `y` of shape `(B, 1)` are fixed likelihood
targets in `[0,1]`, including soft labels. There is **no output latent or output
clamping**: logits `o` are computed directly from the last free hidden latent.
Every hidden prediction is recomputed from the **current lower latent** at each
evaluation. Hidden latents need not remain in `[-1,1]`.

Each layer's precision is an explicit constant `beta_l` in `[1e-6, 1e6]`.
The squared residual is summed over neurons, not averaged over width or depth.
The entire energy, including its latent derivatives, is averaged over batch.
Precision values do not silently change with width, depth, batch size or time.
Duplicating a batch preserves its energy and fixed-latent parameter partials,
but halves each duplicated latent's partial. Thus the same latent step size is
not batch-size invariant; keep batch size and rates matched in comparisons.

All inputs, parameters and latents are finite, nonempty, rank-2 float64 NumPy
arrays with exact shapes. Parameter, latent and precision collections are tuples
with matching nonzero depth. Invalid shape/type/domain values raise `ValueError`;
nonfinite computed values raise `FloatingPointError`. Logits use stable
`logaddexp` BCE and an unclipped stable sigmoid derivative, preserving their
relationship in saturation. Legacy activation/clipping functions are unchanged.

## Exact partial derivatives

Let `q = (sigmoid(o) - y) / B`, and define the prediction-linear derivative
`d_l = -beta_l r_l * (1 - p_l^2) / B`, with elementwise multiplication.

```text
dE/dW_l = z_(l-1)^T d_l
dE/db_l = sum_rows d_l
dE/dV   = z_L^T q
dE/dc   = sum_rows q

dE/dz_l = beta_l r_l / B + d_(l+1) W_(l+1)^T     for l < L
dE/dz_L = beta_L r_L / B + q V^T
```

The downstream term has a negative residual-Jacobian contribution through
`d_(l+1)`. Biases, tanh derivatives, layer precisions and batch normalization
therefore participate in the same scalar objective. These are partials with
other arguments fixed, **not gradients through the inference procedure**.

## Alternating learner update

`CoupledPredictiveCodingNetwork` initializes weights with independent ordinary
fan-in Gaussian draws (`std = 1/sqrt(fan_in)`) and zero biases. This is a declared
initialization choice, not the unimplemented published parameterization in R1.4.

1. Initialize all hidden latents with a target-free feedforward pass.
2. For the requested fixed number of inference steps, recompute the energy and
   all latent partials at one common state, then update every latent
   simultaneously: `z_l <- z_l - inference_learning_rate * dE/dz_l`.
3. Reevaluate at the final inferred state. Record this post-relaxation,
   pre-weight-update energy. Hold latents fixed and update all weights/biases
   simultaneously: `theta <- theta - learning_rate * dE/dtheta`.
4. Validate candidate parameters and their energy before publishing the new
   parameter bundle. A failed step leaves model parameters unchanged.

Both rates must be finite positive native numbers (not booleans); the inference
count is a native integer in `[1,10000]`. A fixed finite count is a computation
bound, not a wall-time or memory preemption guarantee. No adaptive convergence
stop or fallback is implemented here (R1.5). Arbitrarily large finite rates are
not guaranteed to lower energy or avoid a refusal. Tests check descent only in
explicit small-step fixtures, not universal convergence.

Prediction performs target-free feedforward inference; it does not relax using
evaluation labels. `snapshot_parameters` returns detached parameters;
`restore_parameters` accepts only the current topology and copies after complete
validation. These are local parameter operations, not durable checkpoints or
restoration of an external budget, runtime cursor or data-access authority.

## Run a tiny native update

```python
import numpy as np
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork

x = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
y = np.array([[1.0], [0.0]], dtype=np.float64)
model = CoupledPredictiveCodingNetwork(2, (3, 2), seed=23)
before = model.predict_proba(x)
result = model.train_epoch(x, y, learning_rate=0.03,
                           inference_steps=3, inference_learning_rate=0.1)
after = model.predict_proba(x)
assert result.energy_definition == "numpy_coupled_binary_energy_v1"
```

All example rows are synthetic training data. The API does not grant final-data
access or enforce a dataset role; callers remain responsible for the existing
evaluation-isolation rules. The separate [coupled CPC wake path and neutral
control](coupled-neutral-control.md) now exercise the basic circadian gate with
exact multilayer neutral parity. [Owned runtime adapters](coupled-learner-ports.md)
provide complete compatible model snapshots and delayed-label integration.
Replay, sleep, structural changes and comparative evaluation remain separate work.

## Structure and verification

```text
src/core/coupled_energy.py                energy and exact partials; no mutation
src/core/coupled_predictive_coding.py     owned alternating native learner
tests/test_coupled_energy.py             independent differences and autodiff
tests/test_coupled_predictive_coding.py   update order, ownership and refusals
```

Core depends only on NumPy and the standard library. There are no new required
packages, environment variables, outer-layer imports, global RNG changes or
changes to legacy model defaults.

The numerical contract uses tiny 1/2/4-hidden-layer networks, central differences
with epsilon `1e-6`, absolute tolerance `2e-8`, relative tolerance `2e-6`, and a
separate test-only CPU Torch autodiff oracle. The finite-difference tests run
independently of Torch availability. Backend skips must be reported as skips.

```powershell
python -m pytest -q tests/test_coupled_energy.py tests/test_coupled_predictive_coding.py
python -m ruff check src/core/coupled_energy.py src/core/coupled_predictive_coding.py tests/test_coupled_energy.py tests/test_coupled_predictive_coding.py
python -m ruff format --check src/core/coupled_energy.py src/core/coupled_predictive_coding.py tests/test_coupled_energy.py tests/test_coupled_predictive_coding.py
python -m mypy --follow-imports=silent --no-incremental src/core/coupled_energy.py src/core/coupled_predictive_coding.py tests/test_coupled_energy.py tests/test_coupled_predictive_coding.py
```

R1.3 neutral CPC parity is verified for the documented coupled binary wake path.
R1.1 now has a separate [tiny linear reference](linear-pc-reference.md); it does
not reproduce this binary learner. R1.7 now has a separate
[small attribution report](reports/coupled-attribution-20261014.md); R1.4 scaling
remains unfinished. No experiment sweep is launched by these checks.
