# Continuous PC energy

`src/core/continuous_pc.py` supplies a pure continuous task energy and its exact
partial derivatives. It accepts the robot's multi-axis labels; the older
`coupled_energy.py` retains its scalar binary likelihood. There is no learner,
latent inference loop, parameter update, data access, simulator, or role policy
in this module.

## Run the example

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.continuous_pc_example
```

The fixed two-row/four-action example prints initial task-head parity with the
existing BC reference, then the energy and derivatives at a declared perturbed
free state. It performs zero optimizer updates and uses no random draws. This is
a mathematical example, not a trained policy or benchmark.

## Interface and ownership

Use `CoupledParameters` with row biases, a validated `ContinuousActionSpec`, and
finite float64 NumPy arrays. Supported bounds are 1..16 hidden layers, widths
1..256, batches 1..256, inputs 1..1024, and actions 1..16. These are validation
limits, not evidence of deep-network scaling.

- `feedforward_continuous_latents(parameters, features, spec)` initializes an
  owned tuple of hidden arrays without targets.
- `predict_continuous_pc(parameters, features, spec)` returns bounded physical
  actions from feedforward latents, without target-based inference.
- `evaluate_continuous_pc(parameters, features, targets, latents, precisions,
  spec)` returns `ContinuousPCEvaluation`: energy, task loss, hidden penalties,
  predictions, latent and parameter gradients, precision tuple, and definition
  IDs. The caller supplies one native numeric precision in `[1e-6, 1e6]` per
  hidden layer; booleans and nonfinite values are refused.

Inputs must be quiescent during a call and are copied. Returned arrays belong to
the caller and do not alias inputs or one another. The frozen result container
does not make its owned numeric buffers immutable. All arithmetic, including
preactivations before `tanh`, must remain finite; overflow raises an error.

## Equations and comparison scope

Let `B` be batch rows and `A` the number of nonconstant action axes. For layer
`l`, `P_l = tanh(previous_l W_l + b_l)`, with `previous_0 = X` and later inputs
the preceding free latent. Let `R_l = Z_l - P_l` and
`v = tanh(Z_last W_out + b_out)`.

The task term is the existing default BC half mean normalized squared error:
`L = sum_active((v - target_normalized)^2) / (2 B A)`. It is zero if all axes
are constant. Physical predictions use the existing BC convex blend of bounds.
Hidden penalties are `H_l = beta_l sum(R_l^2) / (2 B)`, summing neurons rather
than averaging over action axes. Total energy is `E = L + sum(H_l)`.

Why this: matching the deployed BC action geometry and task normalization avoids
changing the comparison objective when a continuous PC learner is added. The
separate energy ID is `numpy_coupled_continuous_energy_v1`; the task ID remains
`continuous_bc_half_mean_active_axis_normalized_squared_error_v1`.

All derivatives hold other free latents fixed. At feedforward initialization the
hidden residuals and hidden-weight partials are zero, while BP's total hidden
gradients can be nonzero. Only task values, predictions, and output-head gradients
are matched to one-hidden BC there. This module does not claim BP/PC update
equivalence, a converged latent state, or a circadian advantage. The optional BC
binary-gripper likelihood is outside this energy's contract.

## Extend safely

Add bounded latent/parameter updates and an owned learner adapter as a separate
increment. Reuse the stream runner and existing role/state boundaries; charge
actual inference, gradients, diagnostics and probes. Verify a neutral CPC bridge
before any comparative experiment. Final evaluation must remain unavailable to
training, promotion, tuning, or probe selection. Existing negative robot results
and unverified historical thread evidence remain unchanged.

The prospective equations, resource limits and acceptance gates are in
[the protocol](../artifacts/runs/continuous-pc-energy-20261020/PROTOCOL.md).

## Focused development checks

```powershell
python -m pytest tests/test_continuous_pc.py
python -m ruff check src/core/continuous_pc.py tests/test_continuous_pc.py scripts/continuous_pc_example.py
python -m ruff format --check src/core/continuous_pc.py tests/test_continuous_pc.py scripts/continuous_pc_example.py
python -m mypy --follow-imports=silent src/core/continuous_pc.py tests/test_continuous_pc.py scripts/continuous_pc_example.py
```

[Executed commands, costs and skipped work](../artifacts/runs/continuous-pc-energy-20261020/RESULTS.md)
and [independent review](../artifacts/runs/continuous-pc-energy-20261020/REVIEW.md)
bound the accepted claims. Torch checks may skip in a minimal installation;
acceptance evidence requires that the independent autodiff cases actually run.

## Follow-up: bounded latent inference

The separate [continuous inference API](continuous-pc-inference.md) composes this
energy with the existing guarded inference loop and explicit stopping work. It
does not alter these equations or update parameters. Owned learner updates and
neutral CPC comparisons remain later requirements; target-conditioned inference
predictions must remain separate from target-free deployment.
