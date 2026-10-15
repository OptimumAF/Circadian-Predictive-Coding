# Tiny published linear PC reference

Run one fixed synthetic calculation:

```powershell
python -m scripts.run_linear_pc_reference
```

The JSON contains the exact inputs and weights, optimal hidden activities,
residuals, local weight gradients, energy identity error, stationarity residual,
and one simultaneous gradient step at rate `0.01`. No files, model state,
datasets, random initialization, or external services are needed at runtime.

## Source and scope

R1.1 reproduces a tiny special case of equations 1–6 in
[Innocenti, Achour and Bogacz, ICML 2026](https://proceedings.mlr.press/v306/innocenti26a.html).
The full 39-page proceedings PDF and its appendices were inspected, together with
the linked author implementation pinned at
[`18a732e1d495d833cb97cf305f9d6fbca7bdddeb`](https://github.com/thebuckleylab/jpc/tree/18a732e1d495d833cb97cf305f9d6fbca7bdddeb/experiments/limits_paper).
This identifies the code inspected today; the proceedings page does not identify
it as the original revision used for every published experiment.

The reference has identity activations, scalar output, at least one hidden layer,
no biases or skips, unit precisions, standard parameterization (`sp`), and output
factor `gamma=1`. Inputs and targets are clamped. Targets can be any finite real
numbers. All matrices are finite float64, with samples in rows and weights in
`(input_width, output_width)` orientation; this transposes the paper's weights.

Why this: an exact linear equilibrium provides a small independently checkable
reference before implementing or claiming the paper's scaling methods. The
[coupled binary learner](coupled-predictive-coding.md) retains its separate
objective and behavior.

## Equation and code map

Let `A_1,...,A_L` be the row-oriented weights, `Z_0=X`, `Z_L=y`, batch size `B`,
and `P_l=A_(l+1)...A_L` for hidden layer `l`. Let `f=X A_1...A_L`.

| Published quantity | Local calculation | Pinned author symbol |
| --- | --- | --- |
| Eq. 1, feedforward loss | `mse=sum((y-f)^2)/(2B)` | `linear_equilib_energy`, feedforward residual |
| Eq. 2, clamped energy | `e_l=Z_l-Z_(l-1)A_l`; `energy=sum_l sum(e_l^2)/(2B)` | `pc_energy_fn` |
| Eq. 3, hidden optimum | `s=1+sum_l sum(P_l^2)`; `e_l=((y-f)/s) P_l.T`; reconstruct `Z_l=Z_(l-1)A_l+e_l` | `compute_linear_activity_solution`; analytical optimum |
| Eqs. 5–6, equilibrium identity | `energy=mse/s` | `linear_equilib_energy`, `_compute_linear_equilib_rescaling` |
| Eq. 4, local learning | `grad_l=-Z_(l-1).T e_l/B`; all weights updated together | `compute_linear_equilib_energy_grads`, `update_linear_equilib_energy_params` |

The implementation is `evaluate_linear_equilibrium` in
`src/core/linear_pc_reference.py`. The record contains hidden latents only, but
residuals and gradients include the output layer. Output residual is evaluated
against the clamped target after reconstructing the hidden activities.

Each hidden stationarity residual is `e_l-e_(l+1)A_(l+1).T`; dividing by `B`
gives its energy partial. The local weight partial at the exact optimum equals
the derivative of the equilibrated energy by the envelope theorem. In particular:

```text
grad(mse/s) = grad(mse)/s - mse*grad(s)/s^2
```

The second term is essential (paper Eq. 28 and Appendix A.6 Eq. 130). Ordinary
MSE BP and equilibrated PC can differ at finite width. This fixture's standard
parameterization does not establish the paper's asymptotic convergence claims.

## Deviations and limitations

- Explicit tiny tensors replace the author's random initialization, datasets,
  training trajectories, and figures. No width or depth sweep is performed.
- NumPy reconstructs the scalar equilibrium analytically; author JAX/Equinox
  functions are inspected but are not installed or executed. Tests independently
  solve the hidden least-squares problem and differentiate the quotient in Torch.
- No biases, skips, multidimensional outputs, nonlinearities, generalized
  precisions, optimizer state, or published mean-field parameterization are added.
- Existing coupled PC/CPC uses tanh, biases, binary likelihood, finite inference
  steps and optional precision/gates. Its objective is not `mse/s`.
- Numerical checks target tiny well-conditioned float64 fixtures. Nonfinite
  input/arithmetic refuses with `ValueError`; no guarantee covers all finite
  ill-conditioned networks. There is no mutation, checkpoint, or data authority.

## Structure and checks

```text
src/core/linear_pc_reference.py           pure equilibrium and local derivatives
scripts/run_linear_pc_reference.py       fixed synthetic fixture and JSON output
tests/test_linear_pc_reference.py        independent math and input controls
tests/test_linear_pc_reference_cli.py    real command and report checks
```

```powershell
python -m pytest -q tests/test_linear_pc_reference.py tests/test_linear_pc_reference_cli.py
python -m ruff check src/core/linear_pc_reference.py scripts/run_linear_pc_reference.py tests/test_linear_pc_reference.py tests/test_linear_pc_reference_cli.py
python -m ruff format --check src/core/linear_pc_reference.py scripts/run_linear_pc_reference.py tests/test_linear_pc_reference.py tests/test_linear_pc_reference_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/linear_pc_reference.py scripts/run_linear_pc_reference.py tests/test_linear_pc_reference.py tests/test_linear_pc_reference_cli.py
```

No new production dependency or environment variable is introduced. The derivative
oracle uses the existing test Torch installation. Extend scaling under R1.4 with
explicit effective weights, initialization and optimizer factors; do not silently
reinterpret these fixed weights or count this reference as a scaling experiment.
