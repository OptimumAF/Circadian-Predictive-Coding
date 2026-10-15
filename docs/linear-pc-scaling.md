# Linear PC initialization and SGD controls

Run the fixed engineering example:

```powershell
python -m scripts.run_linear_pc_scaling
```

The JSON includes exact raw/effective initialized tensors, all scaling factors,
one simultaneous update for each control, and pre/post equilibrium energy. It
uses seed 23, input width 2, hidden width 3, two hidden layers and rate `0.01`.
These are equation checks, not a width/depth study or a learning comparison.

## Source and mathematical scope

The mean-field mapping follows §2.2 equations 7–15 and Appendix A.3 Table 2 in
[Innocenti, Achour and Bogacz, ICML 2026](https://proceedings.mlr.press/v306/innocenti26a.html),
with author code pinned at
[`18a732e1d495d833cb97cf305f9d6fbca7bdddeb`](https://github.com/thebuckleylab/jpc/tree/18a732e1d495d833cb97cf305f9d6fbca7bdddeb).
The retained full paper and author `_get_param_scalings` /
`configure_param_optim` were inspected. No JAX author runtime is executed.

This is a scalar-output, identity-activation MLP with common hidden width `N`,
input width `D`, `H >= 1` hidden layers and `L=H+1` learned matrices. There are
no biases, skips, residual depth factors or variable precisions. Arrays use row
orientation `(input_width, output_width)`, transposing the paper's matrices.

Why this: a parameterization includes optimizer factors as well as initialization.
Applying only the published initialization would implement a different algorithm.
This separate linear utility does not alter existing binary PC/CPC, legacy
learners, their snapshots, or the completed negative attribution study.

## Raw weights and effective weights

Write `A_l = a_l W_l`, where `W_l` is a raw trainable matrix and `A_l` is the
effective matrix used by the [linear equilibrium reference](linear-pc-reference.md).
All controls draw the same standard-normal matrices in the same order for a
given shape and seed, then apply the declared raw initialization standard deviation.

| Scheme | Raw initialization std | Forward factors `a_l` | Raw SGD factor `r` |
| --- | --- | --- | --- |
| `legacy_gaussian` | `.5` at every layer | all `1` | `1` |
| `fan_in` | first `1/sqrt(D)`; remaining `1/sqrt(N)` | all `1` | `1` |
| `mean_field` | all `1` | first `1/sqrt(D)`; middle `1/sqrt(N)`; output `1/(gamma0*N)` | `gamma0²*N` |

With one hidden layer there is no middle matrix. The fan-in standard deviations
are initialization choices, not forward multipliers during training.

The paper's total output factor is `gamma = gamma0*sqrt(N)` for mean-field;
the author helper receives `gamma0` after incorporating the other width factors.
Do not apply another `sqrt(N)` to the helper's output factor. Mean-field raw
variance is one, as specified by Table 2's `b_l=0`.

The legacy control matches the repository's `.5` Gaussian initialization, not
its nonlinear training behavior. The Gaussian fan-in control is not the author's
experimental SP uniform initializer: that uniform distribution has variance
`1/(3*fan_in)`. These controls are deliberately named separately.

## Gradient and update map

The reference evaluates the clamped squared-residual energy at its hidden
equilibrium using **effective** weights in both the feedforward prediction and
the suffix-product denominator:

```text
E*(A) = MSE(A) / s(A)
grad_A E* = grad_A MSE/s - MSE*grad_A s/s^2
grad_W_l E* = a_l * grad_A_l E*

W_l_new = W_l - learning_rate*r*a_l*grad_A_l E*
A_l_new = A_l - learning_rate*r*a_l^2*grad_A_l E*
```

Thus mean-field effective-coordinate SGD factors are:

| Layer | Factor multiplying the supplied learning rate |
| --- | --- |
| first | `gamma0²*N/D` |
| middle | `gamma0²` |
| output | `1/N` |

Both local controls have effective factor one at every layer. The output factor
is independent of `gamma0`; the first/middle factors are not. This is SGD without
momentum, weight decay or an adaptive optimizer; Adam does not follow this square
factor mapping.

`step_linear_pc` returns a replacement raw state and a **preupdate equilibrium
evaluation in effective coordinates**. Its `weight_gradients` are effective
partials, not raw gradients. Every matrix is updated simultaneously, with the
full weight-dependent denominator derivative. Candidate raw/effective arrays and
the candidate equilibrium are checked before a replacement is returned.

## Ownership and numerical boundary

`LinearScalingConfig` is frozen and validates exact scheme names, native positive
dimensions, native finite positive `gamma0` and representable positive factors.
Local controls require `gamma0=1`, refusing an ignored setting. Before RNG or
allocation, it refuses more than 1,000,000 raw entries (8,000,000 float64 bytes).
This bounds raw tensor storage, not Python metadata, temporaries or process RSS.

`LinearScalingState` validates/copies raw matrices on construction. Arrays remain
owned mutable snapshots in a frozen record; every public calculation revalidates
them. Effective arrays and replacement states are detached. Initialization owns
a local RNG; no continuing RNG or optimizer state exists. This is not a concurrent
model, persistent checkpoint, untrusted decoder or runtime authority.

Invalid schemas, shapes, types, domains and unrepresentable factors raise
`ValueError`. Arithmetic or candidate refusal may raise `ValueError` or
`FloatingPointError`, leaving source state untouched. Large finite rates are not
guaranteed to produce a finite or descending update.

The raw delta is multiplied through binary mantissas/exponents so a partial
`learning_rate*raw_factor` underflow cannot silently erase a representable final
update. This equivalent stabilized float64 arithmetic differs from a bit-for-bit
author runtime implementation. Final subnormal rounding/underflow and genuine
overflow remain float64 limits; no general relative-accuracy or convergence
guarantee applies near those limits. Tests include an exact binary regression
where the naive partial rate is zero but the correct effective update is `2^-22`.

## Use and verify

```python
import numpy as np
from src.core.linear_pc_scaling import (
    LinearScalingConfig, initialize_linear_scaling, step_linear_pc,
)

config = LinearScalingConfig("mean_field", input_dim=2, width=3,
                             hidden_layers=2, gamma0=2.0)
state = initialize_linear_scaling(config, seed=23)
result = step_linear_pc(state, np.array([[0.3, -0.2]]),
                        np.array([[1.0]]), learning_rate=0.01)
# state remains unchanged; result.state owns the replacement raw weights.
```

```text
src/core/linear_pc_scaling.py             validated configuration, state and raw SGD
scripts/run_linear_pc_scaling.py         exact fixed control records and one update
tests/test_linear_pc_scaling.py          independent raw Torch and domain controls
tests/test_linear_pc_scaling_edge.py     independent review arithmetic regression
tests/test_linear_pc_scaling_cli.py      actual user-facing command
```

```powershell
python -m pytest -q tests/test_linear_pc_scaling.py tests/test_linear_pc_scaling_edge.py tests/test_linear_pc_scaling_cli.py
python -m ruff check src/core/linear_pc_scaling.py scripts/run_linear_pc_scaling.py tests/test_linear_pc_scaling.py tests/test_linear_pc_scaling_edge.py tests/test_linear_pc_scaling_cli.py
python -m ruff format --check src/core/linear_pc_scaling.py scripts/run_linear_pc_scaling.py tests/test_linear_pc_scaling.py tests/test_linear_pc_scaling_edge.py tests/test_linear_pc_scaling_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/linear_pc_scaling.py scripts/run_linear_pc_scaling.py tests/test_linear_pc_scaling.py tests/test_linear_pc_scaling_edge.py tests/test_linear_pc_scaling_cli.py
```

Tests use actual CPU Torch raw-coordinate quotient gradients and SGD/effective
equivalence at 1/2/4 hidden layers, including `gamma0=.5/1/2`, plus schema,
ownership, resource and failure controls. Near ordinary settings the declared
gradient tolerance is absolute `2e-14`, relative `2e-12`.
No production dependency or environment variable is added. The separate
[measurement runner](linear-scaling-measurement.md) now records repeated
width/depth norms, cost and isolated heldout results for a fixed small grid.
The [residual utility](residual-linear-pc.md) and its separate
[measurement report](reports/residual-scaling-measurement-20261014.md) now complete
the applicable linear R1.4 scope. Nonlinear/binary/CPC integration remains
unchecked R1.8 and requires its own explicit mapping, tests and measurements.
