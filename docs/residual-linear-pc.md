# Residual linear PC scaling and SGD

Run the fixed CPU equation example:

```powershell
python -m scripts.run_residual_linear_pc
```

It prints exact initial and updated raw/effective weights, fixed skip positions,
initialization/forward/SGD factors and energies for three explicit controls. The
example uses seed 23, input width 2, hidden width 3, two hidden layers and rate
`.01`; each control takes one real update. This is an engineering calculation.
There is no residual width/depth experiment in this increment.

## Source mapping

The mean-field mapping follows §4 Eq. 19 and Appendix A.5 Eqs. 97–100/106–110 of
[Innocenti, Achour and Bogacz, ICML 2026](https://proceedings.mlr.press/v306/innocenti26a.html).
The previously retained full paper and author code pinned at
[`18a732e1d495d833cb97cf305f9d6fbca7bdddeb`](https://github.com/thebuckleylab/jpc/tree/18a732e1d495d833cb97cf305f9d6fbca7bdddeb)
were inspected. Author `_get_param_scalings` in `jpc/_core/_energies.py` supplies
`1/sqrt(N*L)` for middle layers with skips; `configure_param_optim` in
`experiments/limits_paper/utils.py` supplies raw SGD factor `gamma0²*N`.
Its explicit `MLP.__call__` adds fixed identities only in middle layers;
`_compute_linear_equilib_rescaling` includes them in the suffix products.
The cached author `make_skip_model` implementation is not present; these claims
use the equations and inspected explicit forward/analytical paths. No JAX author
runtime is executed.

For input width `D`, common hidden width `N` and `H` hidden layers, define
**`L=H+1` total learned matrices**, with `H-1=L-2` residual middle blocks. Arrays
are row-oriented, transposing the paper's column matrices. All weights are
float64; activation is identity, output width is one, biases are absent and
all energy precisions are one.

```text
first:   A_1 = a_1 W_1
middle:  A_l = I + a_l W_l           l=2,...,L-1
output:  A_L = a_L W_L
```

`W` is trainable raw state; `A` is the complete effective matrix used by the
existing [equilibrium reference](linear-pc-reference.md). The identity has no
trainable parameters and receives no update. No skip is added to the input or
output map, even if its shape happens to be square. `H=1` has no residual blocks
and reduces exactly to the previous MLP expressions.

Why this: a depth factor applies to the residual branch, not the identity. An
identity folded into a trainable raw matrix would define a different optimizer
and initialization. A separate module preserves the accepted MLP implementation
and its [fixed measurement results](reports/linear-scaling-measurement-20261014.md).

## Controls and gradient map

| Scheme | Raw initialization std | Branch factors: first / middle / output | Raw SGD factor |
|---|---|---|---|
| `legacy_gaussian` | all `.5` | `1 / 1 / 1` | `1` |
| `fan_in` | first `1/sqrt(D)`, others `1/sqrt(N)` | `1 / 1 / 1` | `1` |
| `mean_field` | all `1` | `1/sqrt(D) / 1/sqrt(N*L) / 1/(gamma0*N)` | `gamma0²*N` |

All three controls use identical fixed middle skip topology and the same
underlying standard-normal draws for a given shape/seed. The local Gaussian
controls are explicit baselines, not the author's SP uniform initializer or its
nonlinear experiments. Controls require `gamma0=1`; mean-field supports native
positive finite gamma0 when all required factors are representable. The depth
exponent is fixed to the paper-critical `.5`, not selected from final outcomes.

The **complete** matrices enter both prediction and the equilibrium denominator:

```text
prediction = X A_1 ... A_L
s = 1 + sum_(l=1)^(L-1) ||A_(l+1) ... A_L||_F²
E* = half_MSE / s
grad_raw_l = a_l * grad_effective_l E*
W_l_new = W_l - learning_rate * raw_factor * a_l * grad_effective_l E*
```

The denominator excludes the input matrix and includes the output-only suffix.
Its derivative must be retained. `LinearResidualStep.evaluation` is the
**preupdate** equilibrium in effective coordinates; its `weight_gradients` are
effective partials. The trainable raw gradient is the scaled effective partial.

Mean-field effective SGD factors are first `gamma0²*N/D`, middle `gamma0²/L`,
output `1/N`. There is no additional raw learning-rate depth multiplier. Both
local controls have effective factor one. This map applies to plain SGD; Adam,
weight decay, learned skips and altered precision schedules require independent
derivations. It makes no finite-depth stability or convergence guarantee.

## Ownership and refusal

`LinearResidualConfig` composes the unchanged MLP domain/allocation checks. More
than 1,000,000 raw entries are refused before RNG/allocation. This cap bounds raw
storage only; fixed identity matrices, effective arrays and other temporaries
also consume memory. `LinearResidualState` copies raw matrices. Owned arrays
remain mutable within a frozen record and are revalidated on public calculation.
Borrowed sources must be quiescent; this is not concurrent state or a decoder.

Effective matrices and replacement raw arrays are detached. Updates are
simultaneous, using the existing stabilized binary mantissa/exponent delta
product; source state remains unchanged if candidate arithmetic or its full
equilibrium is invalid. Invalid inputs raise `ValueError`; arithmetic may raise
`ValueError` or `FloatingPointError`. Finite tensors/rates do not guarantee a
finite or descending update. Final float64 rounding/underflow/overflow limits
remain; the method does not provide extended precision.

## Modules, checks and extension

```text
src/core/residual_linear_pc.py       config, copied raw state, effective skips, raw SGD
scripts/run_residual_linear_pc.py    fixed control example and JSON
tests/test_residual_linear_pc.py     independent math, schema and ownership checks
tests/test_residual_linear_pc_edge.py exact representable-update regression
tests/test_residual_linear_pc_cli.py actual CLI and complete matrix/energy checks
```

```powershell
python -m pytest -q tests/test_residual_linear_pc.py tests/test_residual_linear_pc_edge.py tests/test_residual_linear_pc_cli.py
python -m ruff check src/core/residual_linear_pc.py scripts/run_residual_linear_pc.py tests/test_residual_linear_pc.py tests/test_residual_linear_pc_edge.py tests/test_residual_linear_pc_cli.py
python -m ruff format --check src/core/residual_linear_pc.py scripts/run_residual_linear_pc.py tests/test_residual_linear_pc.py tests/test_residual_linear_pc_edge.py tests/test_residual_linear_pc_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/residual_linear_pc.py scripts/run_residual_linear_pc.py tests/test_residual_linear_pc.py tests/test_residual_linear_pc_edge.py tests/test_residual_linear_pc_cli.py
```

The script sets OMP/MKL/OpenBLAS threads to one before NumPy imports; no dependency
or persisted configuration is added. A future residual measurement runner must
keep the accepted evaluation isolation, fix its protocol before execution and
retain every seed/refusal. The separate [residual measurement study](residual-scaling-measurement.md)
now supplies one complete fixed grid; [its report](reports/residual-scaling-measurement-20261014.md)
records the finite linear scope. Nonlinear/binary/CPC scaling remains unfinished R1.8.
This code does not extend the binary/CPC learners or validate asymptotic scaling,
published figures, convergence or comparative learning benefit.
