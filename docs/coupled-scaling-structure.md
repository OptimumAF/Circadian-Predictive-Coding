# Raw coupled split and compatible merge

`python -m scripts.run_coupled_scaling_structure --describe` previews the bounded
engineering fixture without model work. Run without `--describe` for complete
PC/neutral/active CPC states, energy/gradient checks and learner continuation.
CPU thread settings are one before imports; output is bounded to2MiB. No new
dependencies, configuration variables, scientific study or final-test data.

## Modules and API

```text
src/core/coupled_scaling_structure.py      pure owned transformation and lineage
scripts/run_coupled_scaling_structure.py   fixed native learner example and JSON
tests/test_coupled_scaling_structure.py    independent math/state/refusal controls
tests/test_coupled_scaling_structure_cli.py actual bounded CLI behavior
```

`split_coupled_scaling_width(raw_state)` returns `CoupledScalingWidthChange`
containing a detached state, `CoupledScalingSplitLineage` and the maximum effective
encoding error. `merge_coupled_scaling_width(raw_state, lineage)` returns the same
result type. Lineage format1 owns the complete original config; its target config
and fixed replication factor2 are derived. It is metadata, not a raw checkpoint,
runtime history, provenance or serving authority.

Why this: the native learner has uniform hidden width and scalar precision per
layer. Replicating every coordinate equally permits energy preservation using
those scalar precisions. An uneven single-unit split would need a different
precision/coordinate contract to offer the same identity. This operation keeps
the current learner equations and validates the target domain before expansion.

## Mathematical contract

Let R duplicate each source coordinate into adjacent pairs, and let S=R^T/2,
so RS=I. For row latents z'=zR:

| Field | Split effective coordinates |
|---|---|
| Width | 2N in every hidden layer |
| Hidden precision | beta/2 per layer |
| First hidden W | WR |
| Later hidden W | SWR |
| Hidden bias | bR |
| Output W | SW |
| Output bias | unchanged |
| CPC chemicals/gates | duplicate adjacent pairs |
| CPC policy/wake count | unchanged |

Duplication commutes with tanh, so hidden predictions and residuals duplicate.
Their squared norms double, compensated by beta/2; output logits stay unchanged.
In exact arithmetic E'(repeat z)=E(z) for arbitrary source latents and every
target latent-gradient clone equals G/2. Simultaneous fixed inference proposals
stay replicated if the latent step size doubles. Guarded decisions can differ
from rounding; a gradient inf-norm tolerance would also need halving. No adaptive
stopping or convergence identity is promised.

Raw parameters are re-encoded under the new width's weight and bias factors.
Gamma, scheme, input/depth and CPC policy stay fixed. Beta/2 must be valid and
exactly reversible. Effective middle/output halves must multiply back exactly
before raw encoding; odd subnormals that lose information are refused. Existing
raw pruning encoding helpers enforce a per-entry8eps relative, zero-absolute
reconstruction bound, with nonzero/sign/zero checks. This is a representation
check, not a global function/error bound. Raw-to-effective decoding can already
round or underflow; an exact original raw tensor roundtrip is not promised.

## Compatible merge and continuation

Merge revalidates every field against the lineage's full target configuration.
Every adjacent raw hidden column, middle row, hidden bias, output row and CPC
chemical/gate pair must be exactly equal. One asymmetric pair refuses; there is
no tuned compatibility tolerance. It selects representatives, doubles effective
middle/output weights, checks finite representability, restores source width and
precision, then re-encodes with the same guard. The current wake count remains.
Symmetric trained states may merge; this does not restore the old trained model.
Invalid or late-failing transforms never mutate the borrowed, quiescent source.

Construct a new `CoupledScalingLearner` for the returned state and choose its
fixed update settings explicitly. The example doubles the latent inference rate
and keeps the raw parameter learning rate fixed. No source/widened raw SGD
trajectory identity follows: effective learning factors change, and effective
bias SGD scaling targets1 while per-clone energy gradients halve; actual stored
float factors are reported. Forks preserve
their chosen settings and complete detached state.

Existing strict raw restore rejects changed geometry/configuration. The existing
topology adapter also rejects changed precision. This increment does not relax
either binding, combine split/prune histories, publish to serving, or add an
optimizer, arbitrary merge, codec or nonuniform learner.

## Checks and example

The mean-field N3/H2/gamma1.5/beta(.8,1.2) example uses two genuine source wakes,
nonzero biases/CPC history, split, arbitrary-latent energy/gradient checks and
immediate merge. A widened learner and owned fork take one matched wake each;
the continued symmetric state merges and is checked again. An asymmetric middle
pair refuses. PC and neutral CPC continued raw parameters remain exactly equal.
Tolerances1e-12 apply only to this fixed numerical fixture. Complete states and
gradients are returned, with explicit changed forward/bias/learning factors.

```powershell
python -m pytest -q tests/test_coupled_scaling_structure.py tests/test_coupled_scaling_structure_cli.py
python -m ruff check src/core/coupled_scaling_structure.py scripts/run_coupled_scaling_structure.py tests/test_coupled_scaling_structure.py tests/test_coupled_scaling_structure_cli.py
python -m ruff format --check src/core/coupled_scaling_structure.py scripts/run_coupled_scaling_structure.py tests/test_coupled_scaling_structure.py tests/test_coupled_scaling_structure_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/coupled_scaling_structure.py scripts/run_coupled_scaling_structure.py tests/test_coupled_scaling_structure.py tests/test_coupled_scaling_structure_cli.py
```

Use a separately derived full-history/configuration contract before integrating
mixed split/prune operations into the runtime. Keep original non-width bindings
and independent serving guards until that transfer has explicit tests and review.
