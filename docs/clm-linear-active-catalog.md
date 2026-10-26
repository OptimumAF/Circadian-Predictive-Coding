# Active tool catalogs for the linear typed provider

`CLMLinearDecisionPredictor` accepts an optional `active_candidates` tuple. It
binds a read-only subset of the full fitted catalog for one provider's lifetime.
The default `None` keeps the original full-catalog behavior, work fields and
readout metadata. No fit, update, encoder extraction or optimizer operation is
performed by choosing a catalog.

Why this: the temporal learner has fixed columns for `defer, inspect, retire,
route`, while v1 permits `inspect, retire, route` and v2 permits `defer, inspect,
route`. Column positions cannot be reused for different tools. The full model
and its original fit metadata remain intact.

```python
from src.adapters.clm_linear_decision import CLMLinearDecisionPredictor

allowed_ids = ("defer", "inspect", "route")  # Explicit public policy declaration.
active = tuple(c for c in action_features.candidates if c.candidate_id in allowed_ids)
predict = CLMLinearDecisionPredictor(
    encoder, fitted_train_control, action_features,
    instructions="Choose the ticket workflow tool.", active_candidates=active,
)
```

`action_features` still contains the **full** original catalog in fitted order,
with the same encoder identity. The active declaration must be an immutable
tuple of2..all source candidates, with original IDs, descriptions and relative
order. Unknown IDs, duplicates, singleton/empty catalogs, reordered candidates
and changed descriptions fail before query encoding. Each request must exactly
match the bound active declaration. It is not a per-request choice driven by
scores, labels or ambient state.

The core selects coefficient columns before stable softmax. An inactive extreme
score cannot win or underflow all active scores through full-catalog
normalization. No projected `WideCategoricalLinearFit` is created: original
rank, error, row count, coefficients and class metadata remain the original
fit's evidence. The projection affects only the current readout array.

The existing role-bound readout port also supports an explicit active-ID tuple:

```python
probabilities = predict_clm_linear_control(
    control, active_choice_batch, control.candidate_ids, report,
    active_candidate_ids=("defer", "inspect", "route"),
)
```

The positional catalog remains the full source IDs. `active_choice_batch.actions`
contains the declared active rows, in order. TRAIN and disposable-probe readout
roles stay separate; this permission cannot make a probe deployable. The typed
provider rejects probe controls before action access or query encoding. No
final-data access, export, promotion or external tool effect is introduced.

The provider owns one full source-control copy and one active action-row copy,
with no binding/model history chain. Advanced row selection already owns its
storage; validation compares row views without repeatedly gathering new tensors.
Source owners/values, including **excluded** weights, active declaration objects,
active action storage, shared encoder work and paid counters are checked around
operations. A lost/unadmitted paid result keeps its uncertain prefix and stops
the provider. The encoder's observed completed return can remain paid even if
corrupted provider counters prevent admitting that return locally.

For explicit active mode only, `work` adds source/active class counts, retained
active action-copy elements, and active readout attempts/returns. Per-readout
metadata records source/active IDs, columns, source coefficient logical payload,
the active coefficient array and read-only projection flag. Logical array sizes
are not all Python copies, validation views, allocation peaks or FLOPs. Caller
budgets must cover full-model guards, projection and encoding. This API grants
no experiment or retry allowance.

A catalog change requires a new explicitly declared provider scope. Shared
encoder limits/work remain cumulative. Older providers reject intervening
encoder work; they do not silently accept counter changes or renew limits.
External workflow permissions and effects retain their existing enforcement.

```text
src/core/wide_categorical_linear_fit.py   pure ordered-column probability readout
src/adapters/clm_linear_fit_control.py    existing role/identity/work boundary
src/adapters/clm_linear_decision.py       typed declaration, owner and encoder guards
tests/test_wide_categorical_active_readout.py  independent literal softmax oracles
tests/test_clm_linear_active_catalog.py       provider, roles and paid-failure checks
```

No new dependencies, configuration or orchestration framework. The explicit
[context workflow mode](clm-linear-context-workflow.md) binds public policy tools,
keeps the six-tool union TRAIN catalog fixed across warm updates, and counts
per-scope copies and shared encoder work. Default driver behavior is unchanged.
Literal correctness is not a native adaptation result.

Focused commands (observe the owning issue's limits before executing):

```powershell
python -m pytest -q tests/test_wide_categorical_active_readout.py tests/test_clm_linear_active_catalog.py
python -m ruff check src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_fit_control.py src/adapters/clm_linear_decision.py
python -m ruff format --check src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_fit_control.py src/adapters/clm_linear_decision.py
python -m mypy --follow-imports=silent src/core/wide_categorical_linear_fit.py src/adapters/clm_linear_fit_control.py src/adapters/clm_linear_decision.py
```

The [software report](reports/clm-linear-active-catalog-20261025.md) records actual
checks, failures, costs, skipped work and self-review scope.
