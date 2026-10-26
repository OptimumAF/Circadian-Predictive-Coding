# Direct categorical fit of frozen CLM features

This disposable control fits ordinary class scores directly from already encoded
CLM query features. It helps separate a representation's observed TRAIN fit from
the behavior of a warm nonlinear CLM head and its optimizer. It creates no native
learner, changes no source head, and loads no encoder or assets.

The method matches the existing categorical linear control: one
`numpy.linalg.lstsq`, fixed `rcond=1e-12`, equal-weight one-hot targets, an appended
constant-one intercept, then stable softmax. It uses the thin rectangular design
directly, rather than normal equations. No regularizer, seed, feature selection,
rescaling, learning rate or early stopping is chosen here. The fit minimizes
squared error, not cross-entropy. Its softmax is not calibrated confidence.

The new `WideCategoricalLinearFit` accepts at most16 rows,4096 features and16
classes. Each readout accepts1..16 rows. Pure core inputs are exact finite float64
arrays in[-1,1]; targets are hard class indices. Immutable coefficients, rank,
singular values and the actual squared residual are returned. The original
`CategoricalLinearFit` API keeps its296-feature/128-row limits and exact model
type. Only numeric solve/softmax internals are shared.

Why this: released CLM features have4096 coordinates, outside the existing game
control's contract. A separate small-row route supports them without broadening
old callers' resource bounds or introducing a different solver.

With existing admitted TRAIN features:

```python
from src.adapters.clm_linear_fit_control import (
    fit_clm_linear_control, predict_clm_linear_control,
)
from src.core.clm_head import FrozenChoiceBatch

report = {}
control = fit_clm_linear_control(train_batch, targets, report)
readout = FrozenChoiceBatch(
    train_batch.identity, train_batch.states, train_batch.actions, role="train",
)
probabilities = predict_clm_linear_control(
    control, readout, train_batch.candidate_ids, report,
)
```

`train_batch` must be an exact `FrozenCatalogTrainingBatch` with matching
`CLMCatalogTargets`. Alternatively an exact `FrozenProbeCatalogBatch` creates a
separate `probe_train` control; it accepts `FrozenProbeChoiceBatch` readouts with
`train`, `probe_train` or `probe_test` role. A TRAIN control accepts
`FrozenChoiceBatch` with `train` or `outer_selection` role. Neither path admits
`final`. Readouts take no labels. No final, promotion, tuning, controller or replay
selection hooks are added. Probe labels never enter the source learner.

The adapter requires detached contiguous CPUfloat32 query/action matrices with
exact logical backing storage and finite coordinates in[-1,1]. Fits require2..16
queries and2..16 ordered candidate IDs. Width must match the declared encoder;
readout encoder and candidate order must match the fitted control. The upstream
encoder supplies normalization; this adapter checks the coordinate bound without
rescaling or certifying unit norms/provenance.

Class scores use query features and the fitted class coefficients. **Action
embeddings are validated but never used in the linear score calculation.** Their
values are not a cross-call candidate-description certificate. This is a fixed
class control, not the contrastive state/action objective or a way to add unseen
candidates. A future typed decision provider must bind actual catalog descriptions
and preserve permission enforcement. Array argmax ties use the first class, as in
the old linear control; a typed provider must apply its own declared tie policy.

Inputs are copied for the detached solve and checked after return. Replaced
owners, changed identity/labels/features/storage/version, or model changes during
readout refuse publication. Output probabilities own separate storage. The
returned control retains no source tensor, learner, optimizer or RNG state, and
offers no native train, restore, fork or installation method. Trusted local Python
guards do not contain hostile code or attest dataset provenance.

Use a fresh report destination: one `linear_fit` key and one
`linear_prediction` key per dictionary. Returned numeric work and successful
publication are distinct: `returns=1` can coexist with `complete=False` after a
post-return guard failure. Exceptions retain attempted rows, error and
`unreturned_work_unknown=True`; missing returns are never zero-cost work. The
control itself has no shared experiment budget. The existing caller/protocol
must count every fit, repeated exposure, readout, source/probe conversion and
encoder operation, including failures. A new dictionary is not renewed authority.

Reports expose planned logical input/guard/design/one-hot/coefficient/probability
array footprints and operation timing. These are not total allocations, cumulative
IO, Python tuple overhead, LAPACK workspace or FLOPs. Zero native-head/encoder/
optimizer fields refer only to this detached operation; producing the features
has its own cost. No unbounded internal success history is kept.

A perfect observed TRAIN fit does not establish fresh-task transfer, closed-loop
success, retention, context-return recovery, or a circadian advantage. An imperfect
minimum-norm fit does not generally prove that no linear classifier could succeed.
Keep all negative fits, frozen comparison directions and spent experimental
allowances. A new native question needs a separately frozen local protocol.

Structure and extension:

```text
src/core/categorical_linear_fit.py         original bounded API/shared arithmetic
src/core/wide_categorical_linear_fit.py    distinct bounded wide model/fit/readout
src/adapters/clm_linear_fit_control.py     existing CLM roles, conversion and costs
tests/test_wide_categorical_linear_fit.py  analytic numeric/legacy references
tests/test_clm_linear_fit_control.py       wide, roles, ownership and paid failures
```

Core imports NumPy and other core modules; the adapter owns Torch conversion.
There is no new configuration, dependency, CLI or orchestration framework. Next
integration should use the existing typed decision/workflow ports, with fixed
class/catalog binding and original external permissions. Native useful behavior
remains a separate qualification.

Tests perform bounded literal LAPACK work. Run them only under a declared
engineering allowance; exact Windows/Linux commands, costs, failures, self-review
and limits are in [the qualification report](reports/clm-linear-fit-control-20261025.md)
and [issue #157](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/157).
