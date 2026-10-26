# TRAIN-only continuous learner conditioning

`NumpyContinuousBCLearner.diagnose_training_conditioning(episode)` describes
standardized inputs, tanh saturation and actual objective gradients without
training. It complements the existing [axis/stage residuals](continuous-bc-diagnostics.md).
A matching complete TRAIN episode and TRAIN policy are required. Guard,
probe_train, probe_test and final_test episodes are refused; callers cannot
override the role. This API does not select probes or evaluation data.

## Structure and responsibilities

```text
src/core/continuous_bc_diagnostics.py     Pure bounded scalar summaries
src/adapters/numpy_continuous_bc.py      TRAIN admission, snapshot, standardization
tests/test_continuous_bc_conditioning.py Independent math and behavior checks
```

The adapter validates an owned full state, standardizes once with existing
frozen statistics (or makes the existing unscaled copy), and calls the core.
There is no fitting, update, RNG draw, publication, or budget renewal. Parameters,
momentum, counters, policy and episode remain exact; a subsequent ordinary update
matches an otherwise identical learner that skipped the diagnostic.

The pure `continuous_bc_conditioning(features, targets, parameters, spec,
binary_gripper_axis=None)` takes already standardized float64 arrays. It reuses
the actual existing forward/loss/gradient kernels. Its immutable result contains
native scalars and bounded tuples, with no model, gradient or activation aliases.
No IO, simulator, training orchestration or new dependency is introduced.

## Reading the report

```python
report = learner.diagnose_training_conditioning(train_episode)
print(report.conditioning.objective_definition, report.conditioning.objective_loss)
print(report.conditioning.feature_magnitude.rms)
print(report.conditioning.hidden_saturation.saturated_fraction)
for axis in report.conditioning.output_saturation:
    print(axis.index, axis.unit, axis.active, axis.saturation)
for gradient in report.conditioning.gradients:
    print(gradient.name, gradient.l2_norm, gradient.maximum_absolute)
print(report.standardization_values, report.conditioning.work)
```

- Feature magnitude: count, maximum absolute value, RMS and exact zero count.
- Hidden and active output saturation: fixed `abs(tanh) >= 0.99` count/fraction,
  minimum and mean `1 - tanh**2`. Constant output axes have `saturation=None`.
  Axis indices preserve distinct axes even when units repeat.
- Actual declared objective/loss and four gradient summaries, ordered
  `w1, b1, w2, b2`: count, L2 norm, maximum absolute value and exact zero count.
  Max-scaled reductions avoid avoidable squaring overflow; a nonrepresentable
  summary raises rather than silently returning infinity.

These are descriptive activation/gradient signals, not a matrix condition number
or proof of learnability. Under binary gripper cross-entropy, a saturated physical
tanh output can still have a nonzero logit gradient. The report preserves that
actual objective derivative. Default MSE and all optimizer/policy defaults remain
unchanged. Comparing initialization reports cannot establish trained skill,
retention, recovery or a circadian advantage.

## Bounds and accounting

Existing bounds apply: 1..256 rows, 1..1024 input dimensions, 1..256 hidden units,
1..16 action axes. One successful readout performs two forwards and one backward.
Why this: reuse working kernels without refactoring their objective just to expose
activations. For `R` rows, `D` inputs, `H` hidden units and `A` outputs, dense
multiply-accumulates are `R * (3*D*H + 4*H*A)`. This excludes elementwise math,
reductions, validation, copies and snapshot overhead.

The work record includes input/hidden/output/parameter/gradient value counts and
float64 gradient bytes. Input/hidden/output counts are batch tensor cardinalities,
not totals across both forward evaluations. The adapter separately counts one standardization call,
rows/values and whether a transform was applied. Failed/unreturned calls may
already have consumed work and do not yield a successful work report. Callers
must retain attempts/failures separately. Numeric payload is not process memory.

## Verification and extension

Run the focused scope with ordinary repository tools:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_bc_conditioning.py tests/test_continuous_bc_diagnostics.py tests/test_continuous_bc.py tests/test_continuous_bc_standardization.py tests/test_continuous_bc_binary_metadata.py tests/test_continuous_bc_metadata_compatibility.py tests/test_guard_only_teacher.py
.venv/Scripts/python.exe -m ruff check src/core/continuous_bc_diagnostics.py src/adapters/numpy_continuous_bc.py tests/test_continuous_bc_conditioning.py
.venv/Scripts/python.exe -m ruff format --check src/core/continuous_bc_diagnostics.py src/adapters/numpy_continuous_bc.py tests/test_continuous_bc_conditioning.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/continuous_bc_diagnostics.py src/adapters/numpy_continuous_bc.py
```

Independent hand calculations and scalar-loop finite differences cover MSE and
mixed cross-entropy, alongside role/corruption refusals, stable norms, explicit
kernel costs, ownership and next-update equivalence. Add new diagnostic summaries
in the pure module with independent numeric tests; preserve the TRAIN gate and
charge extra work explicitly. Any learnability study needs a separately frozen
small protocol and isolated learner before training begins.

Frozen one-run protocol, exact commands, actual outputs and limitations:
[conditioning evidence](../artifacts/runs/continuous-bc-conditioning-20261019/RESULTS.md).
The earlier [native guard rejection](../artifacts/runs/native-guard-promotion-20261019/RESULTS.md)
remains unchanged. Original R2.1 diagnostic acceptance is reused; broader learning
evidence, R4.4/R4.5 and full research/release requirements remain open.
