# Explicit fixed-range normalization for public RGB features

The existing continuous recipe can explicitly use the frozen PickCube RGB recipe
to fit only21 public vector features (qpos9/qvel9/publicgoal3) from TRAIN-A, and
normalize the768 pooled RGB features with fixed mean.5/scale.5. The default
empirical fit across all coordinates remains unchanged.

Why this: stored B-context inputs under the empirical transform had RGB RMS77.86
and maximum1967.31, while RGB sensor values have a declared[0,1] range. Fixed
scaling maps that modality to[-1,1] without a fitted pixel variance or tuned floor.
The vector prefix can still shift and is not bounded by this option. Earlier
native failures and negative corrective/mixed-objective results remain valid;
this diagnostic does not prove their cause.

## Use

```python
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.adapters.numpy_continuous_bc import fit_training_standardization
from src.adapters.continuous_learnability_demo import run_continuous_learnability

recipe = pick_cube_rgb_recipe()
statistics = fit_training_standardization(warm_train, rgb_range_recipe=recipe)
result = run_continuous_learnability(warm_train, new_train,
    expected_standardization=statistics, rgb_range_recipe=recipe)
```

The first call above demonstrates the fitting API separately. The actual fixed
recipe fits once internally; callers reusing a prospective expected transform
should not add another fit without charging it. The stored experiment constructs
the expected prefix from accepted unchanged A statistics and fixed RGB constants.

```powershell
.venv/Scripts/python.exe -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --visual-observation --rgb-range-normalization --warm-teacher artifacts/runs/pick-cube-rgb-features-20261019/teacher-visual.json --new-teacher artifacts/runs/visual-context-return-20261019/teacher-b.json
```

This command performs24 updates; it does not request rerunning accepted evidence.
Declare a separate budget for new scientific cases. The range flag refuses other
modes or missing visual mode before teacher IO. The binary-gripper option remains
independent; the stored normalization control uses MSE only.

Both result and JSON APIs accept `rgb_range_recipe=None` by default. Omitting it
preserves previous delegate calls, math, reports and work. The supplied recipe
must be the exact existing frozen `PickCubeRGBRecipe`, including feature order,
camera/renderer/pooling and pixel-range metadata. There is no adjustable scale,
floor, feature selection, clipping or new observation order.

## Validation, fitting and state

Pure `validate_rgb_range_features` admits only native finite float64 arrays with
shape `(1..256,789)` and RGB coordinates in[0,1], and returns an owned copy.
`fit_rgb_range_standardization` fits existing stable population statistics on
the first21 columns only, then concatenates768 fixed means/scales. Constant vector
coordinates keep the existing scale1 behavior and preserve new-input deviations.
Pure array callers own release authority; the BC fit adapter admits TRAIN roles
before entering core math.

The fixed recipe owns/revalidates both complete TRAIN episodes, all training
observations and terminal next observations before fitting or model creation.
The direct TRAIN fitter also validates its terminal next observation; terminal
rows are never fitted or trained on. B is range-validated, never fitted. The
expected transform gate precedes all model initialization and SGD.

The policy keeps the existing flat owned `ObservationStandardization`; no policy,
snapshot or prediction-kernel schema changes. Fork/restore/actor/runtime paths
therefore retain the entire effective transform as before. Explicit report
metadata identifies which coordinates were empirical versus fixed: fitted_rows
counts the A rows used for the empirical prefix, not768 fitted RGB variances.
Generic raw-array prediction still trusts the caller's feature contract and does
not clip/range-check RGB; the native frozen encoder supplies declared[0,1] values.

## Costs and interpretation

Explicit reports add `input_normalization` with full recipe/definition,
21 fitted/768 fixed dimensions, constants and validation counts. Successful work
changes `fit_feature_values` to Arows*21 and adds `fixed_range_dimensions=768`.
All full789D application/training/diagnostic costs remain accounted as before.

Actual A29/B30:609 fitted prefix values instead of22881 empirical values;
88 training-input validation rows plus3 terminal validation rows,71799 full-vector
coordinates checked. Validation scans/copies, prefix-fit copies, temporary model
templates, RNG allocation and process overhead are separate. This is not a claim
of equal total FLOPs or latency despite equal dense model work.

## Structure and checks

```text
src/core/pick_cube_rgb_standardization.py   Exact recipe/range validation and pure fit
src/adapters/numpy_continuous_bc.py         Existing TRAIN-only fitting boundary
src/adapters/continuous_learnability_demo.py Existing fixed sequence, explicit option
scripts/run_native_stream_demo.py          CLI admission/forwarding
tests/test_pick_cube_rgb_standardization.py Math, ownership, early refusals, integration
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_pick_cube_rgb_standardization.py tests/test_continuous_bc_standardization.py tests/test_continuous_learnability_demo.py -k 'not direct_reference'
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_rgb_standardization.py src/adapters/numpy_continuous_bc.py src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_pick_cube_rgb_standardization.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_rgb_standardization.py src/adapters/numpy_continuous_bc.py src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_pick_cube_rgb_standardization.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_rgb_standardization.py src/adapters/numpy_continuous_bc.py src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py
```

[Frozen protocol, exact executed checks, results and review](../artifacts/runs/rgb-range-normalization-20261019/RESULTS.md).
The offline control uses identical architecture/seed/numeric initialization and
24-update exposures, with physical action half-MSE compared at fixed endpoints.
Different input transforms intentionally change initial predictions. Preserve
all per-context/axis/forgetting tradeoffs. No native case, failure-cause proof,
generalization or circadian advantage follows from implementation correctness or
a small offline improvement. Future work needs its own frozen protocol and costs.
