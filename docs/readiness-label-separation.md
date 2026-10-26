# Readiness input/target consistency

`src/core/continuous_bc_diagnostics.py` provides
`continuous_input_target_consistency(features, targets, spec)` for1..100 finite
float64 rows and four normalized action axes with endpoint gripper targets.
It retains every unordered pair, signed second-minus-first action differences,
robust Euclidean norms, exact-input conflict flags and deterministic nearest
other-row indices. A singleton has no nearest row. Raw action differences and
float32-encoded command differences are separate. No proximity threshold is fit.

`src/adapters/readiness_label_separation.py` composes that helper with existing
teacher/correction ownership and executed-history projection. It accepts an
already qualified-boundary `ReadinessStageBinding` and an exact provisional v2
correction batch from the same TRAIN context, plus original release time:

```python
report = run_readiness_label_separation(
    teacher_binding, corrections, now=original_release_tick,
    recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
)
```

Why this: distinguish conflicting supervision from representation or learning
limitations before spending another training run. External teacher files still
require the existing strict qualification reader. This API does not qualify
mixed-controller labels. Original target actions, actually executed commands,
stage/version/source identities and query ticks remain distinct. Query arrival
is validated on raw evidence before projections or pair arithmetic.

Raw42 and taskspace12 have different units; their distances are not comparable
performance scores. The taskspace view advances history using actual commands,
never queried targets. No learner, RNG, fitting, update, simulator, final data,
probe data or promotion path is exposed. This is a TRAIN diagnosis, not proof
of learnability, causal ambiguity, autonomy or generalization.

Invalid admission raises before projection. Numerical/projection failures return
`complete=False` with an error and attempted/returned work; later views stop.
Projection-internal failed work is bounded but explicitly unknown. Named logical
array/pair payload counts exclude validation, copies, containers and total RSS.

## Scoped validation and saved result

34 focused checks passed on Windows and pinned Linux with independent scoped
acceptance. The separately frozen saved diagnosis retained all930 pairs from
31TRAIN rows. Each view had one shared-reset duplicate with identical targets;
no exact input/target conflict was observed. This does not establish adequate
features or explain the control failure. No model or simulator work was spent.

```powershell
./.venv/Scripts/python.exe -X utf8 -B -m pytest -vv --maxfail=1 tests/test_readiness_label_separation.py
./.venv/Scripts/python.exe -m ruff check src/core/continuous_bc_diagnostics.py src/adapters/readiness_label_separation.py tests/test_readiness_label_separation.py
./.venv/Scripts/python.exe -m ruff format --check src/core/continuous_bc_diagnostics.py src/adapters/readiness_label_separation.py tests/test_readiness_label_separation.py
./.venv/Scripts/python.exe -m mypy --follow-imports silent src/core/continuous_bc_diagnostics.py src/adapters/readiness_label_separation.py tests/test_readiness_label_separation.py
```

The saved-input diagnostic uses a separate prospective protocol; its one-use
driver is not authority to repeat scientific work. [Evidence and limitations](../artifacts/runs/readiness-label-separation-20261020/RESULTS.md).
Extend the pure helper with explicit semantics and small literal oracles; keep
role admission and original command-history projection at the adapter boundary.
