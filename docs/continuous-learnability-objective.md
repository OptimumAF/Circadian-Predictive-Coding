# Explicit objective selection in the fixed continuous recipe

The existing result and JSON APIs accept `binary_gripper_axis=None` (default MSE)
or an explicit action index using the already implemented mixed binary-gripper
objective. The fixed sequence, architecture, seed, optimizer, statistics, label
arrival, budget and physical residual metrics remain unchanged.

```python
result = run_continuous_learnability(warm_train, new_train,
    expected_standardization=frozen_a_stats, binary_gripper_axis=3)
```

```powershell
.venv/Scripts/python.exe -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --visual-observation --binary-gripper-axis 3 --warm-teacher artifacts/runs/pick-cube-rgb-features-20261019/teacher-visual.json --new-teacher artifacts/runs/visual-context-return-20261019/teacher-b.json
```

This command trains; it is not a request to rerun the stored experiment. Reuse
the accepted evidence and declare a new budget before any new scientific case.
The flag is refused for every other scenario before teacher reads. Omitting it
preserves the previous CLI calls and report. Python wrappers forward the keyword
only when explicitly supplied, preserving existing default delegates.

## Validation and semantics

The selected axis must be an exact native integer, have `normalized_gripper`
units and bounds `(-1,+1)`. Both complete TRAIN episodes must have strict endpoint
labels on that axis. Role/spec/identity admission and this validation precede
fitting or model creation. MSE continues to admit its original continuous labels.

There is no new mathematical loss: the existing core uses binary logit `2z`
cross-entropy on that axis and half-MSE on the others, with the existing active
axis denominator. Physical predictions still use bounded tanh. Actual training
receipts and conditioning reports carry the policy's objective definition.
The common physical-action half-MSE and all axis/stage diagnostics stay separate.
Unlike MSE and mixed training losses must never be ranked numerically.

The default report's work keys remain unchanged. Mixed reports add
`binary_gripper_loss_rows`: training rows plus conditioning loss/gradient rows.
For the fixed visual case this is 704+236=940. Dense MACs, gradient shapes and
batch counts stay the same; added CE elementwise math means computation and
latency are not identical. RNG initialization/templates, copies, validation and
process memory are accounted separately.

## Structure and tests

```text
src/adapters/continuous_learnability_demo.py   Existing recipe/policy selection
scripts/run_native_stream_demo.py             Explicit CLI flag
tests/test_continuous_learnability_binary.py  Paired ownership/objective/gate tests
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_learnability_binary.py tests/test_continuous_learnability_demo.py tests/test_continuous_learnability_result.py tests/test_continuous_stream_demo.py tests/test_visual_continuous_stream.py -k 'not direct_reference'
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_learnability_binary.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_learnability_binary.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py
```

Paired tiny fixtures check all eight initial parameter/velocity arrays, actual
receipt/objective/conditioning consistency, fixed endpoints/common metrics and
costs. Invalid labels/axis/unit/bounds and illegal CLI modes refuse early. Direct
references run once within the frozen engineering budget and are excluded from
repeated/platform checks; previously accepted references are reused.

## Scientific limits and extension

The small visual study reuses the accepted unchanged MSE control and executes
one new mixed-loss arm, keeping all endpoints and translation/gripper tradeoffs.
Two extra zero-update templates verify numeric initialization parity. The old
report stores no raw model, so evidence for actual initialization is deterministic
constructor identity, independent tests/templates and initial physical reports.
No new MSE optimization or favorable checkpoint selection is performed.

Full reports preserve mean/max gripper error. Wrong-sign fractions are unavailable
because candidate per-row predictions were not retained; no extra calls are added
after outcomes. The [earlier 42D binary study](../artifacts/runs/continuous-bc-binary-gripper-20261019/RESULTS.md)
contains translation/grasp regressions, and [both MSE native rollouts failed](../artifacts/runs/visual-mlp-native-20261019/RESULTS.md).
Neither history is erased by a new result. No native run, guard/probe/final tuning,
new task, generalization or circadian advantage is established by offline fitting.

Use the existing policy/kernels for later controlled recipes. Freeze seeds,
metrics/directions/exposure/endpoints/costs before changing a comparison; keep
held-out data unavailable. [Frozen protocol, exact checks and observed results](../artifacts/runs/visual-mlp-binary-control-20261019/RESULTS.md).
