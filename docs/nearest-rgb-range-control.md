# Fixed RGB-range nearest correction control

This local study tests an existing distance transform using the accepted nearest
corrective pipeline. Production code is unchanged. The empirical pilot is reused
from saved evidence; it was not retrained or rerun.

```text
artifacts/runs/nearest-rgb-range-control-20261020/
  native_case.py       fixed case; one real fit and one native invocation
  PROTOCOL.md          engineering scope and metadata fixture budget
  NATIVE_PROTOCOL.md   frozen intervention, endpoints and resource limits
  native-case/         original collection/update/candidate rollout reports
  native-result.json  all metrics, saved control and comparisons
tests/test_nearest_rgb_range_case.py   metadata and stub-boundary tests
```

## Intervention

The existing TRAIN-only `fit_training_standardization` fits A29's 21 public
coordinates and uses fixed RGB means/scales 0.5, mapping the declared sensor
range to [-1,1]. The new transform must exactly match the old public prefix plus
those fixed constants before model construction. No weight or variance floor is
tuned. Fit attempts and returns are retained before that validation.

Everything else keeps the empirical recipe: A29, RGB789, max256 prototypes,
TRAIN17/19, official controller, one provisional-label append, and candidate
A17/B19 readouts. Earliest original ties remain. No final, probe, guard or
promotion input is used. Provisional targets do not become qualified teachers.

The transform can change the source trajectory and collected labels. These are
whole-recipe outcomes, not a comparison on identical corrections or matched
compute. Prior MLP range failures remain negative and were not rerun.

## Observed outcome: no task improvement

| Endpoint | Empirical return | Range return | Range controls |
|---|---:|---:|---:|
| Source B19 |0|0|50|
| Candidate A17 |1|1|29|
| Candidate B19 |0|0|50|

All six signed acquisition/retention differences and all nine raw endpoint
differences are zero. A's full29 transitions equal the qualified teacher and
empirical readout. Both B trajectories and prototype selections changed, with
no improvement in task success. This does not prove RGB scaling is irrelevant
or identify the sole failure cause.

One real fit used29 rows ×21 coordinates. One RNG-free nearest constructor,
two appends/79 row exposures and129 controls/645 known control-physics steps
followed. The original source stayed1/29; the candidate was2/79. There were50
provisional geometry labels, no optimizer or gradient calls. The accepted fixed
case is spent; do not rerun it to confirm documentation.

## Verification and commands

The same24 metadata-only tests pass on Windows and pinned Linux, with zero
skips. They stub all actual models, fits, array kernels and native work. Existing
27-case production and normalization/learner/runtime proofs were reused without
changes. Scoped Ruff, format, Linux-target mypy and independent source/actual
review passed. One unused-import lint failure was repaired and preserved.

```powershell
python -m pytest -vv tests/test_nearest_rgb_range_case.py
python -m ruff check artifacts/runs/nearest-rgb-range-control-20261020/native_case.py tests/test_nearest_rgb_range_case.py
python -m ruff format --check artifacts/runs/nearest-rgb-range-control-20261020/native_case.py tests/test_nearest_rgb_range_case.py
python -m mypy --follow-imports=silent --platform linux --python-version 3.12 artifacts/runs/nearest-rgb-range-control-20261020/native_case.py tests/test_nearest_rgb_range_case.py
```

[Exact commands, costs, failures, resource bounds and next action](../artifacts/runs/nearest-rgb-range-control-20261020/RESULTS.md) ·
[Independent review](../artifacts/runs/nearest-rgb-range-control-20261020/REVIEW.md).
No generalization, CPC advantage, physical joint-constraint coverage or broad
repository readiness follows from these scoped results.
