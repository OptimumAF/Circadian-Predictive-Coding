# Canonical state42 readiness corrections

Use the existing private correction pipeline with an explicit canonical feature
view. The original raw trajectory and provisional readiness labels remain the
collection evidence. Feature projection does not qualify arbitrary teacher labels
or establish successful grasping, useful adaptation or a circadian advantage.

## Files and boundaries

```text
src/core/pick_cube_state_corrections.py       pure detached raw/model batch view
src/adapters/continuous_correction_demo.py    existing arrived private updates
scripts/maniskill_continuous_smoke.py         existing query/update/frozen return
 tests/test_state_quaternion_corrections.py   independent composition/failure tests
```

The exact readiness batch owns all original raw42 observations, student actions,
rewards, label targets/stages and arrival ticks. The derived model batch applies
the accepted sign recipe to observation and next_observation, reusing each projected
transition for its paired label. The model batch is explicitly a feature view;
`source_encoding=official_state42_raw` and `model_encoding` identify the distinction.
Projection work includes2n observations/4n quaternion unit checks/16n components
for n labels, plus copied label observation/target payload. No source_recipe/RGB
claim is attached. Fixed-schedule batches cannot silently become readiness labels.

The adapter gates original arrival BEFORE projection/forks. It reuses eight BP/PC/
CPC private updates or one nearest append, preserving frozen statistics and source/
actor. Canonical BP additionally binds its exact native owners and original birth
budget, clocks, permanent receipt object prefix and paid charge through callbacks.
Failed work retains receipts/progress and releases no candidate. Other defaults
stay. Probe/final/guard labels, normalization refitting, deployment and promotion
are absent.

## Existing API composition

Declare a separate native budget before actually invoking this code:

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_corrective_adaptation
from src.core.pick_cube_state_quaternions import state_quaternion_projection

# source is an owned TRAIN learner fitted/trained with this same canonical recipe.
summary = run_corrective_adaptation(
    source,
    output_directory=Path("fresh-canonical-acquisition"),
    visual_observation=False,
    observation_projection=state_quaternion_projection(),
    readiness_corrections=True,
    acquisition_only=True,
)
```

Explicit acquisition_only requires canonical readiness/nonvisual42. It collects
TRAIN17, applies the existing private update, then runs exactly one frozen TRAIN17
return. Protocol `corrective_robot_train_17_acquisition_v1` is base acquisition;
it supplies no B-learning or old-task retention metric. Default False/omitted
keeps the original19 collection →17 return →19 return and old output fields.
Each path has an exclusive output directory, no retry or budget renewal.

Student executed gripper history controls readiness closing progress; proposed
teacher closes alone do not. Raw query pairing/native guards survive projection.
Before more work and after final observers, the canonical driver verifies bounded
prior raw/model evidence and exact source/candidate state. It retains at most one
collection, one update and two return reports, not an unlimited success chain.

## Checks

Exact selected pytest commands and fixture costs are in
[TEST_COMMANDS](../artifacts/runs/state42-canonical-corrections-20261020/TEST_COMMANDS.json)
and [prospective fixture budget](../artifacts/runs/state42-canonical-corrections-20261020/FIXTURE_BUDGET.md).
Focused statics (same four paths for format/type):

```powershell
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_state_corrections.py src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_state_quaternion_corrections.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_state_corrections.py src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_state_quaternion_corrections.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/pick_cube_state_corrections.py src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_state_quaternion_corrections.py
```

No dependency or environment variable was added. Extend the established projection
and existing private update contracts for further recipes; keep raw evidence and
arrival authority separate, and budget native work only after focused validation.

## Verified scope

130 focused cases per platform and scoped lint/format/types pass, independently
accepted. Tests include the joined TRAIN17 fake collection/update/return and paid
failure boundaries. [Executed results and costs](../artifacts/runs/state42-canonical-corrections-20261020/RESULTS.md).
Native acquisition usefulness remains a separate question; previous canonical
source control is negative. Fixture peak RSS was not measured.

The one [native acquisition case](../artifacts/runs/state42-canonical-acquisition-20261020/RESULTS.md)
completed:50ALIGN labels/eight privateupdates400rows lowered successive PRE-UPDATE batch diagnostics (last beforeupdate8), but frozen
TRAIN17 success stayed zero. Full acquisition/reliability requirements remain open.
Next prerequisite is explicit bounded teacher roll-in with per-control provenance
for later-phase acquisition coverage, before another frozen experiment.
