# Explicit sign-stable state42 features

The existing learned-robot command accepts `canonical_state_quaternions=True` or
`--canonical-state-quaternions`. It projects both qualified TRAIN episodes before
the single A normalization fit, then applies the same recipe at frozen inference.
It keeps the original official state42 collector and raw native trace unchanged.

```text
src/core/pick_cube_state_quaternions.py          pure recipe/vector/episode projection
scripts/maniskill_continuous_smoke.py           existing frozen projection and paid work
scripts/run_continuous_robot_learning.py        opt-in training/return composition
tests/test_pick_cube_state_quaternions.py        pure mathematical/role validation
tests/test_state_quaternion_robot_learning.py    actual fit/serving and failure boundaries
```

## Behavior and provenance

The fixed recipe selects the first largest-absolute component of each unit
quaternion and flips all four components when that component is negative. TCP
slice22:26 and object32:36 use the pinned official42 layout. Signed quaternion
zeros become positive zero; all other coordinates/actions/rewards/IDs/contexts/
endflags remain. Full finite input, exact recipe metadata and unit norm tolerance
1e-4 are validated, without rescaling. q/-q invariance and deterministic ties do
not establish a globally continuous orientation chart.

Reports explicitly retain `observation_projection`, `input_source_episodes` and
`input_projection_work`; native reports retain the same recipe, exact-projection
flag and six work counters. The learner consumes derived features while the
native episode records original raw42. Original collection files remain the
qualification/acquisition receipts. No teacher is requalified by projection.

The default False/omitted option keeps original APIs/outputs. True requires
nonvisual state42; visual/clock/actor/intervention combinations are
not admitted by this recipe. Explicit readiness corrections are supported through
the [existing private correction composition](state42-canonical-corrections.md). The existing raw42 saved diagnostic refuses projected
results rather than applying feature statistics to a differently encoded raw trace.

## Usage

This example spends a fresh model fit and simulator episode. Freeze/review a
local budget first; do not repeat a spent evidence destination:

```powershell
.venv/Scripts/python.exe -m scripts.run_continuous_robot_learning --output fresh-canonical-state.json --warm-teacher artifacts/runs/state-readiness-acquisition-20261020/teacher-17.json --new-teacher artifacts/runs/state-readiness-acquisition-20261020/teacher-19.json --readiness-teachers --a-only-control --phase-updates 64 --learner-family bp --canonical-state-quaternions
```

Python: `run_learning_rollout(output, warm, new, canonical_state_quaternions=True)`.
Nonboolean values and incompatible visual mode refuse before IO/fitting. No
dependency or environment variable is added. Existing CPU/simulator environment
requirements remain. [Decision](adr/ADR-0482-project-sign-stable-state42-quaternions.md).

## Focused checks

```powershell
.venv/Scripts/python.exe -m pytest -vv tests/test_pick_cube_state_quaternions.py tests/test_state_quaternion_robot_learning.py
.venv/Scripts/python.exe -m pytest -vv tests/test_continuous_robot_learning.py tests/test_continuous_robot_a_only.py tests/test_readiness_stream_input.py
.venv/Scripts/python.exe -m pytest -vv tests/test_pick_cube_public_vectors.py tests/test_diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_state_quaternions.py scripts/maniskill_continuous_smoke.py scripts/run_continuous_robot_learning.py scripts/diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py tests/test_pick_cube_state_quaternions.py tests/test_state_quaternion_robot_learning.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_state_quaternions.py scripts/maniskill_continuous_smoke.py scripts/run_continuous_robot_learning.py scripts/diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py tests/test_pick_cube_state_quaternions.py tests/test_state_quaternion_robot_learning.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL src/core/pick_cube_state_quaternions.py scripts/maniskill_continuous_smoke.py scripts/run_continuous_robot_learning.py scripts/diagnose_learned_robot_control.py tests/test_state_control_diagnosis.py tests/test_pick_cube_state_quaternions.py tests/test_state_quaternion_robot_learning.py
```

Tests construct models and perform bounded updates/fake controls; they are not zero
work. The public-vector regression also initializes models before restoring fixture
parameters. [Protocol, actual ledger and evidence](../artifacts/runs/state42-quaternion-recipe-20261020/PROTOCOL.md).
Add future representations as explicit validated recipes with matching training/
serving semantics. Keep raw data, final-evaluation isolation and native state guards.

## Executed evidence

Final404focusedcases/platform, scoped lint/format/types and independent functional
review passed. One fixed BP192-update native case completed with zero task success
and four sampled joint violations; whole training report except elapsed matches
raw192. Fixture sharedCPU1/serialization conformity remains withheld after a
recorded scheduling overlap. No safety/useful-adaptation/CPC benefit follows.
[Full results, costs, commands and failures](../artifacts/runs/state42-quaternion-recipe-20261020/RESULTS.md).
