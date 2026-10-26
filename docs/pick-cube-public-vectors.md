# Explicit public-vector model input

The existing frozen rollout can explicitly give a 21D policy the public
`qpos9`, `qvel9`, `public_goal3` prefix of a known 789D RGB observation.
This implementation supplies a matched input-ablation prerequisite. No native
comparison or scientific benefit is claimed by its correctness checks.
The separately frozen follow-up now finds A-only public21 memory sufficient for
the known camera pair; see [native result and limits](public-vector-camera-control.md).

## Structure and boundary

```text
src/core/pick_cube_public_vectors.py       pure, owned fixed projection
scripts/maniskill_continuous_smoke.py      existing frozen rollout API/CLI opt-in
tests/test_pick_cube_public_vectors.py    59 pure and fake-IO boundary cases
artifacts/runs/public-vector-projection-20261019/  protocol, receipts, review
```

Why this: the public prefix already exists in the accepted RGB contract. Using
its exact fixed coordinates permits a simpler matched control without changing
the physical task, labels, camera, renderer or orchestration. The official 42D
state observation is a different input contract and is not substituted here.

`public_vector_projection()` returns the immutable definition
`pick_cube_public_qpos9_qvel9_goal3_v1`, source dimension 789, target dimension
21, order `(qpos9, qvel9, public_goal3)`. Validators require exact native types
and known values. `public_vector_spec` preserves valid action units and bounds.
`public_vector_observation` validates all 789 finite values and the dropped
RGB suffix's `[0,1]` range before returning an owned prefix. It accepts only the
exact default recipe or the explicitly bound known camera recipe.

`public_vector_training_episode` requires a complete known TRAIN17/TRAIN19
episode, or the declared known TRAIN camera context. It owns the full input,
validates current and terminal observations, preserves identity, context,
actions, rewards, flags and seams, and enforces the existing 50-step horizon.
Keep the projection metadata beside a derived episode. Projection grants no
teacher qualification, fitting, evaluation access or training authority.

## API

```python
from src.core.pick_cube_public_vectors import (
    public_vector_projection,
    public_vector_training_episode,
)
from scripts.maniskill_continuous_smoke import run_frozen_rollout

projection = public_vector_projection()
projected = public_vector_training_episode(
    qualified_train_episode, projection, source_recipe=qualified_recipe
)
# Supply an independently prepared, matching 21D learner or actor.
result = run_frozen_rollout(
    output_path,
    learner=owned_21d_learner,
    visual_observation=True,
    observation_projection=projection,
)
```

The snippet illustrates composition; it was not a native experiment in this
increment. A caller supplying a learner or actor to the existing `main(...)`
entry point can add CLI flags `--public-vector-only --visual-observation`.
The standalone zero-policy CLI refuses this selection. A matching explicit 21D learner or actor is
required before a learner fork or native scene creation. Zero-policy, nonvisual,
dimension mismatch and clock combinations refuse. Explicit learner corrections
are now supported through [owned raw/model batches](public-vector-corrections.md);
their actor, clock and camera incompatibilities remain. Known camera selection
is supported for frozen rollouts without corrections. The default `None` preserves existing calls and report
keys. No new learner or stream runner is introduced.

## Evidence and costs

Raw ports, rendering, pooled RGB and episode traces remain 789D. Explicit reports
add projection metadata, `projection_exact` and `projection_work`. Per attempted
projection the source-value upper bound increases by 789; a successful return
charges 789 validated values, 21 projected values and 168 numeric bytes. Attempt
and return counts bind to prediction progress and must all agree on completion.
Paid failed attempts remain visible. No unlimited projected-observation archive
is retained. Metadata and complete owner state are checked around predictions,
before control and at cleanup, including callback mutations.

The correctness gates and their precise scope are in
[RESULTS.md](../artifacts/runs/public-vector-projection-20261019/RESULTS.md).
They do not establish native adaptation, generalization, visual necessity,
neural learning, CPC advantage or repository-wide readiness.

## Accepted follow-up and safe next extension

The separately frozen A29-only nearest input-ablation case is complete; its
exact protocol and both29-control successes are linked above. It used the
accepted original/shifted camera, physical17, controller, horizon, teacher labels
and endpoints. It reused the accepted statistics' first21 values exactly through
the existing typed constructor, without new fit/camera labels/private append.
Reused exposure, memory, distance-coordinate work and unchanged rendering are
charged separately. Preserve the
accepted failed RGB warm actor. The separately frozen24-update MLP A17/B19/A17
sequence now runs on projected TRAIN inputs with explicit initialization/cost
limits; [its final mean errors worsen](public-vector-mlp-learnability.md).
Next validate its saved final state natively under a new frozen protocol,
without retraining or endpoint selection. All prior negatives remain preserved.

## Scoped commands

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_pick_cube_public_vectors.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/pick_cube_public_vectors.py scripts/maniskill_continuous_smoke.py tests/test_pick_cube_public_vectors.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/pick_cube_public_vectors.py scripts/maniskill_continuous_smoke.py tests/test_pick_cube_public_vectors.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_public_vectors.py scripts/maniskill_continuous_smoke.py
```

Use a new prospective protocol for any native execution; these commands exercise
the implementation and fake-IO boundaries only.
