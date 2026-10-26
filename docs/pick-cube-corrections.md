# Provisional TRAIN corrections on student-visited robot states

The [private corrective-adaptation recipe](continuous-corrective-adaptation.md)
now connects this collector to eight arrived private updates and the existing
frozen17/19 rollout. `run_frozen_rollout` returns a usable batch only after the
same final gates as the CLI; default collection behavior remains unchanged.

The existing frozen robot smoke accepts `main(learner=model,
corrective_queries=True)` with TRAIN seed17 or19. It predicts from the usual
public observation, queries the existing fixed teacher on that same state, then
executes the student's original action. Teacher actions are **labels only**.
The raw episode keeps actual float32 controller actions, rewards and termination.

Why this: closed-loop student states can differ from reference demonstrations.
This provides a way to collect provisional supervision there without falsely
presenting unexecuted teacher commands as a successful expert trajectory.
Neither a coverage cause for earlier failures nor correction quality is proven.

## Use and boundaries

```python
from scripts.maniskill_continuous_smoke import main

# Invoke in an application with --output a-fresh-path.json in sys.argv and an
# already owned TRAIN continuous MLP or populated nearest learner.
status = main(learner=model, train_seed=17, visual_observation=True,
              corrective_queries=True)
```

The existing parser also accepts `--corrective-queries` for a Python caller
supplying `learner=`. A stand-alone zero-policy invocation refuses this option.
Actor serving and observation-clock routes refuse it for this first increment;
their existing default behavior remains intact. No optimizer is called during
collection, and there is no model loader or new runner.

Native execution requires the pinned optional CPU simulator stack and a separate
prospectively frozen protocol/budget. The implementation evidence below uses
fake environment boundaries, not new simulator episodes. No native case is
admitted by the completed implementation protocol.

## Records and arrival

`PickCubeCorrectionLabel(step, observation, target, stage)` owns native finite
tuples. Step0..49 determines observation tick `2*step` and label arrival tick
`2*step+1`. These are local logical event order, not wall-clock timestamps or
latency measurements. Fixed stage/gripper signs and grasp/settle zero translations
are validated. Geometry-derived translation quality is not attested.

`PickCubeCorrectionBatch(train_seed, executed_episode, labels)` validates a
complete TRAIN episode, declared17/19 context/identity/controller spec, and one
exact same-observation/step label per executed transition, at most50. Success is
not required: failed student episodes remain honest training-source trajectories.
The core does not attest a physical reset; the infra query additionally binds
the actual port's retained seed and first owner identity.

```python
# Infra query.finish(executed_episode) returns this owned core batch.
features, targets = batch.training_pairs(now=2 * len(batch.labels) - 1)
```

Before the final arrival tick the whole release refuses. Release revalidates
records against corruption, held-out roles, changed pairing and invalid targets.
It returns public features and provisional targets separately; it does not
rewrite `episode.action`, manufacture rewards, authorize promotion or create an
expert episode. Existing `continuous_training_pairs(batch.episode)` still yields
the executed student's actions. A later budgeted private update can consume the
explicit correction pairs through existing learner/runtime ports.

## Query, state and cost guarantees

`PickCubeCorrectionQuery` reads privileged `_geometry` only within infra.
Seed/role/context/episode identity, normalized controller spec, exact current
public observation, step and first port ownership validate before geometry,
after geometry and after the teacher call. A failed query poisons admission;
there is no retry/reset. The frozen private model and original owner are checked
after callbacks, before actual controls, after controls and after cleanup.

Only opted-in reports add `corrections`: provenance
`provisional_fixed_pick_cube_schedule_v1`, geometry attempts/returns, teacher
attempts/returns/schedule steps, bounded raw labels with logical times, and a
small `batch` metadata record when all collection/cleanup/state gates pass.
The original `episode` remains the sole executed trace; no duplicate episode is
stored. Failed collection keeps paid query work and valid returned raw labels,
but `batch` stays null. Native controller/prediction/visual work continues to be
reported by the existing smoke. Query attempts are not claimed as equal-cost
geometry reads or a FLOP measurement, especially for partial failures.

The nominal teacher follows a50-call schedule. Off-reference labels are
provisional corrections, not independently qualified expert actions. No expert
success, new learning, native benefit, generalization or circadian advantage
follows from correct collection. Earlier MSE native failures and binary-objective
tradeoffs remain preserved.

## Structure, checks and extension

```text
src/core/pick_cube_corrections.py       Owned records and arrival-gated pairs
src/infra/maniskill_corrections.py      Privileged query on one TRAIN port
src/infra/maniskill_continuous.py       Retained admitted reset seed
scripts/maniskill_continuous_smoke.py   Existing rollout opt-in
tests/test_pick_cube_corrections.py    Core and fake-native behavioral checks
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_pick_cube_corrections.py tests/test_maniskill_bc_smoke.py tests/test_maniskill_continuous.py
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_corrections.py src/infra/maniskill_corrections.py src/infra/maniskill_continuous.py scripts/maniskill_continuous_smoke.py tests/test_pick_cube_corrections.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_corrections.py src/infra/maniskill_corrections.py src/infra/maniskill_continuous.py scripts/maniskill_continuous_smoke.py tests/test_pick_cube_corrections.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_corrections.py src/infra/maniskill_corrections.py src/infra/maniskill_continuous.py scripts/maniskill_continuous_smoke.py
```

[Exact executed checks, limitations and handoff](../artifacts/runs/train-corrective-queries-20261019/RESULTS.md).
Next extend through existing private-update ports under a small frozen native
collector→private-update→fixed-rollout protocol. Preserve public pairing, original
student trajectory, all negative outcomes, frozen control and separate query,
replay/exposure, memory and compute costs; keep final evaluation unavailable.
