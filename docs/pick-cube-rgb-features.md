# Frozen public RGB features for PickCube

The existing native port, task teacher and frozen rollout smoke now accept
`visual_observation=True` (CLI: `--visual-observation`). The default remains
the original 42D state observation with rendering disabled. The optional
simulator installation and existing TRAIN17/19, controller and horizon rules apply.
No dependencies or environment variables were added.

## Files and boundaries

```text
src/core/pick_cube_rgb_features.py          Pure fixed recipe and encoder
src/infra/maniskill_continuous.py           Public camera IO and work counters
src/infra/maniskill_task_teacher.py         Existing TRAIN teacher, optional RGB
scripts/maniskill_task_teacher_smoke.py     Existing collector CLI
scripts/maniskill_continuous_smoke.py       Existing frozen rollout CLI
tests/test_pick_cube_rgb_features.py        Pixel, layout and ownership checks
tests/test_maniskill_rgb_observations.py    Fake SDK/teacher/control integration
```

The pure core imports NumPy, with no simulator, filesystem, learner or fitting
authority. Infra selects public SDK arrays and supplies owned CPU data inward.
Existing app runtime and nearest/MLP learners are unchanged. ADR0458 records the
recipe decision.

## Recipe and inputs

`pick_cube_rgb_pool16_qpos_qvel_goal_v1` declares the SDK base camera, eye
`(.3,0,.6)`, target `(-.1,0,.1)`, field of view pi/2, near .01, far 100,
minimal shader and `sapien_cpu` rendering. The native input is CPU uint8 RGB
shape `(1,128,128,3)`. The core accepts exact ndarray uint8 HWC `(128,128,3)`
plus exact finite float64 vectors qpos9, qvel9 and public goal3.

Each nonoverlapping 8x8 block becomes its channel mean divided by 255. Features
are qpos9, qvel9, goal3, then row-major 16x16x3 RGB means: **789 values**.
Only the RGB suffix is in [0,1]; public vectors retain their original units.
There are no learned weights, fitting, RNG, pretrained downloads or ResNet claim.

Why this: a deterministic analytic visual control fits the existing <=1024D
continuous interface and exposes camera data without a new encoder dependency.
The pinned SDK hides the goal marker during sensor rendering and explicitly
supplies `extra.goal_pos` in RGB mode, so the public goal is declared as task
instruction. Object poses, relative object vectors, grasp status, TCP pose and
other extras/sensor parameters never enter learner features. The TRAIN teacher
still uses privileged geometry to generate labels; that is separate from inputs.

```python
from src.core.pick_cube_rgb_features import encode_pick_cube_rgb, pick_cube_rgb_recipe

encoded = encode_pick_cube_rgb(frame, qpos, qvel, goal)
assert len(encoded.features) == pick_cube_rgb_recipe().feature_dim == 789
# encoded.frame owns a C-order copy; features are independent immutable values.
```

The port retains only its current owned frame. `rgb_frame` returns another owned
copy and accounts for that export. Replacing a frame does not grow a history.
Malformed selected arrays fail before emitting features. `observation_recipe`
returns immutable fixed metadata; the smoke checks exact recipe types/values
before and after commands and cleanup. No observed recipe change is accepted.

## Existing commands and APIs

Use fresh output paths and a prospectively reviewed local budget for real native
work. These are ordinary examples, not admission to another experiment:

```powershell
python -m scripts.maniskill_task_teacher_smoke --visual-observation --output fresh-visual-teacher.json
python -m scripts.maniskill_continuous_smoke --visual-observation --output fresh-visual-zero.json
```

The second command uses the default zero policy. Python callers can instead pass
a compatible warm learner to `main(learner=model, visual_observation=True)` or
the supported actual stable actor to `main(actor=runtime.actor,
visual_observation=True)`. Existing clock features append to the declared raw
789D observations. Learner and actor remain mutually exclusive. The real case
below used the concrete nearest learner. The later visual context case also
ran the actual A-only actor on B:50controls/return0/truncation with full state
preserved. Its detached candidate succeeded separately; no promotion occurred.

## Work, memory and executed evidence

`visual_work` separately records encoding attempts/returns, input bytes and
feature values, returned numeric bytes, current frame and frame export copies.
One frame is 49,152 bytes; one 789D float64 feature row is 6,312 numeric bytes.
Attempts retain upper bounds when decoding fails. A successful native step
retains its known controller physics work even if subsequent decoding fails.
SDK constructor/reset hidden rendering and whole-lifetime process-tree memory
remain unmeasured. Numeric payloads exclude Python/NumPy object overhead and
temporary buffers; counters do not claim all renderer computation is measured.

80 new tests and final combined **641 Windows / 641 Linux** tests pass with
zero failures, errors or skips. Scoped 12-file lint/format and seven-source
types pass. Independent source/test review preceded exactly one bounded native
case; actual evidence is reviewed separately in the stage review.

The real software-rendered TRAIN17 teacher qualified after 29 controls. One
fit/append of all 29 pairs produced a frozen nearest policy whose fresh TRAIN17
rollout reproduced all 29 transitions and reached sparse return1/final success.
Initial 789D observations and full original learner state were exact. This is
same-initial-state stored trajectory reconstruction, not visual generalization,
new-task adaptation, neural plasticity, a matched state/visual comparison or CPC
advantage. All earlier negative results remain valid.

See [protocol, commands, results, failures and costs](../artifacts/runs/pick-cube-rgb-features-20261019/RESULTS.md).
The full suite, clean-clone and release gates were not run.

## Focused verification and extension

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q -p no:cacheprovider tests/test_pick_cube_rgb_features.py tests/test_maniskill_rgb_observations.py
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_rgb_features.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_rgb_features.py tests/test_maniskill_rgb_observations.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_rgb_features.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_pick_cube_rgb_features.py tests/test_maniskill_rgb_observations.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/core/pick_cube_rgb_features.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py
```

To extend safely, give a different camera or encoder its own explicit recipe
and tests. Do not change this recipe after observing outcomes. The existing
delayed stream CLI now explicitly admits789D fixed-recipe TRAIN17/19 receipts
with --visual-observation and a4MiB cap, preserving raw42D/1MiB defaults.
Keep state/RGB modes distinct, final/probe data unavailable and candidate updates
separate from serving. Same-task visual reset acquisition/retention is now
verified in the fixed case; generalization, domain changes and serving
publication remain unfinished. [Actual evidence](../artifacts/runs/visual-context-return-20261019/RESULTS.md).
