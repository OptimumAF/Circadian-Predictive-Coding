# Continuous TRAIN packets on the existing actor/shadow runtime

`run_continuous_demo(warm_episode, new_episode)` warms a bounded nearest-action
learner from one released TRAIN episode, then replays a second episode as one
delayed label packet through `ActorShadowRuntime`. It exercises the ordinary
inbox, budget and complete native ownership paths. It does not create a simulator,
promote a policy or reconstruct a durable checkpoint.

## Files and dependencies

```text
src/adapters/continuous_stream_demo.py  Continuous recipe and bounded report
scripts/run_native_stream_demo.py      Existing CLI plus qualified TRAIN receipt IO
tests/test_continuous_stream_demo.py   Integration, isolation and refusal checks
tests/test_visual_continuous_stream.py Visual receipt and delayed-runtime checks
```

The adapter depends on existing core continuous records/nearest learner and app
actor/shadow/inbox/budget boundaries. The CLI owns file IO. Core and app modules
are unchanged. No new dependency, runner or checkpoint codec is introduced.

Why this: a warm learner can serve before B labels arrive; an empty nearest
learner intentionally cannot predict. A single complete episode's pairs form one
packet and consume one native-call budget charge. Appended pairs and SGD steps
are different work, so the report keeps them separate.

## Fixed sequence

1. Validate two exact complete TRAIN records, distinct episode identities, equal
   specs and at most256 total rows before fitting or constructing a learner.
2. Fit statistics from warm observations once and append all warm pairs once.
3. Fork the ordinary stable actor and candidate. Configure one inbox identity,
   one update allowance and zero consolidations.
4. Serve B features before each tick's update. Record B features at tick0; hold
   their labels until tick2. Polling at ticks0/1 changes neither candidate nor
   budget. Tick2 appends the whole B packet once to the candidate.
5. Serve again, poll again, and check complete source/actor/candidate state,
   retained A prefix, unchanged frozen statistics and the permanent charge.
6. Read A/B candidate predictions through a detached full native snapshot.

The serving actor remains A-only. The candidate contains A+B and has not been
promoted. Logical arrival is simulated using already saved TRAIN data; no live
asynchronous label service or online rollout is implied.

## Run with saved qualified robot data

In the repository root, with the existing NumPy environment:

```powershell
python -m scripts.run_native_stream_demo --scenario continuous-nearest --warm-teacher artifacts/runs/pick-cube-task-teacher-20261019/teacher-smoke.json --new-teacher artifacts/runs/continuous-context-return-20261019/teacher-b.json
```

Each file is read with a1MiB bound and must be a successful qualified PickCube
TRAIN receipt with the fixed A17/B19 seed, candidate identity/context, raw42D
state and declared normalized four-axis controller. Complete record validators
check finite states/actions, bounds, sequence seams and end flags. Rewards must
be sparse0/1, ending in1. The original A receipt predates top-level context
metadata; only that missing A field is accepted using its validated candidate
context. A conflicting field or missing B field is refused. Held-out/probe roles
are rejected before fitting. These local declarations do not attest an arbitrary
file's physical provenance.

### Explicit visual receipts

Add `--visual-observation` to the same `continuous-nearest` command when both
files contain the fixed public RGB recipe:

```powershell
python -m scripts.run_native_stream_demo --scenario continuous-nearest --visual-observation --warm-teacher warm-visual.json --new-teacher new-visual.json
```

These filenames are examples, not newly qualified native data. Real native
collection needs a separately frozen local protocol and budget. The reader API
is `_read_teacher(path, seed, *, visual_observation=False)`; only exact booleans
are accepted. The CLI flag is refused for the other scenarios before file IO.

The explicit visual path reads at most **4 MiB + 1 byte** per file and rejects
files over 4 MiB. Why this: the accepted native 789D teacher is 1,390,451 bytes,
above the original state cap; 4 MiB is the existing reviewed teacher output
bound. The default state path keeps its **1 MiB** limit and 42D contract.
Visual inputs require all fixed recipe fields, exact JSON scalar/list/element
types and values, 789D observations, declared controller units/bounds, and
explicit matching top-level/candidate TRAIN17 or TRAIN19 context. Missing fields
are not backfilled. Duplicate visual JSON object keys, recipe drift, null recipe,
held-out roles/seeds and mixed state/RGB inputs refuse before fitting/runtime.
State inputs accept absent/null recipe metadata but refuse a non-null recipe.
The legacy absent A top-level context exception applies only to state receipts.

All visual observation and next-observation pixel coordinates (suffix21:) must
lie in [0,1]. The first21 qpos/qvel/public-goal coordinates keep physical units
and are required to be finite, without an image-range constraint. The usual
complete-episode, finite/action/shape/seam/end/reward validators still apply.
This format check does not attest source-camera provenance or pixel necessity.

The ordinary runtime receives admitted complete episodes; its logic, delayed
tick2 charge, private candidate and frozen warm serving are unchanged. Visual
reports additionally include `input_observation` with mode `public_rgb`, exact
recipe and `per_file_byte_limit=4194304`. State/default reports retain their
previous shape. Existing work/memory formulas use the actual789D spec. Previous
capture/rendering costs belong to source evidence; this saved replay does not
run the camera or claim those costs were zero.

`--seed` and `--diagnose-units` do not apply to this scenario. Teacher file
arguments are refused for other scenarios. Existing four scenarios and default
seed23 behavior remain. The result goes to stdout as ordinary JSON; JSON is an
integration report, not restorable model/experiment state.

## Report and work

The report records ticks, actor versions/actions, applied native receipts and
candidate rows/publications. Experiment schema and the in-memory snapshot type
are separate. Full-state checks include the post-update empty poll, source and
actor immutability, frozen statistics, A-prefix retention and detached readout.

Fitting, warm append, runtime append/charge, normalized rows/calls, output bytes,
distance work and per-owner numeric storage are explicit. `StableActor` returns
actions rather than a nearest-work report, so its distance counts are labeled as
derived from the exact unchanged typed actor and query shapes. Candidate counts
come from actual `NearestPrediction` results. Per-owner storage excludes snapshot
copies, Python objects and temporaries; it is not whole-process peak memory.

This path neither uses probe/final data nor updates deployed/benchmark state.
Predictions on saved labels cannot establish native success, generalization,
new-task plasticity, comparative learning benefit or a circadian advantage.
The separately accepted native context case and all earlier negatives remain.

## Checks and safe extension

```powershell
python -m pytest -q -p no:cacheprovider tests/test_visual_continuous_stream.py tests/test_continuous_stream_demo.py tests/test_native_stream_demo.py tests/test_actor_shadow.py tests/test_experience_inbox.py tests/test_toy_execution_budget.py tests/test_nearest_continuous_bc.py tests/test_pick_cube_rgb_features.py
python -m ruff check src/adapters/continuous_stream_demo.py scripts/run_native_stream_demo.py tests/test_continuous_stream_demo.py tests/test_visual_continuous_stream.py
python -m ruff format --check src/adapters/continuous_stream_demo.py scripts/run_native_stream_demo.py tests/test_continuous_stream_demo.py tests/test_visual_continuous_stream.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/continuous_stream_demo.py scripts/run_native_stream_demo.py
```

Preserve the generic runtime and add recipes at this adapter boundary. More
packets require explicit retained-identity, update and prototype capacity limits.
Serving publication, generalizing visual adaptation and domain recurrence remain
unfinished. The public visual port and one native same-task candidate acquisition/
return case are independently verified; saved-input replay alone establishes
neither native skill nor generalization. [Native case](visual-context-acquisition.md).
[Prospective protocol and executed evidence](../artifacts/runs/continuous-delayed-stream-20261019/PROTOCOL.md).

[Visual reader protocol and scoped evidence](../artifacts/runs/visual-delayed-stream-20261019/PROTOCOL.md).
