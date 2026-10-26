# Learning from qualified readiness demonstrations

The existing stream and learned-robot CLIs accept `--readiness-teachers`. It
selects distinct qualified geometry-readiness TRAIN17/19 receipts for both inputs.
The old default reader and fixed-teacher IDs are preserved; declaring the new
teacher definition requires the explicit route. Do not rename readiness episodes
or reuse provisional correction batches as successful demonstrations.

## Responsibilities and data flow

`scripts/run_native_stream_demo._read_readiness_teacher` reads one bounded JSON
receipt and returns an owned complete TRAIN episode plus retained acquisition
metadata. It binds definition, IDs, task/controller/seed/context, success and work
through the existing `TaskTeacherResult.training_episode` qualification. Exact
42-state or base RGB789 observations are supported. The RGB recipe must be complete.
Duplicate/nonfinite JSON, wrong roles, unsuccessful/incomplete collection, work or
stage/action mismatches, context interventions and correction provenance refuse.
The explicit mode excludes camera/mass/lighting drift; no implicit role conversion.

Inputs flow through the existing continuous stream, learner and frozen-robot APIs.
No new optimizer, objective, collection loop, model loader or dependency. Retained
teacher work is acquisition that already happened; receipt decoding incurs new
validation/IO work, not simulator work. Geometry remains teacher-side. New labels
are never substituted for the student's executed action.

## Usage

These commands spend training or simulator work and require a prospective local
protocol. They are examples, not authority to repeat an accepted run:

```powershell
.venv/Scripts/python.exe -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --warm-teacher readiness-17.json --new-teacher readiness-19.json --readiness-teachers --visual-observation --learner-family bp --phase-updates 8
.venv/Scripts/python.exe -m scripts.run_continuous_robot_learning --output fresh-learning.json --warm-teacher readiness-17.json --new-teacher readiness-19.json --readiness-teachers --visual-observation --learner-family bp --phase-updates 8
```

The stream JSON retains `input_teachers` metadata. The robot composition preserves
distinct source episode IDs inside the training report; its existing default
report schema is unchanged. Keep the original qualified input receipts as the
complete acquisition evidence. Robot output paths are exclusive, paid partial
results survive failure, and full candidate state must remain frozen at serving.

Existing public21 projection can be used on the visual MLP stream with
`--public-vector-only`; it does not change the robot CLI's fixed raw789/state42
contract. Existing separately reserved probes retain their roles and isolation;
TRAIN demonstrations cannot be relabeled into probe or final inputs.

## Checks and interpretation

Focused admission, old-route and end-to-end tests, scoped statics and independent
review are recorded in [the current results](../artifacts/runs/readiness-demonstration-learning-20261020/RESULTS.md).
The previously accepted readiness teacher solved two known TRAIN resets in20/22
controls. That success qualifies demonstrations, not a trained student. The prior
readiness correction intervention had all50 ALIGN labels and no native success;
complete-phase demonstrations address that observed coverage gap. Fixed-budget
learning may still fail. Preserve every negative and all earlier costs.

Extend admission only for a separately defined/qualified teacher. Keep learner
changes in existing adapters and freeze any new native or comparative experiment
before execution. Native rewards/endflags establish scoped task performance;
imitation fit and source preservation do not establish retention, recovery,
safety, reliability, generalization or CPC benefit.

## Executed scoped evidence (2026-10-20)

494 focused cases pass Windows/pinnedLinux, scoped three-file Ruff lint/format and
Windows-run Linux3.12-target mypy pass; independent combined implementation review
accepted. One initial nullable-candidate typing error was repaired explicitly and
preserved. The tests include a real tiny BP→frozen fake-robot path; they establish
input/runtime composition, not native skill.

The one prospectively reviewed default BP8/8/8 raw789 case trained24updates496rows
from full-phase readiness demonstrations, then used50controls250knownphysics on
TRAIN17 with zero return/any/final success and truncation. Candidate state stayed
exact. Eleven of51 sampled joint readings exceeded panda_joint6 upperlimit, peak
0.0018639564514160156rad. No safety or useful neural acquisition claim.
Saved-data diagnosis (zero model/fit/SGD/native work) found max normalized action
error0.3805268406867981 at the exact reset observation and support departure from
step1. Other student states have no admitted teacher error label. Large normalized
features are descriptive; prior range/public21-depth negatives remain preserved.

| Boundary | Evidence |
|---|---|
| Qualified input admission | Known TRAIN17/19 IDs, definition, complete stages/work and strict state/RGB payload |
| Passing composition |494focusedcases/platform and scoped statics/independent review|
| Native BP usefulness |Not demonstrated: one known TRAIN17 rollout failed|
| Acquisition versus interference |Unresolved; final A fit improved beyond postA8, so B is not shown causal|
| Follow-up |[A-only robot composition](continuous-robot-a-only.md) implemented; bounded native result remains negative|

Use these scoped developer checks (they rerun the tiny fixture and consume its
separately declared correctness work):

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_readiness_stream_input.py
.venv/Scripts/python.exe -m ruff check scripts/run_native_stream_demo.py scripts/run_continuous_robot_learning.py tests/test_readiness_stream_input.py
.venv/Scripts/python.exe -m ruff format --check scripts/run_native_stream_demo.py scripts/run_continuous_robot_learning.py tests/test_readiness_stream_input.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL scripts/run_native_stream_demo.py scripts/run_continuous_robot_learning.py tests/test_readiness_stream_input.py
```

Full combined selection, exact native/diagnostic commands, costs and failures are
in RESULTS; these commands do not establish clean-clone or release readiness.
