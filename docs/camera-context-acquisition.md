# Native camera acquisition and return

One fixed local case now executes the explicit5cm camera context through the
existing TRAIN teacher, actual stable actor, ordinary actor/shadow runtime and
detached nearest candidate. Production source is unchanged in this increment.

## Structure

```text
artifacts/runs/camera-context-return-20261019/
  PROTOCOL.md       prospective sequence, endpoints and budgets
  native_case.py    existing teacher/runtime/frozen CLI composition
  native_guard.py   accepted observer, only stage/module/cache binding changed
  run_check.py      accepted root-only serial command recorder
  REVIEW.md         independent source/gate admission and actual readback
  RESULTS.md        exact commands, outcomes, costs and limitations
  teacher-b.json, b-before.json, b-after.json, policy-rollout.json
  prototype-states.npz, rollout-audit.json, rollout-supervisor.json
tests/test_camera_context_return_case.py 54 synthetic-boundary cases
```

Camera metadata, APIs and two production CLI options are documented in
[pick-cube-camera-context.md](pick-cube-camera-context.md). This case adds no
learner, simulator runner, dependency, checkpoint codec or configuration variable.

## Sequence and observed outcomes

All episodes use physical seed17. A is the original camera; B changes only the
camera eye from(.3,0,.6) to(.3,.05,.6), with the same target and public789D recipe.
Reuse accepted A29 teacher/reference. Collect B once and qualify it before any
fit or learner creation. Initial vector21 matches A exactly;766 of768 pooled RGB
coordinates differ, mean absolute change.0213310292, maximum.2026960784.

Fit A observations and append A29 once. Create the existing stable actor/private
runtime with one-update budget, run the warm actual actor in B, then record the
complete B packet observed at logical0. Poll0/1 without labels: no update/charge.
At2, release labels and append B29 once to the private candidate. Freeze original
A statistics/prefix and all source/actor state, validate the receipt before and
after an empty poll, then run detached candidate B and fresh A return.

| Endpoint | Owner | Controls | Return | Any/final success | End |
| --- | --- | ---: | ---: | --- | --- |
| A reference, reused | accepted A-only control |29 |1 |true/true |terminated |
| B teacher, new | fixed geometry teacher |29 |1 |true/true |terminated |
| B before append | actual A-only stable actor |50 |0 |false/false |truncated |
| B after append | detached private candidate |29 |1 |true/true |terminated |
| Fresh A return | same private candidate |29 |1 |true/true |terminated |

Candidate-minus-warm B gains are+1 for all three frozen endpoints; A reference
minus return losses are0. Both successful candidate traces exactly reconstruct
their respective complete teacher trajectories. All native action bounds pass;
other physical constraints remain unmeasured. Source and deployed actor stay
A-only/publication1/29rows. Candidate becomes publication2/58rows, with one
permanent charge. No promotion or deployed-policy improvement occurs.

This is observed same-task, known-camera stored-memory adaptation and return.
It does not establish neural plasticity, generalization, visual necessity,
invariance, CPC benefit or fullR4.5. Every earlier neural/offline/native negative
is retained. The failed warm baseline is not tuned or replaced.

## Costs

- New qualified supervision:29 teacher geometry queries/actions/labels;58 total
  ingested A+B pairs, zero SGD/replay/refit on B. One A fit of29×789 values.
- Prototype payload per learner:183976→367952B; added183976B. One retained inbox
  packet183976B; frozen statistics12624B per owner. One bounded warm/final NPZ
  552998B. These are named numeric payloads, not total process peak or all copies.
- 137 controls/685 known physics/141 returned RGB projections;108 predictions/
  4814 prototype distances/3798246 distance coordinates/3456 prediction bytes.
  Normalization110 calls/166rows. Unknown constructor/reset physics stays unknown.
- Inner51.220184763s, supervised53.145147833s, launcher58.510267800s<90/105s.
  Observed process highwater sum1099788288B, final CLI SELF1056935936B<2GiB;
  sampled-tree/final-self observation is not a lifetime-tree memory proof.

[RESULTS.md](../artifacts/runs/camera-context-return-20261019/RESULTS.md) includes
all renderer/frame/feature/log/cache/fit/append costs and exact bounded artifacts.

## Checks and commands

54 new cases pass on Windows and pinned Linux with identical IDs, zero failed,
error or skipped cases; scoped Ruff/format/types pass. They use real nearest
learner/runtime owners under synthetic IO:25 tiny appends/482rows per invocation,
zero SGD. Accepted unchanged690-per-platform camera implementation checks and
earlier learner/runtime evidence are reused rather than rerun. Initial mypy
tuple-inference errors were repaired and remain recorded. Independent source/gate
GO preceded the one native invocation; actual readback is separate in REVIEW.md.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_camera_context_return_case.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check artifacts/runs/camera-context-return-20261019/native_case.py artifacts/runs/camera-context-return-20261019/native_guard.py artifacts/runs/camera-context-return-20261019/run_check.py tests/test_camera_context_return_case.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check artifacts/runs/camera-context-return-20261019/native_case.py artifacts/runs/camera-context-return-20261019/native_guard.py artifacts/runs/camera-context-return-20261019/run_check.py tests/test_camera_context_return_case.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux artifacts/runs/camera-context-return-20261019/native_case.py
```

The exact native command and pinned interpreter are in PROTOCOL/RESULTS. This
one-use stage is already consumed: existing outputs refuse before new work.
A new native experiment needs its own prospective small protocol, not deletion
of receipts or a retry under this completed allowance.

## Next implementation

Add an explicit owned public-vector21-only projection of qpos9/qvel9/public-goal3
and a matching opt-in at the existing frozen rollout boundary. Preserve default
RGB789, physical reset/controller, all records and camera/renderer work. No such
projection exists today; the existing42D state route is a different input contract.
Then freeze one small A29-only nearest control against the same two cameras to
test whether extra camera labels/prototypes were necessary. Keep the same labels,
frozen A statistics prefix and endpoints; separately account for smaller distance
and memory costs. No privileged inputs, pose/seed selection, new learner/runner or
neural/CPC claim. FullR4.4/R4.5 lighting/object/dynamics/constraints and research/
final/UI/release remain open.
