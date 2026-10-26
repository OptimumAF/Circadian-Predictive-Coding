# Bounded PickCube ambient-light context

## Structure and boundaries

```text
src/core/pick_cube_lighting_context.py   Fixed declaration and role/context admission
src/infra/maniskill_lighting.py          Native light IO, owned readback and paid work
src/infra/maniskill_continuous.py        Existing episode port and RGB encoding
tests/test_pick_cube_lighting_context.py Synthetic boundary and failure cases
src/infra/maniskill_task_teacher.py      Typed teacher collection and qualification
scripts/maniskill_continuous_smoke.py    Existing frozen rollout API and CLI
scripts/maniskill_task_teacher_smoke.py  Existing teacher CLI
tests/test_lighting_context_interfaces.py Teacher/rollout integration and callbacks
docs/adr/ADR-0475-bind-ambient-light-before-visual-encoding.md
artifacts/runs/pick-cube-lighting-context-20261019/ Protocols, checks and review
```

Why this: lighting is a concrete missing R4.5 domain axis. The existing episode
port remains the integration boundary. No learner, optimizer, runner, dependency,
environment variable or final-evaluation permission is added.

## API

```python
from src.core.pick_cube_lighting_context import pick_cube_lighting_context
from src.infra.maniskill_continuous import ManiSkillPickCubeEpisode

context = pick_cube_lighting_context()
with ManiSkillPickCubeEpisode(
    "lighting-train-17", context.context,
    seed=17, visual_observation=True, lighting_context=context,
) as episode:
    measured = episode.lighting_readback  # owned original values; no native reread
    work = episode.lighting_work
    features = episode.observation       # refreshed under the admitted light
    # Ordinary episode.step(...) remains the sole control path.
```

This is one explicit visual TRAIN17 context, `pick-cube-ambient-half-train-17`.
The ambient multiplier is exactly 0.5. Arbitrary declarations, heldout roles,
wrong seeds, nonvisual use, missing reserved binding and implicit camera/mass
mixing are refused before environment creation. The original camera/feature
recipe stays frozen. Defaults never resolve scene lighting or refresh observations.

After explicit reset, read one native float32 RGB vector of shape `(3,)`, each
coordinate finite in `(0, 1]`. Set its exact representable positive float32 half
once and require exact readback. Unrepresentable subnormal halves fail before
the setter. Only ambient light is intentionally set; no measured directional,
camera or physics invariance follows from that intent.

Discard the original reset observation and call the SDK's public `get_obs()`
once. It performs ordinary evaluation/rendering work without a control step.
Verify light and identities before accepting that refreshed packet. Each later
control verifies before stepping and before encoding returned observations;
episode export verifies again. Mutation during native/identity callbacks is
refused permanently. Repairing the value does not admit a failed session again.

## Work and errors

Lighting work records read/write/refresh attempts and returns, plus one discarded
reset observation. Successful construction costs three validated reads, one
setter and one refresh. Each successful control adds two reads; complete export
adds one. A refresh return is a successful native API return, even if its packet
later fails encoding. Existing visual work still counts encoding attempts/returns
and owned frame copies. Do not count the discarded reset rendering as zero work.

Readback is an owned immutable context and source/applied RGB tuples. It is
available only after exact setter verification. Failed construction attaches
`lighting_work`, partial `lighting_source_ambient_rgb` and requested target to
the original exception, with verified `lighting_readback` when available. A
cleanup failure is attached as `lighting_cleanup_error`; it does not replace
that original failure. Failed controls retain work in the episode. Storage stays
constant apart from the existing bounded trajectory; no success-history chain.

## Focused checks

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_pick_cube_lighting_context.py tests/test_maniskill_continuous.py tests/test_pick_cube_mass_context.py
.venv/Scripts/python.exe -m ruff check src/core/pick_cube_lighting_context.py src/infra/maniskill_lighting.py src/infra/maniskill_continuous.py tests/test_pick_cube_lighting_context.py
.venv/Scripts/python.exe -m ruff format --check src/core/pick_cube_lighting_context.py src/infra/maniskill_lighting.py src/infra/maniskill_continuous.py tests/test_pick_cube_lighting_context.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent src/core/pick_cube_lighting_context.py src/infra/maniskill_lighting.py src/infra/maniskill_continuous.py tests/test_pick_cube_lighting_context.py
```

Exact selected RGB/camera regressions, executed outcomes and remaining limits
belong in the stage results. Synthetic pixels prove routing/ownership, not a real
lighting effect. Native validation and later adaptation remain separately gated.

## Next extension

The joint appearance comparison below has completed, with a descriptive fit/
retention tradeoff and an UNVERIFIED one-thread gate. Full protocol acceptance
remains open; both arms are spent. Future explicit WSL environment forwarding was
verified without models. Next implement scoped physical joint-limit telemetry,
preserving defaults/actions/features; no automatic normalization/horizon variant.

## Accepted scope, 2026-10-19

323 identical ordered focused cases pass on Windows and pinned Linux; scoped
Ruff7/format7/types5 and independent implementation/gate/native acceptance pass.
The sole frozen zero-control native case observed exact ambient RGB
0.30000001192092896→0.15000000596046448, with 48,854 changed channel values and
mean change −12.908467610677084. Fresh default frames, all features and light
restored exactly. All 2,304 pooled feature values were independently recomputed
from saved raw frames using only stdlib. Initial typing failures are retained.

Three constructions/resets and four public camera packets include one discarded
reset observation and one explicit refresh; three encodings/owned frame exports,
three internal light reads plus two external default reads, one setter. Hidden
constructor/reset/render/evaluation/physics work is unmeasured, not zero. No
control, teacher, learner, heldout read or promotion ran. Memory evidence is
sampled-process highwater plus final SELF; it does not prove a whole-lifetime
process-tree bound. Full learning/return/recurrence/constraint claims remain open.

At this boundary milestone, teacher/frozen-rollout integration was still pending;
it is now accepted below. The original native allocation is spent; historical
commands do not authorize retries.
[Exact commands, outcomes, costs, failures and skips](../artifacts/runs/pick-cube-lighting-context-20261019/RESULTS.md) ·
[independent review](../artifacts/runs/pick-cube-lighting-context-20261019/REVIEW.md).

## Teacher and frozen-rollout integration

`collect_task_teacher(..., lighting_context=pick_cube_lighting_context(),
visual_observation=True)` now returns typed lighting context, original verified
readback and paid work. Only complete final-success visual TRAIN17 data with
exact binding, frozen RGB recipe, stage schedule and controller/light/visual
lifecycle counts qualifies for `training_episode()`. Missing or corrupted
reserved metadata, guard/probe/final data and camera/mass mixing are refused.
Unsuccessful collection remains unqualified and retains valid costs/receipts.
The original port exception carries unverified source/request measurements;
the teacher result carries paid work and verified readback when available.

`run_frozen_rollout(..., lighting_context=..., visual_observation=True)` and both
existing CLIs support this context. Frozen MLP and actual nearest StableActor
inference preserve source/actor arrays, momentum, statistics and counters. Late
metadata or owner callback changes clear success. Pure final checks compare
owned original readback, cumulative work and actual encoder storage after all
external getters. Partial failures may retain paid frame exports; successful
teacher/rollout lifecycles require zero exports and exactly n+1 encodings.

Defaults keep their established CLI JSON keys and factory calls. Python
`TaskTeacherResult` intentionally appends three fields defaulting to None;
raw `asdict` therefore gains those keys. The CLIs strip inactive None keys.
Lighting requests require visual TRAIN17 and refuse camera/mass/guard/corrective
or public-vector combinations before fork, actor versions or native creation.
No new runner, learner, optimizer, dependency or environment setting is required.

```powershell
# Native usage examples: freeze a separate local protocol before executing.
.venv/Scripts/python.exe -m scripts.maniskill_task_teacher_smoke --output fresh-light-teacher.json --visual-observation --lighting-drift
.venv/Scripts/python.exe -m scripts.maniskill_continuous_smoke --output fresh-light-rollout.json --visual-observation --lighting-drift

# Synthetic focused checks; these do not create a simulator or train a learner.
.venv/Scripts/python.exe -m pytest -q tests/test_lighting_context_interfaces.py
.venv/Scripts/python.exe -m ruff check src/infra/maniskill_lighting.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_lighting_context_interfaces.py
.venv/Scripts/python.exe -m ruff format --check src/infra/maniskill_lighting.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_lighting_context_interfaces.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent src/infra/maniskill_lighting.py src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_lighting_context_interfaces.py
```

411 identical selected cases pass on Windows and pinned Linux (94 new plus
317 affected regressions), with scoped Ruff9/format9/types6 and independent review.
Two initial typing failures are retained. Per platform, six tiny constructors and
45 prediction rows execute; no fit, SGD, prototype append or native simulation.
These checks prove usable integration on fake IO. At this API milestone native
acquisition/return remained unmeasured; the fixed memory case below now supplies
that narrow evidence. PC/CPC benefit and broader generalization remain open.
[Exact selection, commands, failures, costs and next action](../artifacts/runs/lighting-context-interfaces-20261019/RESULTS.md) ·
[independent acceptance](../artifacts/runs/lighting-context-interfaces-20261019/REVIEW.md).


## Fixed lighting acquisition and A return

```text
artifacts/runs/lighting-context-return-20261019/
  native_case.py    Existing API composition, saved A restore and delayed B append
  native_guard.py   Existing Linux observer with this case's fixed output bounds
  run_check.py      Root serial command receipts
  PROTOCOL.md      Prospective single-use scientific/resource limits
  RESULTS.md       All six endpoints, costs, failures/skips and exact next action
  REVIEW.md        Independent source, gate, one-use GO and actual acceptance
tests/test_lighting_native_context_case.py  Actual nearest/runtime with fake RGB IO
```

The saved fullRGB789 nearest baseline has29A prototypes and frozen statistics.
Restore it without refitting; collect one qualified visualTRAIN17 half-ambient
teacher. The actual unchanged StableActor serves B-before. Existing runtime
observes features at0, withholds labels through polls0/1, and applies one private
29row append at2. Source/actor stay1update/29rows, candidate2/58. Empty polls cannot
charge twice. Exact owners/prefix/statistics/original readback and strict retained
JSON contents/receipts are checked before dependent work, including after label
callbacks; failures preserve paid work. No new production runner or optimizer.

One frozen native case observed warmB50controls/return0 -> candidateB29/return1,
then freshdefaultA29/return1. All three B gain metrics+1; all three A retention
losses0. Both candidate readouts exactly reconstruct their teacher trajectories.
All29lighting/A teacher action labels and public physical trajectories are the
same; only appearance is deliberately changed. This demonstrates bounded same-
task/same-seed memory recovery, not a new physical skill, neural generalization
or a circadian advantage. Full R4.5 remains open.

Costs:137controls/685knownphysicssteps/141acceptedencodings/108predictions,
4814nearest candidates/3798246feature distances,3light setters/refreshes/discards,
228light reads. Per-owner prototype payload183976->367952B; frozen statistics
12624B. One permanent update charge, zero SGD/replay/fit/guard/probe/final/promotion.
Wrapper51.25s; observed sampled process-tree+finalSELF1.107GB<2GiB, not a lifetime-
tree proof. Reset/render/intrinsicphysics/geometry/copy work is unmeasured. Full
receipts/costs/raw evidence in RESULTS.md; the native allowance is SPENT.

```powershell
# Focused fake-IO verification, no simulator or SGD.
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_lighting_native_context_case.py tests/test_lighting_context_interfaces.py
.venv/Scripts/python.exe -m ruff check artifacts/runs/lighting-context-return-20261019/native_case.py artifacts/runs/lighting-context-return-20261019/native_guard.py artifacts/runs/lighting-context-return-20261019/run_check.py tests/test_lighting_native_context_case.py
.venv/Scripts/python.exe -m ruff format --check artifacts/runs/lighting-context-return-20261019/native_case.py artifacts/runs/lighting-context-return-20261019/native_guard.py artifacts/runs/lighting-context-return-20261019/run_check.py tests/test_lighting_native_context_case.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent artifacts/runs/lighting-context-return-20261019/native_case.py artifacts/runs/lighting-context-return-20261019/native_guard.py artifacts/runs/lighting-context-return-20261019/run_check.py tests/test_lighting_native_context_case.py
```

110 identical ordered Windows/Linux cases pass, scoped Ruff4/format4/types4 and
independent acceptance. Initial five typing failures remain recorded. Real source
admission and independent stdlib array/trajectory/receipt verification supplement
fake-IO tests; no release/full-suite claim follows.
[All outcomes, commands and next implementation](../artifacts/runs/lighting-context-return-20261019/RESULTS.md) ·
[independent acceptance](../artifacts/runs/lighting-context-return-20261019/REVIEW.md).


## Saved lighting teachers in the existing stream

```text
scripts/run_native_stream_demo.py       Existing CLI, bounded input and dispatch
  _read_lighting_teacher               Full qualification + original paid metadata
  _require_fixed_context_fields        Complete serialized light/mass declarations
tests/test_lighting_stream_input.py     Input isolation, ownership and CLI composition
artifacts/runs/lighting-stream-input-20261019/
  PROTOCOL.md / RESULTS.md / REVIEW.md   Limits, exact commands and acceptance
  saved-input-admission.json            Actual saved A/B admission, no fitting
```

Use --lighting-drift with --visual-observation on continuous-nearest or
continuous-mlp-learnability. Warm input is default TRAIN17; B is fixed half-ambient
TRAIN17. One owned4MiB JSON payload per input reader is checked before fitting.
Duplicate/nonfinite fields, missing/extra/untyped context declarations, incomplete
readback/work/stages, non-TRAIN roles and camera/mass/guard/projection mixing refuse.
Default/mass routes reject nonNone lighting fields; None dataclass additions and
original default report fields remain compatible. The shared check also prevents
missing mass context evidence from being silently filled by Python defaults.

The reader reuses current TaskTeacherResult qualification rather than treating
qualified=true as sufficient. It retains native original source/applied light,
setter/refresh/discard, visual/controller/geometry costs and source bytes under
input_context. These describe historical acquisition, not new simulator work.
Raw public collector receipts may omit the CLI-only optimizer_updates field;
when present it must be exact native zero. Full stage schedule is validated and
retained in the original unchanged source receipt.

Existing MLP --a-only-control, --phase-updates, --binary-gripper-axis and
--rgb-range-normalization compose without altering their defaults. No new runner,
learner, dependency, environment variable, final/probe/guard permission or native
allowance is introduced. Command shape (a real fit requires the separately frozen
joint treatment/control protocol first):

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --warm-teacher artifacts/runs/pick-cube-rgb-features-20261019/teacher-visual.json --new-teacher artifacts/runs/lighting-context-return-20261019/teacher-b.json --visual-observation --lighting-drift
# The jointly frozen equal-exposure control uses the same command + --a-only-control.

# Scoped zero-model checks (full selected set below intentionally omits one learning fixture).
$lightingInputSelection = Get-Content artifacts/runs/lighting-stream-input-20261019/selection.txt
.venv/Scripts/python.exe -X utf8 -B -m pytest -q @lightingInputSelection -k 'not test_should_run_actual_delayed_789d_stream'
.venv/Scripts/python.exe -m ruff check scripts/run_native_stream_demo.py tests/test_lighting_stream_input.py artifacts/runs/lighting-stream-input-20261019/run_check.py artifacts/runs/lighting-stream-input-20261019/validate_saved_inputs.py
.venv/Scripts/python.exe -m ruff format --check scripts/run_native_stream_demo.py tests/test_lighting_stream_input.py artifacts/runs/lighting-stream-input-20261019/run_check.py artifacts/runs/lighting-stream-input-20261019/validate_saved_inputs.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent scripts/run_native_stream_demo.py tests/test_lighting_stream_input.py artifacts/runs/lighting-stream-input-20261019/run_check.py artifacts/runs/lighting-stream-input-20261019/validate_saved_inputs.py
```

375 matching Windows/Linux cases pass; three equivalent test stub repairs were
rechecked separately, and scoped statics/independent acceptance pass. Both initial
typing failures remain recorded. One stubbed actual CLI admits saved A29/B29 in
789D and preserves original metadata/input bytes; no model/fit/forward/normalization/
SGD/RNG/native work ran. This proves usable input/dispatch, not neural appearance
adaptation, a new physical task, generalization or CPC benefit. Next jointly freeze
both learning arms, all six metrics/curves/exposures/costs before either fit.
[All commands, failures, costs and next action](../artifacts/runs/lighting-stream-input-20261019/RESULTS.md) ·
[independent source/gates/actual acceptance](../artifacts/runs/lighting-stream-input-20261019/REVIEW.md).


## Joint MLP appearance evidence and resource exception

Both treatment A8/B8/A8 and equal-exposure A-only8/8/8 were frozen jointly before
any fit, using the existing saved-data CLI/defaultMSE/H64/seed17/A-only statistics.
Full policies/1578statistics, first8updates/receipts and checkpoints0/8 match exactly.
Each arm completed24paid updates/696rows, with all owner/receipt/isolation gates.

| Updates | Treatment A/B half-MSE | A-only A/B half-MSE |
| ---: | --- | --- |
| 0 | .1796567 / .4416800 | .1796567 / .4416800 |
| 8 | .0711919 / .4414361 | .0711919 / .4414361 |
| 16 | .0822618 / .2706067 | .0383871 / .4421263 |
| 24 | .0410874 / .3119902 | .0274792 / .4428719 |

B covariate updates improve B fit while A-only updates achieve better A retention.
All six original metrics/directions,24losses perarm, full curves/axis harms and
costs are retained without a composite winner. Complete B hidden saturation and
zero measured first-layer gradients at every recorded endpoint limit inference;
carried momentum can still move weights. Large gripper errors remain. Same physical
action trajectory under changed appearance is not a novel task/generalization,
native utility or CPC advantage result.

**Full protocol acceptance is withheld:** Windows-parent thread settings were not
forwarded into WSL, so the intended one-thread limit was not verified. Both arms
remain SPENT, with no replacement fit or metric/tolerance retcon. Numerical
comparison gates passed; they do not establish that resource gate. A separate
zero-model check verified explicit future command configuration:

```powershell
# Linux child settings belong inside WSL, before its Python command.
wsl.exe -d Ubuntu --exec /usr/bin/env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/avery/.cache/circadian/maniskill-expert-numpy1-20261019/env/bin/python -X utf8 -B -c "import os; print(os.environ['OPENBLAS_NUM_THREADS'])"
```

This repairs future command shape, not historical thread measurements. Each arm
paid56forward/1624rows,32backward/928rows including8diagnostic gradients232rows,
129756672denseMAC and one29rowfit/50752normaldraws. GNUtime command-process maxRSS
was62861312/63119360B, with unmeasured tree/copy/parallelism details; no runtime
performance ranking. Scoped Ruff/format/types pass; unchanged accepted80learning/
A-only checks and375input+3repair gates reused without more fixture training.
[All six metrics, exact commands, costs, failures/skips and next action](../artifacts/runs/lighting-mlp-joint-20261019/RESULTS.md) ·
[independent numerical review and withheld resource acceptance](../artifacts/runs/lighting-mlp-joint-20261019/REVIEW.md).

Next concrete implementation: verify public pinned SDK joint-limit units/order,
then add a pure bounded signed-exceedance report and opt-in read-only binding in
the existing episode port. It currently validates normalized action bounds but
lacks physical joint-limit telemetry. Keep raw actions/default features/behavior;
any tolerance must be declared prospectively. Collision/velocity/general safety
and full R4.5 remain open; no new native allowance is implied.
