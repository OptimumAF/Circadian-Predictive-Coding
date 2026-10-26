# Bounded PickCube mass context

## Accepted executed taskspace history binding

The existing frozen rollout can compose the fixed mass-only TRAIN17 context with
`PickCubeTaskSpaceHistoryRecipe(mode="executed")`. Mass changes dynamics; it does
not change the admitted Panda table/controller WORLD axes or add an observation
feature. Original raw42 export, actual nativefloat32 command history, mass
readback/work and frozen source checks are all retained. No successful native
mass transfer or adaptation result is implied by this software extension.

After the existing teacher boundary qualifies and releases a complete mass TRAIN
record, encode its model input with the explicit same context:

```python
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_action_history import (
    action_history_training_episode,
    validate_action_history_training_episode,
)

mass = pick_cube_mass_context()
recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
model_episode = action_history_training_episode(qualified_raw_train, recipe, mass_context=mass)
owned_model = validate_action_history_training_episode(model_episode, recipe, mass_context=mass)
```

The pure conversion owns the input and preserves role/context/ID/actions/flags.
It does not qualify an external teacher, release labels, fit/update a learner or
include mass itself as a feature. Omitting metadata on a mass episode refuses;
ordinary17/19/20 inputs retain their existing default API. For serving:

```python
from scripts.maniskill_continuous_smoke import run_frozen_rollout

# Requires a separately frozen local native protocol and an exclusive new path.
result = run_frozen_rollout(
    output_path, learner=frozen_candidate, train_seed=17,
    action_history_recipe=recipe, mass_context=mass,
)
```

Only exact native BP/PC-family learners with the admitted12D policy and original
fixed mass metadata are accepted. Zero/47D recipes, alternate TRAIN seeds,
nearest/actor/clock/visual/camera/lighting/guard/probe/query/rollin/decoder/joint
mixing refuse before source fork or scene creation. Equal-valued replacement or
mutation of an admitted causal mass request refuses. A failed callback/control
retains paid predictions/native-work evidence and cannot claim completion.
Default mass raw42 and ordinary history paths retain their existing behavior.

169focused synthetic checks pass per Windows/pinnedLinux (52new+117purecore),
with final scoped statics and independent actual acceptance. No fitting/training
or actual simulator physics. A separately frozen native case now succeeds at mass-x2 TRAIN17 and a fresh
ordinary17return in17controls each, with no update beyond source reconstruction.
Independent review accepts exact readback/work, raw initials, history and costs.
This fixed condition needed no adaptation. Broader dynamics adaptation/reliability
and R4.5 remain open; older raw42 mass negatives below remain unchanged.
[Physical outcomes, exact commands and costs](../artifacts/runs/learned-mass-transfer-20261021/RESULTS.md).
[Protocol, commands, costs and review](../artifacts/runs/taskspace-mass-binding-20261021/RESULTS.md).

## Files and boundaries

```text
src/core/pick_cube_mass_context.py        Pure fixed TRAIN17 declaration/binding
src/infra/maniskill_continuous.py         Existing episode port, native readback/work
tests/test_pick_cube_mass_context.py     Synthetic boundary behavior (no physics)
src/infra/maniskill_task_teacher.py       Existing qualified teacher with optional mass
scripts/maniskill_continuous_smoke.py     Existing frozen rollout/CLI with optional mass
scripts/maniskill_task_teacher_smoke.py   Existing teacher CLI with optional mass
scripts/run_native_stream_demo.py        Qualified saved inputs/existing runner dispatch
tests/test_mass_context_interfaces.py    Teacher/rollout/qualification/default tests
tests/test_mass_native_context_case.py   Actual runtime/nearest with fake native IO
tests/test_mass_stream_input.py          Qualified saved input and CLI routing
tests/test_continuous_a_only_control.py  A-only isolation, costs and CLI behavior
docs/adr/ADR-0474-declare-bounded-mass-only-context.md  Decision and limits
artifacts/runs/pick-cube-mass-context-20261019/         Protocol/check receipts/review
artifacts/runs/mass-context-return-20261019/           Case/tests/failed launch/repair
artifacts/runs/mass-context-return-legacy-20261019/    Corrected native outcome
artifacts/runs/mass-stream-input-20261019/             Input gates/review
artifacts/runs/mass-mlp-learnability-20261019/          Fixed MLP A/B/A diagnostic
artifacts/runs/mass-a-only-control-20261019/            Matched information control
```

Why this: object/dynamics context is a concrete remaining R4.5 dependency. Use
the existing episode boundary, preserving default state/RGB/camera behavior.
No new runner, learner, label permission, dependency or environment variable.

## API

```python
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.infra.maniskill_continuous import ManiSkillPickCubeEpisode

context = pick_cube_mass_context()
with ManiSkillPickCubeEpisode(
    "mass-boundary", context.context, seed=context.seed, mass_context=context
) as episode:
    original_measurement = episode.mass_readback  # owned; no implicit live reread
    # Ordinary bounded controller actions go through episode.step(...).
    # episode.episode() requires ordinary completion and checks current mass.
    measured_work = episode.mass_work
```

This optional simulator API is not an authorization to repeat experiments;
freeze a small local native protocol before actual physical/learning claims.
The fixed context is TRAIN17 only, mass multiplier2.0; arbitrary values, wrong
roles/seeds, relabeling, absent reserved metadata and implicit mass+camera mixing
are refused before scene creation. Visual observations may use the unchanged
base frozen recipe under the distinct physical context.

After explicit reset, require one float32 native mass in(0,0.5]kg, call public
Actor.set_mass once, and verify exact doubled float32 result <=1kg. The mass
setter is the only intentional physical-property operation. Inertia/COM effects
are unmeasured; this is not density scaling or an observed invariant claim.

Source and immediate readback plus initial encoding guard cost3validated reads;
each successful control has before/after mass guards (2reads); complete episode
export costs1read. Reads validate CPU float32 shape(1,), finite positive bounds.
Counters distinguish attempts from validated returns and setter attempts/returns.
Returned metadata is owned, immutable and constant size. Each external getter/
setter is followed by actor/context identity checks; changed mass/identity or
metadata permanently refuses further controls/extraction. Original source/target
measurements are retained, without a success-history chain. Invalid setters
close the failed construction; no automatic rollback, reset or retry occurs.
Default calls do not resolve cube/mass or introduce factory arguments.

## Checks

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_pick_cube_mass_context.py tests/test_maniskill_continuous.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/pick_cube_mass_context.py src/infra/maniskill_continuous.py tests/test_pick_cube_mass_context.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/pick_cube_mass_context.py src/infra/maniskill_continuous.py tests/test_pick_cube_mass_context.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent src/core/pick_cube_mass_context.py src/infra/maniskill_continuous.py
```

Exact executed commands/outcomes and limitations are recorded in the
[results](../artifacts/runs/pick-cube-mass-context-20261019/RESULTS.md) and
[independent review](../artifacts/runs/pick-cube-mass-context-20261019/REVIEW.md).

## Next safe extension

The matched A-only control and bounded native lighting boundary are accepted.
Next propagate lighting context/original readback/paid work through the existing
teacher and frozen-rollout interfaces, preserving qualification and default
reports before a separately frozen acquisition/adaptation case. Those APIs still
expose camera/mass only. [Lighting API, evidence and checks](pick-cube-lighting-context.md).
Preserve this negative mass control, all neural failures and accepted R2.1/update
evidence; full R4.5 remains open.

## Verified scope

216identical Windows/Linux cases pass (new89 plus127affected boundary regressions),
scoped Ruff6/format6/types3 pass. Separate one-use native API check completed:
0.06400000303983688kg→0.12800000607967377kg,104mass-port reads/one setter during
50controls250knownphysics, then fresh-default0.06400000303983688kg. Default
observations have no controls; two external reads are counted separately. Native
zero-action episode returns0/truncates. Summary retains controls and selected
state views, not every intermediate observation. No learning benefit or observed
inertia/COM invariance follows. Full R4.5 remains open; later interface/native
case evidence follows below. See RESULTS/REVIEW for limits and acceptance.

## Teacher and frozen-rollout integration

The existing `collect_task_teacher(..., mass_context=pick_cube_mass_context())`
and `run_frozen_rollout(output, ..., mass_context=pick_cube_mass_context())` now
carry this explicit physical context and original native readback/work. Both
existing CLIs support `--mass-drift`; it cannot mix camera/corrective/guard
allocations. Default CLI fieldsets are unchanged. Python TaskTeacherResult adds
optional None fields; raw dataclass asdict includes them (CLI strips inactive
fields). No label or training permission is inferred from a partial result.

```powershell
# Native examples require a separately frozen small local protocol before use.
.venv/Scripts/python.exe -m scripts.maniskill_task_teacher_smoke --output fresh-mass-teacher.json --mass-drift
.venv/Scripts/python.exe -m scripts.maniskill_continuous_smoke --output fresh-mass-rollout.json --mass-drift
```

Teacher qualification requires complete sparse final success, fixed identity,
role/spec, exact original measurements and actual one-export costs. A failed
teacher retains observed work and cannot release training data. Frozen rollouts
preserve source parameters/momentum/counters, retain paid failures, and reject
late metadata/owner changes. Pure final port metadata validation follows source
owner checks without new physics reads. Postclose checks validate original
measurement ownership, not live mass after simulator cleanup.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_mass_context_interfaces.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_mass_context_interfaces.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py tests/test_mass_context_interfaces.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent src/infra/maniskill_continuous.py src/infra/maniskill_task_teacher.py scripts/maniskill_continuous_smoke.py scripts/maniskill_task_teacher_smoke.py
```

210final combined cases pass on Windows/pinnedLinux, scoped Ruff8/format8/types4
pass, independent source/evidence acceptance. Initial Optional type failure and
one fixture-dispatch test failure are preserved. These synthetic checks do not
establish actual native mass teacher success or learning benefit. The next
implementation is a fixed small mass acquisition/return case through the existing
runtime, after a frozen reviewed protocol, accepting all failures/negative results.
[Exact commands/costs/skips/limits](../artifacts/runs/mass-context-interfaces-20261019/RESULTS.md) ·
[independent review](../artifacts/runs/mass-context-interfaces-20261019/REVIEW.md).

## Fixed native mass acquisition and original-context return

The saved A-only nearest baseline is restored without fitting. On the fixed
TRAIN17 doubled-mass context it fails (50controls/return0). After one private
29-row append at logicaltick2, its detached candidate succeeds on mass-B
(29controls/return1) and freshdefaultA (29controls/return1). Original source and
serving actor remain unchanged. B gain return/any/final is+1/+1/+1; A retention
loss is0/0/0. Candidate2updates/58rows, original1/29; no promotion or final access.
This is same-trajectory TRAIN memory, not neural/generalization/CPC advantage.

Full four-scene cost includes teacher29:137controls/685knownphysics,
108predictionrows/4814nearest candidates, prototypepayload10672to21344B.
The original launch failed before physics on legacy reference shape and remains
failed/spent. Its narrow repair passed actual saved-source admission; a separately
prospective corrected allocation completed within unchanged aggregate bounds.
All failures and the unsuccessful original-actor B trajectory are retained.

```powershell
# Synthetic runtime/IO regressions; these commands do not launch the simulator.
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_mass_native_context_case.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check artifacts/runs/mass-context-return-20261019/native_case.py tests/test_mass_native_context_case.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check artifacts/runs/mass-context-return-20261019/native_case.py tests/test_mass_native_context_case.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent artifacts/runs/mass-context-return-20261019/native_case.py
```

Original47case platform runs plus repaired10source cases/40unchanged evidence
reuse are recorded precisely; no single final50case invocation is claimed.
[Native outcomes/costs/commands/next action](../artifacts/runs/mass-context-return-legacy-20261019/RESULTS.md),
[original failure/repair](../artifacts/runs/mass-context-return-20261019/RESULTS.md),
[independent review](../artifacts/runs/mass-context-return-20261019/REVIEW.md).
Both native allocations are spent; rerunning the historical exclusive outputs
is refused. Future experiments require a separately frozen small local protocol.

## Saved mass input in the existing stream CLI

`--mass-drift` selects a qualified new mass TRAIN17 receipt while warm remains
original TRAIN17. Default behavior still reads new TRAIN19. One bounded owned
parse preserves exact context, original mass readback/work and full existing
TaskTeacherResult qualification before fitting. Default routes refuse retained
guard/camera/mass markers rather than silently dropping them. Original None or
absent metadata is compatible. Final/probe/guard episodes cannot enter training.

The report's `input_context` holds original mass metadata, source bytes/cap and
historical teacher costs from the same parsed receipt. Those are retained prior
acquisition, not new simulator work. State42 and full baseRGB789 are supported;
visual receipts require exact complete encoding costs and report
`teacher_visual_work`. Mass+`--public-vector-only` is refused before IO because
that projection has no declared mass binding. No new dependency/environment
variable, runner or optimizer is introduced.

```powershell
# Boundary tests: all runners stubbed, no fitting or simulator calls.
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_mass_stream_input.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check scripts/run_native_stream_demo.py tests/test_mass_stream_input.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check scripts/run_native_stream_demo.py tests/test_mass_stream_input.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent scripts/run_native_stream_demo.py tests/test_mass_stream_input.py
```

264 identical selected Windows/Linux cases pass, scoped checks and actual A29/
massB29 read-only admission pass; independent acceptance and exact selection are
in [input results](../artifacts/runs/mass-stream-input-20261019/RESULTS.md).

A separately frozen one-use existing CLI run used unchanged default MSE/H64/
seed17/lr.01/momentum.9,8updates per A/B/A phase. Final A/B half-MSE is
0.043576914149010604/0.04419919483834617 after24updates696row exposures. A error
also improves during B; no forgetting occurs in this case. Costs include8extra
conditioning backward diagnostics,7,750,656denseMACs and638repeated-row exposures.
No new native run, heldout access, promotion or generalized neural skill claim.
Final gripper maximum errors remain>1.52; lower training loss does not prove
successful control. A-only equal-exposure control is still needed before causal
attribution to B labels or plasticity machinery.

[All four checkpoints, six signed metrics, costs, failure/limits and next action](../artifacts/runs/mass-mlp-learnability-20261019/RESULTS.md).
The historical one-use command is in that protocol; its budget is spent. Freeze
and review the next small scientific protocol before additional comparative fits.

## Matched A-only control

Both public learnability APIs accept `a_only_control=False` by default. Opting in
trains warm A during all three phases, while checkpoints still diagnose the actual
released TRAIN A and B. Phase labels stay warm/new/warm_return; event context and
inbox/receipt episode IDs identify the actual A update source. B never enters an
update. Both inputs must pass original role/spec/identity/finite admission before
fitting, including B used for read-only diagnostics. Default reports are unchanged.

The existing MLP CLI supports `--a-only-control`, composed with `--mass-drift`,
`--phase-updates` and the existing binary objective where applicable. It refuses
non-MLP scenarios before input IO. Control-only work fields distinguish actual
training rows by context, unique training rows, repeated exposure and diagnostic
rows by context (for each conditioning/residual/actor group).

```powershell
# Synthetic focused checks, including one small direct momentum reference.
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_continuous_a_only_control.py tests/test_continuous_learnability_demo.py -k "not test_should_match_one_direct_reference_every_endpoint_full_momentum_and_all_24_losses"
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_a_only_control.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_a_only_control.py
.venv/Scripts/python.exe -m mypy src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_a_only_control.py
```

The 80 selected cases pass on Windows and pinned Linux, with scoped statics and
independent review. Initial typing failure and its repair remain recorded.
A separately frozen one-use A-only comparison completed 24 updates / 696 rows;
the original treatment was read-only. Policy, all 84 normalization statistics,
first eight updates and checkpoints 0/8 matched exactly. Final A/B half-MSE was
0.04239958105626931 / 0.04338906405810711, below treatment
0.043576914149010604 / 0.04419919483834617. Warm acquisition tied; the other five
original metric directions favored A-only. Treatment gripper residuals were
slightly better, while all three translation axes were worse. Both conditions
retain large gripper errors and no observed forgetting.

This establishes no aggregate contribution from B labels under this exact related
TRAIN recipe; it does not establish generalization, native skill or CPC advantage.
Costs include read-only B conditioning gradients: 56 forward passes / 1,624 rows,
32 backward passes / 928 rows and 7,750,656 dense MACs in each arm. Source exposure
is A696/B0 versus treatment A464/B232. The one-use allocation is spent.
[All four curves, six differences, axis tradeoffs, costs, commands and skips](../artifacts/runs/mass-a-only-control-20261019/RESULTS.md) ·
[independent acceptance](../artifacts/runs/mass-a-only-control-20261019/REVIEW.md).
