# Provisional readiness corrections through the existing learner

The geometry-only default described here remains v1. For opt-in official current
grasp feedback and distinct records, see [observed grasp corrections](observed-grasp-corrections.md).

A frozen student's own observations/actions/rewards remain the executed trace.
Explicit readiness queries offer separate provisional targets based on current
teacher-side geometry. These are not qualified expert labels at arbitrary
student states. The original fixed-schedule records/default behavior stay intact.

## Structure and boundaries

```text
src/core/pick_cube_readiness_corrections.py  distinct owned labels/arrival batches
src/core/pick_cube_readiness_teacher.py      optional executed-gripper feedback
src/core/pick_cube_public_corrections.py     same-family raw789→public21 view
src/infra/maniskill_corrections.py           existing query lifecycle, opt-in mode
src/adapters/continuous_correction_demo.py   existing private eight/one updates
scripts/maniskill_continuous_smoke.py        existing collection→update→17/19
 tests/test_readiness_correction_demo.py      independent complete-slice tests
```

Exact ReadinessCorrectionLabel/Batch types preserve2*step observation and
2*step+1 arrival ticks, TRAIN17/19 context, normalizedaction units, full executed
trace and exact public-observation pairing. The batch revalidates at arrival
release. Distinct `provisional_geometry_readiness_v1` provenance and
`pick_cube_geometry_readiness_v1` teacher definition survive projection and
updates; fixedtime batches cannot silently become readiness records or converse.
No reward/action substitution, B/statistics refit, replay, extra diagnostic
forward, final/probe/guard access, publication or budget renewal.

Query mode uses the PRECEDING actual native student gripper command. Only negative
closing intentions after a grasp-stage query advance the close counter; open
commands or lost geometric readiness reset it. Six proposed close labels alone
cannot advance lift. This is still not contact/force/attachment verification.
The self-executing teacher's default `executed_gripper=None` behavior is unchanged;
its accepted two-reset qualification is reused rather than rerun.

Original mode/teacher identity, typed scalar progress, port identity, paid counters
and immutable preceding labels validate around opaque metadata/geometry observers.
Final direct TRAIN authority check follows public getters before privileged
geometry. A corrupted/replaced sequence is poisoned, refuses further work/release,
and reports original bounded paid evidence without throwing from cleanup.
At most50labels/decisions; there is no unbounded history. Newrecords use current
stage readiness rather than a rule inferred from elapsed student steps.

## Programmatic use

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout, run_corrective_adaptation
from src.adapters.continuous_correction_demo import run_continuous_correction_update

# Actual work: declare a separate local native budget first.
collected = run_frozen_rollout(Path("fresh-readiness-collection.json"), learner=source,
    train_seed=19, visual_observation=True,
    corrective_queries=True, readiness_corrections=True)
if collected.report["complete"]:
    batch = collected.corrections
    updated = run_continuous_correction_update(source, batch,
        now=batch.labels[-1].labels_arrived_at)

# Alternative fixed composition, exclusive directory, same existing runtime:
summary = run_corrective_adaptation(source,
    output_directory=Path("fresh-readiness-adaptation"), visual_observation=True,
    readiness_corrections=True)
```

The examples are alternatives, not a request to repeat accepted runs. They
require an already owned exact TRAIN learner with frozen statistics/defaultMSE
basepolicy. NeuralBP/PC/neutral/active keep eight whole correction exposures;
nearest appends once. Full source/actor/native momentum/wake/config and paid
budget/receipt guards remain. Same-family public21 projection is supported through
existing `observation_projection`. Source/actor stay fixed and candidate private.

Existing CLI parser accepts `--readiness-corrections` alongside explicit
`--corrective-queries`; it still requires a programmatically supplied learner.
There is no new model loader. Nonboolean/missingquery/actor/clock/contextmix refuses
before SDK. Newmode/report/batch type binds before update and after opaque reads.
A wrongreturnedfamily cannot trigger privatework. Output layout remains collection,
update, updated17, updated19, summary; newmode adds its teacher definition only.
OldFalse keyword dispatch/CLI JSON remain unchanged.

## Work and verification

Training dose/clocks/budget never depend on observed success. PC uses its actual
native event/rowweight work; BP uses existing dense formulas; nearest reports
prototype exposure/distance separately. Label/geometry/renderer/prediction/
normalization/payload/copy/validation costs are not interchangeable with updates.
No equal-information/compute claim from equal eight updates or different student
trajectories. Paid failures retained/no candidate if release fails.

339focused cases pass Windows and pinnedLinux with scoped lint/format/type checks
and independent combinedtree acceptance. New56 cover all five tinylearner
families, actual executed-open/close feedback, mode/history/role refusal, paid
failures, public21 RGB pipeline and activePC19→update→17/19. Unchanged direct
mathreference reused. [Exact protocol, failures, commands, costs and native result](../artifacts/runs/readiness-corrections-20261020/RESULTS.md).

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_readiness_correction_demo.py tests/test_continuous_correction_demo.py tests/test_continuous_pc_corrections.py tests/test_nearest_corrective_adaptation.py tests/test_pick_cube_corrections.py tests/test_pick_cube_public_corrections.py tests/test_pick_cube_readiness_teacher.py -k 'not direct_reference'
```

Scoped Ruff/mypy commands on all seven changed source/test files are recorded in
the linked receipts. No new dependency, environment variable or configuration.
Native usefulness, off-reference correction quality, learned retention/recovery,
generalization, safety and CPCbenefit require separate evidence. Preserve all
old zero-success, normalization, failed budgets and incomplete comparative runs.

## Observed native result and next input dependency

The bounded conventional BP case is independently accepted as a negative result:
source19, candidate17 and candidate19 each used50 controls and had zero native
success. Eight updates on50 unique labels (400 row exposures) gave zero signed
TRAIN19 gains. All labels were alignment targets; this does not teach the full
pick sequence. Initial warm serialization failure, its paid work and an explicit
prospective same-question engineering amendment are retained in RESULTS/REVIEW.
No original-attempt conformity or independent replication claim.

The existing stream CLI reader still requires old fixed-teacher IDs. Explicit
admission of the already qualified readiness20/22-step complete-phase TRAIN
demonstrations is the next concrete dependency before another conventional
learning-to-control check. Preserve provenance/defaults and freeze that check
separately; additional fitting, seeds or PC/CPC comparisons are not automatic.

## Canonical state42 feature view

The [explicit canonical readiness composition](state42-canonical-corrections.md)
retains raw42 evidence and derives model labels after original arrivals. Its
acquisition_only option collects17 and returns17 once; the original19 adaptation
route remains the default.130focusedcases/platform and independent functional
acceptance; real task usefulness remains separately measured.
