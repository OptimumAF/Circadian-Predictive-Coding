# MiniGrid weighted recurrent acquisition control

The existing weighted recurrent CE learner now has a MiniGrid composition.
This is a distinct objective, alongside the accepted unweighted recurrent
control; the original baseline and its negative results stay unchanged.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_weighted_bc_smoke --describe
```

`--describe` constructs no model/environment and reads no reference file.
The fixed reference is
artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json. Run the
separately reviewed small protocol to a new destination only:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_weighted_bc_smoke --output artifacts/minigrid-weighted-a.json
```

Output is strict finite JSON, at most16MiB and exclusive; an existing destination
or missing parent is refused before native setup. Dependencies are the existing
optional Torch/MiniGrid packages; no new settings or dependencies. Protocol and
case readiness are recorded in artifacts/runs/r21-minigrid-weighted-a-20261016/.

## Fixed comparison and isolation

Reuse the accepted six TRAIN sources,128 original A packets, unweighted receipts
and0/32/64/96/128readouts. Only the weighted arm performs new training128/1152rows,
with seed17/RNN148/H8/K7/lr.03/momentum.9/max16 and one-tick delayed labels.
Native loss is sum(weight*CE)/sum(observed target weights); current-episode
class mass is n/(present classes*count), absent-class weight1. No weight knobs,
seed/dose search or metric changes. The existing array wrapper binds each update
to its next exact owned episode, full state and optimizer continuation.

The fixed reference validates before training and is admitted as a detached
input. Source/actor/receipts/clock/budget/paid readouts remain guarded. No opaque
historical checkpoint is reconstructed or claimed. New initial raw probabilities
and native observation/action/probability law must match the retained initial
control within1e-7absolute/0relative tolerance. Mismatch refuses comparisons.
All new training finishes before new readouts. Historical TRAIN readouts are
retained evidence and are not passed to the learner factory. No final D/probe E,
health diagnostic, replay, RL or promotion enters this protocol.

Raw unweighted CE/accuracy/exact sequence uses both original A cues at all five
checkpoints. Live first-argmax/all7native actions uses fresh148-coordinate RNN
sessions at0/128. Both exact raw cue sequences AND both native goals are needed
for joint acquisition. Keep those flags separate; wholecase failure leaves
scientific comparisons null. Signed changes are weighted-minus-unweighted
CE0-minus-CE128, success-count change and mean-return gain. All points, cues and
negative results remain. This is not a CPC or generalization comparison.

## Costs and extension

New work is128SGD/1152rows,10raw forwards/90rows,4live episodes<=64steps and0
new teacher episodes. Historical unweighted128SGD/1152rows and original source/
readout costs remain paid provenance. Aggregate exposure is256/2304, not256new
updates. Equal exposure does not mean equal computation. Wholecase180body/
210parent seconds/1GiB, inherited768MiB per-arm budget, CPU intra1/interop1/
BLAS1/GPU0. JSON payload size is not total memory. Schedule records, seals and
byte copies, checkpoints and guard computation count toward elapsed/RSS.

The wrapper's measured128episode snapshot improved from.1410124s to.018863s
by comparing both records and their seal as whole bytes, keeping exact-field/
owner/native-state refusals. Original and postrepair readonly calibrations ran
zero SGD/predictions/environments; their budgets and results are separate.

```text
src/adapters/minigrid_weighted_bc_pilot.py  reference/new-arm composition
src/adapters/minigrid_policy_evaluation.py TRAIN A-only weighted148 readout
src/adapters/torch_train_weighted_learner.py existing objective/fast schedule guard
scripts/run_minigrid_weighted_bc_smoke.py  fixed describe/run/exclusive output
tests/test_minigrid_weighted_bc_pilot.py   weights/roles/streams/reference/failures
```

Use the existing guarded stream and evaluator for future controls. Keep method
identity, native loss and measured cost explicit. Retention/recovery requires
prior useful acquisition; no automatic dose extension follows a failed result.
See ADR-0357 and stage RESULTS.md for exact qualification commands and outcomes.

## Focused checks

From the repository root in PowerShell, the affected scope can be checked with
ordinary commands. The recorded receipts above retain the original executions.

```powershell
$weightedScope = @('src/adapters/minigrid_weighted_bc_pilot.py', 'src/adapters/minigrid_policy_evaluation.py', 'src/adapters/torch_train_weighted_learner.py', 'scripts/run_minigrid_weighted_bc_smoke.py', 'tests/test_minigrid_weighted_bc_pilot.py')
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_weighted_bc_pilot.py tests/test_torch_train_weighted_learner.py tests/test_minigrid_fixed_a_evaluation.py tests/test_minigrid_history_evaluation.py tests/test_minigrid_fixed_a_acquisition.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @weightedScope
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @weightedScope
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @weightedScope
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @weightedScope
```

## Observed local result

The sole case completed in79.3266 parent/76.1832 pilot seconds with389103616B
sampled pilot peak RSS. All144 focused tests, scoped Ruff/format and win32/Linux
type checks pass. This is scoped qualification; clean-clone and release checks
were not run.

The initial observable law matched. Both controls finish at17/18 raw labels and
one native goal: unweighted succeeds for red and fails blue; weighted succeeds
for blue and fails red. Both joint acquisition flags remain false. Weighted
minus unweighted CE gain is−.0117768311; success-count and mean-return gain
differences are0. The distinct objective supplied no live benefit in this case.
All checkpoints and negative terminals are retained. New work was128SGD/
1152rows/10raw90rows/4live48steps, zero new teacher work; the historical control
was not rerun. [Commands, failures and limits](../artifacts/runs/r21-minigrid-weighted-a-20261016/RESULTS.md).
