# TRAIN readout port for a warm reward checkpoint

The existing MiniGrid evaluator now admits a distinct `recurrent_warm_reward`
method for owned native checkpoints128 and136. This port addresses variable
reward episode lengths; original BC methods keep their nine-rows-per-update law.
The accepted runtime smoke completed136updates/1274rows, which its old evaluator
could not read out. No new native experiment runs in this component scope.

## Contract

`evaluate_checkpoint` in src/adapters/minigrid_policy_evaluation.py accepts one
new optional keyword, `native_training_examples: int | None = None`:

| Method/checkpoint | Required row metadata |
|---|---|
| recurrent_warm_reward /128 | Exact native int1152 |
| recurrent_warm_reward /136 | Exact native int1160..1280 |
| Existing methods | None; original checkpoint/9rows-per-update law |

Eight complete reward episodes each contain1..16 transitions, explaining the
second range. Actual snapshot counters must match the declared counts. Numeric
metadata grants no release or receipt authority: the caller's completed-work
guard must bind actual eight receipts, transition-row sum, original budgets,
full warm proof, source/actor/candidate/checkpoints and preceding readout evidence.

Only exact predefined TRAIN A/B/C_train layouts and six original nine-row148/K7
cue-ordered TRAIN packets are admitted. No A-only, D/E/final/probe, other point,
public-history option or action mask. Native policy remains148/H8/K7/seed17/
lr.03/momentum.9/max16, with six fresh zero-state recurrent sessions and all seven
first-argmax native actions. Scores are unchanged raw unweighted CE, accuracy
and exact sequence accuracy. New-method reports add native_training_examples;
old method defaults/output and final gates are unchanged.

## Caller example

Use actual owned native snapshots before teardown; the saved JSON is not a model.

```python
report = evaluate_checkpoint(
    source,
    owned_endpoint,
    method="recurrent_warm_reward",
    updates=136,
    native_training_examples=1152 + recorded_reward_rows,
    layouts=predefined_layouts()[:3],
    role="train",
    sources=original_six_train_packets,
    budget=readout_budget,
    check_boundary=completed_work_guard,
)
```

This is the evaluator port used by the explicit recipe option described below.
Recipe composition must finish all training before new before/after readouts and preserve the
completed training phase's original charges/clocks. Its45-second training cap
must not be renewed or charged as readout time; use an explicit bounded readout
phase under the original whole deadline, with frozen completed-work validation.
Count six raw forwards/54rows, six sessions/environments and up to96native live
steps per checkpoint, plus checks/copies/native allocations. Retain partial
costs and mandatory cleanup if a state/receipt/resource guard fails.

## Verification and extension

```text
src/adapters/minigrid_policy_evaluation.py        existing evaluator/new method
tests/test_minigrid_warm_reward_evaluation.py     independent fake-boundary tests
```

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_warm_reward_evaluation.py tests/test_minigrid_learning_pilot.py tests/test_minigrid_fixed_a_evaluation.py tests/test_minigrid_history_evaluation.py tests/test_minigrid_weighted_bc_pilot.py tests/test_minigrid_warm_prefix.py
$readoutScope = @('src/adapters/minigrid_policy_evaluation.py', 'tests/test_minigrid_warm_reward_evaluation.py')
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @readoutScope
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @readoutScope
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @readoutScope
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @readoutScope
```

187 focused cases pass,37 new+150 existing, with scoped statics. These use fake
native boundaries, zero SGD/physics. Exact initial typing failures/commands and
review evidence: artifacts/runs/r21-minigrid-warm-readout-20261016/RESULTS.md.
This is scoped component qualification; recipe and native evidence are recorded
separately below, and scientific/release gates remain open. Two endpoints can describe
new B/C change and A endpoint retention; they cannot establish context-return
recovery, causal reward benefit, CPC advantage or generalization. Keep negative
acquisition findings and all matched baseline identities unchanged.

## Existing warm recipe composition

The explicit option is:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --warm-start --train-readouts --describe
# Run only within a separately frozen local protocol; choose a new output path.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_reward_smoke --warm-start --train-readouts --output artifacts/minigrid-warm-train-readouts.json
```

The option requires `--warm-start`. Cold and warm-only paths keep their existing
protocols. Both added checkpoint readouts occur after all eight reward updates.
They reuse the six original known TRAIN packets; final layouts and isolated
fresh-task probes are unavailable. Each checkpoint gets six raw cue scores
and six fresh recurrent live episodes using the original seven-action argmax.

Why this: a 136-update reward learner has variable native training rows, and
JSON cannot reconstruct its missing weights or optimizer. The recipe passes
its actual owned checkpoints to the accepted evaluator instead.

```text
src/adapters/minigrid_reward_pilot.py           existing runtime composition
src/adapters/minigrid_warm_start.py             guarded original TRAIN sources
src/adapters/minigrid_warm_readouts.py          two bounded diagnostic outputs
scripts/run_minigrid_reward_smoke.py            explicit CLI option
tests/test_minigrid_warm_readout_pilot.py        phase/isolation/failure cases
```

The original training phase completes its final memory and receipt checks
before its clock and charges are frozen. Diagnostics cannot advance, replace,
renew or forgive changes to that completed budget. Their separate zero-update
budget has at most 60 seconds within the original 180-second whole deadline,
sharing the original RSS sampler. The reward phase retains its 45-second cap;
all original source, actor, candidate, receipt, row, sampler and prior-evidence
checks still apply. See [ADR-0360](adr/ADR-0360-separate-completed-training-and-readout-time.md).

Two checkpoints add twelve raw forwards/108 rows and twelve fresh sessions,
with at most 192 live predictions/physics steps. Checks, copies, snapshots,
partial failures and RSS also count. Before/after changes describe known TRAIN
contexts only; failed acquisition remains a failed premise for retention.
They do not establish within-stream task-return recovery or a causal reward
advantage. No new teacher, baseline, replay, probe or final evaluation is run.

Combined code qualification passes142 focused cases, with a corrected exact
final-callback regression and final scoped statics; independent SOURCE/ACTUAL
acceptance is recorded in the [recipe evidence](../artifacts/runs/r21-minigrid-warm-readout-compose-20261016/RESULTS.md).
The original native attempt stopped after six reward commits at the original
45-second cap, before either readout. Its134 updates/1242rows remain permanently
paid. At that stage native completion was open; this is
a timeout, not proof of a learner defect. No retry or budget increase followed.


## Accepted repaired native composition

A separately qualified guard repair retains full proof before admitting work
and after the final external snapshot callback, while deferring duplicate large
payload checks only inside a synchronous completed read-only warm guard. Nine
new corruption/scope regressions bring the focused set to151 passing cases.
The separately reviewed native case completes eight reward updates and both
TRAIN readouts under the unchanged original limits; earlier failures remain
SPENT. [Outcome, costs and negative results](reports/minigrid-warm-reward-readouts-20261016.md).
Cumulative four-case training is540updates5032rows, permanently charged.
This establishes scoped runtime behavior. All contexts still fail joint
acquisition, and endpoint CE/success worsens. Additional-update momentum and
paired runtime composition remain open prerequisites for causal comparison.
