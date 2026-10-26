# Realized BP TRAIN wake/replay diagnostic

The existing prior-gradient helper cannot show the displacement produced by
optimizer momentum. `diagnose_bp_updates` now measures one fixed native BP cycle
on an owned copy: B-blue TRAIN wake, A-blue replay, A-red replay. It returns scalar
fit, direction and paid-work records, with no model state or deployed update.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.diagnose_minigrid_bp_cycle --describe
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.diagnose_minigrid_bp_cycle --train-report artifacts/runs/joint-catalogue-study-20261018/bp.txt
```

The public engineering example reuses saved TRAIN sources, validates their causal
histories, creates the original invariant BP H8/seed17/LR0.03/momentum0.9 learner,
performs one A-blue warmup, then the diagnostic. Four native calls/36rows total;
no new collection, teacher, physics, reserved F, final data or joint128 rerun.
One warmup does not reproduce the historical warm BP state or explain its F result.

## API and interpretation

```python
from src.adapters.torch_bp_update_diagnostics import diagnose_bp_updates
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

budget = ToyBudgetSession(
    ToyExecutionBudget(max_training_updates=3, max_replay_examples=18), clock
)
report = diagnose_bp_updates(snapshot, b_blue_train, (a_blue_train, a_red_train), budget=budget)
```

Only exact original invariant BP snapshots and complete nine-step TRAIN histories
are admitted. Source IDs/context/order and full current/next histories are checked;
this cannot authenticate historical acquisition of arbitrary external weights.
The fresh budget's owner, full limits, clock, start, progress, sampler and exact
observed charges stay bound. Native attempts and replay rows are charged before
calls; exceptions retain a detached incomplete paid prefix. Original snapshot
values/autograd metadata are pinned before the first callback. Inputs, source,
RNG and caller grad mode are checked; inference mode refuses before native work.

Each step measures the original eighteen-row old-A mean CE before/after, the
current training gradient, actual native parameter displacement, and the carried
momentum displacement `-0.03 * 0.9 * previous_buffer`. The full updated buffer is
`gradient + 0.9 * previous_buffer`, with a zero carried component on a cold step.
Whole/per-parameter signed dot/cosine and formula errors are reported. Negative
old-A gradient dot displacement denotes local first-order descent; actual CE is
also measured. Neither alignment nor one counterfactual cycle establishes causality.

## Costs and verification

One diagnostic costs3native updates/27rows, including18separately counted replay
rows; additional diagnostics15CE forwards/135rows and9backwards/81rows. Native
training adds its own3forwards/backwards/optimizer steps. Admission adds54current/
next projection rows; the public six-source loader adds108 separately. Logical
arrays exclude Python objects, validation/copies/model/optimizer/RSS/time costs.
Public failure output preserves paid warmup attempts/returns and source admission.

Independent NumPy tanh/CE gradients verify actual cold/warm SGD deltas, buffers,
old-A CE and geometry. **105 distinct scoped cases accepted**:91final scalar/
regression plus9repaired native and5reused early refusals; public CLI also executed.
Initial9native failures and3typing errors remain preserved with repairs. Four-path
Ruff/format,2source Windows/Linux mypy and independent SOURCE/PROTOCOL/ACTUAL GO.
Across failed/repaired/public work:29native calls/261rows,139CEforwards/1251rows,
107backwards/963rows, within declared limits. No comparative experiment ran.

The public example's old-A CE changes1.8992741108→1.6870722175, buffer error0 and
maximum parameter-step formula error7.45e-9. This verifies functionality only.
The preserved historical warm BP rehearsal result remains worse than its control.
[Exact commands, failures, receipts and review](../artifacts/runs/bp-realized-update-20261018/RESULTS.md).

## Structure and next extension

```text
src/adapters/torch_bp_update_diagnostics.py  owned native cycle/readout and accounting
scripts/diagnose_minigrid_bp_cycle.py       saved TRAIN IO and bounded public example
tests/test_torch_bp_update_diagnostics.py   independent math/isolation/failure checks
tests/test_minigrid_bp_update_cli.py        resource-free routing and failure prefixes
```

## Existing joint pilot integration

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --joint-catalogue --distinct-probe --old-task-readouts --return-to-a --bp-update-diagnostics --describe
```

Remove `--describe` to execute the fixed local case after freezing a scientific
protocol. The opt-in observer runs at owned checkpoint128 before reserved F
collection. It uses original B-blue TRAIN wake and A-blue/A-red TRAIN replay;
only a disposable learner is updated. A separate3-update/18-replay-row budget
shares the remaining whole-pilot wall/RSS envelope. Its owner/full limits/clock/
sampler/start/charges are pinned; completed counters and clock cannot later change.
Pure `--train-fit-bounds` can compose; other diagnostic, encoding, weighting,
rehearsal or game options refuse. Disabled dispatch/protocol shapes are unchanged.

`report.bp_update_diagnostics` holds the detached observed cycle and paid prefix.
The parent's existing evidence guards protect owner/value before each later
probe; failure refuses further work and restores original published evidence.
`work.bp_update_diagnostics` separates observed native attempts/returns, charges,
projections, CE/backwards and unknown work. Overall/per-method native and replay
totals include this work:195calls/7515rows/18replay; classifier work stays94/846.
Observer15 diagnostic forwards/135rows plus3 native CE forwards/27rows and9
diagnostic backwards/81rows plus3 native backwards/27rows. Overall predictions+
CE forwards304/8496rows, backwards204/7596rows. Admission54projection rows127872
array-equivalent bytes and diagnostic319680 returned array bytes are additional.
API validation counter1 excludes the integration's separate checkpoint validation/
copy and repeated parent guards; none of these fields is total process memory.
Observer elapsed time is nested in pilot time, so do not add it twice.

The sole synthetic native public case passed at checkpoint128 with original
benchmark128/6912rows, separate diagnostic3/27rows, then unchanged F/A probes.
Final evaluation remains not_run.237 distinct scoped checks/366 executions pass,
six-file Ruff/format/three-source Windows/Linux mypy and independent review GO.
Only an unused-import lint failure needed repair. No native/science retry.
Synthetic old-A CE1.5372253060→1.5357107520 and zero momentum formula error verify
runtime composition; they do not explain historical real-source BP interference.
[Exact commands and evidence](../artifacts/runs/minigrid-bp-update-route-20261018/RESULTS.md),
[ADR0433](adr/ADR-0433-bp-readout-at-owned-joint-checkpoint.md).

Additional structure:

```text
src/adapters/minigrid_bp_update_route.py    pure admission/description/cost reduction
src/adapters/minigrid_learnability.py      existing pilot owns checkpoint/allowance/guards
scripts/run_coupled_attribution.py         existing CLI dispatches explicit option
tests/test_minigrid_bp_update_route*.py    scalar admission/failure and one native case
```

Dependency direction: CLI → existing pilot → owned BP diagnostic → native learner;
pure route helper has no model or IO dependency. The small reviewed real-source
TRAIN case is now complete below; scalar reports lacked native128 weights/momentum,
so a bounded reconstruction was required. Keep final/probe selection excluded,
preserve all negatives and avoid momentum/dose/seed tuning. Expanded neural
learning/science/release claims remain open; the next implementation is below.

## Real-source checkpoint128 observation

The sole frozen public case completed and matched all270 benchmark probability
rows to the accepted original BP run exactly. Old-A CE decreased.3937437683→
.3906135261; all3aggregate carried-momentum dots were locally descent-signed,
with mixed parameter-block signs retained. There is no momentum intervention or
F input in this cycle, so it does not explain historical F/rehearsal interference
or establish acquired retention, recovery or CPC benefit. See the
[full report/costs/limitations](reports/bp-checkpoint128-train-cycle-20261018.md).
No further observer/seed/dose variant is the next implementation: proceed to a
separate bounded multiclass classification control through existing TRAIN-fit
ports, leaving all original recipes and negative results frozen.
