# Fresh-task learning at return checkpoints

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-return-probe --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-return-replay-probe --describe
```

Omit `--describe` to run one condition. Both retain the fixed A64/B64/A64 mean-CE
recipe. The second uses the existing bounded episode replay policy. After stream
construction and again after192 wakes plus stream.finish(), the existing E-task
probe trains a disposable owned checkpoint fork for16 updates. Probe checkpoints
0/4/8/16 retain CE, accuracy, class confusion/support/recall and ordered sequence
decisions. CE gain is CE0 minus CE16; both endpoints are always reported.

Why this: good fitting of a returning context does not show that a learner can
still acquire another task. The existing isolated probe measures that behavior
without adding E training to the benchmark learner or choosing from final scores.
E uses the existing fixed16 probe_train and2 probe_test packets. Probe_test labels
are used only for metrics; final A/B/C/D data do not exist until both probes and
all benchmark work finish. Every curve is retained, including worse outcomes.

## Source structure

```text
src/adapters/train_acquisition_pilot.py  existing recipe with explicit probe composition
src/adapters/return_task_probes.py      benchmark checkpoint/budget guard around probes
src/adapters/key_door_learnability.py   existing independent16-update probe slot/helper
src/app/fresh_task_probe.py            existing disposable learner training/scoring loop
scripts/run_coupled_attribution.py     existing CLI dispatch
```

The small adapter requires the selected copy to equal the authoritative stream
checkpoint at the current wake count before admitting probe work. Around external
fork/restore/train/predict callbacks it checks original outer budget bindings,
source/actor/candidate, receipts, saved states and replay evidence. It does not
advance outer budget clocks during a probe; the probe's original independent
session enforces the invocation deadline and RSS cap. Successful native updates,
charges and valid returned diagnostics are recorded before a later guard can fail.
Failed probes retain partial work and stop before later benchmark/final work.

Old probe slots still default to0/8; explicit new slots are0/192. Default helper
kwargs, timing/native calls and report shapes remain unchanged. The new callback
is optional. No new dependencies, settings, checkpoint format or training runner.

## Work and interpretation

Each condition adds32 disposable updates/256 transition exposures,16 prediction
calls/128 scored rows and18 teacher episodes. The off condition totals224 native
updates/1792 rows; replay-on totals320/2560. Benchmark clocks remain192 wakes,
with192 native updates off or288 on; probe updates never advance those clocks.
Array payload byte counts exclude Python/native-state/copy overhead; invocation
time and sampled process RSS include probe/diagnostic/guard costs.

Comparisons retain initial and post192 CE/accuracy/sequence curves. Different
starting losses and optimizer histories limit causal interpretations of gain
changes. This E test repeats a deterministic trace domain; it does not establish
new-task live success or broad generalization. Replay adds50% benchmark training
work, so off/on are not equal-compute controls. No circadian benefit follows from
implementation correctness.

Safely extend through these existing stream and probe ports, preserving fixed
protocols and independent budgets. Freeze/review a small local protocol before
new native work; do not tune from probe_test or final outcomes. Commands and
observed evidence are recorded in the stage RESULTS.md after execution.

## Observed paired case

| E-probe source | CE0 → CE16 | Accuracy0 → Accuracy16 | Complete sequences |
|---|---|---|---|
| Initial, either condition |1.618712 →1.325128|.125 →.5|0 at all points|
| Post192, replay-off |2.235092 →.490127|.25 →.8125|0 at all points|
| Post192, replay-on |2.710471 →.785617|.3125 →.75|0 at all points|

All four0/4/8/16 curves remain. Both trained states improve reserved E fitting
within16 disposable updates; complete E sequences and new-task live success are
unproven. Replay-on's larger CE reduction starts from a worse loss and ends worse
than off, so it does not show a plasticity advantage. Original benchmark outputs
match the accepted off/on reports exactly; final B retention and held-out C/D
success remain unmet.

One frozen paired case used544 native updates/4352 training rows,6115 unrolls,
363 predictions,124 environments/586 physical steps and26.8575 parent seconds.
Max observed RSS325468160B combines condition samples and serialization readings.
All original limits passed.239 distinct focused cases have passing latest results
following two preserved CLI dispatch failures and a four-case correction recheck;
Ruff/format/both-platform mypy and independent SOURCE/GO/ACTUAL passed.
[Exact commands, costs, failures and open claims](../artifacts/runs/r21-return-fresh-task-probes-20261016/RESULTS.md).

The original R2.1 implementation remains accepted; useful game adaptation and
release claims remain open. Next is a recorded-run inspector using these actual
layout/identity/transition/outcome/cost records; no additional training is needed.
