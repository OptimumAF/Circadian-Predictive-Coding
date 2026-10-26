# Train-only episodic acquisition diagnostic

Run from repository root:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-train-fit --describe
```

Omit `--describe` for the fixed local diagnostic. This creates two actual A cue±
complete train-role teacher traces and repeats their ordered episodes for exactly
64 mean-CE updates on H8 seed17, learning rate.03/momentum.9. There are16 collected
source records and512 training exposures. Numeric values can repeat within those
records; unique packet IDs do not create independent samples.

All five checkpoints0/8/16/32/64 score the same two train packets on separate
owned forks. Reports retain CE, accuracy, per-class support/confusion, ordered
decisions, exact-sequence accuracy and gains against a fixed train-label empirical
prior/majority rule. The budget never stops early or extends based on fit.
These are resubstitution metrics. They do not measure held-out generalization,
new-task acquisition after a context shift, retention, recovery or game success.

Why this: earlier fixed-budget adaptation recipes did not establish useful
acquisition. Checking whether the unchanged learner can fit complete teacher
episodes helps distinguish a fitting bottleneck from later adaptation failures.
No final/probe-test packets, live rollouts, replay, reward, weighted objective,
promotion, deployed model or benchmark learner participates in this recipe.

## Structure and extension

```text
src/app/coupled_attribution.py             existing stream, explicit training_only
src/core/context_return_metrics.py        pure fixed endpoint arithmetic
src/adapters/train_acquisition_pilot.py    fixed recipe and train-fit reporting
scripts/run_coupled_attribution.py         existing CLI dispatch
tests/test_training_only_stream.py        admission/lifecycle/continuation
tests/test_train_acquisition_pilot.py      recipe/isolation/cost/failure checks
tests/test_context_return_metrics.py       signed metrics/invalid inputs
tests/test_train_context_return_pilot.py   phases/charges/isolation/return recipe
```

`AttributionStream(..., training_only=True)` requires the exact empty heldout
tuple `()`. It preserves ordinary train-role validation and native receipts,
budgets, observers and continuation. `finish()` seals the complete stream with
`evaluation_scope="training_only"`, no scored checkpoints and zero heldout
forwards. A checkpoint observer may independently perform explicitly train-only
diagnostics. Default stream behavior still requires held-out final data and
defers its scoring until training finishes. Mode changes refuse at existing
authority boundaries.

The adapter checks source/isolated-source/serving state around diagnostics and
at readback. Failed runs retain attempted/completed score work, permanent charged
updates, partial curves and available native counters/receipts; primary and
secondary errors remain distinct. Input-array byte counts exclude Python/native
state/copy/graph/report allocations; total wall time and sampled RSS include them.

No dependencies, settings or environment variables are added. Existing CPU and
thread controls apply. Focused exact commands, failures, limits and results are in
[the local stage](../artifacts/runs/r21-train-acquisition-20261016/RESULTS.md).
For a new diagnostic, keep train/held-out roles explicit, isolate its learner,
record all exposures and cost, and freeze a small protocol before native work.
Do not adjust seeds, metrics or budgets to turn a failed fit into success.

## Fixed A→B→A context return

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-train-return --describe
```

Omit `--describe` for the separately frozen return recipe: A64/B64/A64 updates
on one continuous optimizer and one192update budget. Four train-role teacher
traces are collected once, then copied into192unique-ID packets:1536exposures
from32collected source records. Every fixed checkpoint0/8/16/32/64/72/80/96/128/
136/144/160/192 scores all four A/B packets. There are52diagnostic forwards and
416scoring rows. The ordinary64update recipe and its report remain unchanged.

`src/core/context_return_metrics.py` owns pure signed endpoint arithmetic and
strict validation; it does not select checkpoints or drive training. At0/64/128/
192, the report retains both layout values and initial A gain, new B gain, old A
loss, return A gain and return gap. CE gains are earlier minus later; accuracy
and sequence gains are later minus earlier. Positive old loss/return gap are
worse. The initial A fit is explicitly marked incomplete when either sequence
was not fitted at64; later improvement cannot certify restoration of a fully
acquired skill.

Pooled and per-layout prior/majority references use all declared training labels,
including future-phase labels. These are descriptive offline references, not
causal online controls. B predictions before its training phase are diagnostics
on owned forks; neither labels nor reports change the deployed/benchmark learner,
native training schedule, optimizer or budget. Final/probe-test data stay absent.

The original A prefix must be recreated because the previous JSON lacks the
opaque optimizer/gradient state needed for continuation. Its original five A
score/decision curves match exactly; this is necessary continuation work, not
an independent baseline replication. All negative results and original costs
remain valid. See [protocol, results and exact commands](../artifacts/runs/r21-train-context-return-20261016/RESULTS.md).

The measured case fits both A training sequences by192, but B accuracy declines
from87.5% at128 to50% after returning to A. Initial A fitting was incomplete, and
no fixed-A equal-exposure control ran in that original study; there is no recovery-speed, savings,
transfer, generalization, live-game success or circadian-benefit conclusion.
Next validate a separately frozen closed-loop outcome through the existing
cached/live evaluator, after all training; do not tune training from final scores.

## Fixed A128 equal-A-exposure control

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-fixed-a-exposure --describe
```

Omit `--describe` for the explicit mean-CE control. It uses the original two A±
TRAIN traces, H8 seed17/rate.03/momentum.9,128 alternating updates and1024rows.
Nine points0/8/16/32/64/72/80/96/128 score both original packets on owned forks.
Why128: the accepted A64/B64/A64 recipe already contains128A updates; this fills
its documented missing equal-A-exposure control without selecting a dose from
weighted failures or searching for favorable outcomes.

Compare raw A metrics at control0..64 with return0..64, and control72/80/96/128
with return136/144/160/192. Retain return post-B128 at64A exposures separately.
Report all fixed points/cues: return minus control CE, control minus return
accuracy/complete-sequence accuracy. Control128SGD/1024rows versus return192/
1536 (128A+64B) is unequal total work; source, scoring and diagnostic costs also
differ. InitialA64 remains incomplete. This comparison alone cannot establish
fully acquired-skill restoration, recovery savings, causal transfer or CPC benefit.

After all128 commits and finish, the control additionally plays both TRAINcues
at owned0/128 with the existing legal greedy evaluator/max12steps. The train
prior plays both once, with the existing explicit actual-zero-mass refusal and
null-comparison behavior. This standalone live readout is not interchangeable
with historical unmasked return live episodes. Final/probe data stay absent;
diagnostics never train/promote/select. Original64/default/weighted/return routes
remain unchanged. No new native objective, runner, dependency or setting is added.

`train_acquisition_pilot` owns the schedule/CLI composition and original stream
authority. `train_policy_diagnostics` accepts only explicit endpoints64(default)
or128, keeping six fixed slots and original budget/state/receipt/clock guards.
New behavior tests are `test_fixed_a_exposure_pilot.py` and
`test_train_policy_diagnostic_endpoints.py`; existing controls remain qualified
through focused integration tests. Exact executed checks/costs/observations:
[fixed exposure stage](../artifacts/runs/r21-fixed-a-exposure-20261016/RESULTS.md).

Observed scoped evidence:329 focused cases and one bounded native CLI pass.
At128, A trace CE.3364245/accuracy.9375/complete-sequence accuracy.5; blue succeeds
live, red opens the wrong door, success.5/meanreturn0. A remains partially acquired.
At equal128A exposure the accepted return endpoint has CE.1325612/accuracy1/
sequence1, with64extraBupdates. All nine comparisons/post-B128 are retained;
this does not isolate B history from extra total work or show recovery savings.
Both priors refuse with undefined comparisons. Native128SGD/1024rows/1207unrolls/
57predictions/8env/61steps/8priorcallbacks cost12.4170836parentseconds within caps.

Next extend the existing fully fitted A192 return prefix with C_train64/A64,
using fixed TRAIN-only A/C raw and matching legal readouts192/256/320 after all
training. Verify initial A192 under that law; otherwise retain an inconclusive
run, with no schedule extension/gating/tuning. C_train is new to that prefix,
not unseen generalization. An A192 equal-total-native control remains deferred
for causal B-history attribution; descriptive retention/relearning does not
require claiming such an effect.
