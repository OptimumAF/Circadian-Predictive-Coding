# Acquisition, retention and return after a fitted A prefix

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-acquired-return --describe
```

Omit `--describe` to execute the fixed local recipe. The existing delayed stream
runs A64/B64/A64/C_train64/A64,320 mean-CE updates on H8seed17/rate.03/momentum.9.
The first192 updates reproduce the accepted return recipe. Its JSON has no native
optimizer tensors, so regeneration is needed for owned continuation state.
Saved analysis requires exact original A/B source, raw score/decision projections
and first192 receipts; recorded scores do not themselves prove old tensor identity.

Why this: the A128 control still only partially acquired A. The existing A192
prefix supplies a previously fitted task from which to measure new C_train
learning, A loss, A return and C loss. C_train is a declared training context new
to that learner prefix, not unseen generalization or a context selected from
historical final success. An A192 equal-total-native control remains deferred
for causal B-history attribution; this recipe makes descriptive claims only.

## Data, readout and fixed interpretation

Six original TRAIN traces48records produce320 unique episode IDs/2560exposures.
All six are scored at21 fixed points (the original13 return points plus
200/208/224/256/264/272/288/320):126raw forwards/1008rows. The pooled train prior
uses all declared training labels, including future C labels; it is an offline
reference, not a causal online control. Neither metrics nor future labels affect
the prescribed SGD schedule. Final/probe data, replay and promotion are absent.

Only after all320 commits and `finish()`, owned checkpoints192/256/320 each play
A± and C_train± with the SAME legal greedy law and12-step maximum. All twelve
episodes remain, including failures. There are no extra prior rollouts. Raw
complete A192 teacher fit and actual A192 successes on both cues jointly establish
the initial condition for interpreting retention. If either fails, all phases
and costs remain and the interpretation is inconclusive. No gating, extension,
retry, seed selection or criterion-driven dose change is allowed.

For raw CE use direction−1; for accuracy, sequence, live success and return use+1:

| Metric | Fixed signed difference |
|---|---|
| New C gain | d × (C256 − C192) |
| Old A loss | d × (A192 − A256) |
| A return gain | d × (A320 − A256) |
| A return gap | d × (A192 − A320) |
| C loss after return | d × (C256 − C320) |

Positive gain is better; positive loss/gap is worse. These are training-domain
observations, not recovery savings, causal transfer, generalization or CPC benefit.
Historical unmasked live outcomes are not substituted for this matching-law run.

## Source boundaries and verification

```text
src/adapters/train_acquisition_pilot.py    existing stream/receipt/budget orchestration
src/adapters/acquired_task_return.py      twelve fixed owned live slots and metrics
src/core/context_return_metrics.py       pure strict three-endpoint arithmetic
scripts/run_coupled_attribution.py        explicit CLI dispatch
tests/test_acquired_task_return.py        core signs, admission, state/evidence faults
tests/test_acquired_return_pilot.py       independent real-stream caller behavior
tests/test_acquired_return_publication.py completed-result final-hook refusal
```

The helper never trains, selects, accesses final data or does IO. Original source,
selected states, receipts, budget/clock/charges and preceding episode records are
checked before more work. Storage is bounded by twelve slots and one caller seal
of the completed report. Derived acquisition is published after the final helper
guard; caller validation/readback also refuses later replacement or corruption.
Native snapshot format/math and existing default/weighted/control routes remain.
No dependency, environment variable or setting is added.

Exact executed checks, failures, resource observations and scientific outcomes:
[stage results](../artifacts/runs/r21-acquired-task-return-20261016/RESULTS.md) and
[independent review](../artifacts/runs/r21-acquired-task-return-20261016/REVIEW.md).
Extend using explicit fixed protocols through this runner, with isolated diagnostic
forks and prospectively reviewed small native budgets; preserve all negative
results and final isolation. Full useful adaptation and release remain open.

## Observed fixed case, 2026-10-16

The native case met the initial condition: both A192 cues succeeded. After C64,
both C cues succeeded and both A cues failed; after A64 returned, both A cues
succeeded and both C cues failed. Raw complete sequences agree with that pattern.
This demonstrates TRAIN acquisition/relearning with failed retention; it does
not establish recovery savings, generalization or CPC benefit.320updates/
2560training rows plus1008diagnostic rows and120live steps cost39.4725parent
seconds in the scoped instrumented process. See stage results for all counters,
RSS observations, negative outcomes and verification limits.

## Existing replay controls

The same stream also accepts explicit `torch-key-door-acquired-recent48-replay`
and `torch-key-door-acquired-coverage-replay` scenarios. Use `--describe` to see
one arm without running native work. Existing recent-only and declared-context
coverage policies each retain at most48transitions/12096encoded-array bytes and
replay one complete8-step released episode after every2wake updates. The fixed
320wake/2560row recipe therefore costs160replay/1280rows,480native updates total.
No new replay algorithm or orchestration layer is introduced.

Raw checkpoints keep their wake labels. Native counts are wake + floor(wake/2);
owned192/256/320 live readouts restore288/384/480native states and report both
clocks. All21raw points and12legal TRAIN episodes remain. Original wake/replay
budgets, clocks, receipts, charges, source isolation and preceding-result guards
apply. Publication and final readback bind the declared control and acquisition
claims; no default report schema or no-replay behavior changes.

Replay starts at wake2, so each arm must establish its own initial A192 complete
teacher fit AND both live successes. C256 must independently meet those same
criteria before a C320 loss is interpreted as forgetting. Missing premises leave
all fixed phases/results intact and make that retention interpretation
inconclusive. The readouts never gate training or choose a dose, seed or policy.

For a prospective comparison, both arms share architecture, optimizer, teacher
source, wake/replay counts, payload ceilings and raw/live readout. Coverage uses
stronger declared context/cue metadata, and recent-only selection can depend on
cue parity; these are disclosed differences. The existing no-replay320 result
is an unequal-compute reference, not a matched control. Pooled A/C live success
and mean return at320 use coverage-minus-recent; retain all endpoints/cues,
raw curves and signed gain/loss metrics regardless of sign. No generalization,
causal mechanism, recovery savings or CPC benefit follows from correctness.

New behavioral modules are `tests/test_acquired_replay_diagnostics.py` and
`tests/test_acquired_replay_pilot.py`; production changes extend the existing
acquired helper, replay accounting, caller and CLI. Frozen local comparison
qualification and observed results are recorded in
[replay stage](../artifacts/runs/r21-acquired-replay-20261016/RESULTS.md).

Observed paired case: both arms acquired A and C, then lost the other task at each
switch. Coverage improved some teacher-forced fit (C320accuracy.8125 vs.375), but
both C320live cues failed. Primary pooled320live success/return differences are0;
no live retention gain or equivalence established. Each480updates/3840training
rows cost57.54seconds recent-only and55.15seconds coverage in the scoped parent
process; all additional diagnostics and retained payloads are reported in the
stage results. No baseline, seed, metric or dose was tuned after these outcomes.
