# Continuous TRAIN replay

## Fixed comparison example (accepted fixed pilot)

`scripts/continuous_replay_example.py` reuses the accepted saved no-replay cohort
and runs only four new replay arms of the same synthetic A/B/A/fresh-task recipe.
All six benchmark metrics and both disposable probe views at all four wake
checkpoints retain raw baseline/replay values and signed differences. Acquisition
and recovery gains are larger-better; retention loss/deficit is smaller-better.
Added replay is explicitly extra training/compute, not a matched-compute trial.

After the separate frozen protocol, source review and correctness gates pass:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.venv/Scripts/python.exe -X utf8 -B -m scripts.continuous_replay_example
```

This fixed local comparison requires the original Windows Python3.14.7/NumPy2.4.6
producer environment and the saved artifact already present in this checkout.
It does not reconstruct old parameter matrices or train the baseline again.
Do not rerun a spent invocation without a separately justified prospective scope.
The single prospectively budgeted comparison completed and was independently
accepted: 56 new updates/116 training-row exposures, with saved controls reused.
The 45 metadata cases passed in the same order on Windows/Linux; scoped lint,
format and types passed. No production learner/runtime mathematics changed.

Replay improved warm A acquisition/retention but reduced B acquisition, reduced
incremental return recovery and worsened B retention after A returns: three
favorable and three adverse benchmark directions for every family. All still
learned B and recovered A in this fixed case. Final fresh TRAIN/test absolute
errors were higher for all four families. The inherited fresh-task one-update
gains remained negative; zero-velocity gains stayed positive but smaller at the
final checkpoint. These gain values are not losses. No overall replay or CPC
advantage, matched compute, native skill or generalization is established.

Replay adds 12 updates/24 row exposures across the four families. Probe scoring,
gradients, zeroing preparations, bank payloads and native PC/CPC work are recorded;
validation/copies/container/process overhead is additional. All 152 signed
comparisons, adverse results and original failed-control provenance are retained.
[Exact results and commands](../artifacts/runs/continuous-replay-comparison-20261020/RESULTS.md) ·
[Independent acceptance](../artifacts/runs/continuous-replay-comparison-20261020/REVIEW.md) ·
[Frozen protocol](../artifacts/runs/continuous-replay-comparison-20261020/PROTOCOL.md).


## Behavior and scope

`ContinuousReplayPlan` opts the existing A/B/A continuous learner into replay of
its already released warm A TRAIN episode. The A-only fitted standardization,
learner initialization, wake packet sequence and actor remain unchanged. Each
scheduled event trains one whole bank batch on an independent candidate fork and
publishes it through `ActorShadowRuntime.consolidate`. Replay never uses the
future B labels, probe TRAIN/test, guard or final data. No automatic selection,
promotion, deployed update, seed search or comparative benefit is implied.

Why this: the existing shared replay buffer encodes categorical/binary targets.
Continuous vectors need their own small fixed-bank boundary, while the existing
runtime can already publish a complete validated consolidation transform.

## Python API

```python
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.adapters.continuous_learnability_demo import run_continuous_learnability

# warm and new are separate complete ContinuousEpisode records with role=train.
# With two-row warm data and phase_updates=1 this permits three replay batches.
result = run_continuous_learnability(
    warm, new, phase_updates=1, learner_family="pc",
    replay_plan=ContinuousReplayPlan(every_updates=1, max_replay_examples=6),
)
if result.report["complete"]:
    candidate = result.candidate  # detached learner, not the serving actor
```

The optional final keyword is also supported by the JSON-only
`run_continuous_learnability_demo` wrapper. Omit it to preserve the existing
no-replay report and calls. No replay CLI flags or new environment settings are
introduced. The existing isolated `probe_plan` composes with replay, but each
probe trains solely on its reserved fresh-task TRAIN episode with a separate
local allowance. Its global native counter starts from the benchmark candidate;
its local budget starts at zero.

## Authority, memory and cost

Wake receipts, logical clock and `progress.updates_completed` count wake packets.
Native state and CPC wake chemistry count both wake and replay updates. Opt-in
checkpoint `wake_updates`/`native_updates` and probe `benchmark_wake_updates`
make that distinction explicit. A separate original `ToyBudgetSession` pays
replay row exposure before native training. Failure stops further replay;
charges, returned diagnostics and observed postcommit receipts remain recorded.
Unknown final native counts on incomplete work are reported as null.

The immutable bank is bounded by exact row and numeric-byte caps (defaults128
rows/1MiB, at most four complete TRAIN episodes in the generic app helper). The
runner supplies warm A only. Numeric storage counts observation, next observation,
action and reward as float64 payload. Container/metadata/validation-copy/native
array and process overhead is additional. Event records are finite, bounded by
the declared maximum384 wake packets; no unbounded model-history chain is kept.

`report.replay` distinguishes attempted, returned and valid-receipt committed updates and paid
row exposures; malformed publication observations are explicitly unknown. Its
`bank_signature_bytes` records the additional retained serialized bank size. Successful BP work adds selected replay normalization/forward/
backward/gradient/dense costs; PC work uses actual returned native event records.
Replay input payload is separate from wake inbox payload. Operation forks and a
transient model owner are counted, not presented as measured process memory or
matched total computation. Original source, actor, budget birth identities,
limits, clocks, receipts and full native/CPC state are checked; no budget renewal.

## Files and extension

```text
scripts/continuous_replay_example.py        fixed saved-control comparison
tests/test_continuous_replay_example.py     zero-model admission/failure checks
src/app/continuous_stream_rehearsal.py       fixed bank, charges, consolidation
src/adapters/continuous_learnability_demo.py existing A/B/A composition and costs
tests/test_continuous_stream_rehearsal.py    fake-port authority/failure behavior
tests/test_continuous_replay.py              four families and isolated probes
```

Extend with a new separately frozen protocol using this runner. Do not infer
benefit from correctness fixtures, import fresh/final labels into replay, or
change a seed/metric after observing results. Any future dynamic bank, selection
or real deployment needs its own substantive requirements.

## Validation status

R2.1-CONTINUOUS-REPLAY implementation accepted: Windows and pinned Linux each
passed the same76 ordered cases with zero failures/errors/skips. Scoped five-file
Ruff lint/format/mypy and independent source/actual-results review passed. The
first static pass had two unused imports and16 typing errors, all repaired before
fixtures; failed receipts remain. See [exact commands and outcomes](../artifacts/runs/continuous-replay-20261020/RESULTS.md)
and [independent acceptance](../artifacts/runs/continuous-replay-20261020/REVIEW.md).

Per platform the frozen fixtures use58 native updates129 training-row exposures,
six A-only fits12rows24featurevalues and six initializers1536 normal draws.
Benchmark conditioning/residual/actor readouts each48calls120rows; isolated probe
24score calls56rows with diagnosticsFalse (no probe direction gradients). PC
18updates39rows, at most90 energy entries/72latent proposals,39candidate-forward
rows. Additional validation, copies and scalar/fake-port checks are not presented
as matched total computation. No standalone comparison/native trial ran under that implementation scope.

Run the exact test selection and scoped static commands from the linked receipts;
exclude the old direct-reference test unless separately budgeted. That implementation-stage comparison is now complete; see the accepted pilot
above. Next extend the existing frozen TRAIN robot-learning bridge with explicit
optional replay and separate wake/native handoff checks. Its current admission
requires native updates equal wake updates, so it rejects replay candidates.
Keep default None unchanged and use a separately reviewed protocol before any
native scene; prior zero-success native runs and every negative result remain.
Full R2.1/R2.3/R4.5/useful adaptation/scientific benefit/native serving/final/UI/
release requirements remain open; prior negative and inconclusive results stand.
