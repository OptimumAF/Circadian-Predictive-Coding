# Clean and corrupted feedback during native live service

[R7.7-CAT.RN / issue255](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/255)
completed its sole frozen two-arm native invocation. Source/procedure/result
**self-review**, not independent acceptance. Full repository goal remains active.

The result is counterintuitive: at identical realized five-update/60-row exposure,
the clean candidate still fails legacy, while the retained-corruption candidate
learns it and retains both old tasks. Both original actors continue serving with
the same failures. This is a descriptive known-task observation, not evidence
that wrong supervision is useful or generally safe. Do not tune, rerun or select
another seed to make the result less surprising.

## Fixed protocol and observed behavior

One existing `run_case` per arm, in fixed clean→retained_corruption order. Original
245chooser clean→wait/mistakes→wake1, source96updates/576rows, BPseed17/D7/H3/K6/
lr.03/momentum.9 and full optimizer state unchanged. Each restores its own
qualified source for its original live owner; both new prefixes are fully paid.
No baseline/controller refit, new runner, production edit or old-budget renewal.

Both use the existing `live_new_retained_noise` recipe: eight mixed twelve-row
new-context packets, delay2, `LiveServingPlan(9,(2,6))`. Only the retained-noise
flag differs: cleanFalse/noisyTrue, changing row7 target inspect→defer. Features,
IDs, ordering, timings and all other targets remain fixed by the existing builder.
The changed label is explicitly experimental noise, not verified teacher truth.

| Observation | Clean | Retained corruption |
| --- | --- | --- |
| Private new-task count | 1/2→1/2 | 1/2→2/2 |
| Private old-task count | 2/2→2/2 | 2/2→2/2 |
| Private invoice/parcel/password | All succeed | All succeed |
| Private legacy | Fails | Succeeds |
| Actual updates / training rows | 5 / 60 | 5 / 60 |
| Full candidate history | 101 / 636 | 101 / 636 |
| Arrived / unarrived labels | 7 / 1 | 7 / 1 |
| Arrived-unused / total pending | 2 / 3 | 2 / 3 |
| Original live episodes complete / successful | 9 / 6 | 9 / 6 |
| Original live requests | 29 | 29 |

Five FIFO receipts per arm use original qualified-96 sources and c0 learner:
batch0..4 labels arrive2..6, updates apply3/4/5/7/8. Both reserved2/6ticks defer
for serving. Last label due9 is not released into the tick0..8window. No forced
drain, shortened delay, extended horizon or new decision/budget allowance.
Original eight-update/120s512MiB wake budget, separate24row replay allowance,
gate admissions, clocks and source/frozen policy/recipe/caller RNG remain bound.

All private task readouts are complete. Neither candidate is published or used
to generate controller feedback. The original generation0/qualified-96/native96
actor serves every request. Invoice0/4/8, password3/7 and parcel2 succeed; invoice4
recovers from the planned stale revision, legacy1/5fails, revoked parcel6refuses
commit. Both have zero observed applied permission/revision violations. These
failures remain; there is no demonstrated deployed learning benefit in this pair.

Private endpoints and live outcomes never supply TRAIN labels, promotion,
tuning, replay, probe selection or controller fit. No final/sealed/probe data,
selected replay, guard evaluation, promotion, PC/CPC, physics or new fit.

## Actual costs

Native body12.018982520s, sampled peak processRSS340643840B; parent14.9041293s,
EXIT0/no timeout, stdout1855B/stderr0. Sampler started before Torch/input admission;
main and serving workers share synchronized counters. Original per-owner120s/
512MiB and cumulative180s/512MiB/parent195s limits hold. Sampling is not an
allocator hard bound or process-tree measurement.

| Instrumented work | Per arm | Whole invocation |
| --- | ---: | ---: |
| Fits / rows / backward / SGD | 5 / 60 / 5 / 5 | 10 / 120 / 10 / 10 |
| Model components / initializer slots / draws | 107 / 4173 / 214 | 214 / 8346 / 428 |
| Full snapshots / logical elements | 636 / 86688 | 1272 / 173376 |
| Native forwards / rows | 72 / 215 | 144 / 430 |
| Typed requests / sandbox dispatches | 59 / 119 | 118 / 238 |
| TRAIN-bank certification dispatches | 36 | 72 |
| Case owners / selections | 1 / 9 | 2 / 18 |
| Tensor clones / logical elements | 4737 / 61888 | 9474 / 123776 |
| Tensor deepcopies / logical elements | 9152 / 109824 | 18304 / 219648 |
| Source+fit reads / logical bytes | 99 / 42793641 | 202 / 92489518 |
| Code-pin reads / bytes | 0 / 0 | 66 / 1446504 |

Per-arm forward count reconciles5training+8monitors+59typed; rows60+96+59=215.
Typed59 includes47serving/phase requests and12private candidate decisions.
Live29requests are a subset. Both full original TRAIN banks were prepared even
with selected replay0. Startup/preservation adds one source and three fit reads
outside arm deltas:4reads6902236B. Frozen-fit validation performs six historical
utility consistency checks, newutility observations0. No hidden refit or source
optimization. All declared permanent counters and storage/time limits pass.

The arms have equal observed optimization exposure and recorded logical work
counters. This is not a matched-total-computation/physical-IO/timing study:
fixed order, shared process/page caches and instrumentation remain. Live predictor
p50/p95/max: clean15.372049/17.873943/19.822864ms;
noisy15.367387/18.122815/18.473888ms. Episode elapsed additionally includes queue,
reservation and workflow. No latency SLO or efficiency ranking.

Snapshot/build/clone/deepcopy counters overlap; logical elements are not bytes or
exact FLOPs. Prior245discarded48fits240rows/48replayrows/228prediction calls468rows/
216prepared bank rows remain separate from whole source study56fits288rows/
304predictions624rows/288prepared bank rows and original96fits576rows/source costs.
These groups overlap and are not summed. Exact fit overhead is inseparable;
all-in utility remains unavailable. No old baseline/controller is retrained.

## Commands and validation

Local ignored evidence: `artifacts/runs/catalog-live-noise-native-20261027/`.
Protocol/current pins/observer and `review-go.json` froze before models. Existing
offline Ubuntu NumPy1.26.4/Torch2.14.0+cpu, numerical threads1. Exact native command:

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout --kill-after=2s 190s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-live-noise-native-20261027/run.py
```

ParentPID120540 observed the same process to terminal/nativePID363. No restart,
retry, remaining-arm continuation or protocol/budget amendment. The allowance
is now SPENT. Original exclusive attempt/result/ledger/parent/raw logs remain.
Case completion and whole-protocol qualification both passed. Result1004894B,
SHA256 `7b35cc35a3c24f95776152e6912951b6df2cf7c9ffd0bf38591ff015b109234c`.

Source/fit hashes were verified inside the native invocation before models and
preserved by original guards:

```text
source ee40be1162f38d22730b3050726687e063c5a54cde6238ade158f9fc1efc6b27
fit    9db55756ddc3a81a4aecf40f3e3fee20c9c6313dcbac1b3d5f1fcd2322212c79
```

Software observer commands (no project/NumPy/Torch/scientific payload imports):

```text
./.venv/Scripts/python.exe -B artifacts/runs/catalog-live-noise-native-20261027/check.py
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-live-noise-native-20261027/check.py
```

14stdlib checks/platform PASS,0failures/errors, Windows0.108353500s/Linux0.278974963s.
Permanent costs, lost returns, dimensions, worker/shared counters, portable
pins, pair recipe difference, unknown failure, negative tasks, unequal exposure
and forbidden promotion are covered. Adding `--pins` passed once/platform:
33code files723252B, Windows0.053277600s/Linux0.144153462s. Linux distribution
metadata matched without Torch import. One preparation pass read the same33code
files separately. Scientific-source preflight reads0. Reuse254's242software
cases per platform unchanged; no model-fixture rerun.

Scoped statics, `<files>` = stage `run.py check.py`:

```text
./.venv/Scripts/python.exe -m ruff format --check <files>
./.venv/Scripts/python.exe -m ruff check <files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <files>
```

Final all PASS in `static-corrected.json`. Earlier `static.json` and
`static-final.json` retain test-only mixed-dictionary inference diagnostics at
check.py128 then130. Explicit `dict[str,Any]` on both literal fixture records
fixes these without ignores/runtime changes; passing behavioral tests reused.
No models preceded the fixes/pins/GO. Measured static total27.0585344s; initial
format call was not separately timed/not reported as0.

One saved-only stdlib actual read produced compact `readback.json` (30271B) and
35PASS checks of parent/protocol status, caps/runtime/pins, original source/
actor/RNG, causal receipts/budgets/native counts, pending arrivals, complete
service/private readouts and version/permission/revision evidence. Subsequent
interpretation used the projection, no actual/source reread, model rebuild,
prediction or rescore. Original result remains available; assertions alone do
not establish the scientific interpretation. Self-review inspected source,
procedure and actual negative/positive outcomes.

Mtime-selected cache113279220B (footprint/not growth, inherited separate) below
160MiB; stage1093247B/metadata86498B before result-review/closeout below8MiB/1MiB.
No native/test failure, retry or cap change. One guessed old-report filename
was absent; actual source imports resolved orientation, with no model work.
Skipped full suite/clean clone/wheel/install/browser/independent review, unchanged
baseline/math/simulator cohorts, comparative/PC-CPC/recovery/long-duration and
release checks. No production/dependency changes, installs/downloads/GPU/CI/
agents/Git mutations/reset/raw publication. Historical plans/log stay frozen
under AGENTS; current handoff lives in issues255/18.

## Interpretation and precise next action

The corrupted label happens to accompany a better new-task endpoint in this
one known grammar. It is still an incorrect target. Neither contamination
resistance nor beneficial augmentation, generalization, controller/CPC advantage
or reliable sequential recovery follows. Clean acquisition failed at the
declared partial horizon; retain that negative result. Both old tasks pass at
the endpoints, but this is not relearning after actual forgetting or context
return. Serving remains unchanged because neither candidate was published.

Earlier245negative utilities,248OPEN FAILED-SPENT comparison unknowns,251private
success,253clean publication and59/60fixed-dose observations remain in their
original frozen scopes/spent allowances. This pair is distinct actual LIVE
feedback robustness at five updates, not a renamed retry of those studies.

Before changing the learner, dose or data validation, resolve the surprising
decision difference using already-paid scores. Existing
`scripts/diagnose_ticket_training.py` handles original four-checkpoint or
policy-study reports and reconstructs NumPy references; it does not admit this
two-arm live report. Existing inspector likewise has three older report families.

Next implementation action: add a bounded saved-score comparison route to the
existing ticket diagnosis command for this report family. Read each arm's
already-paid per-step scores/selected actions, identify the first divergent
decision and expose the corresponding action margins alongside source task
outcomes and marked corruption. No model reconstruction, NumPy forward, teacher
generation, new prediction/training, final/probe read or dose/seed/metric change.
Start pure fixture checks, then freeze one bounded saved-only read of255for
diagnosis in its own scope; this allowance is not permission to rerun models.
This diagnoses a concrete learning-to-action uncertainty, not another capture,
publication, source-loader or audit increment. Full R2/R4/R7/long-duration
deployment/recovery/comparative science/release and repository goal remain open.
