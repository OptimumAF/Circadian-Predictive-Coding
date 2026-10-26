# Original-MSE BP head Gauss–Newton update

## Fixed old-information control (2026-10-21)

Same source/JSC reconstruction/eight115-row updates, replacing newA50 with fixed
oldA[i%15], recovers A17 in15 controls and loses B19 after50. Curated115 succeeds
on both. Every control imitation-error group improves despite native B failure.
Independent checks accept source/lineage/runtime/native/cost evidence; no general
success or CPC benefit. Row/update/context dose and solver shapes match; unique
information, stage distribution, memory and historical queries differ.
[Protocol, exact commands, costs and negative result](../artifacts/runs/matched-old-information-control-20261021/RESULTS.md).
One accepted disposable default-SGD18-row probe from corrected32/1736 improves
reserved training/test errors (+.00387405/+.00496754) while old A/B fit worsens
slightly. All20 original GN/four-family comparisons retain mixed gain/absolute
endpoint rankings; no matched-state/total-compute or new physical-task claim.
[Probe commands, costs and endpoints](../artifacts/runs/post-correction-fresh-probe-20261021/RESULTS.md).
Next expose the accepted simultaneous correction recipe through a reusable
adapter; no further checkpoint/probe/dose series.

## Behavior and boundaries

The existing `NumpyContinuousBCLearner` exposes two explicit methods:
`train_head_gauss_newton_episode(episode)` and
`train_head_gauss_newton_batch(features, targets)`. They propose one bounded
head-only step for the original normalized active-axis half-MSE. Default
`train_batch`/`train_episode` still perform SGD; PC/CPC objectives are unchanged.

```python
# learner is an existing owned BP learner; this episode is already released TRAIN.
result = learner.train_head_gauss_newton_episode(released_train_episode)
print(result.diagnostic.value, result.loss_after, result.work)
```

Only exact BP/TRAIN/common-MSE policies are admitted. The episode path refuses
probe_train/probe_test/inner_guard/final_test roles. Raw arrays require the same
caller release authority as existing train_batch; no data-role inference or
conversion is supplied. Existing frozen observation statistics are applied,
never refitted. Input/hidden widths≤64, rows≤128, actions≤16.

On acceptance, hidden weights/bias and hidden momentum remain unchanged; head
weights/bias take the proposal and their SGD velocities reset tozero. Why: the
curvature step is not an SGD velocity; adding stale head momentum would change
the declared proposal. This does not freeze hidden weights on later ordinary
SGD calls. One accepted update adds1 completed_updates and N training_examples.
Diagnostic value reports the BEFORE loss, matching existing native convention;
loss_after is separate. Caller can fork the existing learner for isolation.

Rejections and numerical failures raise existing TrainingUpdateFailed, with
owned native work counters, and publish no update. Detected native-source
mutation is retained and reported, never restored away. Source checks inspect
actual native state rather than relying on a replaceable public snapshot getter.
No concurrent-use, promotion, deployment or task-success guarantee is implied.
No new CLI, runner, dependency, environment variable or configuration is added.

## Fixed mathematical rule

Let `Z=[tanh(X W1+b1),1]`, `u=tanh(Z C)` with C the current head, N rows and K
active axes. For each active axis a:

- `r_a=(u_a-target_a_normalized)/sqrt(N*K)`;
- `J_a=diag(1-u_a**2) Z/sqrt(N*K)`;
- solve `[J_a; sqrt(1e-3) I] delta_a = [-r_a; 0]` once with fixed
  `numpy.linalg.lstsq(..., rcond=1e-12)`;
- combine ALL axis deltas, cap their global Frobenius norm to1, and evaluate
  the resulting head on the ORIGINAL unregularized normalized half-MSE;
- accept only a finite STRICT decrease. No backtracking, retry, adapted damping,
  target clipping or seed/threshold selection occurs.

Damping penalizes the proposed local step, not the whole model coefficients.
Constant axes have zero residual/Jacobian and no solve; their head parameters
stay intact. An all-constant or saturated non-improving case rejects. A no-op is
not counted as an accepted update. Returned mathematical arrays are detached.
The prior inverse-tanh diagnostic coefficients are neither read nor installed.

Why this: fixed hidden features fit the recorded TRAIN labels constructively,
but the unchanged SGD source did not attain faithful control. This is a bounded
optimizer hypothesis using the original objective. These engineering constants
are fixed prospectively, not tuned for a circadian comparison. Native usefulness,
reliability and extrapolation remain unproven; large condition/norm and coverage
problems from earlier evidence remain relevant.

## Work and extension

Each active axis has exactly one solver attempt; a returned malformed solver
output remains a completed, paid return. Completed-forward dense MAC, operand
sizes, Jacobian/residual/design payloads and accepted/publication counters are
separate. Solver arithmetic/workspace, failed partial arithmetic, elementwise
operations, validation copies, snapshots, RNG in caller initialization, Python
objects and RSS are additional. Exception `.work` returns owned copies.

```text
src/core/continuous_head_gauss_newton.py  pure linearization/proposal/work
src/adapters/numpy_continuous_bc.py       existing learner opt-in/atomic commit
tests/test_continuous_head_gauss_newton.py  independent math/state checks
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_head_gauss_newton.py
.venv/Scripts/python.exe -m ruff check src/core/continuous_head_gauss_newton.py src/adapters/numpy_continuous_bc.py tests/test_continuous_head_gauss_newton.py
.venv/Scripts/python.exe -m ruff format --check src/core/continuous_head_gauss_newton.py src/adapters/numpy_continuous_bc.py tests/test_continuous_head_gauss_newton.py
.venv/Scripts/python.exe -m mypy --platform linux --follow-imports silent src/core/continuous_head_gauss_newton.py src/adapters/numpy_continuous_bc.py tests/test_continuous_head_gauss_newton.py
```

The exact focused platform commands and prospective work limits are in
[the correctness protocol](../artifacts/runs/bp-head-gauss-newton-20261020/PROTOCOL.md).
The existing stream also supports explicit `head_gauss_newton=True`:

```python
result = run_continuous_learnability(
    warm_train, new_train, learner_family="bp", head_gauss_newton=True
)
print(result.report["native_updates"], result.report.get("work"))
```

Import `run_continuous_learnability` from
`src.adapters.continuous_learnability_demo`. Its JSON-only wrapper accepts the
same option. False preserves the previous call shape, schema and optimizer.
The selected path initially refuses PC/CPC, mixed likelihood, replay, probes,
RGB, stage-balanced and A-only modes before fitting or model construction.
Complete TRAIN episodes must have at most128 rows each and64 input dimensions.

`continuous_head_learnability.py` is a small existing-native-port facade. It
records returned work before app callbacks, retains bounded scalar success
records and one latest private model snapshot, and validates the preceding
success before further work. Source/actor/candidate/readout and final observer
guards reject changed evidence without restoring corruption away. Incomplete
runs return no candidate and preserve preceding receipts and paid native work.
GN solves and hidden/head passes are counted separately from read-only
conditioning gradients and residual/actor forwards. They are not SGD gradient
costs or complete FLOPs. Snapshot/copy/prefix-validation costs remain additional.

Focused stream commands (same scoped lint/format/type patterns as above):

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_head_stream.py
```

Scientific and native work require separate frozen small protocols.

## Accepted software scope

28 focused tests per platform passed with scoped statics and independent actual
acceptance. One accepted TRAINepisode update and explicit failure/refusal/default
paths ran; this is software correctness, not a robot study.
[Kernel evidence and costs](../artifacts/runs/bp-head-gauss-newton-20261020/RESULTS.md).
The stream integration passed23 focused checks per platform and independent
actual review, including default SGD, actual delayed charges, a failure after
paid success and late candidate corruption.
[Stream evidence and costs](../artifacts/runs/bp-head-stream-20261021/RESULTS.md).

## Fixed TRAIN study and native control

ONE fixed A15/B22/A15 H64seed17/8updates-per-phase study completed24updates/
416rows with identical initial checkpoint to saved SGD. Final A normalized
half-MSE .0019148053 vs .0602251222 improves; final B .0860333708 vs .0646035121
worsens. All6metrics and4checkpoints remain. This is an acquisition/forgetting
tradeoff, not uniform benefit or native usefulness.96solves7904augmentedrows
and diagnostic work are charged separately; total computation is not matched.
[Full results](../artifacts/runs/bp-head-train-study-20261021/RESULTS.md).

The fixed native control question is now accepted: ONE fullycharged24/416 GN
reconstruction exactly reproduced the whole accepted report excluding elapsed,
then the existing frozen soft serving completed known TRAIN17 in15controls/
75knownCONTROLphysics, sparse1/anyTrue/finalTrue/terminated. Source/candidate/
frozen/history exact,16sampledjointpositions0violations, not a safety result.
No production change or native optimization was needed. [Native commands, costs
and limits](../artifacts/runs/bp-head-native-control-20261021/RESULTS.md).

Controlled native adaptation and reliability remain open. The optional detached
phase observer is implemented and accepted at56checks/platform. One fixed native
study finds Acontrol1→0 after B, Bcontrol0→0, reused returnA1. Bdoes grasp/transport
but does not complete the task. Allscoregains/costs/negativeoutcomes remain.
[ObserverAPI](continuous-phase-readouts.md) · [Study](../artifacts/runs/gn-native-adaptation-20261021/RESULTS.md).

## Arrived corrections

`run_continuous_correction_update(..., head_gauss_newton=True)` and existing
`run_corrective_adaptation(..., head_gauss_newton=True)` now compose the same
bounded GN port with eight arrived correction updates. Default False preserves
SGD. Exact TRAIN BP commonMSE D/H<=64, N<=128, A<=16 only; binary loss
and step diagnostics are refused for this mode before forks/native work.
Reference rehearsal uses the existing ContinuousReplayPlan and TRAIN-only bank.
The native helper preserves TRAIN19 collection then frozen17/19 returns.
Existing goal-consistent labels remain separate from executed student actions.

`continuous_head_corrections.py` summarizes actual successful GN work, separate
from SGD gradients or unknown solver/copy/validation costs. The existing adapter
owns scheduling and permanent charges; no new runner or core/infra dependency.
Native successes retain at most8scalarrecords and one latest modelpin. Corrupted
prior evidence or final observers release no candidate and do not erase charges.

21 focused checks/platform, scoped statics and independent actual acceptance.
[Costs, exact commands and failures](../artifacts/runs/continuous-head-corrections-20261021/RESULTS.md).

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_head_corrections.py tests/test_continuous_head_correction_replay.py
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_head_corrections.py src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_continuous_head_corrections.py tests/test_continuous_head_correction_replay.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_head_corrections.py src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_continuous_head_corrections.py tests/test_continuous_head_correction_replay.py
.venv/Scripts/python.exe -m mypy --platform linux --follow-imports silent src/adapters/continuous_head_corrections.py src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_continuous_head_corrections.py tests/test_continuous_head_correction_replay.py
```

Extension: add a separately bounded composition only after measuring a concrete
need; retain existing runtime authority/paid evidence and freeze a small protocol
before new native/comparative work. Probes and final evaluation remain separate.

### Fixed arrived-correction evidence

Known B19 succeeds in18controls after50arrivedv2labels/eightprivateGNupdates;
A17remains0/50. Fullbasebenchmark/sourceisolation and all3primaryscoregains are
independently accepted, with original negatives retained.118controls590known
physics, source+private32GN816rows,128solves11584augrows2121984knownMAC.
This is localknown-context adaptation, not retainedA/reliability/generalization/
matchedcompute/CPCbenefit. [Full evidence and limitations](../artifacts/runs/gn-arrived-corrections-20261021/RESULTS.md).
That same-source, same-label defaultSGD control is now accepted: bothSGDreturns
fail50controls, reusedGN B19succeeds18. No newteacher/collection or baseline tuning.
The original failed1ULPguard attempt remains, and a prospective same-task repair
verifies exact method-native reductions instead of cross-reduction bitidentity.
[Comparison, costs and limitations](../artifacts/runs/same-label-sgd-corrections-20261021/RESULTS.md).
GN plus reference rehearsal is now accepted at42 focused checks/platform, scoped
statics and independent review. Pass replay_plan and reference_episodes to the same
correction API; each whole-bank event is separately charged before work. At most
16 native scalar records and one latest model pin; final callback corruption
releases no candidate and retains paid work. GN wake/replay totals reflect actual
native results, with repeated wake rows separate from reference rows.
[Full evidence and commands](../artifacts/runs/continuous-head-correction-replay-20261021/RESULTS.md).
That frozen question completed negatively: fourA15replays/60reference rows plus
eightBupdates failboth17/19 at50controls, all3differencesvsnoreplayA0/B-1.
No A recovery and lost corrected B, extra work preserved. No retry/tuning.
[Study](../artifacts/runs/gn-reference-rehearsal-20261021/RESULTS.md).
The reserved23/24 probe composition now executes through existing detached
observer/probe APIs. All four one-update defaultSGD probes improve training/test
imitation error, with mixed old-task changes; fullsource benchmark exact and
independent acceptance. This is not GN adaptation or native task success. All4
controls/costs/absolute errors retained. [Evidence](../artifacts/runs/gn-checkpoint-probes-20261021/RESULTS.md).
Next check original A15/arrivedB50 TRAIN target compatibility without more model
work before selecting a joint-fit/capacity repair.

Affected files:

```text
src/adapters/continuous_head_corrections.py   actual work summary
src/adapters/continuous_correction_demo.py   existing runtime opt-in
scripts/maniskill_continuous_smoke.py        existing native composition
tests/test_continuous_head_corrections.py   original no-replay/default checks
tests/test_continuous_head_correction_replay.py reference composition checks
```

## Fixed joint TRAIN supervision through the existing runtime

An artifact-local case composes ActorShadowRuntime + RecordedContinuousHead
on a detached B16/296 source: eight equal-row A15+B50 batches,65 rows each.
A15 qualified and B50 provisional lineage/clocks/commands remain distinct;
derived packet availability99/99 is explicit. Original budget, source/actor,
prior receipts/native records and latest model continuation are checked; no
unbounded successful-model chain, new production interface or dependency.
Four existing prediction/residual calls retain all axis/stage errors.

The executed candidate preserves B19 success19 controls but fails A17 at50,
although both recorded-label errors fall. All signed comparisons and prior
negatives remain. Independent actual validation accepted the evidence; reviewer
closeout60.210s conservative bound leaves strict60s adherence unproven. Full
study checkbox stays open; no retry/waiver. This is not matched total compute,
reliable control, unseen-task adaptation or CPC benefit.
[Protocol, run commands, costs and next coverage dependency](../artifacts/runs/joint-supervised-corrections-20261021/RESULTS.md).

## Returning-A curated correction result

The fixed existing-runtime composition consumes all115 released TRAIN rows:
original A15, arrived B50 and post-hoc curated A50. Eight new GN updates after
the exact reconstructed joint checkpoint recover A17 in17 controls while B19
succeeds in18. Source benchmark/deployed actor stay unchanged. New data AND
920 extra row exposures were added; a matched old-information control is next.
Old A imitation error worsens despite native recovery; every endpoint is retained.
No unrestricted retention/reliability/generalization/CPC claim follows.
[Commands, work, lineage and accepted evidence](../artifacts/runs/returning-a-curated-corrections-20261021/RESULTS.md).
