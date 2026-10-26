# Bounded categorical PC proposals

`src/core/categorical_pc_inference.py` proposes a detached TRAIN update. It
accepts the same CPU float32/64 tensors, explicit class weights, geometry and
positive rates as `class_weighted_pc_update`, plus `CoupledInferencePolicy`.
This is a core proposal API; it does not publish an update to a learner.

```python
from src.core.categorical_pc_inference import bounded_class_weighted_pc_update
from src.core.coupled_inference import CoupledInferencePolicy

result = bounded_class_weighted_pc_update(
    features, targets, parameters,  # four CPU tensors: Wfh, bh, Who, bo
    class_weights=(1.0, 1.0, 1.0),
    learning_rate=0.03,
    inference_steps=64,             # maximum, in 1..64
    inference_learning_rate=0.04,
    role="train",
    policy=CoupledInferencePolicy(
        min_steps=1,
        gradient_tolerance=1e-6,
        relative_energy_tolerance=None,
    ),
)
# Owned candidate tensors; caller must provide an atomic publication boundary.
proposal, diagnostics = result.update, result.diagnostics
paid_work = diagnostics.work
```

## Objective and stopping

The objective is the existing target-weighted mean of categorical cross entropy
plus half the hidden residual squared sum. Latent steps follow `B*dE/dz` to
preserve native PC finite-step conventions; parameter gradients are the partial
`dE/dtheta` holding the accepted latent state fixed. The explicitly named final
scaled gradient infinity norm measures `B*dE/dz`.

A candidate must have finite state, finite nonincreasing energy, finite direction
and finite norm before acceptance. Unsafe candidates return the preceding fully
accepted state without retries or step-size adjustment. Safety can stop before
the declared minimum. Convergence checks apply after accepted minimum steps;
gradient tolerance takes precedence over relative energy tolerance. Relative
energy drop divides by `max(1, abs(previous_energy))`. Maximum bounds attempts.
With both tolerances `None`, safe paths preserve fixed-loop outputs exactly.

## Costs and failures

Diagnostics are frozen scalars. `.work` returns a fresh dictionary of integer
and Boolean counters compatible with `TrainingUpdateFailed`. Accepted nominal
latent rows in the numerical proposal do not include rejected work; diagnostics
separately retain attempted/accepted/rejected steps and attempted/accepted rows.
Even zero accepted steps still pays final partial gradients and one parameter
proposal at the prior. Initial prior, inference energy, direction, attempted
norm reads, energy comparisons, final parameter gradients, parameter proposals,
final energy and legacy diagnostic work remain separate.

Inference energy/direction completions mean a finite helper returned. Norm
counts are attempts. Finalization completions mean arithmetic returned before
the shared final finite check; an overflowed final candidate may have completed
all four final arithmetic stages and still fail. Numeric initial/final failures
raise `TrainingUpdateFailed` with observed work; unexpected exceptions retain
observed work and set `failure_work_may_be_unobserved=True`. Admission failures
raise `ValueError` before numerical proposals. Donor tensors, gradients and RNG
states remain unchanged; returned tensors own their storage.

## Verification and extension

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_categorical_pc_inference.py tests/test_categorical_pc_weighted.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/categorical_pc_inference.py src/core/categorical_pc_weighted.py tests/test_categorical_pc_inference.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/categorical_pc_inference.py src/core/categorical_pc_weighted.py tests/test_categorical_pc_inference.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/categorical_pc_inference.py src/core/categorical_pc_weighted.py tests/test_categorical_pc_inference.py
```

Repeat type checking with `--platform linux`. The recorded supervised commands,
87 passing scoped cases, work counts and retained initial type-check failure are
in [RESULTS](../artifacts/runs/categorical-bounded-inference-20261018/RESULTS.md).

R1.5-T ordinary core/atomic/scheduled support is accepted below. Broader managed
capture/prune/probe consumers and scientific benefit remain separately unqualified.
Existing S2/S64/default controls,
active CPC, BP and the binary coupled bounded solver retain their behavior.
No new learning, acquisition, retention, recovery or circadian benefit follows
from these mathematical and regression checks. No final evaluation or probe
selection is available through this core API; callers must retain role isolation.

## Atomic TRAIN episode API

```python
from src.adapters.torch_pc_inference import train_bounded_weighted_episode
from src.core.coupled_inference import CoupledInferencePolicy

result = train_bounded_weighted_episode(
    learner, train_episode, class_weights,
    policy=CoupledInferencePolicy(min_steps=1, gradient_tolerance=1e-6),
)
print(result.diagnostic.value, result.inference.reason, result.work)
```

`learner` is an exact InvariantTorchFeatureLearner with bound raw296/K7 PC or
neutral_cpc policy; `train_episode` is one complete released TRAIN episode of
1..16rows and explicit class weights have7positive finite entries. Native
inference_steps remains the maximum (1..64). The caller retains release/budget
authority and must keep reserved probe/final roles unavailable. The result owns
no model tensors; snapshots/forks use existing complete native interfaces.

The native update is staged on independent state and atomically published only
after the original donor/source/gradient/RNG/mode checks and an exact copied
stopping-policy check pass. Foreign changes cause refusal without rollback.
A safety return with zeroaccepted latent steps still pays the final parameter
proposal and native transition at the prior. Full native history/traffic/age and
neutral chemistry are included; a bare proposal.parameters assignment is invalid.

`.work` contains24kernel counters plus8adapter counters (32total):
feature_projection_attempts, native_state_restores_attempted,
staging_context_evaluations_attempted, native_transitions_attempted/completed,
candidate_validations_attempted/completed and published_updates. Projection
includes encoding/tensor conversion; staging context recomputes prior/logits/
error/residual for the native transition. Attempt-only names do not claim success;
completed means the corresponding call returned. Snapshot/admission/copy/closing
checks are outside these numerical counters. Unexpected adapter failures retain
known work and mark unknown, including failures after a staged physical update.
Only the trusted core error can contribute core counters; unrelated exception
receipts cannot overwrite them. The original exception remains the cause.

Admission/inference-mode failures precede native work and raise ValueError;
subsequent failures use TrainingUpdateFailed. Successful scalar results are
frozen and .work returns a fresh dictionary. Existing fixed episode/catalogue/
stream routes, snapshots and defaults are unchanged.

Run the40new checks with:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_pc_inference.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/torch_pc_inference.py src/adapters/torch_pc_weighted.py tests/test_torch_pc_inference.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/torch_pc_inference.py src/adapters/torch_pc_weighted.py tests/test_torch_pc_inference.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/torch_pc_inference.py src/adapters/torch_pc_weighted.py tests/test_torch_pc_inference.py
```

Repeat type check with --platform linux. [96executed scoped cases and exact
supervised commands](../artifacts/runs/categorical-bounded-adapter-20261018/RESULTS.md)
include56fixed-path regressions. They establish adapter correctness, not learned
adaptation. The two existing delayed3update stream cases are fixed regressions;
new bounded ordinary stream support is now qualified separately below. The
older96case evidence remains scoped to the episode API and its fixed regressions.

## Scheduled TRAIN stream

```python
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.core.coupled_inference import CoupledInferencePolicy

learner = TorchTrainWeightedLearner(
    invariant_pc_or_neutral_policy,
    released_train_episodes,
    inference_policy=CoupledInferencePolicy(min_steps=1, gradient_tolerance=1e-6),
)
# Use the ordinary AttributionStream/run_attribution_stream with original budgets.
```

The new keyword defaults to None, retaining exact base diagnostics and original
fixed PC/neutral/BP/recurrent behavior. Bounded configuration admits only bound
PC/neutral complete TRAIN episode schedules (not BP/recurrent/catalogues). It is
copied before native construction. The inference_policy property returns a copy;
policy owners and primitive scalar/type/float-bit birth values are revalidated
before operations. Fork/deepcopy own independent configuration. Existing native
snapshot/restore formats remain unchanged: the wrapper retains declared stopping
configuration; a native snapshot cannot authenticate historical stopping choices.

Returned CategoricalPCTrainingDiagnostic is an exact TrainingDiagnostic subtype.
Existing stream asdict retains definition/value plus inference_policy, inference,
native_work and stream_work. Work fields serialize as immutable key/value pairs;
.work and .scheduling_work return fresh dictionaries. All data is scalar, with
no borrowed tensors. The native32 scope is the atomic bounded episode's private
candidate update. The outer7 scope counts candidate fork attempts/returns,
publication restore attempts/returns, publication validation attempts/returns,
and actual scheduled publication. A successful one-call receipt has all outer7
and native8adapter counters1; core work retains actual attempted/accepted/rejected
latent counts. Every partial failure retains actual counts rather than forecasts.

ScheduledCategoricalUpdateFailed extends the existing failure port, preserving
native .work (up to32counters) and outer .scheduling_work separately. Native
published_updates1 can accompany outerpublished0 when a completed private
candidate was discarded. Such work is not free and does not count as a committed
stream update. Unexpected failures preserve known work and flag unknown; the
actual inbox stops and does not retry the label. Foreign changes are not rolled
back. Administrative snapshots/admission/copies/closing checks stay outside these
numerical counters and can contribute additional or unknown cost.

The ordinary cursor accepts only exact base or exact complete validated bounded
diagnostics. Ordinary AttributionStream supports pause→refused paused run_until
(no extra native work/charge)→resume→successful next updates with original rich
receipts, actor and budgets. It has no deferred-poll API; previous managed runtime
deferred-poll evidence is unchanged. No managed capture/prune/probe support is
inferred from ordinary cursor acceptance.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_bounded_pc_stream_adapter.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/categorical_training_diagnostic.py src/core/inbox_cursor.py src/adapters/torch_train_weighted_learner.py tests/test_bounded_pc_stream_adapter.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/categorical_training_diagnostic.py src/core/inbox_cursor.py src/adapters/torch_train_weighted_learner.py tests/test_bounded_pc_stream_adapter.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/categorical_training_diagnostic.py src/core/inbox_cursor.py src/adapters/torch_train_weighted_learner.py tests/test_bounded_pc_stream_adapter.py
```

Repeat typing with --platform linux. [Final303case execution, all failed checks
and exact commands](../artifacts/runs/categorical-bounded-stream-20261018/RESULTS.md)
include48new cases and255affected fixed/cursor regressions. No new comparative
experiment or circadian benefit is established. Next compose explicit reserved
probe_train episodes on an owned disposable bounded learner through the existing
fresh-task probe, preserving native+outer failure costs. Do not relabel reserved
roles or update the benchmark/serving actor. Freeze/review any comparison first.

## Isolated reserved-task probe

```python
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.core.coupled_inference import CoupledInferencePolicy

probe_source = TorchTrainWeightedLearner.from_probe_snapshot(
    warm_snapshot,
    reserved_episodes,  # 1..16 complete original probe_train records
    inference_policy=CoupledInferencePolicy(min_steps=1, gradient_tolerance=1e-6),
    return_episodes=return_update_sources,  # empty or exactly16 original TRAIN records
)
# Pass probe_source and warm_snapshot to existing run_fresh_task_probe.
# Its training batches match reserved_episodes in order; it trains only its fork.
# Optional runner return_training contains TWO templates, alternated for16 calls.
# Bind those sixteen calls explicitly in return_update_sources, in the same order.
```

The factory admits PC/neutral only and retains full warm native state and absolute
counters. Schedule progress subtracts the owned warm base. Fork/deepcopy/restore
retain that base and stopping policy. Native snapshot formats do not change;
snapshots alone cannot authenticate historical role/configuration choices.

Original source roles are preserved. Separate train_bounded_probe_episode and
probe_episode_class_weights admit probe_train exclusively; public TRAIN APIs
remain strict. Labels for weights come only from the current complete episode.
Wrong/future arrays fail before new native update work; probe_test/final_test
records are never admitted for training. The existing runner owns probe-only
evaluation, immutable readouts, checkpoints, separate fresh/return budgets and
source isolation. No result selects training, promotion, tuning or final data.

Successful exact rich diagnostics serialize stopping policy/inference/native32/
outer7work; scheduled failures preserve partial native work and outer costs only
when received at train_batch. External callbacks cannot claim those update costs.
Legacy base diagnostics serialize unchanged. Native publication on a discarded
private candidate remains paid even if scheduled publication fails.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_bounded_probe_schedule.py tests/test_probe_rich_diagnostics.py
```

[628final scoped checks, failures, commands and resource accounting](../artifacts/runs/categorical-bounded-probe-20261018/RESULTS.md).
The actual native fixtures establish execution/isolation, not learning benefit.
The permissive early-stop test tolerance is an engineering fixture, not a selected
scientific operating point. Managed capture/prune consumers and comparative
acquisition/retention/recovery/release gates remain separate and open.

## MiniGrid F16 and A16 probe recipes

The existing MiniGrid pilot and CLI accept `probe_inference=None` (the unchanged
original recipe), `fixed-weighted`, or `bounded-weighted`. Explicit recipes require
joint public invariant history, an explicit PC/neutral method (S2 or named S64),
distinct F, old-task readouts and A return. Existing weighted-catalogue selection
is retained. BP/active and extra live/gradient diagnostic combinations are refused.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-minigrid-history-learnability --invariant-history --qualification-method pc_relax64 --joint-catalogue --weighted-catalogue --distinct-probe --old-task-readouts --return-to-a --probe-inference bounded-weighted
```

Use `fixed-weighted` for the same weighted objective and cap with minimum equal
to maximum and tolerances disabled. Both share last-accepted-state safety fallback,
so fixed does not guarantee all configured steps are accepted. Bounded uses minimum1,
existing default scaled-gradient tolerance1e-6, relative toleranceNone. These are
explicit recipes; tolerance selection is not automated. Benchmark policy/objective/
sources/dose, all raw metrics and original control recipes stay unchanged.

Original `_run_probe` creates a full-state reserved source inside its existing
source/checkpoint/resource guards. F16 and TRAIN A16 update only the app's fork.
Original warm clocks, independent budgets, preceding-success checks and complete
native32/outer7 receipts remain. `work.probe_inference` reduces known update
receipts separately; partial unknown work is retained. Copies/construction/scalar
reductions and unreturned operations contribute wall/RSS and may add unknown cost.

Module boundaries: `minigrid_bounded_probe.py` owns recipe admission, source binding
and observed scalar-cost reduction; `minigrid_learnability.py` retains orchestration;
`fresh_task_probe.py` retains isolated F/readout/return execution. No new runtime.
To add another supported recipe, qualify its native objective and role boundary
first, then add an explicit descriptor and matched scoped tests.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_bounded_probe.py
```

Exact scoped statics, regression commands and the separately frozen paired local
study are in [execution evidence](../artifacts/runs/minigrid-bounded-probe-route-20261018/RESULTS.md).
Unqualified scientific/release requirements remain open; implementation correctness
alone does not establish new-task acquisition, mature retention or circadian benefit.
