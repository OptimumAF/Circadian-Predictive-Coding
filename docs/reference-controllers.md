# Causal fixed and heuristic references

The existing native BP serving demo supports four opt-in reference policies:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --reference periodic --output periodic.json
```

Choices: `no_learning`, `periodic`, `drift`, `budget_aware`. The output must be
fresh. This uses the qualified old-task source already required by the
[serving demo](native-ticket-serving.md); no warmup is retrained. Omitting
`--reference` retains its original full delayed-stream/publication behavior.

## What the chooser can observe

`core/reference_controller.py` contains immutable scalar policies, observations,
arrived TRAIN mistake aggregates and pure `choose_reference`. No learner, native
array, target, task result, final/probe data or callback reaches the chooser.
Feedback must have ordinary TRAIN role, have arrived, and match the original
serving actor version. Invalid/future/stale feedback fails even for no-learning.

The existing script reuses ActorShadowRuntime, ResourceSharedRuntime and the
existing arrival helper. Eight new-context batches, delay1, ticks0..8 give every
policy the same nine opportunities and hard eight-update envelope. Original actor
predictions are made before each batch's labels arrive. All policies pay the same
eight feedback prediction batches/48feedback row reads, including no-learning.
Only then can an aggregate mistake count enter a decision. The actor remains
unchanged until the schedule ends. Feedback is global TRAIN classification error,
not a direct measurement of closed-loop task utility or statistical drift.

| Policy | Frozen rule, after readiness/credit checks |
| --- | --- |
| no_learning | Always wait |
| periodic | Wake one if tick modulo2 is zero |
| drift | Wake one if latest arrived actor error exceeds0.1 |
| budget_aware | Reserve four credits; wake when unreserved credit covers remaining opportunities |

No parameters are fitted here. Choices enter the finite ID admission API; the
original gate, budget, clocks, roles, pending identities and receipts retain
authority. A finite observation horizon may leave batches unapplied. Reports
state that count explicitly; no forced drain occurs and AttributionStream.finish
still requires complete training on its original path.

## Reading the evidence

Controller events record observation, selected ID/reason and returned poll.
Counters/timers include common monitoring, observation construction, pure choice
and admission execution. Admission time includes native training; it is not pure
controller overhead. Total elapsed time also includes task readouts, guard/native
construction, serialization and framework work. Instrumented single-case timings
are not comparative deployment-latency estimates. Bounded pending prediction
records and nine-event reports do not create an unlimited success history.

Four isolated candidate ordinary task readouts measure new-task gain and old-task
drop. They remain separate from actual guarded actor publication/refusal. The
original inner-template guard thresholds, effect-time permissions, stale-revision
checks and complete rollback are unchanged. Neither candidate task scores nor
guard outcomes can select the dose, reserve, threshold or future action.

The frozen development case produced:

| Policy | New updates / rows | Candidate new tasks /2 | Old tasks /2 | Promotion |
| --- | --- | --- | --- | --- |
| no_learning |0 /0|1|2|refused|
| periodic |4 /24|1|2|refused|
| drift |8 /48|2|2|accepted|
| budget_aware |4 /24|1|2|refused|

The two four-update payloads were identical and did not acquire legacy. The drift
payload exactly matched the saved fixed-eight-update result. These results show
an exposure tradeoff; they do not isolate an advantage from drift detection or
establish optimized heuristics/generalization. All old tasks were retained in
this known-template case, so no recovery from forgetting was demonstrated. The
accepted actor solved all four ordinary tasks, recovered two stale cases and
refused two revoked commits. Rollback restored the original legacy failure.

All16new updates/96rows and monitoring/evaluation work remain charged; prior
96updates/576rows are shared historical source work, not four new warmups. No
learning still pays common monitoring; it is not an optimized zero-monitoring
control. Equal opportunities do not mean equal update count or compute.
PC/CPC/scientific negatives remain unchanged and no circadian benefit is claimed.

## Focused checks and extension

```console
python -m pytest -q tests/test_reference_controller.py tests/test_torch_serving.py
python -m ruff check src/core/reference_controller.py scripts/run_bp_ticket_serving.py tests/test_reference_controller.py
python -m ruff format --check src/core/reference_controller.py scripts/run_bp_ticket_serving.py tests/test_reference_controller.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/reference_controller.py scripts/run_bp_ticket_serving.py tests/test_reference_controller.py
```

Final57checks/platform and scoped statics passed. Tiny reference fixtures use two
batches and cannot establish task utility; deliberately lost returns and failed
choices retain their prefixes and charges. One fixed four-reference native cohort
ran after the gates, without numerical retries or tuning. See
`artifacts/runs/reference-task-controllers-20261023/` for protocol, exact commands,
all positive/negative traces, work counts, timings, hashes and limitations.

Why this: establish causal fixed/heuristic controls before learned adaptation.
Next declare the outer objective and separate controller development/validation/
sealed evaluation chronology before controller training. Preserve all original
learner final/probe barriers; do not select a new metric or tune these policies
against this one tiny guard/task set. Advanced action executors, broader recurring
streams, continuous rebase, learned control and scientific/release gates remain
separate unfinished requirements.
