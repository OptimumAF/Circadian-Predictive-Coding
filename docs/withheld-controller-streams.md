# Frozen controllers under recurring and noisy TRAIN receipts

The existing native serving CLI can evaluate its frozen controllers on two
declared receipt shifts. No controller is fitted by this command:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --controller-comparison --fitted-report artifacts/runs/learned-contextual-controller-20261023/actual.json --output comparison.json
```

Use the existing native dependencies and exact qualified96-update/576-row BP
source described in [native serving](native-ticket-serving.md). The completed
[contextual fit](contextual-controller.md) supplies immutable policy parameters.
Both inputs must retain their original provenance. Output must be fresh.
`--fitted-report` is required only with `--controller-comparison`; other modes
are exclusive. New report cap8MiB; original/default/reference1MiB, outer2MiB and
learned4MiB caps remain intact. The HTML inspector does not accept this schema.

## What changes between streams

`scripts/ticket_controller_streams.py` builds ordinary TRAIN packets from the
existing six-row new/old teacher packets. It copies source buffers and assigns
unique recipe/batch/row IDs. Recipes admit at most8batches and delay1or2.

| Recipe | Eight contexts | Received labels | Delay |
| --- | --- | --- | --- |
| diluted_return |old,new,old,old,new,new,old,new|Selected6teacher rows plus6clean old rows|1|
| noisy_return |old throughout|Two copies of6old rows; ready invoice row1 changes inspect->defer|2|

Every packet has12rows. The corruption changes exactly1of12received TRAIN
targets, explicitly logged as experimental noise. It changes neither the source
teacher nor clean final task truth, and is not described as a correct expert
target. Unsupported roles/shapes/recipes are refused before model work.

This produces different error fractions and recurrence/arrival patterns from the
six-row controller fit. One mistake/12examples falls below the original drift
threshold0.1 while the fitted table treats any positive mistake count as its
mistakes context. This motivates a diagnostic comparison, not a predicted win.
Grammar, features and pretraining overlap. These are withheld receipt schedules,
not novel tasks/features or independent statistical confirmation.

## Execution and isolation

Four original references—no-learning, periodic2, drift0.1, budget-aware reserve4—
and the original frozen fitted chooser receive identical packets/opportunities/
common TRAIN monitoring within each recipe. Neither thresholds nor objective
weights are changed. All start from separate original96owners; no warmup refit.

The existing schedule uses the original arrival helper and shared wait/wake
admission. Readiness counts labels that have actually arrived. It runs n+delay
opportunities, including the final arrivals; no forced drain. Ready packets keep
their FIFO order. A wake triggered by current error can therefore update an older
pending packet; reports expose actual applied sample IDs. This is a real limit of
the action space and partial observation, not a license to skip old work.

Explicit deferred recipe mode performs no workflow task readouts or promotion.
It returns the complete candidate and checks original source/actor/RNG ownership.
The comparison refuses more candidates or evaluation after failed/unknown work.
Input fit bytes and immutable protocol are checked before further work.

Only after all10native candidates finish are their records frozen. Then the
existing detached full-state readonly evaluator opens the clean task source.
Every case pays before/after readouts:24typed calls/8episodes. No later TRAIN,
promotion, policy selection, controller refitting or probe work occurs. Final
labels/results cannot enter the chooser, replay or promotion guards. The actual
live-serving path and `AttributionStream.finish` retain their original behavior.
This study evaluates private candidates, not deployments; rejected or bad
candidate behavior cannot change the original serving actor.

## Reading costs and utility

Each final record contains complete external new/old task outcomes and the
original unit-weight normalized utility. Its incremental compute proxy is native
case callback wall time plus readonly task callback wall time; memory is their
maximum sampled process RSS. Applied violation counts and all raw physical work
remain visible. Common setup, copying/freeze bookkeeping, framework residency,
serialization and outer costs are separately recorded. These instrumented times
are not comparative deployment latency; fixed order/first-use and allocator
effects can dominate small utility differences.

Historical fit16updates96rows are carried alongside the complete original
24-update/144-row fit-and-application study cost. That saved record does not
separate every fit overhead from its applications. Exact all-in utility remains
unavailable, not zero-filled. Incremental values cannot establish net/amortized
controller benefit. Different update counts mean this is not matched compute.
All policies still pay common monitoring and evaluation, including no-learning.

## Modules, checks and evidence

```text
scripts/ticket_controller_streams.py     bounded TRAIN recipe metadata/builder
scripts/ticket_controller_comparison.py frozen cohort then readonly evaluation
scripts/run_bp_ticket_serving.py        existing admission, deferred option, CLI
tests/test_ticket_controller_streams.py role, ownership and exact-corruption tests
tests/test_ticket_controller_comparison.py causal/cohort/final barriers and CLI
```

```console
python -m pytest -q tests/test_ticket_controller_streams.py tests/test_ticket_controller_comparison.py tests/test_contextual_controller.py tests/test_ticket_controller_learning.py tests/test_controller_objective.py tests/test_controller_outer_study.py tests/test_reference_controller.py tests/test_torch_serving.py
python -m ruff check scripts/ticket_controller_streams.py scripts/ticket_controller_comparison.py scripts/run_bp_ticket_serving.py tests/test_ticket_controller_streams.py tests/test_ticket_controller_comparison.py
python -m ruff format --check scripts/ticket_controller_streams.py scripts/ticket_controller_comparison.py scripts/run_bp_ticket_serving.py tests/test_ticket_controller_streams.py tests/test_ticket_controller_comparison.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/ticket_controller_streams.py scripts/ticket_controller_comparison.py scripts/run_bp_ticket_serving.py tests/test_ticket_controller_streams.py tests/test_ticket_controller_comparison.py
```

153focused checks/platform and scoped statics passed, with source/procedure/result
self-review rather than independent acceptance. Full updated native snapshots
are not bit-identical across Windows/Linux; raw work counts for the new tiny
comparison match, while first differences include float32 momentum buffers.
Original Linux producer is used for the fixed study. No tolerance or source
algorithm was changed to force cross-platform equality.

Exact protocol, commands, failures, raw results and costs remain local under
`artifacts/runs/withheld-controller-streams-20261023/`.
Live handoff: [issue54](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/54).

The sole fixed study completed28updates/336rows:

| Recipe | Policy | Updates | Final new tasks /2 | Old tasks /2 |
| --- | --- | --- | --- | --- |
| diluted_return |no-learning /periodic /drift /budget /fitted|0 /4 /0 /4 /4|1 for all|2 for all|
| noisy_return |no-learning|0|1|2|
| noisy_return |periodic|4|2|2|
| noisy_return |drift|0|1|2|
| noisy_return |budget-aware|4|2|2|
| noisy_return |fitted|8|1|2|

The noisy four-update candidates show incidental transfer from old-task TRAIN
data; they did not receive new-task labels. This is not evidence that label
corruption improves learning generally. Periodic/budget candidates have identical
full states within each recipe. Fitted diluted updates consume FIFO batches0..3
(old,new,old,old), despite four later error-triggered decisions. These observed
limits do not justify changing queue order, controller thresholds or dose using
the final results. No old-task forgetting/recovery was demonstrated.

All incremental utilities were negative. No fitted-controller advantage was
established; its historical fitting costs also remain. The first no-learning
callback took8.869s, others roughly1–1.47s including final readouts. Fixed-order
first-use effects prevent interpreting small wall/RSS differences as efficiency.
Overall42.732s metered/60.563s outer; all28updates returned,240final typed calls,
0newfits/0finalTRAIN/0promotions. Original sources and caller RNG unchanged.

Why this: test an actual distinction between policies before more mechanisms;
do not rerun equivalent six-row schedules or tune on these task outcomes.
The initial objective/causal controller-training, validation and sealed-readout
chronology is exercised across#52–54. Broader meta-generalization, long-duration
and release requirements remain separate unfinished work.
