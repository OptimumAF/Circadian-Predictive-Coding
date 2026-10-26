# Typed serving during causal native updates

For separately declared delayed/noisy TRAIN packets, see
[live TRAIN recipes](live-training-recipes.md). The default plan and preserved
observations below continue to use the original clean six-row batches.

For optional guarded publication at the causal drain, use
[drained publication mode](drained-ticket-publication.md). The original mode and
its preserved observations below continue to publish after the serving window.

Run the existing BP serving demo with one finite live-service worker:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --serving-during-updates --output live-serving.json
```

Use a fresh output path. The source is the exact qualified96update/576row native
BP artifact; no warmup fit is repeated. The mode uses the existing frozen drift
reference, eight6row TRAIN batches and original delayed inbox/gate/budget. It is
incompatible with other CLI study modes or custom deferred/fitted/counterfactual
recipes. Other modes retain their defaults.

`LiveServingPlan` optionally configures1..64opportunities, sorted contention ticks
and at most10second coordination waits. The CLI freezes24ticks, contention2/6.
The concept cycle is invoice,legacy,parcel,password. Invoice ticks4/12/20 inject
one stale observation; parcel6/14/22 revoke commit permission. Every episode has
at most six actions. The app dispatcher independently checks permissions/revisions.
These are existing local public-template workflows, not new/generalized tasks.

## Execution and isolation

`scripts/ticket_serving_timeline.py` owns one worker, at most one pending episode
and a finite record per opportunity. Training remains in the existing
`_reference_schedule` loop. Scheduled phase requests and live episodes use its
original serving gate. At contention ticks a real worker reserves the serving
lease before the admission decision and begins prediction after that decision;
the wake attempt defers without native work. On other admitted wakes the original
counted native callback starts a service worker while training admission is held.
Serving does not preempt an in-flight optimizer call. Each worker joins before
the next tick; failed/unknown work retains its prefix and stops further work.

Each request records actual actor generation/version/native clock, whether TRAIN
was active at prediction start, duration and completion. Episode lease admission
also records actual TRAIN state. An admitted overlap does not prove simultaneous
optimizer execution; measured flags remain explicit. Only the main thread mutates
the original ToyBudgetSession. Worker source/plan checks do not race that clock.

Live outcomes never become training labels, controller feedback, promotion guards,
fit objectives or final/probe selection. Only original causally arrived TRAIN
predictions/labels feed the controller. Guarded publication and rollback occur
after the finite stream, using their unchanged development-template guards. No
final/probe data source is opened. The helper stores bounded scalar/request/workflow
records rather than native model histories; no new runner or durable format.

Why this: the existing demo already has the native learner, typed predictor,
promotion guards and causal schedule. Adding serving opportunities at that
boundary exposes contention and the delay between learning and deployed use.

## Observed outcome and costs

The sole frozen24tick trace completed8updates/48rows, two serving-priority
deferrals and24live episodes with78typed requests. Eight service leases began
under native admission; four request predictions began while TRAIN was active.
Every live request used generation0/qualified-96/full96state. The candidate reached
104updates/624rows; source and caller RNG stayed unchanged.

Live invoice6/6 and password6/6 succeeded; ordinary parcel3/3 succeeded, and all
three revoked parcel commits were refused. All three stale invoice episodes
recovered. All six live legacy episodes completed the wrong workflow. The old
actor remains deployed until the stream ends, so shadow learning is not immediate
deployed adaptation. The unchanged guard then accepted the candidate; all four
ordinary tasks succeeded after publication. Complete rollback restored original
96state and legacy failure. These are scoped known-template outcomes.

Instrumented predictor-call durations: p50=15.23ms,p95=18.16ms,max=21.07ms
(nearest rank,78attempts). They exclude queue/reservation and workflow dispatch;
per-episode queued/lease/finished timestamps and elapsed time retain those costs.
Profiling, startup, one fixed ordering and a tiny trace prevent latency SLO,
tail-distribution or controller-efficiency claims.

Full physical case:8SGD calls48rows/8backwards,588head constructor entries/
22932constructor matrix-slot upper,168native forward invocations1345snapshot
constructions179952logical tensor elements,148typed attempts172local dispatches,
0PC/finalTRAIN/probe/replay/newfit. Internal11.6784s,meter13.4301s,outer21.1199s;
sampled peak334913536B. Constructor slot counts are conservative uppers (including
initialization templates), not exact random draws/FLOPs; memory is current-process
sampled RSS, not a process-tree peak or hard preemption guarantee.

## Checks and remaining requirements

```console
python -m pytest -q tests/test_ticket_serving_timeline.py
python -m ruff check scripts/run_bp_ticket_serving.py scripts/ticket_serving_timeline.py tests/test_ticket_serving_timeline.py
python -m ruff format --check scripts/run_bp_ticket_serving.py scripts/ticket_serving_timeline.py tests/test_ticket_serving_timeline.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/run_bp_ticket_serving.py scripts/ticket_serving_timeline.py tests/test_ticket_serving_timeline.py
```

81selected combined checks/platform and scoped statics passed. Seven unchanged
reference-cohort/failure cases were explicitly reused/deselected. All commands,
failed initial typing check, software work and native receipts:
`artifacts/runs/live-ticket-serving-20261023/RESULTS.md`.

This is single-agent source/procedure/result self-review, not independent
acceptance or repository readiness. Full R7.7 still requires long-duration,
noisy/poisoning-feedback and broader utility/reliability evidence; R7.1 gate/sleep/
combinations also remain open. No controller or CPC advantage is established.
