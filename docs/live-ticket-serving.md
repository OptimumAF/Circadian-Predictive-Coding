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

The separate [frozen catalog API](contextual-controller.md#learn-privately-while-the-original-actor-serves)
now supports the same live worker with the original catalog's nine-opportunity
cap and a private candidate. It performs no publication or comparison scoring;
its software checks are separate from the fixed-reference native observations
below. [Explicit mixed/corrupted live TRAIN recipes](contextual-controller.md#mixed-or-corrupted-feedback-during-live-service)
can now be composed with the private catalog path under the same nine-opportunity
cap; delayed labels beyond it remain unarrived. Their native usefulness remains
unqualified. Custom deferred and terminal-publication combinations remain refused.

## Optional isolated diagnostic after a private catalog window

The existing Python API can run the fixed warm/fresh learning diagnostic after
a successful, fully captured private catalog live window:

```python
from scripts.run_bp_ticket_serving import run_case
from scripts.ticket_serving_timeline import LiveServingPlan

result = run_case(
    source_path,
    catalog_policy=frozen_policy,
    serving_plan=LiveServingPlan(9, (2, 6)),
    probe_after_window=True,
)
```

Use the original qualified source and frozen catalog policy as in the catalog
API above. This is an exact Boolean opt-in; trials, publication and deferred
comparisons refuse it before source access. Defaults remain unchanged. Existing
explicit live TRAIN recipes retain their original horizon and pending labels.

The diagnostic reuses `probe_ticket_policy_learning.run_arm` and the existing
disposable probe:16fixed TRAIN-derived policy-change updates followed by16old
TRAIN-return updates, checkpoints0/4/8/16 and typed task readouts. One warm seed
uses the **in-memory terminal candidate's full optimizer state**; another uses
the original fixed fresh BP initialization. Separately counted diagnostic seeds
avoid inheriting the live learner's benchmark callbacks. The four original
diagnostic sessions reserve64updates/192rows total; each keeps the existing
16update/120s/512MiB limit, with a separate180s aggregate diagnostic deadline.
No budget is restored or renewed. Original live budget clocks/counters stay
fixed; total elapsed time and RSS include the diagnostic.

`catalog_window_complete` reports the preceding window separately from optional
diagnostic completion. `isolated_probe` contains both arms, training/return
costs, seed-construction attempts/unknown returns, task outcomes and failed
prefixes. Failed or replaced prior evidence prevents further work. Original
paid records are retained after refusing an edit; this restores evidence only,
never model state or authority. Retention stays bounded to the original window
and at most two diagnostic arms.

The deployed actor, private candidate, source, pending labels, receipts, budgets,
sharing and caller RNG are guarded throughout. Probe updates never enter live
training, replay, promotion, final evaluation or controller selection. These
known TRAIN-derived tasks are a diagnostic, not unseen-task generalization or
proof of plasticity loss. Original comparison utility refuses this new report
family because it does not price diagnostic work.

[Software qualification and failures](reports/catalog-candidate-probe-20261027.md)
cover the scripted connection. The [single native diagnostic](reports/catalog-candidate-probe-native-20261027.md)
now completes: warm keeps both old tasks but still fails the changed invoice;
fresh solves none of the four endpoint tasks. Row loss improves without full
invoice acquisition. Warm never loses the old tasks, so return is not recovery.
The run costs70updates228rows including its live prefix; no circadian advantage,
plasticity-loss or generalization claim. Both diagnostic and earlier native
allowances remain spent; no automatic retry or dose tuning.

## Preserved native live observations

[Native catalog live qualification](reports/catalog-live-native-20261027.md)
now records one complete9opportunity application: private new tasks1/2→2/2,
old2/2retained with6updates36rows, while the original actor serves9episodes
(6success/29requests). It remains private: both live legacy requests still fail,
and two TRAIN packets remain pending. This establishes no comparative advantage
or broader long-duration reliability; original fixed-reference results remain below.

For explicit publication after that frozen catalog window ends, see
[terminal publication](contextual-controller.md#publish-after-the-live-window-ends).
It uses the original inner guards, retires old training without draining pending
packets, and reports post-window typed service separately. Its current evidence
is software-only; it does not alter the original native observations below.

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
