# Public-context serving on the existing worker

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-head-serving-live --output local-context-live.json
```

The output must not exist. This runs one finite local availability case. It uses
qualified77 weights as data and separately eligible TRAIN packets, never private
probe78 weights. The [previous guarded-learning result](context-head-serving.md)
is reused; its frozen comparison is not rerun.

## Delivered functionality

`LiveServingPlan.context_tasks` optionally declares exactly one immutable known
`(policy_version, concept)` pair per opportunity. The existing single-worker
`TicketServingTimeline` includes that public context in every request. Default
plans preserve original v2 requests without context markers. Plan/source tampering
and failures retain prefixes and stop the original owner.

`ContextServedDecision.native_updates` aliases the complete learner counter for
the existing timeline port. It does not mean Torch-only computation. Each typed
read retains its actual atomic generation, model and version. No new scheduler
or runtime was introduced.

The existing reference schedule owns delayed TRAIN arrivals, fixed periodic
decisions, shared admission and drained publication. Live outcomes never become
labels, policy feedback or promotion guards. All applied receipts survive a
later schedule failure. Source/data/RNG/plan checks are read-only in the worker;
only the main thread mutates the original budget clock.

## Sole observed case

The fixed 14-opportunity case completed all 14 worker episodes and 46 typed
requests. Serving priority deferred one admission at tick2. Two updates then
consumed nine TRAIN rows and made two mistakes, at ticks4/6. Candidate states
exactly match the two previously observed79 learning states; no new learning
benefit is inferred from repeating these integration updates.

After labels and receipts drained at tick6, the original guard accepted a complete
publication: new6/6 rows versus actor5/6, old12/12 unchanged. All seven subsequent
worker episodes succeeded, covering all six known tasks; one had the original
stale-ticket scenario. The revoked commit at tick6 remained denied. Two earlier
legacy failures were retained. Rollback restored the complete source and its
5/6 ordinary workflows; costs were not refunded.

**Negative timing result:** two service leases entered while training admission
was active, but every typed request started after training became inactive.
No request interval intersected either delegated model-call window (3.13ms and
4.37ms). This verifies integration, priority, publication and rollback. It does
not demonstrate typed inference during model execution, speedup or a service SLO.
There was no forced rendezvous, enlarged workload or retry to manufacture overlap.

Observed predictor latency p50=16.73ms, p95=29.57ms, maximum48.85ms under profiling;
source checks before predictor entry and queue/episode work cost extra. Recipe
2.4791s/meter3.4892s; sampled case peak48,820,224B and boundary48,893,952B. Full
worker/model/copy/guard/teacher work is counted on both threads. History ends at
14updates81rows18mistakes; the original12/72/16 remains charged. Known TRAIN
templates, one case and unequal comparison work do not establish heldout,
efficiency, CPC or release claims.

## Verification and next outcome

71 focused checks pass on each platform, with zero optimization in fixtures.
Scoped five-file Ruff/format/fresh Windowsmypy pass. Linuxmypy remains unavailable;
no installation or repeated failed check. Finite independent read-only source,
procedure and saved-result review accepted this scope, not independent numerical
execution or the missing overlap claim.

See [protocol/results/commands](../artifacts/runs/context-head-live-20261023/RESULTS.md),
[ADR0596](adr/ADR-0596-contextual-serving-worker.md), and
[issue80](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/80).

Next useful outcome: compare the already implemented fixed controller references
on this learnable conventional policy, reusing the accepted periodic result.
First add an explicit reference-policy selection to this existing mode, preserving
periodic defaults, and bind readouts to the existing controller objective. Freeze
only the missing no-learning/drift/budget-aware cases before execution. Do not
tune their defaults or chase another overlap/capture increment. Broader unseen
stream meta-generalization and long noisy deployment remain open.
