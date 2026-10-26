# Guarded contextual serving

For an installed CPU package without research files, use
[the bundled qualified source](context-example.md):
`python -m scripts.run_bp_ticket_serving --example-context-source --context-head-serving --output fresh-context.json`.
The explicit research-source command below remains available.

The existing serving command now supports the conventional contextual learner:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-head-serving --output local-context-serving.json
```

The output must not exist. Use the qualified source77 record; private probe78
weights never enter this path. The command runs one finite local development
demonstration, not a persistent server. Original native serving modes remain.

## Behavior and boundaries

`core/context_perceptron.py` permits mixed contexts for readonly prediction,
grouping validated rows by public onehot context and scattering original order.
TRAIN still requires one homogeneous context and commits one complete head pass.
`adapters/context_serving.py` provides typed atomic actor reads and generic guard
callbacks. Each typed request owns one complete generation/version/model snapshot;
failures latch. Complete canonical serialized record bytes include policy,
coefficients and all counters. They exclude Python heap, copies and process RSS.

`scripts/context_head_serving.py` composes the existing AttributionStream,
PromotableActor and ServingPromotionController. A frozen control and guarded arm
start from the same qualified weights. The guarded arm trains one v1legacy
three-row packet, then one v1invoice/password six-row packet. Labels arrive one
tick after observation. Both guard evaluations use detached inner_guard data
projected from eligible TRAIN templates. Identity separation does **not** make
these repeated known template values held-out data.

The first boundary measures and discards any accepted ticket; publication is
allowed only at the second boundary. No published state feeds later training.
Original accuracy/gain/retention/latency/action thresholds remain; the explicit
resource metric is `complete_context_record_bytes`, capped at4096B. A successful
final publication is followed by one-step rollback. Episodes run synchronously
between updates; concurrent availability is not demonstrated here.

## Fixed observed result

| Phase | Frozen serving | Guarded serving | Candidate guard decision |
| --- | --- | --- | --- |
| Initial | 5/6 | 5/6 | Not evaluated |
| After new-task update | 5/6 | 5/6, original actor | Rejected: old row fit12/12→11/12 |
| After old-task update | 5/6 | 6/6, published candidate | Accepted: new6/6, old12/12 |
| Rollback | Not run | 5/6, original complete actor | Original receipt consumed |

The original four workflows remain successful in every served phase. The new
success is v1legacy; v2legacy was already successful initially. Candidate row
regression is measured; candidate closed-loop task loss/recovery was not read
out in this case. The separate [private probe](context-head-probe.md) retains its
own learning/loss/return evidence and all negative outcomes.

The sole study used2updates9newrows2mistakes. Qualified history12updates72rows
16mistakes remains charged; candidate ends14updates81rows18mistakes, even after
rollback restores actor12/72/16. Guard evaluation adds72full-class rows;42episodes
use126typed decisions. Raw result and all six failed workflows are preserved.
Recipe1.6442s; meter2.2803s; sampled process peak48,971,776B. Coefficients768B;
canonical source835B, candidates837/839B. These metrics are not process memory.

Known TRAIN templates, fixed order, unequal work and two declared public contexts
limit interpretation. This is useful scoped guarded adaptation, not generalization,
matched-compute efficiency, concurrent service, repository readiness or CPC benefit.

## Verification and extension

Focused Windows and Linux each pass112distinct checks, including both failure
prefix checks (113executions with one repeated regression per platform). Scoped
six-file Ruff/format/Windowsmypy pass. Linuxmypy is unavailable; no dependency was
installed. Independent read-only source/procedure/result review accepted the
repaired scope, without executing models. Details, commands, limits and failures:
[local evidence](../artifacts/runs/context-head-serving-20261023/RESULTS.md),
[ADR0595](adr/ADR-0595-contextual-guarded-serving.md),
[issue79](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/79).

The [existing live-worker slice](context-head-live.md) now supports fixed public
contexts, priority admission and drained publication. Its sole case preserved
these learning results but observed zero request/model-call overlap. Defaultv2
requests remain unchanged. Next compare the existing fixed controller references
on this learnable policy using the accepted periodic evidence, not another
timing/capture increment; freeze missing controls before new numerical work.
