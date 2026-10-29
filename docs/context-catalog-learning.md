# Context catalog on the existing live learner

The library entry `context_controller_learning.run_study(source, catalog=True)`
implements six isolated TRAIN trials and an original-source frozen application
using the existing D9 paired-head learner, full catalog, fitter, live schedule
and ordinary task readouts. It performs real local model work and requires a
separately frozen allowance. The sole issue262 genuine invocation is **FAILED /
SPENT**:serialization lost its results/counters. Do not rerun its `actual.py`.

The implementation passes scoped synthetic-source integration checks. Genuine
task/controller qualification remains unknown; this is not a successful learned
controller demonstration or release-ready command.

## Behavior and boundaries

`ContextCatalogControl` reuses the original catalog's common pin/chooser/capture
contract while replacing its D7 BP source/bank/diagnostic assumptions with the
explicit qualified-12 D9 context backend. The old native BP backend stays strict.
The original three arms are wait,wake1,replay6-then-wake; the bank is the
[accepted ordinary contextual TRAIN bank](context-catalog-replay.md).
No model, optimizer, selector algorithm or alternate runtime is introduced.

`context_head_live.run_case` optionally admits an exact `catalog_trial` plus full
`catalog_base`, or an exact `catalog_policy`. Conflicting reference/recipe/
feedback/promotion options refuse before source/model work. Full source version,
digest, state and original counters are bound. Default reference modes, guarded
publication and rollback are unchanged. Catalog mode keeps the original actor
deployed and learns only in its private candidate:zero publication/rollback.

Cases keep the existing14opportunity worker, contention2, stale events4/12 and
revocation6. Exact scenario table is frozen in the protocol. Ordinary arrived
TRAIN labels alone feed the shared FIFO, catalog observations and model. Independent
original finite wake/replay sessions keep their limits/clocks/permanent charges.
Native history is original12/72 plus observed wake receipts and paid replay events/
rows. Source/actor/policy/packet/bank/previous evidence changes or replacements
refuse further work; failed returns retain known partial cost and unknowns.

Replay may complete before serving priority defers the following wake. The replay
stays paid. A later clean observation can select wait and leave TRAIN pending;
no forced drain, refund, duplicate allowance or fabricated success is added.
Reports distinguish complete observed window from completed training.

For an explicitly different execution contract, pass `resume_catalog_wake=True`
to `context_head_live.run_case` in catalog mode. After a returned, verified replay
success and a late serving/pause/resource deferral, the next ordinary opportunity
finishes only its original wake. A clean fresh observation cannot cancel that
pending part or cause another replay. One bounded pending record binds the
original selected action, receipt and FIFO target; no credits or clocks renew.
The original exact returned poll and sharing counter deltas must be observed
before another selection. Changed/missing evidence or unknown native work stops
the continuation. Capture reports pending/completed state, never resumable
authority. This option does not change `run_study` or existing frozen policies'
default execution. See [executed smoke and retention loss](reports/context-catalog-continuation-20261027.md).

Six ordinary private tasks are measured before/after on frozen copies. They are
the existing old4/new2 known TRAIN templates, explicitly `controller_train`, not
heldout/generalization data. Live worker outcomes and guard/probe/final labels do
not select training or fit. All tasks must execute completely before utility is
read; failure to solve a task remains a valid measured outcome.

The existing unit-weight objective/time120s/RSS512MiB/violation1 and wait-first
tie order remain. Source feedback must actually match declared new-v1=mistakes,
old-v1=clean before trials; otherwise stop without substituting a prototype.
All six empirical estimates must complete before freezing; frozen application
cannot refit. Serial time/RSS include instrumentation and retained earlier
reports, so they are not isolated causal costs or matched-compute evidence.

## Output

The new catalog report projects controller snapshot dataclasses into ordinary
JSON records. Existing classifier/task modes keep their previous defaults.
For a future producer with its own declared protocol, use the bounded writer:

```python
from pathlib import Path
from scripts.context_catalog_learning import write_catalog_report

# report and observed_work belong to that separately authorized producer.
saved_complete = write_catalog_report(
    Path("fresh-catalog-report.json"), report, observed_work=observed_work,
)
```

Outputs are exclusive; existing files are never replaced. Declared limit is
32KiB..24MiB. Unknown objects/nonfinite values/oversize output save a bounded
FAILED scalar-cost summary (<=32KiB), with detailed outcomes, fit and validated
receipts explicitly unavailable. Missing or malformed counters remain `null`.
Observer/producer scalar reports are not independent native validation. Filesystem
write failures still raise. Callers must preserve their observer in their own
failure handling; this writer cannot recover a process that already terminated.

The original genuine issue262 wrapper incorrectly supplied a whole-report byte
encoder as a JSON default converter. Its controller contained `CatalogEstimate`
records; no detailed output or counters survived. The repaired projection/writer
is verified only using bounded literal metadata and does not recover that run.

## Checks and extension

```powershell
python -m pytest -q tests/test_context_catalog_control.py tests/test_context_catalog_learning.py tests/test_context_catalog_output.py
python -m ruff format --check scripts/context_catalog_control.py scripts/context_catalog_learning.py
python -m ruff check scripts/context_catalog_control.py scripts/context_catalog_learning.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/context_catalog_control.py scripts/context_catalog_learning.py
```

These tests execute conventional fixtures; honor their prospective model budgets.
The output-only module uses no models/tasks/source. Actual executed commands,
failures, costs and skipped gates are in
[the report](reports/context-catalog-learning-20261027.md).

The next functional action is explicit continuation of a selected replay-then-wake
after a verified successful replay and serving-priority wake deferral. Keep one
bounded pending step tied to the original observed receipt/source/budgets, resume
only the remaining wake, and preserve old default behavior. Do not repeat replay,
renew credits, override unobserved state or rerun the failed six-trial study.
