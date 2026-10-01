# P6.10a: Complete original resource field and scope inventory

Status: implementation and first correctness gate pass; related correctness
and actual publication/repetition/readback acceptance are pending. Keep
P6.10a unchecked. Original P6.10/P6.10b/P6.12 remain unchecked.

## Module boundaries

```text
src/app/continual_confirmation_resource_contexts.py
  Independently derive context guard/work/storage and recorded capacity points.
src/app/continual_confirmation_resource_fields.py
  Every arm resource field, unit, scope, status and original source pointer.
src/app/continual_confirmation_resources.py
  Complete fixed input/scope/work/capacity/run inventory and explicit gaps.
src/app/continual_confirmation_resource_rendering.py
  Deterministic exhaustive field/status/scope Markdown, no outcome ranking.
src/infra/continual_confirmation_resource_bindings.py
  Current complete saved source/input/request/recorded authority and full reads.
src/infra/continual_confirmation_resource_artifacts.py
  Exclusive request/result/Markdown/audit/failure/claim and reconstruction.
scripts/run_p610_resource_inventory.py
  Fixed local publish/readback CLI with no scientific or budget overrides.
```

Dependency direction is CLI → infra → app/core; infra does not import the CLI.
The public pure builder requires entire original cost/report byte identities,
reuses unchanged complete report declaration validation and checks all 560
original costs and all four original train/scored audit bodies. The private
development seam grants no original source/execution authority. Complete
decoded proof objects are discarded between the two original input reads.
No dependencies/environment variables or pinned scientific sources change.

## Original field mapping and interpretation

| Original P6.10 field | Stored source and scope | Inventory treatment |
| --- | --- | --- |
| Wall time | Original train/scored audit elapsed and worker elapsed, four process segments | Measured run rows; individual arm time unmeasured |
| Wake updates | Raw method counters, complete original work and live aggregate audit | Derived per-arm original counts, independently reconcile every group/total |
| Latent iterations | Raw wake/applied/rejected loop and example counters | Derived formula-based work, exclude guard prediction/selection/CPU/FLOP claims |
| Replay exposure | Applied presentations, original committed IDs and shared supply observations | Per-arm committed/distinct exposure and separate shared supply; retain rejected execution |
| Sleep/guard overhead | Raw attempts and original context guard calls/examples | Counts retained/reconciled; isolated per-arm duration unmeasured |
| Peak memory | Four original sampled absolute process RSS segments | Measured with original sampling interval, PID/count/start/peak and process scope |
| Replay bytes | Explicit owned fields where present, shared FIFO observations, original group work | Distinguish occupancy, owned arrays and shared supply before copies; no RSS allocation |
| Initial/final/peak parameters | Three actual checkpoint capacities plus raw peak and epoch/transaction points | Actual checkpoints measured; peak geometry derived including split before prune/rollback |
| Parameter history | Complete available checkpoints and recorded wake/epoch/transaction points | Preserve all named points/pointers; not a continuous wall-time or every-update profile |
| Accuracy/forgetting versus cost | Subsequent complete report consumer | P6.10b remains required; no composite winner in this inventory |

Every one of the 560 arm rows has an explicit field dictionary. Each field
stores value, unit, measured/derived/unmeasured status, scope, reason and exact
JSON pointers into the original complete cost inspection. Every context has
its full raw context byte identity and pointer, shared/owned storage, guard
work and history scope. Complete proof arrays remain in the exact bound
original artifact; they are not duplicated per arm. Every available history
point remains in the companion JSON, with original checkpoint counts and
derived `4*width+1` geometry for the fixed two-input, one-output head.

The original 15,210 optimizer calls include 46 rejected replay calls. A
rollback changes the committed state but does not refund execution. Occupancy
is not summed across successive checkpoints as if it were a peak or exposure.
Original retained array bytes exclude checkpoint copies and are not total RAM.
Baseline zero latent loops do not mean zero compute. Whole-process elapsed
time/RSS cannot be divided by the number of arms, and subtracting the train
segment from a scored segment does not isolate final evaluation overhead.

## Current saved evidence and authority

This inventory reads **both complete original cost inspections**, both report
bodies, every shared proof context, every arm/capacity, and all four original
train/scored audits. Current source/request/scope/configuration/environment,
all fourteen original report inputs, both report bundles and recorded complete
reader validations are checked before/after reading and publication/readback.
The complete original report and cost readers have already independently
reproduced those exact input bytes; their bound terminal evidence is retained.

The consumer proves current complete saved-byte preservation and independently
reconstructed inventory consistency with that recorded authority. It does
**not** claim another complete scientific reader execution, new profiling,
model/source construction, training or final-data access. P6.10b preserves
its own unchanged official complete-reader acceptance. ADR-0165 records this
scope and the alternatives. Original P6.10 remains unchecked until audited
against the inventory, outcome presentation and genuinely required measurements.

## Prospective source, correctness and local budget

The source/request record precedes development inventory fixtures:
`artifacts/runs/p610-resource-inventory-source.json`, **40,280 bytes**, SHA
`2b383bcea4ae012c39f6f191bb052c2674432fc8d8acddb8cafa621b25f36b78`.
All **114** sources are pinned; map
`3b16adb6caab2d76d537217fad6792ef8899dc431b955001fd24a85b4322b4f5`.
AST closure checks cover all 90 runtime imports, plus preserved evidence
producers in the original 106-source set. Source count is a deliberate
superset of runtime imports; it is not claimed as an exact 114-import closure.
The previous matrix consumer is preserved independently. No frozen production
byte changes after this record. Historical pending flags remain immutable.

The fixed **120-second derivative budget** is declared before fixtures or
actual publication. Prior complete cost operations took 74–75 seconds; this
inventory reconstructs saved inputs and does not repeat scientific reader
execution. Original 16,000-update / 600-second / 512-MiB scientific caps remain.
First app gate: 11 passing tests / 10.04 seconds. First five-module gate:
99 passing tests / 29.47 seconds, zero skipped. Related gate is pending.
New tests cover all six genuine first development families after model/data
seals, full 560-row metadata-only dispatch, scope/arithmetic/units/history/
guard/storage corruption, source/input/request drift, repeated inventory,
exclusive occupied/partial files, ownership/failure-marker handling and CLI.
The metadata dispatch is not reserved-source reproduction; IO spies are not
scientific readback authority.

## Local commands and exact next action

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p610_resource_inventory --publish --output-dir artifacts/runs/p610-resource-inventory
.\.venv\Scripts\python.exe -m scripts.run_p610_resource_inventory --read-only --output-dir artifacts/runs/p610-resource-inventory
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q --tb=short tests/test_continual_confirmation_resources.py tests/test_continual_confirmation_resource_rendering.py tests/test_continual_confirmation_resource_bindings.py tests/test_continual_confirmation_resource_artifacts.py tests/test_p610_resource_inventory_cli.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_resource_contexts.py src/app/continual_confirmation_resource_fields.py src/app/continual_confirmation_resources.py src/app/continual_confirmation_resource_rendering.py src/infra/continual_confirmation_resource_bindings.py src/infra/continual_confirmation_resource_artifacts.py scripts/run_p610_resource_inventory.py tests/test_continual_confirmation_resources.py tests/test_continual_confirmation_resource_rendering.py tests/test_continual_confirmation_resource_bindings.py tests/test_continual_confirmation_resource_artifacts.py tests/test_p610_resource_inventory_cli.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

Publication refuses occupied outputs before any input reads. Preserve current
artifacts and failures; use read-only reconstruction for complete bundles.
After the related gate, freeze correctness/test/producer identities, then run
two local publications and two independent full saved-input reconstructions
within each prospective budget. Only terminal passing evidence can close a.
P6.10b then adds every original accuracy/forgetting outcome against the scoped
work/storage/capacity ledger, retaining all failures/nulls/negative/inactive
results. Map missing measurements to prospective next actions without changing
scientific endpoints, seeds, baselines or caps. No broad sweeps are required.
