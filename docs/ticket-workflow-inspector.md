# Inspect saved task learning and failure

Export an existing report through the shared offline inspector:

```powershell
python -m scripts.build_key_door_inspector --source-artifact artifacts/runs/bp-new-contexts-20261023/actual.json --output bp-tickets.html
python -m scripts.build_key_door_inspector --source-artifact artifacts/runs/native-ticket-family-controls-20261023/pc.json --output pc-tickets.html
```

For a checkout-independent example, use `--example-ticket bp` (also `pc`,
`neutral_cpc`, `gated_cpc`) instead of `--source-artifact PATH`. Exact saved
resources ship in the wheel; see [portable usage](portable-ticket-examples.md).

Open the fresh HTML file locally. It includes its source data, styles and script;
no server, Torch, learner or network connection is required. The existing source
loader admits bounded UTF-8 JSON, rejects duplicate keys/nonfinite values, and
publishes exclusively without overwriting another file. Neutral/active CPC
reports use the same command with `neutral_cpc.json` or `gated_cpc.json`.

## Reading the page

The endpoint table shows the original success flags for invoice/parcel (old)
and legacy/password (new input contexts). The signed changes retain the original
definitions and directions. Initial success can be transfer; zero forgetting
without initial competence is not retention. Three endpoints are shown as a
table, without fabricating rolling learning curves.

Episode outcomes distinguish capture completion, ordinary success, stale recovery
and revoked-commit refusal. A failed revocation slot that never reaches its event
does not prove refusal. Full task steps show the validated selection, observed
revision, state at effect time, resulting state and capabilities. The original
raw predictor return is pre-validation; its `validated:false` field is retained
in the complete source and does not override subsequent validation/accounting.

Click any underlined recorded value to open its exact JSON pointer and complete
value. **Open source report** shows the entire detached original input. Unknown,
recorded null, zero and false remain distinct. Source claims are not authenticated
by this viewer, including completion and isolation flags. No metric is recomputed.

Checkpoint rows separate local phase updates from full native history. Topology,
unit IDs/parents/ages and sleep counters appear only when recorded. BP lineage and
unrecorded sleep proposals/acceptance remain unknown. Native receipt clocks,
diagnostics, teacher/readout work, prior warmup charges and new budgets stay
separate. Optional `--resource-artifact costs.json` is always labeled
**association UNVERIFIED**; supplying it does not bind it to the report.

Saved BP acquires legacy and retains both old tasks in this fixed synthetic
case; password succeeds initially. PC/neutral/active CPC fail full tasks at the
same BP-informed dose. Equal exposure is not equal computation, objectives,
optimization or whole-runtime work. No general method ranking, circadian benefit,
safety or release readiness follows. This inspector adds no training/promotion,
final/probe/replay selection or native state reconstruction.

## Structure and focused checks

```text
src/app/ticket_workflow_inspector_projection.py   Pure bounded saved-report view
src/app/key_door_inspector.py                    Existing schema dispatch
src/infra/key_door_inspector_files.py             Existing exclusive HTML exporter
src/infra/coherent_stream_inspector.html          Existing table/source-dialog UI
tests/test_ticket_workflow_inspector.py          Saved, partial, fault and IO checks
```

```powershell
python -m pytest -q tests/test_ticket_workflow_inspector.py tests/test_taskspace_memory_inspector.py tests/test_continuous_comparison_inspector.py tests/test_key_door_inspector_files.py
python -m ruff check src/app/ticket_workflow_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py tests/test_ticket_workflow_inspector.py
python -m ruff format --check src/app/ticket_workflow_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py tests/test_ticket_workflow_inspector.py
python -m mypy --explicit-package-bases --follow-imports=silent src/app/ticket_workflow_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py tests/test_ticket_workflow_inspector.py
```

Why this: existing task evidence becomes inspectable through the shared renderer,
without another model run or viewer framework. Add future report families as
explicit bounded projections with source-pointer checks; keep training and
scientific metrics outside the presentation layer. Full shared-inspector,
comparison packaging and release requirements remain open under R4.7/R4.8.

Scoped evidence: [commands, failures, browser checks and limitations](../artifacts/runs/ticket-inspector-20261023/RESULTS.md).
The final view has21ticket checks/platform plus89unchanged related checks,
scoped static checks and19real Chromium interaction checks. Desktop1280px and
narrow390px screenshots were inspected; this is not all-browser/device readiness.
The in-app browser/default Playwright build were unavailable; an existing explicit
Chromium executable was used without installation. The two final HTML examples
remain local artifacts; they are not yet bundled installed-wheel examples.
