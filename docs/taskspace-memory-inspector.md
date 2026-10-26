# Inspect saved taskspace-memory evidence

Use the existing standalone inspector command from the checkout:

```powershell
.venv/Scripts/python.exe -m scripts.build_key_door_inspector --source-artifact replay.json --output memory.html
# Optionally: --resource-artifact costs.json
```

Open the new HTML file in a browser. No server, SDK, learner or network is needed.
The destination must be fresh. Both `taskspace_memory_demo_v1` (offline replay)
and `taskspace_memory_native_v1` (fresh native rollout) are supported, including
failed partial reports. Inputs remain bounded by the shared 8MiB JSON loader.

The page shows fixed scene endpoints, historical native outcomes, the original
198-row baseline, raw native returns, delayed TRAIN events, release charges,
checkpoint memory composition and recorded costs. Click an underlined value to
open its exact JSON pointer. Open source report shows the entire detached input.

Offline action agreement is not newly measured physical success. A raw native
return can precede a wrapper failure; validated endpoint summaries are displayed
separately. Missing values remain unknown, explicit null remains null, and zero
remains zero. Separately supplied resources have **UNVERIFIED** source association.
Recorded claims are not authenticated by the viewer. Numeric memory payloads do
not measure Python/snapshot/RSS overhead. No comparative curves, circadian benefit,
generalization or absent topology/lineage/sleep evidence are invented.

Why this: a pure app projection reuses the existing source-cell table renderer and
exclusive file publication. New report families should add a bounded projection
and dispatch entry; training and metric computation belong outside this viewer.

Focused checks:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_taskspace_memory_inspector.py tests/test_coherent_stream_inspector.py tests/test_key_door_inspector_files.py
.venv/Scripts/python.exe -m ruff check src/app/taskspace_memory_inspector_projection.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent src/app/taskspace_memory_inspector_projection.py
```

Current scoped evidence and known browser limitation:
[commands, outcomes, failures and independent review](../artifacts/runs/taskspace-memory-inspector-20261021/RESULTS.md).
Interactive/narrow-screen validation remains open; saved-page source and script
checks do not establish browser usability or repository-wide readiness.
