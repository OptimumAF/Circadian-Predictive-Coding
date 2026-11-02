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

The same command also opens `taskspace_memory_adaptation_v1` (CPU isolated
retained adaptation) and `taskspace_memory_native_adaptation_v1` (five final
known TRAIN scenes). No new command or optional dependency is needed:

```powershell
.venv/Scripts/python.exe -m scripts.build_key_door_inspector --source-artifact saved-adaptation.json --output adaptation.html
```

For these families, the page separates source preparation, private acquisition,
optional old TRAIN return, all recorded old/new residuals, and native observation
scope. Each phase retains its own clock, permanent updates, exposures, stored
payload and recorded work. `fresh` is the score-only reserved `probe_test`;
`fresh_training` is the new `probe_train`; `old_0` through `old_3` are original
TRAIN readouts. Lower normalized half MSE is better. Acquisition/fit/return gains
are before minus after; old retention loss is after minus before (positive is
worse). These are saved metrics; the inspector computes no scientific metrics.

Native adaptation has fixed Pick17, original probe23, Pick19, Pick20 and Push17
slots. The four-scene legacy formats retain their original bounds. Explicit
`observed_native_phase` and `unobserved_native_phases` stay visible. A missing
before/after phase cannot establish recovery; an unchanged old task never shows
forgetting merely because a return update was charged. Raw native success,
wrapper completion and independent acceptance are different claims. A failed
root stays failed even if a raw child reports success. Partial captures and
absent/null/false/zero values remain distinct. All imported claims are unverified.

Optional cost files remain unverified supplemental metadata. Do not add child
and aggregate costs, or logical/physical counters that overlap. The private
learner is separate from the serving and benchmark learners. Projection never
reads referenced paths/URLs, runs models, trains, promotes or accesses final
evaluation. The existing file adapter reads only explicitly supplied inputs and
writes one fresh HTML destination; direct app projection has no IO.

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
.venv/Scripts/python.exe -m pytest -q tests/test_taskspace_adaptation_inspector.py
.venv/Scripts/python.exe -m ruff check src/app/taskspace_memory_inspector_projection.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent src/app/taskspace_memory_inspector_projection.py
```

Current scoped evidence and known browser limitation:
[commands, outcomes, failures and independent review](../artifacts/runs/taskspace-memory-inspector-20261021/RESULTS.md).
Interactive/narrow-screen validation remains open; saved-page source and script
checks do not establish browser usability or repository-wide readiness.

Adaptation support: [scoped checks, one saved consumer import and review limits](reports/taskspace-adaptation-inspector-20261102.md).
The renderer is unchanged; this increment verifies inert HTML/source-cell and
ordinary file/CLI integration, without new interactive or visual acceptance.
