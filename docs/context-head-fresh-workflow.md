# Fresh paired-head private adaptation

The ordinary recurring workflow can train its public-context source from zero,
qualify its terminal tasks, and run the existing private adaptation controls
without a historical checkpoint file. Run from the physical repository on D:.

```powershell
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --context-return --separate-context-heads --private-adaptation --output artifacts/fresh-private-workflow.json
```

Output must be new. This explicit option performs model work; it is not a preview.
It requires the existing perceptron/context-return/separate-head recipe with
`--replay none` (the default). Other modes are refused before data or output work.
Normal commands without `--private-adaptation` retain their original behavior.

The same module is available from an installed wheel, using `python -I -B -m
scripts.recurring_ticket_workflow_example` from a standalone directory. The
current wheel was built and offline-installed into clean Linux Python3.12.3
with NumPy1.26.4, then qualified without checkout or historical-data reads.
See [installed artifact, commands and limits](reports/context-head-installed-20261028.md).
This scoped compatibility check does not establish full release readiness.

The source pays its original12 TRAIN updates/72 rows and24 workflow readouts.
All four ordinary source tasks at update12 must succeed before any private
model is constructed. The source stream stays open so its existing candidate
guard remains usable. Its original12-update/30-second/512MiB budget stays active
through private work, then the source stream seals normally. No renewed clock,
budget or relaxed native counter is used.

Private copies use the [existing matched rehearsal](context-head-rehearsal.md):
extra-F versus fixed old-TRAIN replay,128 native passes/480 rows total, with
separate finite F/return/replay sessions. The source and actor receive no private
updates. Final evaluation and promotion are unavailable. Lost native work and
failed replay remain charged; failed source qualification prevents adaptation.

Source and private work appear separately in `training_work` and
`private_adaptation.training_work`; a completed invocation pays140 passes/552
rows combined. `probe_updates` includes all returned private native passes,
including return and replay; `replay_examples` is the private replay exposure.
Source budget counters remain12 and0 replay. In-memory provenance is explicit;
the result makes no historical source-SHA claim.

Output uses the existing4MiB bounded writer; the ordinary command's512KiB
encoder remains unchanged. Model/copy/readout/runtime/output costs still matter;
matched row and update counts do not establish equal computation.

Focused checks:

```powershell
.venv/Scripts/python.exe -B -m pytest -q tests/test_context_head_fresh_workflow.py
.venv/Scripts/python.exe -B -m ruff check scripts/recurring_ticket_workflow_example.py scripts/context_head_rehearsal.py tests/test_context_head_fresh_workflow.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/recurring_ticket_workflow_example.py scripts/context_head_rehearsal.py tests/test_context_head_fresh_workflow.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/recurring_ticket_workflow_example.py scripts/context_head_rehearsal.py tests/test_context_head_fresh_workflow.py
```

These commands describe developer usage, not permission to repeat a spent
experiment. See the [executed qualification](reports/context-head-fresh-workflow-20261028.md)
for exact finite commands, costs, failures, skips and limitations. This is a
known-template conventional workflow; it establishes no held-out, circadian,
statistical reliability or release benefit. Extend through existing stream,
source-guard and disposable probe ports; do not import private weights into
serving or bypass source qualification to keep a demonstration running.
