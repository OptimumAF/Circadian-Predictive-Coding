# Matched private context-head rehearsal

Run from the physical repository directory on D:, using the existing environment.
This command composes the existing disposable probe, old-TRAIN return and
StreamRehearsal. It adds no dependencies, learning rule or serving authority.

```powershell
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.context_head_rehearsal --input artifacts/runs/context-head-control-20261023/actual.json --output artifacts/rehearsal-preview.json --preview
```

Preview verifies the original teacher packets and describes the two fixed
controls, tasks, bank and native exposure. It never reads the source or constructs
a model; the input path may be absent. Output must be new and distinct from input.

Without `--preview`, the command restores the exact qualified source and runs
ONE pair: extra F training versus fixed old TRAIN replay. Each private arm pays
64 native passes/240 rows: F phase48/144, then old return16/96. Replay's logical
F16 clock includes32 additional replay passes/96 rows. The fixed4-ticket bank
occupies5216 numeric bytes; retention cap12 native transitions/8192B. All original
budgets, native clocks, source/RNG/data and preceding paid-evidence guards apply.
Final evaluation and deployed/benchmark learners are unavailable to these updates.

Do not repeat the completed scientific case merely to exercise the CLI. The
[executed result](reports/context-head-rehearsal-20261028.md) and its frozen
protocol preserve the sole paired observation. New experiments need their own
prospective question, limits and protocol, under the GitHub workflow.

Focused checks:

```powershell
.venv/Scripts/python.exe -B -m pytest -q tests/test_context_head_rehearsal.py
.venv/Scripts/python.exe -B -m ruff check scripts/context_head_rehearsal.py scripts/probe_ticket_policy_learning.py tests/test_context_head_rehearsal.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/context_head_rehearsal.py scripts/probe_ticket_policy_learning.py tests/test_context_head_rehearsal.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/context_head_rehearsal.py scripts/probe_ticket_policy_learning.py tests/test_context_head_rehearsal.py
```

The executed finite check commands/costs, failures and scoped skips are recorded
in the report. The plain pytest command above describes future developer usage;
it does not authorize repeating spent research or engineering allowances.

Extend through the existing probe and stream ports. Preserve exact roles,
complete episode certification, separate replay charges and native counters;
do not equate logical F updates with actual learner work. This is a conventional
same-template control, not a generalization, CPC or equal-compute result.
