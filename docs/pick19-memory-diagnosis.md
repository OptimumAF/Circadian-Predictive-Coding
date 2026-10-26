# Saved Pick19 memory diagnosis

`scripts/diagnose_pick19_memory.py` compares the complete B22 demonstration
with the B50 corrective labels and reproduces an accepted failed Pick19 trace.
It composes existing episode qualification, executed-action history, consistency
and nearest-memory functions. No learner, simulator or new labels are created.

```powershell
.venv/Scripts/python.exe -m scripts.diagnose_pick19_memory --output artifacts/pick19-diagnosis.json
.venv/Scripts/python.exe -m pytest -q tests/test_pick19_memory_diagnosis.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_pick19_memory.py tests/test_pick19_memory_diagnosis.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent scripts/diagnose_pick19_memory.py tests/test_pick19_memory_diagnosis.py
```

The output must be fresh and its parent must exist. The fixed accepted source
files are checked before and after analysis; changed evidence fails publication.
The report retains every pair, every trace step, partial results and paid work.
The research execution uses the separately frozen stage commands and limits.

Why this: B22 previously supported successful control, whereas the larger bank
using B50 failed. B50 targets describe desired corrections at student inputs;
their history must not be regenerated as though those actions had executed.
B22 and the observed trace use their actual executed histories.

The B22 readout describes stored actions at observed failed states. It cannot
predict rewards or success of a new rollout and is not additional supervision.
Keep any subsequent repair as a separate prospectively budgeted training change;
reuse the existing stream and preserve this result, original data and tie order.

[Protocol and executed evidence](../artifacts/runs/pick19-memory-diagnosis-20261021/RESULTS.md)

## Complete-trajectory coverage test

The next fixed treatment appends the complete B22 demonstration after all198
existing rows in `continuous_stream_sequence.run_sequence`, using explicit
`fixed_input=True, maximum_releases=6`. The original3-release default, explicit5
scope and256-row capacity remain. No packet merges, dropped labels or tie-order
changes are needed. This is a new training coverage hypothesis, not a conclusion
from offline alternative actions. The stage protocol separately gates the one
native repair/retention case.
