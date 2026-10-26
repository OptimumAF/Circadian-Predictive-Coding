# Bounded continuous learnability phases

The existing continuous A/B/A runner accepts an explicit fixed number of updates
per phase. `phase_updates=8` remains the default: A8 → B8 → A8,24 total updates.
Changing the count uses the same learner, delayed-label runtime, objective,
frozen A-only statistics, four endpoints and six acquisition/retention/recovery
metrics. It does not select a count from outcomes or start a native rollout.

Why this: the accepted24-update neural cases still fail the fixed native task,
while the simpler public21 nearest control succeeds locally. A count fixed
before outcomes permits studying optimization depth and its cost. More training
has not been shown to solve the neural failure. All historical negative runs,
settings, clocks, receipts and budgets remain intact.

## Existing entry points

```python
from src.adapters.continuous_learnability_demo import run_continuous_learnability

# warm/new must be owned complete released TRAIN episodes with matching specs.
result = run_continuous_learnability(warm, new, phase_updates=2)
assert result.report["settings"]["max_training_updates"] == 6
# An incomplete run exposes its paid progress but releases no candidate.
candidate = result.candidate
```

The JSON-only `run_continuous_learnability_demo` accepts the same keyword. Both
reject bool, NumPy integer, float, string, `None`, zero and values outside1..128
before episode admission, fitting or model construction. The upper limit gives
384updates/768logical ticks; source rows/dimensions still determine actual memory
and computation. This is API capacity, not a scientific experiment allowance.

```powershell
python -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --warm-teacher A17.json --new-teacher B19.json --phase-updates 2
```

The CLI rejects the option for other scenarios or an invalid count before file
reads. It composes with existing `--visual-observation`, `--public-vector-only`,
`--rgb-range-normalization` and `--binary-gripper-axis` subject to their unchanged
compatibility rules. Omission preserves existing runner calls and report keys.
Explicit8 has the same default behavior. Historical stage scripts still bind
their own frozen24-update protocols; the new option does not renew them.

## State and cost accounting

For admitted count `p`, A rows `a`, B rows `b` and parameter count `n`:

| Quantity | Accounted value |
|---|---:|
| Training update and inbox budgets | `3*p` |
| Next logical tick after completion | `6*p` |
| Training row exposures | `p*(2*a+b)` |
| Unique source rows | `a+b` |
| Repeated training rows | `p*(2*a+b)-(a+b)` |
| Checkpoints / context diagnostics per kind | `4 / 8` |
| Diagnostic rows per kind | `4*(a+b)` |
| Normalization calls | `3*p+24` |
| Forward / backward passes | `3*p+32 / 3*p+8` |
| Gradient values / nominal bytes | `(3*p+8)*n / (3*p+8)*n*8` |

The existing formulas also account for dense math, training/diagnostic/actor
rows, input normalization values, inbox payloads and prediction bytes. Source
and serving actor remain unchanged; only the private candidate learns. Every
packet must defer before its label arrives, validate its successful receipt and
full owners before the next packet, and preserve permanent charges on failure.
The final empty poll and read-only endpoint checks still apply. Incomplete runs
retain events, applied receipts and observed progress; planned full-run work is
released only after completion. Memory payload estimates exclude copies,
validation, elementwise/process overhead and do not prove a peak-memory bound.

## Changed files and verification

```text
src/adapters/continuous_learnability_demo.py  count admission, budget and costs
scripts/run_native_stream_demo.py            existing CLI option and early refusal
tests/test_continuous_phase_updates.py        synthetic/default/failure/CLI behavior
docs/adr/ADR-0472-bounded-continuous-phases.md decision and limits
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_continuous_phase_updates.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_phase_updates.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_phase_updates.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py
```

This implementation increment runs synthetic tests only. A real optimization-
depth comparison needs a separate small prospective protocol choosing ONE count,
fixed seeds/objective/metrics/directions and an explicit budget before outcomes.
Use the existing runner; retain every negative result and identify unequal
exposure/computation. Do not vary the count to obtain a desired result.
Final evaluation stays unavailable to training, tuning, promotion and probes.
No comparative neural/CPC/generalization or release claim follows from tests.

[Exact checks and handoff](../artifacts/runs/continuous-phase-updates-20261019/RESULTS.md) ·
[Protocol](../artifacts/runs/continuous-phase-updates-20261019/PROTOCOL.md) ·
[Independent review](../artifacts/runs/continuous-phase-updates-20261019/REVIEW.md).

## Subsequent fixed p32 evidence, 2026-10-19

One prospectively reviewed96-update arm improves final A/B mean TRAIN fit with4xtraining exposure and higher math, while three metric differences and worst gripper errors worsen. 72checks perplatform and independent readback accept scope. All negatives retained; native/generalization/CPC benefit unproven. See docs/public-vector-mlp-depth.md and its linked protocol/results/review.
