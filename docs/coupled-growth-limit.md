# Bounded coupled split growth

Configure a width ceiling on the existing uniform raw split consolidator:

```python
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator

split = CoupledScalingSplitConsolidator(max_width=8)
runtime.consolidate("split-1", split)
```

`max_width` is a keyword-only exact positive Python integer below2**63, or `None`.
The read-only property exposes the configured value. The default `None` preserves
existing behavior and global configuration limits. A target equal to the ceiling
is admitted; a larger target raises `ValueError` before the expanded split kernel
allocates tensors or models. Existing complete current-state and target metadata
validation runs first and may allocate copies at the current geometry.

Why this: an owner can need a smaller growth bound than the global geometry cap.
The check belongs before expansion, rather than after a large candidate already
exists. It does not change split mathematics, merged/pruned states, learning
settings or the existing unsupported width/replay options in `ToyBudgetSession`.
This is trusted consolidator configuration, not tamper-proof process authority or
a serialized runtime budget; callers must retain their original configured policy.

## Runnable control

```powershell
python -m scripts.run_coupled_growth_limit --describe
python -m scripts.run_coupled_growth_limit
```

Description performs no operational work. The fixed example uses PC, neutral CPC
and active CPC, seed23/mean-field D2/width2/two hidden layers, gamma1.5 and original
precisions(1,1). Each source gets two genuine raw training steps. The private
candidate accepts widths4 and8, then refuses proposed16 under ceiling8. The failed
third attempt remains consumed; duplicate and exhausted-quota probes refuse before
callbacks. Complete candidate state/history is unchanged by these refusals, and
the primed width2 actor/cache stays unchanged throughout.

One arrived delayed-label training update then demonstrates candidate continuation
at width8 under its original single-update allowance. Nine raw SGD calls/36 energy
evaluations total; no guard scoring, promotion, final evaluation data or tuning.
Full CPC policy/chemistry/gates/count and rooted structural history are retained,
with count2 through splits/refusal and count3 after continued training. Matching
PC/neutral phases remain exact. This does not claim identical learning trajectories
across widths or a comparative benefit from circadian gating.

The report retains complete states, cursors, refusal reasons and actual logical
array/cursor/cache/RSS measurements. Process RSS is sampled at explicit boundaries
under a512MiB ceiling and60-second run bound. Sampling is observational; this tiny
run does not establish continuous hard preemption or long-run memory safety.

## Structure and checks

```text
src/app/coupled_structural_consolidation.py  optional pre-expansion ceiling
src/adapters/coupled_growth_limit.py         native/runtime/cache integration
scripts/run_coupled_growth_limit.py          bounded JSON CLI
tests/test_coupled_growth_limit.py           independent limit and native controls
```

```powershell
python -m pytest -q tests/test_coupled_growth_limit.py tests/test_coupled_structural_consolidation.py
$growthFiles = @('src/app/coupled_structural_consolidation.py', 'src/adapters/coupled_growth_limit.py', 'scripts/run_coupled_growth_limit.py', 'tests/test_coupled_growth_limit.py')
python -m ruff check @growthFiles
python -m ruff format --check @growthFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @growthFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @growthFiles
```

Actual native checks require a prospective local budget. See ADR0279 and
`artifacts/runs/coupled-growth-limit-20261015/` for current scope, commands, results
and independent review. No dependency or environment configuration added. Extend
through explicit owner policies before allocation; preserve consumed IDs and
complete native history. Full recovery, long runs and scientific gates stay open.
