# Matched warm-start BC-only control

The existing key/door runner now offers `torch-key-door-warm-start-control`.
It compares an eight-demonstration BC-only learner with an identically initialized
eight-demonstration/eight-reward learner. Full native parameters, policy, gradients,
momentum and clocks must match after demonstrations, before any reward collection.
Both arms finish training before any final classification or live evaluation.

```powershell
# Preview the frozen small protocol without native training or evaluation.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door-warm-start-control
```

Omitting `--describe` executes the fixed local comparison. The sole accepted case
and all costs are [reported here](reports/warm-start-control-20261016.md); rerunning
is new experimental work requiring its own prospective budget. Existing pure-RL,
single warm-start, BC and matched four-method routes remain available.

## What is matched

Both arms start fresh with the same seed17 Elman8/8/5 architecture, rate0.03,
momentum0.9 and exact eight train-role demonstrations,64 rows each. The control
uses one original eight-update budget and keeps its post-BC state frozen. Its
actual endpoint is native8. The reward arm uses one original sixteen-update
budget and samples eight episodes with private PCG64 seed29/max12 steps/gamma1.
Its endpoint is native16. No budgets, optimizer or gradient state reset.

The control has checkpoints0/8 and sixteen live episodes; the reward arm has
0/8/10/12/14/16 and48 episodes. Two bounded completion closures defer all final
work until both train successfully; preceding evidence/state/RNG and original
budgets are rechecked. Source/actor changes refuse further work, and the first
final failure prevents the other arm's evaluation. Partial charges and all
unreached slots remain visible.

All eight final A/B/C/D×two-cue endpoints are required exactly once. Reports retain
both values and signed CE reduction BC−RL, accuracy/success/return gains RL−BC,
per context and overall. Native8 is never relabeled16. Final data do not train,
promote, tune or select probes. D is held-out geometry; known-layout teacher
trace scores are distinct from successful live navigation.

## What is not matched or established

BC uses8SGD/64 rows; the measured reward arm uses16SGD/75 rows. The additional
reward-phase exposures and computation are explicit. All sampled rewards in this
case are zero: carried BC momentum still moves parameters. Lower CE alone is no
evidence of reward-gradient benefit. Every measured live episode fails; useful
acquisition, recall, recovery, generalization and circadian advantage remain open.

187 focused fake cases qualify; final five-path lint/format/types and independent
review gate one budgeted native pair. [Exact commands, failures and caps](../artifacts/runs/r42-warm-start-control-20261016/RESULTS.md).

```text
src/adapters/key_door_reward_pilot.py       existing recipe, frozen control/pair
scripts/run_coupled_attribution.py         explicit CLI scenario
tests/test_warm_start_control_pilot.py     independent runtime/matching/fault tests
tests/test_warm_start_comparison.py        nonzero signed metrics/completeness
tests/test_warm_start_pair_integrity.py    isolation and dependent-final refusals
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_warm_start_control_pilot.py tests/test_warm_start_comparison.py tests/test_warm_start_pair_integrity.py
```

Why this: a matched warm start isolates added reward-phase work without rebuilding
the runtime or changing either native objective. [ADR-0333](adr/ADR-0333-match-warm-start-state-before-reward-comparison.md).

Next add bounded per-decision/class diagnostics to the existing reserved fresh-task
probe, using already returned predictions with no extra native calls. Preserve
aggregate CE/accuracy, original binary defaults, source/actor isolation and probe
budgets. This addresses the observed CE/accuracy disagreement and earlier majority-
level probe accuracy before further learner changes or experimental tuning.
