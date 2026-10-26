# Explicit frozen observation scale floors

## Behavior and purpose

An opt-in transform prevents an explicitly declared physical divisor from falling
below a positive floor. It retains the original TRAIN population means/scales and
records the floor and effective scales separately:

`z[i] = (x[i] - means[i]) / max(scales[i], scale_floors[i])`

Legacy exactconstant coordinates keep their fitted scale1. No clipping, new fit,
data-driven floor, extra feature, or additional learner is introduced. Omitted
floors preserve the original transform class, serialized three fields, defaults
and forwarding kwargs. Floor values are in the units of the model input after
any explicit feature recipe. This is preconditioning, not a safety bound.

Why this: saved TRAIN trajectories exposed large standardized excursions under
tiny nonzero empirical spreads. That observation motivates an explicit option;
it does not prove that scaling caused control failure or choose useful thresholds.

## Existing API and CLI

```python
from src.adapters.continuous_learnability_demo import run_continuous_learnability

# Declare values before a separately budgeted run; never choose from final data.
result = run_continuous_learnability(warm_train, new_train, scale_floors=(1.0,) * 42)
```

The same transform is shared by BP, PC, neutral CPC and active CPC in the existing
stream runner, diagnostics, private corrections, snapshots, forks and serving.
Exactly one A-only TRAIN fit supplies frozen statistics; new-task/probe/final
observations cannot refit them. Floors must be exact native tuples of positive
finite numbers matching the complete input dimension; bool/NumPy/foreign scalar
aliases are refused. The native robot API accepts exactly42 nonvisual values;
the existing canonical quaternion recipe remains composable. RGB-range recipes
and floors are explicitly separate.

Existing `scripts.run_continuous_robot_learning` adds
`--observation-scale-floors <42 positive values>`. It requires the existing
output/teacher paths and supports the existing learner/phase/readiness/A-only/
canonical options. The selected candidate must retain the requested floors before
native work; an omitted request refuses a candidate that introduces floors.
Running the CLI performs training and simulator work and needs a prospective
local protocol; a demonstration of parsing is not experiment authority.

Saved raw42 control diagnosis reconstructs and validates all selected scale
metadata, including redundant effective divisors. It refuses incomplete or changed
metadata and cannot silently fall back to empirical scales. Existing raw42
diagnosis still refuses canonical projected inputs; that limitation is explicit.
This increment adds no checkpoint loader or arbitrary saved-model authentication.

## Source responsibilities

```text
src/core/observation_standardization.py       pure fit/transform/metadata payload
src/adapters/numpy_continuous_bc.py           TRAIN admission and policy ownership
src/adapters/continuous_learnability_demo.py  existing four-family stream option
src/adapters/continuous_learnability_modes.py selected statistics payload accounting
src/adapters/continuous_correction_demo.py    private statistics payload accounting
scripts/run_continuous_robot_learning.py      selected request and frozen native handoff
scripts/diagnose_learned_robot_control.py     validated saved raw42 diagnosis
tests/test_observation_scale_floor*.py        independent arithmetic and integration
```

No new folders, dependencies, environment variables or orchestration layer.
Selected metadata has four vectors (32bytes/dimension) instead of the legacy two
(16bytes/dimension). Existing work reports now count this numeric payload; Python
containers, copies, scalar metadata validation/max comparisons and process costs
are additional overhead, not a matched-computation claim.

## Executed verification

77 focused cases pass on Windows and pinned Linux, including all four families
through actual tiny stream/private updates and fake native IO. Nine-file scoped
Ruff lint/format and mypy win32/linux pass; independent final-tree acceptance.
An initial ten-error type check was repaired before fixtures; its receipt remains.
No native physics or task benefit is established by these correctness fixtures.
[Exact argv, budgets, evidence and limitations](../artifacts/runs/state42-scale-floor-20261020/RESULTS.md).

```powershell
python -m pytest -q tests/test_observation_scale_floor.py tests/test_observation_scale_floor_integration.py
```

Exact affected regressions and nine-file lint/format/type command arguments are in
the linked stage's TEST_COMMANDS.json and recorder receipts. Tests perform bounded
model work; they are not a comparative study. Preserve selected metadata through
any future consumer and test request binding before adding a new route.

## Native result

The separately frozen unit-floor BP192 TRAIN17 case completed with zero success.
Final A imitation loss was worse than the reused empirical-scaling reference; B
read-only loss was lower, with zero B optimizer exposure. No reliability, safety
or circadian-benefit conclusion follows. [Exact protocol/commands/negative/costs](../artifacts/runs/state42-scale-floor-acquisition-20261020/RESULTS.md).
