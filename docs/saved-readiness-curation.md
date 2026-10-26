# Curating a saved TRAIN return

## What it provides

`curate_saved_readiness` annotates one already recorded PickCube TRAIN return
with the existing goal-consistent observed-grasp teacher. It never executes
those target commands. Actual source observations/actions and unknown historical
clocks remain intact. New query and arrival ticks describe the later curation.

Why this: the joint learner grasped in40 recorded A-return states but did not
finish the task. Its original15 expert states did not cover that return. These
labels support a future coverage/adaptation question; they do not prove its cause
or successful recovery. All old negative results remain preserved.

```python
from src.adapters.saved_readiness_corrections import (
    curate_saved_readiness,
    project_curated_training_pairs,
)

# source is an authorized raw42 TRAIN ContinuousEpisode, seed17 or19, <=50 rows.
# Same default Panda table/root/world-axis setup as the existing TS12 recipe.
result = curate_saved_readiness(
    source, train_seed=17, source_available_at=100, started_at=100,
    source_observed_at=None,  # Original control ticks are unknown, not zero.
)
if result.batch is None:
    # Retain attempted/returned work and the paid partial label prefix.
    print(result.report)
else:
    features, targets = project_curated_training_pairs(result.batch, now=199)
    # Caller may convert these tuples for an existing authorized training port.
    # A separate experiment budget is required; curation performs no learning.
```

For50 rows, queries100,102,...198 and arrivals101,103,...199 are NEW logical
curation times. Supplied original observation ticks are preserved separately
and must precede source availability. Before the last arrival, extraction refuses
before model projection. One2-row batch instead finishes at103; choose `now`
from the actual last recorded arrival, not a guessed horizon.

## Data and work boundaries

Only exact raw42 normalized four-axis TRAIN17/19 returns are admitted. Probe,
final and guard roles, wrong contexts, malformed clocks/flags/non-native actions
are refused before queries. Existing legacy correction batches and their
2*step+1 clocks remain unchanged; no automatic schema conversion is provided.

The public raw42 displacement vectors determine a virtual TCP origin with
cube=object-TCP and goal=cube+(goal-object). Existing teacher rules are
translation invariant. World/root axes align only in the declared default Panda
setup; rotated/custom setups need a separately specified geometry adapter.
`official_state42_relative_world_axes_panda_table_v1` and
`provisional_saved_state42_goal_consistent_v1` explicitly distinguish this new
public-state curation from privileged historical labels. It is provisional,
not expert qualification or a claim of bit-identical old root-pose decisions.

Each source row is queried once, up to50. The teacher sees current grasp feedback
and the preceding ACTUALLY executed gripper command (0 at the first row). Returned
labels and attempt/return counters are owned before later source checks. Failure
or mutation returns batch=None with partial work; no rollback or retry. Extracted
TS12 history also follows actual commands, never teacher targets. Original source
bytes/permission/provenance binding and overall query budget remain the caller's
responsibility, as with existing training ports.

```text
src/core/curated_readiness_corrections.py   records, source/clock validation, release
src/adapters/saved_readiness_corrections.py fixed curation and existing TS12 view
tests/test_saved_readiness_corrections.py  behavior and boundary fixtures
```

No simulator/model/RNG/normalization/optimizer/replay/probe/promotion work, IO,
new dependency, environment variable or configuration system is added. Geometry,
query and projection work are separate from later learner/benchmark costs;
validation/copy/JSON/interpreter costs are additional.

## Verified scope

43 focused checks pass on Windows and pinned WSL, with scoped lint, true format
and host Linux-target typing plus independent review. One fixed A17 curation
produced50 labels:4align/4descend/2grasp/40lift/0settle; all50 targets differ from
executed commands. Release198 refuses,199 projects; original source unchanged.
Independent cached validation accepts the evidence without querying a teacher.

A worker's uninstrumented broad source search has an upper bound above its4MiB
engineering limit; conformance remains unproven. It is preserved, not waived.
This does not establish task recovery, uninterrupted retention, reliability,
generalization, CPC benefit or repository-wide release readiness.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_saved_readiness_corrections.py
.venv/Scripts/python.exe -m ruff check src/core/curated_readiness_corrections.py src/adapters/saved_readiness_corrections.py tests/test_saved_readiness_corrections.py
.venv/Scripts/python.exe -m ruff format --check src/core/curated_readiness_corrections.py src/adapters/saved_readiness_corrections.py tests/test_saved_readiness_corrections.py
.venv/Scripts/python.exe -m mypy --platform linux --follow-imports silent src/core/curated_readiness_corrections.py src/adapters/saved_readiness_corrections.py tests/test_saved_readiness_corrections.py
```

These commands execute synthetic teacher fixtures; use the declared fixture
allowance rather than repeatedly running them. [Exact executed commands, budget,
results and incidents](../artifacts/runs/saved-readiness-curation-20261021/RESULTS.md).

## Next extension

Use these arrived provisional targets only on an owned TRAIN candidate through
the existing runtime. Freeze the recurrence protocol before model work, account
all115 available rows and per-context exposures, and retain A/B scores even if
recovery fails. Keep probes/final data and deployed/benchmark state isolated.

### Subsequent executed recovery question

The fixed returning-A study used all115 released rows through the existing
runtime: A17 recovers17controls while B19 succeeds18. This goes beyond curation
correctness, but adds both supervision and optimization. A same-dose old-information
control is required before attributing recovery to new data alone. No general
reliability or circadian advantage follows. [Native evidence/costs](../artifacts/runs/returning-a-curated-corrections-20261021/RESULTS.md).
