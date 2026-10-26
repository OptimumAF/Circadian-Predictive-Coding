# Fresh A4 TRAIN collection

The existing affine workflow collection port accepts an explicit fixed A4 layout:

```python
from scripts.minigrid_affine_workflow import _collect_recurring_train_sources
from src.core.minigrid_key_door import MiniGridKeyDoorLayout

# guard is the caller's original cumulative wall/RSS/work budget boundary.
records = []
episodes = _collect_recurring_train_sources(
    guard, records, layout=MiniGridKeyDoorLayout("A", branch_row=4)
)
```

This is an internal composition seam, not a new experiment runner. Omitting
layout keeps original G TRAIN collection/identities/nine-row contract. Explicit
input supports only original G or unrotated, unswapped, grey A branch4. Other
layouts/probes/final tasks refuse before callbacks or environment construction.

A4 creates new `minigrid-source-A4-train-blue/red` episodes with `train` role from
birth, seed17/max16 and a fresh public-observation teacher per cue. The port
returns existing296-coordinate public initial/current history episodes, validates
positive terminated1..16row native records/cleanup/provenance/reward agreement,
and retains attempted physical records on failure. The caller must keep the
original cumulative budget and validate preceding records before more work.
Teacher-construction failure now also closes the already-created environment.

Why: a read-only development game exposed a blue turning loop. Learning from
fresh TRAIN coverage is a separate operation; old development/probe_test data
cannot be relabeled or replayed. This collector performs no fitting, learner
readout, promotion or saved-model access. No new dependency/environment variable,
global role, model math or orchestration layer was added. The separate
[A3/A4/A3 adaptation mode](minigrid-a4-adaptation.md) now consumes these fresh
TRAIN records through the existing stream; its qualified results remain scoped
to known TRAIN tasks.

Focused tests: `pytest -q tests/test_a4_train_collection.py`, plus the three
original collector cases in `tests/test_ordinary_fourth_task.py`. Use scoped Ruff
check/format and mypy with `--follow-imports=silent` on changed files. Observe
issue170's consumed engineering/native allowances before running collection.
Source/procedure/result self-review is not independent acceptance. See
[#170](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/170) for
current evidence, costs, preserved failed checks and next action.

The [native collection](reports/a4-train-collection-20261025.md) completed both
cues in nine steps each, yielding18 fresh TRAIN rows with zero learner fitting.
That collection allowance is spent. Learned acquisition/old-task retention and
return remain to be evaluated under a separately frozen small protocol.
