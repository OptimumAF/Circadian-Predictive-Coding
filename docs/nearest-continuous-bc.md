# Bounded nearest-observation continuous BC

This control stores released observation/action pairs and retrieves the physical
action of the nearest stored observation. It uses frozen TRAIN population
statistics, squared L2 distance, and the earliest stored row when computed
distances tie. It has no optimizer, action interpolation, refitting or silent
eviction. A successful replay from a demonstration's initial state establishes
one-trajectory reconstruction, not generalization or a strong BC benchmark.

## Files and boundaries

```text
src/core/nearest_continuous_bc.py           Pure retrieval and work counts
src/adapters/numpy_nearest_continuous_bc.py Owned bounded learner and snapshots
scripts/maniskill_continuous_smoke.py       Existing frozen native actor boundary
tests/test_nearest_continuous_bc.py         Independent math/state/fake IO checks
```

The core takes exact finite float64 query and prototype matrices, each1..256
rows, using the existing continuous spec (1..1024features,1..16actions). Prototype
coordinates are already normalized; actions retain declared physical units and
bounds. `NearestPrediction` returns owned actions, selected indices, candidate
count `B*P` and coordinate count `B*P*D`. Per-query `(P,D)` temporaries avoid a
`(B,P,D)` allocation. Nonfinite/unrepresentable distances are refused. Ordinary
float64 rounding/underflow may produce computed ties; there is no metric repair.
The core grants no data-role, fitting, simulator or persistence authority.

## Usage and ownership

```python
import numpy as np

from src.adapters.numpy_continuous_bc import fit_training_standardization
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy, NumpyNearestContinuousBCLearner,
)

# episode must be a complete, released TRAIN ContinuousEpisode.
statistics = fit_training_standardization(episode)
model = NumpyNearestContinuousBCLearner(NearestContinuousBCPolicy(
    episode.spec, statistics, max_prototypes=256,
))
diagnostic = model.train_episode(episode)
snapshot = model.snapshot_state()
child = model.fork()
model.restore_state(snapshot)
query = np.asarray([episode.transitions[0].observation], dtype=np.float64)
prediction = model.predict_with_work(query)
actions = prediction.actions
```

Statistics are mandatory, frozen, dimension-matching and owned. The policy also
owns the action spec, capacity1..256 and training role. Episode updates require
the matching released role/spec before publication; raw batch callers retain
release responsibility. Separate probe-training learners remain separate, and
final evaluation cannot be an episode update. No fitting happens inside this
learner. Appending once normalizes the new observations and stores them with
physical labels; every retained row stays until explicit state replacement.
Over-capacity updates refuse atomically without eviction. Conflicting duplicate
observations retain the earliest-row decision rule.

Full snapshots contain owned normalized prototype features, physical actions,
policy/statistics and counters. Empty initial arrays are valid state but cannot
predict. Examples equal retained rows; completed updates count successful append
publications, not gradient steps. The diagnostic
`continuous_nearest_bc_added_pairs_v1` reports rows added, not an optimizer loss.
Fork/restore validate complete state and refuse different policy/configuration.

## Existing frozen robot path

The existing `main(learner=model)` accepts this exact learner type alongside the
MLP, with a separate exact typed snapshot comparison. Empty nearest models and
non-TRAIN policies are refused before scene creation. Original/private prototype
arrays, statistics, policy and counters are checked around control and cleanup;
the MLP parameter/momentum checks remain. Native traces still contain actual
raw observations and float32 actions. The default zero-action CLI is unchanged.

Receipts distinguish source append publications from optimizer updates, stored
prototype rows/numeric bytes, prediction attempts/returns, pre-call candidate/
coordinate upper bounds, completed work and selected rows. An unreturned call
may have partial distance work; upper/completed counts do not relabel it zero.
Use separately frozen native budgets. Existing evidence is exclusive and refuses
overwrite. No new runner, persistence codec or simulator adapter is introduced.

## Executed evidence and limits

683 focused WSL cases (79new+604prior) and79 Windows cases pass; scoped lint,
formatting and3production/2native-helper type checks pass. Tests include a
hand oracle distinguishing L2 from L1, normalized distances reversing raw choices,
ties, bounds/overflow/capacity, ownership/corruption/roles and actual fake-native
commands/full state/failure costs. Independent source and actual review pass.

One fixed TRAINseed17 native case fit statistics and appended all29 qualified
pairs once. It selected0..28, reproduced every teacher transition exactly and
completed29steps with reward1/final success/terminatedtrue/truncatedfalse. No SGD
update, probe/final read, refit or promotion. This proves same-initial-state
trajectory reconstruction only; stronger baseline quality and adaptation remain
open. All prior negative MLP/probe/teacher/CPC results remain unchanged.

Stored prototypes10672numericB plus672numericB statistics, NPZ11188B. Runtime
29predictions928raw outputB,841distance candidates35322coordinates,29normalizer
rows and29controls145known physics; append normalization adds29rows. Reset
physics and whole-lifetime tree memory remain unknown. Memory/capacity/compute
differ from MLP; there is no equal-parameter/update-count/compute or speed claim.
[Protocol, exact commands, costs, independent review and next action](../artifacts/runs/continuous-nearest-bc-20261019/RESULTS.md).
