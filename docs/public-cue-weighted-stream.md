# Public-cue weighted episodes in the delayed-label stream

The existing `TorchTrainWeightedLearner` accepts `PublicCueRecurrentPolicy` and
an owned tuple of complete TRAIN episodes. It binds each array update to its
exact next episode and calls the explicit class-balanced method. It uses the
existing `AttributionStream`, source/actor forks and original execution budget.

```python
from time import monotonic
from src.adapters.minigrid_public_cue_recurrent_learner import PublicCueRecurrentPolicy
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.adapters.minigrid_learning_pilot import _batch
from src.app.coupled_attribution import AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

source = TorchTrainWeightedLearner(PublicCueRecurrentPolicy(), released_train_episodes)
count = len(released_train_episodes)
stream = AttributionStream(
    source, tuple(_batch(ep) for ep in released_train_episodes), (),
    training_only=True, label_delay=1, checkpoint_updates=(0, count),
    budget=ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=count, max_wall_seconds=90), monotonic
    ),
)
stream.run_until(count)
report = stream.finish()
```

Each chronological episode needs distinct IDs so its row IDs remain globally
unique. The schedule must contain1..512 complete raw148/K7 TRAIN episodes, each
at most16steps. Public pixel/cue admission precedes native construction; neither
reserved probe nor final roles enter this port. Weights use only the current
episode's fixed class-count rule. No episode metadata is invented from arrays.

Snapshots keep both full native heads and validate each head's updates and rows
against the original schedule prefix. Restore admits only a matching healthy
prefix. Forks own their arrays and schedule evidence; sessions retain their own
frozen learned state. Old single-RNN/feature/probe routes keep their contracts.

Public-head prefix counts are computed once from each actual first public frame
at schedule admission. The immutable table has at most513 four-integer rows;
each fork builds its own table. Existing lifetime witnesses still check every
mutable episode/transition field and immutable observation identity before use.
Replacing even an equal table refuses. No TTL is needed: changed source, owner
or policy refuses, and only a new owner admits new data. Native tensor, gradient,
momentum and exact per-head update/row validation remains in every state read.
See [component measurements and limits](weighted-prefix-validation.md).

Public-cue failures bypass the old single-RNN atomic rollback path. One bounded
scalar record retains paid counters and uncertainty; all further train/predict/
session/fork/restore calls refuse without another native operation. If the inner
failure's full preceding heads are verified, its failed snapshot remains
readable. Corrupted source/schedule or unavailable terminal proof refuses snapshot
export; a healthy-looking checkpoint cannot erase the failure. A failed native
attempt is separate from the stream's committed receipt and budget counters.

Qualification:69distinct scripted cases and one tiny native parity case pass on
Windows and Linux (68combined plus8affected after the final failure-bound fix).
A failed second stream attempt preserves the first receipt/budget
charge and source/actor. Native delayed updates, per-head momentum, prefix
restore, owned forks, source/actor/RNG and genuine sessions match existing native
weighted updates exactly. These literal rows establish interoperability, not task
success, retention/recovery, circadian advantage or release readiness.
All32allowed failure counters survive; when no slot remains for a verification
flag, its absence is unverified. Mutating a returned failure cannot change the
retained refusal record. The error-only fix reuses unchanged native parity.

Exact commands/costs/failures/skips:
`artifacts/runs/public-cue-weighted-stream-20261025/RESULTS.md`.
The [class-balanced recurring case](weighted-recurrent-context.md) returned all
128 fits but exhausted its readout allowance. Its unrun endpoints remain unknown;
the prefix optimization does not renew that allowance or recover those results.
