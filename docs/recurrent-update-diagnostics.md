# Observe actual recurrent TRAIN updates

`summarize_recurrent_update(before, after, rows=...)` in
`src.adapters.torch_recurrent_gradient_diagnostics` reads two consecutive owned
native snapshots. It returns parameter, gradient, momentum and actual/nominal
SGD step norms for all five parameter blocks, including the recurrent matrix.
It reports the maximum absolute SGD residual without imposing a success cutoff.
No learner, optimizer, forward pass, backward pass, random draw or update runs.
Validation copies and scalar reductions are additional diagnostic work.

`summarize_public_cue_update` composes the same observation for one changed
public-cue head. Both full snapshots must have the same policy, no failed work,
consecutive update/row clocks, exactly one changed head, and an unchanged other
head. It returns the selected head index and its local clocks. Unsupported or
corrupted state refuses before a summary is returned. Results contain detached
JSON scalars, not weights, gradients, activation arrays or resumable state.

Use the existing delayed stream's `observe_checkpoint` callback. For a fixed
four-update TRAIN-only diagnostic, select checkpoints `(0, 1, 2, 3, 4)` and keep
one previous callback snapshot:

```python
from src.adapters.torch_recurrent_gradient_diagnostics import summarize_public_cue_update

previous = []
records = []  # Four scalar records, bounded by the declared case.

def observe(updates, state, batch):
    if updates:
        records.append(summarize_public_cue_update(
            previous[0], state, rows=len(batch.features)))
    previous[:] = [state]

# Pass observe_checkpoint=observe to the existing AttributionStream.
# Its original budget, delayed labels, actor and source guards still apply.
```

The caller owns actual observed operation provenance, TRAIN release and budgets.
Snapshots do not encode objective/data role; consecutive metadata alone cannot
prove how weights were produced. A nonzero residual can expose an inconsistent
delta, but a small residual does not establish useful learning, acquisition,
retention, recovery or the cause of an existing native failure. This is an
explicit readout, never a training/promotion/controller/probe-selection rule.
Final evaluation remains unavailable, and no labels enter the summary API.

Existing whole-pair mean-CE gradient and activation diagnostics remain available;
their roles differ from observing an actual committed step. Default learners,
objectives and stream behavior are unchanged.

[Scoped checks and costs](reports/recurrent-update-norms-20261031.md),
[issue #374](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/374).
