# Structural matched backpropagation

`StructuralScaledMatchedBackpropLearner` composes the existing raw-coordinate
matched BP learner with the existing rooted structural snapshot. It allows BP
to continue after compatible split, merge and prune state changes without
loosening the original fixed-configuration port.

```python
from src.adapters.structural_scaled_matched_backprop_learner import (
    StructuralScaledMatchedBackpropLearner,
)

learner = StructuralScaledMatchedBackpropLearner(initial_raw_state, learning_rate=0.01)
learner.train_batch(features, targets)
snapshot = learner.snapshot_state()
owned_copy = learner.fork()
owned_copy.restore_state(snapshot)
```

`initial_raw_state` is a PC-only `CoupledScalingState`. Training and prediction
delegate to `ScaledMatchedBackpropLearner`: the original feedforward BCE,
representation, raw/effective learning-rate factors and float64 updates remain.
There is no latent inference, chemistry, new objective or optimizer state.

## State and ownership

Snapshots are complete `CoupledStructuralState` records: original full config,
current raw parameters/config and the rooted transition chain (at most 64
transitions). Restore
and `build_from_state` first validate and construct an owned strict BP learner,
then publish one replacement. Earlier valid histories can be restored for
rollback. Forks and caller-owned snapshots do not share mutable arrays.

The adapter rejects CPC policy/wake arrays, incompatible origins, malformed or
unrooted transitions. The learning rate belongs to the owning adapter and is
preserved when building/forking; snapshots grant no training policy or runtime
authority. The existing fixed-config BP learner still refuses changed configs.

Compatible PC-only tensor data do not prove which algorithm or operation
produced them. Structural history describes layout compatibility, not provenance.
The port neither executes transforms nor chooses pruning/growth policies;
existing application operations own those decisions and admission gates.
Restoring a learner never renews budgets, clocks, consent or spent receipts.

## Existing stream example

```powershell
python -m scripts.run_stream_rehearsal_pilot --describe --scenario capacity-bp
python -m scripts.run_stream_rehearsal_pilot --run --scenario capacity-bp
python -m pytest -q tests/test_structural_scaled_matched_backprop_learner.py tests/test_structural_bp_capacity.py
```

The fixed engineering recipe adds only two BP arms to the retained capacity
example: fixed final width and scheduled split. It reuses the existing stream,
per-arm runner, capacity helper and final scoring, with seed 23, four two-row
batches, width 2→4 and the original raw factors. The native budget is eight
updates, sixteen row exposures, twelve score forwards and two split calls.
CPU1/GPU0, 60 seconds, 256 MiB RSS and 2 MiB output limits apply. Declare a fresh local envelope
before another execution. No dependencies or environment variables are added.

Affected-scope static checks (include the two new test modules when editing them):

```powershell
python -m ruff check src/adapters/structural_scaled_matched_backprop_learner.py src/adapters/capacity_pilot.py scripts/run_stream_rehearsal_pilot.py
python -m ruff format --check src/adapters/structural_scaled_matched_backprop_learner.py src/adapters/capacity_pilot.py scripts/run_stream_rehearsal_pilot.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/structural_scaled_matched_backprop_learner.py src/adapters/capacity_pilot.py scripts/run_stream_rehearsal_pilot.py
```

Repeat the last check with `--platform linux` for that target. These scoped
commands do not establish repository-wide or release readiness.

The original six-arm PC/CPC descriptor and execution stay unchanged. Historical
scores are read only in post-run analysis, never by the producer or training.
Final data remain unavailable to training, transforms, selection and tuning.
Stored-array costs differ from transient allocations/RSS; duplicate units and
width-dependent factors prevent equal-compute/effective-capacity conclusions.
The small fixture already has perfect initial B accuracy, so it is not evidence
of fresh-task acquisition or general adaptation.

The original R2.6 comparison criteria are now independently accepted in the
[full capacity comparison](capacity-comparison.md). Its 23 frozen cohorts use
this port and the existing runner, report actual costs and outcomes, and retain
the rejected prunes and zero-eligible renewal results. This does not establish
scientific benefit. The smaller qualification above retains its original scope;
preserve native baseline math and all failed/negative evidence when extending it.
