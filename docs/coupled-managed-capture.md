# Complete coupled managed-state capture

`capture_coupled_managed_composite` returns a detached observation of a paused,
quiescent managed owner using exact `CoupledLearner` PC or CPC models. It includes
current and previous serving bundles, pending promotions and checkpoints, retired
and current candidates, inbox history, consent, retention and resource records.

```python
from src.adapters.coupled_composite_capture import capture_coupled_managed_composite
from src.core.managed_composite_state import CompositeCaptureLimits
from src.core.managed_lifecycle_state import LifecycleCaptureLimits

# owner and gate are the existing managed owner and its ServingPriorityGate.
gate.pause()
capture = capture_coupled_managed_composite(
    owner,
    limits=CompositeCaptureLimits(
        records=LifecycleCaptureLimits(max_records=64, max_identifier_bytes=128),
        max_nodes=8192,
        max_depth=64,
        max_array_bytes=65536,
        max_dimension=32,
    ),
)
state = capture.state
```

In application code, retain the sharing gate when constructing the owner and
call that gate's public `pause()` method. Capture requires the existing ownership
registry and configured lifecycle measurement ports; it does not create them.
Busy holders, unsupported values, invalid schemas, overdue retention, exhausted
resources or copy limits raise without publishing a partial result. A failed
capture after reservation keeps the attempted copy charge spent.

## Representation and accounting

Each learner is a `SourceRecord` containing all five stored learner fields. Its
model is another exact record containing native class identity, explicit input
and hidden dimensions, precisions, parameters and every CPC config, chemistry,
plasticity and wake-count field. Pending checkpoint states retain the existing
typed `CoupledSnapshot`. Validation checks stored dimensions against array shapes
and validates complete update settings and CPC gate consistency.

Arrays are copied once per original object across the graph. Aliases, float64
values, C/F layout and read-only flags are preserved. Distinct overlapping views,
cycles, opaque values, added fields and nonfinite arrays are rejected. This
preserves payload identity within the detached graph; it does not reconstruct
live model identity.

Why two byte bounds: `max_array_bytes` independently bounds native and raw arrays
in the complete graph. The original lifecycle copy budget charges its retained
**raw** bytes once before graph deepcopy. Learned parameters and temporary Python
allocations are outside that logical raw ledger. The unchanged coordinator
preflights the graph before bounded temporary pending-checkpoint integrity
encoding; corrupt checkpoints still fail before a new charge. Both copy passes
use the same memo, refreshing only final clock and resource observations.

Capture invokes no native model methods, native constructors, training, prediction,
snapshot/restore operations or native checkpoint codec encoding. `capture.authority`
holds the original live references separately. Those references confer no new
permission, and detached state grants no restoration or process restart authority.
There is no disk IO, new configuration or dependency in this API.

## Checks and extension

Focused behavioral and regression command:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_coupled_managed_composite_capture.py tests/test_managed_composite_capture.py tests/test_coupled_checkpoint_codec.py --deselect=tests/test_coupled_checkpoint_codec.py::test_should_restore_native_semantics_without_any_codec_model_construction --deselect=tests/test_coupled_checkpoint_codec.py::test_should_continue_complete_native_state_exactly_after_restore
```

The retained validation controller additionally enforces constructor, thread and
file budgets; its exact command and outcomes are in
`artifacts/runs/coupled-composite-capture-20261015/RESULTS.md`.
For a new native family, add an explicit adapter schema and projection plus
refusal, alias and admission tests. Keep the existing app coordinator and original
authority boundaries. A [bounded complete data format](coupled-composite-codec.md)
preserves this graph within its explicit domain; restoration remains unfinished.
See [ADR-0301](adr/ADR-0301-complete-coupled-managed-capture.md).
