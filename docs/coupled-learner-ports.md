# Coupled learners in the adaptation runtime (R3.8a)

Run the complete local delayed-label stream:

```powershell
python -m scripts.run_native_stream_demo --scenario coupled-neutral --seed 23
```

Both coupled PC and its neutral CPC control execute four native updates through
the existing `ActorShadowRuntime`. Their serving actors stay stable. Labels enter
training two logical ticks after observations. The JSON records experiment and
model format versions, actual native diagnostics and CPC chemistry/gates/counts.
It uses the same fixed unlabeled probe batch as the original
[runtime example](native-stream-demo.md), not a strict online evaluation split.
No final data, scoring, promotion or comparative learning claim is involved.

## Native adapter

`CoupledLearner` implements the existing `NativeLearner` and `ForkableLearner`
interfaces for exact native `CoupledPredictiveCodingNetwork` and
`CoupledCircadianPredictiveCodingNetwork` instances. Construction and `fork()`
reconstruct independent owned native state through its public snapshot/restore
boundary. The source must remain quiescent during copying. Unsupported custom
subclasses are refused rather than treated as certified native implementations.

The three training settings are fixed per adapter: `learning_rate`,
`inference_steps` and `inference_learning_rate`. Rates are positive finite native
numbers excluding booleans; steps are a native integer in `[1,10000]`. Forking
preserves these settings. Restoring model state does not change training policy.
The adapter returns native energy as `TrainingDiagnostic` with the existing
`numpy_coupled_binary_energy_v1` definition; no diagnostic is renormalized.

```python
import numpy as np
from src.adapters.coupled_learners import CoupledLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork

x = np.array([[0.3, -0.2], [-0.5, 0.4]], dtype=np.float64)
y = np.array([[1.0], [0.0]], dtype=np.float64)
learner = CoupledLearner(CoupledCircadianPredictiveCodingNetwork(2, (3, 2), seed=23),
                         learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1)
learner.train_batch(x, y)
saved = learner.snapshot_state()
fork = learner.fork()
first = learner.train_batch(x, y)
fork.restore_state(saved)
continued = fork.train_batch(x, y)
assert first == continued
np.testing.assert_array_equal(learner.predict(x), fork.predict(x))
assert fork.snapshot_state().wake_state.wake_updates == 2
```

## Complete, versioned model state

`CoupledSnapshot` format 1 contains:

- All hidden/output parameter arrays and the compatible precision tuple.
- For PC, `wake_state=None`.
- For CPC, a `CoupledWakeState` containing all four chemistry-policy scalars,
  chemistry and plasticity vectors, and the completed native wake count.

Input/hidden/output dimensions are encoded by parameter shapes and must match
the receiving model's topology. Precisions and CPC policy must match its current
definition. Restore refuses cross-family snapshots, unsupported format, malformed
types/shapes, nonfinite arrays, negative chemistry, nonnative/negative counters
and gates inconsistent with the recorded chemistry/policy. A zero-wake CPC
snapshot requires zero chemistry and unit gates. Every mutable array is copied
and every check finishes before any committed state changes.

Why this: parameter-only restore would lose chemistry or reuse a gate from a
different wake, changing the next native update. Full snapshot continuation
therefore preserves chemistry, gates and counters explicitly. These operations
are synchronous and require one owner; they do not implement serving locks.
Inherited `restore_parameters()` retains its narrower parameter-only behavior.

These are trusted local in-memory model snapshots, not a file decoder or durable
restart protocol. Restoring a model does **not** restore or renew runtime budgets,
experience receipts, permissions, logical clocks or serving versions. Runtime
state remains owned by the existing application layer. The integration tests
distinguish native wakes from runtime updates: a pre-trained CPC with count 1
continues to count 2 after one admitted runtime update, while its actor stays at
the original model state.

## Boundaries and tests

```text
src/core/coupled_predictive_coding.py            typed snapshot and PC validation
src/core/coupled_circadian_predictive_coding.py  complete CPC continuation state
src/adapters/coupled_learners.py                owned native ports and fixed policy
src/adapters/native_stream_demo.py              existing runtime composition
scripts/run_native_stream_demo.py               scenario selection and JSON
tests/test_coupled_model_state.py               complete CPC restore/refusals
tests/test_coupled_learner_ports.py              independent adapter/native parity
tests/test_coupled_runtime.py                    actual arrival/actor/budget/CLI
```

Core imports no adapter or application module. The adapter depends inward on
native models and existing ports; no port signature changed. There are no new
dependencies, environment variables, remote services or downloads.

```powershell
python -m pytest -q tests/test_coupled_model_state.py tests/test_coupled_learner_ports.py tests/test_coupled_runtime.py
```

The tests compare complete native and adapter state at 1/2/4 layers, including
active CPC gates, repeated updates, restore, fork and continuation. They verify
source/snapshot/output ownership, refusal atomicity, future/final-label exclusion,
stable actors and exact-once budget charging. The CLI uses neutral CPC: matching
predictions are a software control, not a benefit measurement.

Replay/privacy lifecycle adapters, sleep, structure changes, durable checkpoints,
coordinator-loss recovery and scientific comparisons retain separate gates.
