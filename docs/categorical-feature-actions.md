# Categorical actions through the existing native feature learner

`TorchFeaturePolicy.num_classes` binds an explicit native integer2..16, default2.
The existing BP, PC, neutral CPC and active CPC adapters can now train a three-
action classifier using their original native objectives. This is the native
prerequisite for key/door imitation; full R4.2 remains unfinished.

## Use the existing port

```python
import numpy as np
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy

policy = TorchFeaturePolicy("pc", 2, 3, 17, num_classes=3)
learner = TorchFeatureLearner(policy)
features = np.array([[0.3, -0.2], [-0.5, 0.4], [0.1, 0.6]], dtype=np.float64)
labels = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
diagnostic = learner.train_batch(features, labels)
probabilities = learner.predict(features)  # owned float64 shape (3, 3)
saved = learner.snapshot_state()
copy = learner.fork()
copy.restore_state(saved)
```

These are the exact native fixture inputs used in the focused three-class tests.
This usage is a classifier example, with no held-out evaluation or reward policy.
The supported method IDs remain `bp`, `pc`, `neutral_cpc`, `gated_cpc`.
No new dependency, environment variable, encoder or dataset is introduced.

| Policy | Hard training labels | `predict` output |
| --- | --- | --- |
| Default or explicit2 | exact float64 B×1 indices0/1 | owned float64 B×1 class1 probability |
| Explicit K≥3 | exact float64 B×1 integral indices0..K−1 | owned float64 B×K all-class probabilities |

Class count is appended after existing policy arguments, preserving the previous
positional signature. Finite fractional, soft/one-hot, negative or out-of-range
targets are refused before native training. Features keep their existing finite
float64/float32 conversion rules and128-row/16-feature bounds. The fixed hidden
width remains at most32. No action is selected, masked, clipped or repaired here;
the environment evaluator must own legal actions and recurrent-state rules.

## State and objectives

Class count binds native constructors, saved/live metadata, output parameter
shapes, BP momentum and gradient buffers. Snapshot/restore/fork owns complete
native state, including optimizer/traffic/chemistry/clocks/local RNG. Wrong counts
or corrupt geometry refuse atomically; failed staged updates do not publish.
Prediction validates complete state and native logit geometry, returning a detached
NumPy copy. Global Torch/NumPy RNG and caller arrays remain owned by their callers.

Snapshot format1 remains the current typed, owner-local snapshot contract. This
does not establish compatibility with historical pickles/dataclass objects that
lack the new field, durable restart or application budget/actor authority recovery.
No receipt, charge, runtime budget or publication history is renewed by copying.
For the tested D2/H3 geometry, K3 has21 float32 parameters/84 parameter bytes,
versus K2's17/68; K16 has73/292. These are geometry-derived parameter payloads,
excluding optimizer, gradients, chemistry, copies and Python/native runtime.
Observed process memory for the full focused run is recorded separately.

Native BP cross entropy and PC/CPC energy keep their separate diagnostic IDs and
units. The neutral CPC control remains mathematically equal to PC for this path;
active CPC retains its existing mechanics. All rates, inference steps, native
core implementations and original binary objectives are unchanged. Correct
categorical updates establish no learning or circadian advantage.

## Verification and next extension

From the repository root with the existing CPU Torch environment:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_feature_learner.py tests/test_torch_categorical_feature_learner.py tests/test_torch_feature_rehearsal_pilot.py tests/test_line_rollout_pilot.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/torch_feature_learner.py tests/test_torch_categorical_feature_learner.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/torch_feature_learner.py tests/test_torch_categorical_feature_learner.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/torch_feature_learner.py tests/test_torch_categorical_feature_learner.py
```

Repeat the last command with `--platform linux` for the other checked target.
The recorded budgeted worker sets OMP/MKL/OpenBLAS/Torch/interop1 before native
work; reuse its saved evidence when unchanged, rather than rerunning to select
outcomes. [Results](../artifacts/runs/r42-categorical-feature-port-20261016/RESULTS.md)
record exact commands, outcomes, costs, failures and skipped checks.
[ADR-0323](adr/ADR-0323-bind-native-feature-actions-to-explicit-class-count.md)
explains the design.

The native feature evidence above remains scoped to that component.
[Categorical demonstration replay](categorical-trajectory-rehearsal.md) now carries
explicit class counts, whole packets and categorical scores through the existing
stream/rehearsal/continuation path, with isolated final evaluation and original
budgets. Its accepted synthetic case changes no accuracy and is not a real game.
[Actual key/door demonstrations and live policy episodes](key-door-navigation.md)
now execute through these ports, with all320 learned failures retained. Next
trainable recurrent behavior cloning with episode-state resets. Original hidden
cue, A→B→C→A, changed layouts/visuals, old-room
recall, held-out layouts and separately evaluated recurrent RL remain substantive
R4.2 requirements. Binary LineNavigation evaluation remains unchanged.
