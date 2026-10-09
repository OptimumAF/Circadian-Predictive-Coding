# Native learner ports

R3.1 adds one synchronous orchestration path for the existing circadian NumPy
learner and ordinary-gradient `BackpropMLP`. Their equations and native model
files are unchanged. This is a software interface, not a new study or loss.

## Structure and responsibilities

```text
src/core/learner_ports.py        generic native learner and diagnostic contracts
src/app/learner_step.py          one complete wake step using ToyBudgetSession
src/adapters/numpy_learners.py   owned CPC/backprop adapters and model state
tests/test_learner_step.py       opaque non-array inputs and budget boundaries
tests/test_numpy_learner_ports.py native parity, state isolation and refusals
```

Dependencies point inward: adapters -> core; app -> core and the existing app
budget session. Core imports neither app nor infra/adapters. Inputs, predictions
and model snapshots have separate generic types. Orchestration calls the port;
it does not inspect tensors, infer a method from its loss, or normalize diagnostics.

| Adapter | Native diagnostic | Model-state boundary |
|---|---|---|
| `BackpropLearner` | `numpy_binary_bce_preupdate_v1` | `BackpropSnapshot` format 1, complete detached model dictionary |
| `CircadianLearner` | `numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1` | Existing `CircadianNetworkSnapshot` format 2 and native restore validation |

These values describe different native diagnostics. Comparing them does not
establish a common optimization objective or held-out performance.

## Ownership and snapshots

Each adapter copies the supplied model at construction. Later training leaves
that supplied model unchanged. Snapshot and restore copy the full model graph;
backprop preserves all hidden layers, parameters, traffic, counters and legacy
aliases to the first hidden arrays. Restore validates dimensions, complete field
membership, list containers, aliases, array shape/dtype/finiteness and traffic
before replacing owned state. CPC keeps its existing replay/RNG/configuration/
topology validation. Neither adapter adds a new learning rule or sleep operation.

Snapshots are trusted local in-memory model state. Their nested state remains
mutable; later edits to a returned snapshot do not mutate the learner. They are
not serialized checkpoints, untrusted-data decoders or source/build certificates.
Learning rates and inference settings remain adapter-owned training policy;
restoring model state does not select or change that policy. Actor concurrency,
versioned promotion and experience permissions remain separate R3 tasks.

Why this: copying the entire state graph preserves aliases and model channels
that a parameter-only copy would omit. Reuse CPC's existing state contract;
the ordinary head needs an outer state adapter without changing old checkpoints.

## Budget and failure semantics

`update_learner` calls `before_update`, the native complete update, `record_update`
and the existing post-update clock/RSS check. A pre-update refusal executes no
native update. A post-update stop retains the completed work and model state;
it does not pretend that the update never occurred. Native training errors
propagate and do not count as successfully returned updates.

The step supports wake-call and wall-time boundaries, and RSS when the caller
has attached the existing sampler. It rejects unattached RSS, width and replay
limits rather than treating them as enforced. Sampled RSS and complete-boundary
wall checks are soft observations, not hard allocation or preemption guarantees.
Replay quotas, sleep work, latent-iteration matching, capacity controls and live
serving contention need explicit later composition. No final-role scoring,
dataset construction, label-arrival authorization, resource preflight, promotion,
privacy decision, external service or environment variable is introduced here.

## Example and verification

After the normal dependency installation, use the existing models and rates:

```python
from time import monotonic
import numpy as np
from src.adapters.numpy_learners import BackpropLearner
from src.app.learner_step import update_learner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP

learner = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic)
features = np.array([[0.3, -0.2], [-0.5, 0.4]])
targets = np.array([[1.0], [0.0]])
state = learner.snapshot_state()
diagnostic = update_learner(learner, features, targets, budget)
learner.restore_state(state)
```

Run `python -m pytest -q tests/test_numpy_learner_ports.py tests/test_learner_step.py`.
The tests compare native diagnostics, predictions and entire model state, including
continued training, and exercise corruption, incompatible state and resource refusal.
A text/dictionary fixture demonstrates that orchestration assumes no array layout.
An extension implements the core port in another outer adapter and supplies its
own native diagnostic and complete model-state tests before using the same step.


## Historical dependency status at 2026-10-06 closeout

R0.3/G0 has reopened for the repair PR remote CI diagnosis. R3.1 remains unchecked
although its new software ports pass local targeted tests and both type targets.
The existing owner retains that diagnosis; no dependent experiment or additional
port implementation is launched. Earlier baseline/document receipts keep their
recorded scope. See [the active roadmap](../RESEARCH_ROADMAP.md).


## Current acceptance — 2026-10-07

R0.3/G0's engineering hold is resolved by the source-bound inventory/count/control/static/storage completion R0.3i. The existing R3.1 ports are accepted from their unchanged complete source/config/package evidence and retained85 passing cases; no new tests or training ran. Later four presentation files match their separately accepted full626-file type snapshot. The native interface, ownership and budget limitations above remain unchanged. Next: separately budget deterministic R3.2 arrival/clock/permission contracts. Actor/promotion/scientific/native enforcement and final-role authorization remain unfinished. See the active roadmap and artifacts/runs/r03-committed-inventory-successor-20261007/.
