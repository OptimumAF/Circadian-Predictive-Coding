# Stable actor and detached shadow learner

R3.3 serves one fixed actor while an independently owned candidate learns from
arrived training events or consolidates a detached state snapshot. Both existing
NumPy learners use the same runtime, their own native diagnostics and unchanged
training equations. This increment adds no promotion policy or performance claim.

## Structure and ownership

```text
src/core/actor_ports.py        owned-fork port and versioned result records
src/app/actor_shadow.py        stable serving, exclusive candidate operations
tests/test_actor_shadow.py    concurrency, ownership, budgets and native parity
docs/actor-shadow-runtime.md  supported use and extension boundaries
docs/adr/ADR-0200-own-stable-actor-and-shadow-forks.md
```

`ForkableLearner` extends the existing `NativeLearner` port with `fork()`: return
an independent complete model and the same native training policy. It does not
change the old port. NumPy adapters reuse their existing owned constructors to
copy the entire model graph, including topology, aliases, traffic, chemistry,
RNG/replay channels and configuration. They retain learning and inference rates.

Construction requires a trusted source that remains quiescent while it is forked.
The runtime owns two distinct forks and exposes neither mutable learner handle.
An identity check refuses a fork returning its source or the same object for
actor and candidate. A malicious custom fork that shares nested model state is
outside this trusted port contract; implementors must prove full state ownership
with native tests. This is not arbitrary Python object graph certification.

`StableActor` returns `VersionedPrediction(actor_version, prediction)` and detached
`VersionedState`. It serializes its private learner's prediction/snapshot calls
with one lock and copies input/output payloads. The candidate uses a different
lock. Training and consolidation never acquire the actor lock or mutate its model.
Returned arrays, payloads and snapshots can be edited without changing either
owned model. The supplied source can also change after construction independently.

Version names are nonempty distinct actor/candidate identifiers. The actor's
version remains fixed throughout this runtime's lifetime. These names identify
local outputs, not parameter hashes, global release permission or promotion
certificates. Candidate snapshots report base actor version, candidate version,
completed wake count, consolidation attempts and successful consolidation count.

## Arrived training and exclusive candidate operations

`record_experience`, `record_label` and `train_ready` reuse R3.2's `ExperienceInbox`.
Training still requires a train-role source with explicit permission and a matching
train-role label; both logical arrival ticks must be reached. Original duplicate,
clock, label-first, stable ordering and version/ownership rules remain intact.
No held-out scoring or final-label release is introduced.

Candidate registration, training, snapshots and consolidation have one exclusive
nonblocking gate. Concurrent or nested candidate operations raise a useful `busy`
error; the caller may defer them. They do not queue, block serving or create an
implicit background thread. Actor prediction is available throughout candidate
work. Separate actor read calls are serialized; no latency fairness, p50/p95
guarantee or serving contention scheduler is claimed before R3.5.

## Detached consolidation and failures

`consolidate(event_id, operation)` gives trusted local code a detached candidate
snapshot. It must return `ConsolidatedState(state, native_diagnostic)`. Copy its
returned state again, restore it through the native learner port, then retain an
`AppliedConsolidation` receipt. The transform receives no mutable actor/candidate
learner. An outer adapter can reconstruct its native model, run an existing sleep
operation and return that complete snapshot. CPC tests execute a real component
sleep with the existing component-test fixed downscale factor0.8, assert actual
weights change, and match the complete direct-native state and sleep clock.
Ordinary-gradient learners are not given an invented sleep algorithm.

Lifetime event IDs and `max_consolidations` bound attempted transforms, including
failed copies/transforms and malformed results. Mark an attempt before trusted
code runs; never retry it under the same ID. A pre-budget refusal consumes neither
ID nor quota. Transform failures leave the private candidate untouched and
propagate; a native restore failure may have partly changed it, so stop the
candidate. Actor serving and committed receipts remain available. No automatic
rollback is claimed; complete rollback is R3.4 work.

Wake failures propagate the inbox's read-only `stopped` status to the candidate
owner. Native cancellation/BaseException also stops the inbox after uncertain
mutation; exceptions retain their original type. The existing budget-stop phase
distinction remains: pre-update refusal stays pending, a native budget-typed
partial failure stops it, and completed late-stop work retains its identity.

## Resource limits

Use one exclusively owned `ToyBudgetSession` for the runtime. Wake-call quota and
wall/RSS checks use the existing shared native step without double counting.
Construction and consolidation use existing before-sleep/post-completion wall
and sampled RSS checks; consolidations have their own finite attempt quota.
Completed wake/consolidation receipts survive a post-check stop and cannot replay.
Count bounds include consumed IDs, so retaining history does not grow indefinitely
within the declared record limits. Payload byte size is a separate concern.

These are synchronous complete-boundary soft limits. A native call/transform may
overrun before its post-check; there is no hard preemption or allocation cap.
Reject width/replay limits and unattached RSS before constructing forks rather
than treating unsupported limits as enforced. A consolidation callback's inner
work must be configured prospectively; it does not obtain replay/data/scientific
authority from receiving a snapshot. Broad live resource sharing and supported
checkpoint cursors remain R3.5 work.

## Example

```python
from time import monotonic
import numpy as np
from src.adapters.numpy_learners import BackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock

features = np.array([[0.3, -0.2], [-0.5, 0.4]])
targets = np.array([[1.0], [0.0]])
source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
clock = LogicalClock()
runtime = ActorShadowRuntime(
    source, actor_version="actor-0", candidate_version="candidate-0", clock=clock,
    budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic),
    max_experiences=4, max_consolidations=0,
)
before = runtime.actor.predict(features)
runtime.record_experience(Experience("s1", "e1", 1, "actor-0", features, "train",
                                     ExperiencePermissions(training=True)))
runtime.record_label(LabelArrival("label-1", "s1", "e1", 3, "actor-0", targets))
clock.advance_to(2)
assert runtime.train_ready() == ()
clock.advance_to(3)
runtime.train_ready()
after = runtime.actor.predict(features)
assert before.actor_version == after.actor_version == "actor-0"
np.testing.assert_array_equal(before.prediction, after.prediction)
candidate = runtime.candidate_snapshot()  # no promotion has occurred
```

## Verification and extension

```powershell
python -m pytest -q tests/test_actor_shadow.py tests/test_experience_inbox.py tests/test_numpy_learner_ports.py
python -m ruff check src tests scripts
python -m ruff format --check src/core/actor_ports.py src/app/actor_shadow.py src/app/experience_inbox.py src/adapters/numpy_learners.py tests/test_actor_shadow.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

No dependency, environment variable, remote service or new scientific experiment
is required. Tests pause partial candidate wake/restore and real detached native
sleep with Events; concurrent read futures finish while candidate work is paused.
They use bounded waits and no timing sleeps. Native fixtures preserve seed23,
rates0.03/0.2, width4 and inference2; comparison is exact engineering parity.

App depends on core contracts and existing app inbox/budget; native outer adapters
depend on core. No infra dependency is added. Extend with another conforming owned
fork adapter and its complete-state/policy tests. R3.4 must add acceptance policy,
atomic full actor/cache/state swaps and rollback before a candidate can be served;
R3.5 handles contention. Preserve all arrival/role/duplicate/failure gates.


## Compatible R3.4b serving composition

Default `StableActor` behavior remains fixed. Optionally pass a `PromotableActor` as `actor=` at construction, with matching initial actor version. Candidate base version is now explicitly frozen; promotion never relabels old inbox work. `with_candidate_snapshot(operation)` holds the same exclusive candidate gate through a trusted callback over detached state. Serving takes no candidate gate. The serving controller supplies matched guard issuance/exact generation binding/atomic complete rollback; see [serving promotion](serving-promotion.md). After promotion start a fresh candidate runtime from an independent full new actor copy and matching native policy; the previous candidate remains tied to its old base.


## Compatible R3.5a admission

`train_ready()` retains full default drain behavior. Optional `max_updates=` bounds a poll; `before_each_update=` leases an exact-bool context around each eligible native update before copying its payload. Denial retains remaining ready pairs and committed receipts, and original failure poisoning/late-stop identity rules remain. Compose the runtime with [cooperative resource sharing](resource-sharing.md) for bounded actual serving calls and per-update priority/pause/resource/quota checks. These gates do not persist/restore complete cursor state or establish live latency; R3.5b/c remain required.
