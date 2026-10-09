# Managed record capture

`capture_managed_records(owner, limits=...)` captures the complete consolidation
and lifecycle records during one original owner lease interval. It returns
detached immutable metadata and 59 original live reference slots. It executes no
model operation, clock read, measurement callback, cleanup or worker operation.

## Modules and boundaries

```text
src/core/managed_record_state.py         paired records and pure validation
src/app/managed_record_capture.py        original source guards and leased capture
tests/test_managed_record_capture.py     actual zero-update and synthetic controls
```

The app depends on core and the existing lifecycle capture. Core does not import
app, adapters or infra. No dependency, environment variable or configuration
format was added. Existing standalone capture signatures remain unchanged.

## Complete observation

| Record | Preserved content |
| --- | --- |
| Runtime observation | Base and current serving versions, learner version, revision, consolidation limit, stopped/retired/ready flags, consumed budget updates, inbox completed updates/last observed tick/stopped flag, current candidate enrollment |
| Consolidation cursor | All ten original fields, consumed attempt IDs, every committed receipt and native diagnostic; failed attempt gaps remain gaps |
| Lifecycle metadata | Every current catalog, consent, provenance, optout, revocation, declaration clock, policy, quota, retention/fault/auxiliary epoch, driver, copy charge and retained enrollment field |
| Authority | The original 48 lifecycle slots plus 11 runtime/inbox/clock/budget/lineage/gate slots; every value remains the original object |

Exact source guards enumerate all 15 current runtime and 17 inbox private
fields. Unknown or missing fields are refused. Current lifecycle guards retain
their complete 67-field contract. Metadata validation checks exact record types,
bounded counts and UTF8 strings, matching cursor/runtime revisions and flags,
original reference aliases, candidate enrollment, consumed budget and required
uncertain stop relationships before copying. Aggregate capacity counts the owner
observation, eight history collections, and all their entries together.

## Lease and alias behavior

Original nonblocking leases are acquired in the existing lifecycle order:
driver operation/state, manager, registry, every live enrolled holder, retention
time, optional copy budget, and sharing. Strong references pin live weak holders
until detachment ends. Contention and reentrance refuse without resetting an
allowance or mutating authority. The driver state lock is reentrant by design;
the operation lock still forbids nested capture.

Why this: calling standalone consolidation capture while its candidate lease is
already held would fail. The new internal cursor reader uses that held lease.
Promotable serving version is read from its current slot under the held original
actor gate; its public version property would try to reacquire that same gate.

Both metadata components are deepcopied together exactly once, preserving their
immutable aliases. Live references never enter the copy. The current serving
version can differ from the candidate base. Lifecycle and inbox observations can
lag each other; capture preserves both and their original clock identity without
probing time.

Managed retention currently refuses arbitrary consolidation transforms. The
refusal increments the runtime revision but consumes no transform ID and invokes
no transform. Tests preserve that negative result. Richer receipt histories and
closed/fault states are synthetic metadata controls, not successful managed
transforms or executed recovery.

## Constructor-only example

```python
from unittest.mock import patch
from src.adapters.numpy_learners import BackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_record_capture import capture_managed_records
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits

class WallClock:
    blocked = False
    def __call__(self):
        assert not self.blocked, "capture read the clock"
        return 0.0

def forbidden(*args, **kwargs):
    raise AssertionError("capture invoked an original port")

wall = WallClock()
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), wall)
runtime = ActorShadowRuntime(
    BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.01),
    actor_version="actor", candidate_version="candidate",
    clock=LogicalClock(0), budget=budget,
)
owner = ManagedExperienceOwner(
    ResourceSharedRuntime(runtime, ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=forbidden)),
    limits=LifecycleLimits(4, 4),
)
lifecycle = ManagedDataLifecycle(
    owner, policy=DataRetentionPolicy(4096, 20, PayloadOwnershipLimits(8, 12)),
    measure_payload_bytes=forbidden,
    native_footprint=lambda value: ReplayPayloadErasure(0, 0, 0),
    native_erase=forbidden,
)
wall.blocked = True
with patch.object(LogicalClock, "now", forbidden), patch.object(BackpropLearner, "snapshot_state", forbidden):
    captured = capture_managed_records(owner, limits=LifecycleCaptureLimits(64, 128))
assert captured.metadata.owner.revision == runtime._revision
assert captured.metadata.owner.budget_updates == budget.updates_completed == 0
assert captured.metadata.consolidation.attempted_ids == ()
assert len(captured.authority) == 59
assert {item.path: item.value for item in captured.authority}["runtime.root"] is runtime
```

## Local verification and next extension

Declare a fresh bounded allowance for constructor/admission fixtures and use an
unused pytest directory. Prior fixture and worker scopes remain spent.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_managed_record_capture.py tests/test_lifecycle_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=<new-directory>
.\.venv\Scripts\python.exe -B -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/app/actor_shadow.py src/app/managed_lifecycle_capture.py src/core/managed_record_state.py src/app/managed_record_capture.py tests/test_managed_record_capture.py
```

Next, implement R3.5b2e4c2: complete explicit paired bytes through CheckpointCodec,
independently bound to the original paired observation, policies, source, content
and authority. Preflight all wire/alias/relationship bounds before construction.
Metadata agreement does not prove source provenance or authorize restore.
Full paired byte, native/inbox/composite/live/disk/model/coordinator loss,
scientific and human criteria remain open.
