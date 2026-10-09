# Complete managed lifecycle state contract

This component defines complete typed records and original reference relationships
for future lifecycle capture. It validates and detaches supplied metadata. It
does **not** acquire an atomic live capture, encode durable bytes, restore an
owner, renew a clock or release a spent allowance. Those acceptance gates remain
open in R3.5b2e4b/R3.5b2e4b2 and their parents.

## Structure

```text
src/core/managed_lifecycle_state.py       complete immutable records and reference slots
src/core/managed_lifecycle_validation.py  complete record/bound/relationship validation
src/app/managed_lifecycle_schema.py      exact native field guard and record detachment
tests/test_managed_lifecycle_state.py    full preservation/refusal/source controls
docs/adr/ADR-0229-model-complete-lifecycle-state-and-original-references.md
```

The core depends only on existing inner record definitions. The application
module guards supported exact owner classes and all 66 current private fields:
manager 10, lifecycle 32, driver 16, registry 5, copy budget 3. Its explicit field
maps link each stored metadata field to the corresponding record. Remaining
fields map to original live reference slots. Unknown or missing native fields
refuse instead of being dropped.

## Complete record families

| Record | Preserved state |
| --- | --- |
| ManagedOwnerState | Original limits, catalog in declaration order, complete provenance/consent/retention, optouts, revocations, declaration tick and seconds mappings |
| LifecycleAccountingState | Full retention policy, admitted ingress bytes, last tick/seconds, auxiliary epoch, cleanup failure and retention fault |
| RetentionDriverRecord | Original limits/state/polls/purges/cleanup attempts/created epoch, held/pending/error, stop/wake/purge levels, thread presence/aliveness |
| OwnershipRegistryState | Original shared policy, lifetime enrollment count and every retained enrollment, including ready/initializing/dead weak-reference observations |
| CopyBudgetState | Original shared copy policy and monotonic consumed copy charges |
| ManagedLifecycleCapture | Complete metadata graph plus all 47 named original authority references |

Reference slots retain original roots, owner/holder/time/copy/driver/sharing gates,
callbacks, weak-holder mapping, lineage, budget and policy objects, driver token,
events, thread, clock/progress/sampler and related ports. The original objects
are retained by identity. They are never deepcopied, compared through foreign
equality, invoked or serialized by this component. Source identity and actual
lease ownership must be established separately by the future live capture.

Why keep two parts: metadata can be detached; live authority must remain the
original authority. Treating locks, callback references, driver tokens or a thread
as portable scalar summaries would lose the relationships needed for recovery.
`None` records represent an actually unconfigured driver/copy policy; `ready=None`
represents an actual dead retained weak entry. They cannot stand in for configured
state or conceal omitted source fields.

## Validation and detachment

`validate_lifecycle_metadata` revalidates exact complete nested native record
schemas, original policies, supported consent, unique catalog/anchor/revocation
identities, original quota/count/clock relationships, retention faults, driver
counters and copy charges. It preserves the original shared holder/copy policy
aliases. Independently supplied `LifecycleCaptureLimits` bound the aggregate
number of all metadata entries and UTF8 bytes per identifier. Counts use exact
nonnegative integers below `2**63`; clock observations retain finite nonnegative
native seconds. These bounds are not a total process RSS or cumulative payload
copy allowance.

`validate_lifecycle_capture` additionally checks all required reference slots,
original root aliases, configured driver/copy presence, required locks/tokens/
events/ports, original held retention token and thread presence. Matching opaque
objects alone cannot prove that a caller supplied the real original owner.

`detach_lifecycle_record` validates before copying metadata as one graph, then
revalidates the detached record. It keeps the authority tuple and every object
inside it unchanged. It grants no live lease and resets or refunds nothing.
`require_lifecycle_source_schema` checks actual exact native root classes, all
fields and configured root identities without invoking a port or reading time.

## Metadata-only example (zero native work)

```python
from copy import deepcopy

from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.managed_lifecycle_state import (
    LifecycleAccountingState, LifecycleCaptureLimits, LifecycleMetadata,
    ManagedOwnerState, OwnershipEnrollment, OwnershipRegistryState,
)
from src.core.managed_lifecycle_validation import validate_lifecycle_metadata
from src.core.payload_ownership import PayloadOwnershipLimits

holders = PayloadOwnershipLimits(2, 8)
policy = DataRetentionPolicy(2048, 10, holders)
metadata = LifecycleMetadata(
    1, ManagedOwnerState(LifecycleLimits(0, 0), (), (), (), (), ()),
    LifecycleAccountingState(policy, 0, 0, False, None, None, False),
    OwnershipRegistryState(holders, 2, (
        OwnershipEnrollment(1, "actor", True),
        OwnershipEnrollment(2, "candidate", True),
    )), None, None,
)
limits = LifecycleCaptureLimits(2, 128)
validate_lifecycle_metadata(metadata, limits)
detached = deepcopy(metadata)
validate_lifecycle_metadata(detached, limits)
assert detached == metadata and detached is not metadata
assert detached.registry.limits is detached.lifecycle.policy.holders
assert detached.registry.limits is not holders
```

These are synthetic records with no live authority. The example executes no model
operation, capture, cleanup, driver start or restore. The fresh integration fixture
constructs an untrained original runtime/manager/lifecycle/copy/registry and an
unstarted driver solely to validate all 66 current source fields. Its original
budget permits zero updates and native operations are spied and prohibited.

## Commands

Each future invocation needs a fresh declared engineering envelope and unused
basetemp. Existing spent scopes remain spent.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest tests/test_managed_lifecycle_state.py -q -o addopts= -p no:cacheprovider --basetemp=artifacts/runs/lifecycle-state-unused
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -X utf8 -m ruff format --check src/core/managed_lifecycle_state.py src/core/managed_lifecycle_validation.py src/app/managed_lifecycle_schema.py tests/test_managed_lifecycle_state.py
```

## Exact next implementation

Complete coherent capture under original managed owner, holder, driver, time,
copy-budget and sharing leases. Existing driver `stop`, `wake`, `_failure` and
worker terminal transitions do not all participate in its operation lease.
Acquiring that lease alone cannot establish an atomic record of all state and
events. Extend the actual synchronization protocol without changing original
epochs, token identities, clock domains, consumed work or supported behavior.
Test busy/failure and complete live history before explicit bounded byte encoding
or any composite/disk/live/native/model/coordinator recovery acceptance.

## Coherent actual capture (R3.5b2e4b2)

Current layout:

```text
src/app/managed_lifecycle_capture.py   original-owner capture and lease orchestration
src/app/retention_expiry.py            short synchronized driver transitions
src/core/managed_lifecycle_state.py    complete immutable records/original references
tests/test_managed_lifecycle_capture.py actual histories, contention and worker joins
docs/adr/ADR-0230-capture-lifecycle-under-original-nonblocking-leases.md
```

The current complete source contract expands to67fields/48original reference slots
when the original driver's state lock is added. Earlier66/47contract receipts remain
historical evidence. Capture acquires original operation/state,manager,registry,
every supported live holder,time,copy and sharing leases nonblocking. Busy and
reentrant capture refuse and release acquired leases. Retained dead weak entries
are recorded without pruning. No port,clock,measurement or model operation is
called;last observed epochs and charges remain unchanged.

All driver mutations participate in the short state lock. Callbacks and joins run
outside it so stop can signal during cleanup. The operation lease continues to
serialize native cleanup and reject in-flight capture. Liveness is observed once;
the returned thread/event/token/gate/callback references remain original objects.
Exact source schemas and independent aggregate/UTF8 bounds are checked before
metadata detachment. Policy aliases are copied together as one metadata graph.

Self-contained actual capture example (constructor-only,zero model work,no worker):

```python
from src.adapters.numpy_learners import BackpropLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_lifecycle_capture import capture_managed_lifecycle
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

budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
runtime = ActorShadowRuntime(
    BackpropLearner(BackpropMLP(3, 2, seed=23), learning_rate=0.01),
    actor_version="actor", candidate_version="candidate",
    clock=LogicalClock(0), budget=budget,
)
owner = ManagedExperienceOwner(
    ResourceSharedRuntime(runtime, ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)),
    limits=LifecycleLimits(4, 4),
)
lifecycle = ManagedDataLifecycle(
    owner, policy=DataRetentionPolicy(4096, 20, PayloadOwnershipLimits(8, 12)),
    measure_payload_bytes=lambda value: 0,
    native_footprint=lambda value: ReplayPayloadErasure(0, 0, 0),
    native_erase=lambda value: ReplayPayloadErasure(0, 0, 0),
)
result = capture_managed_lifecycle(owner, limits=LifecycleCaptureLimits(64, 128))
assert result.metadata.registry.total_enrollments == 2
assert result.metadata.lifecycle.last_tick == 0
assert len(result.authority) == 48
assert budget.updates_completed == 0
assert next(ref.value for ref in result.authority if ref.path == "root.lifecycle") is lifecycle
```

Verification commands below require a fresh declared fixture allowance and an
unused basetemp;the recorded local run spends its own scope.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_managed_lifecycle_state.py tests/test_managed_lifecycle_capture.py -q -o addopts= -p no:cacheprovider --basetemp=<new-directory>
.\.venv\Scripts\python.exe -B -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/app/managed_lifecycle_capture.py src/app/retention_expiry.py src/core/managed_lifecycle_state.py src/core/managed_lifecycle_validation.py tests/test_managed_lifecycle_state.py tests/test_managed_lifecycle_capture.py
```

Next extension:implement explicit bounded lifecycle bytes with independent original
source/policy/content/authority bindings. Consume the complete capture,retain every
catalog/consent/revocation/epoch/charge/enrollment and original reference criterion.
This API neither serializes live authority nor restores a model,renewed budget or
coordinator. Original e4/e4b/composite/live/disk/native/scientific gates remain open.
