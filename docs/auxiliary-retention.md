# Retention age for all owned auxiliary content

Supported nonempty metadata and cache dictionaries now receive an elapsed age
anchor even when their numeric arrays occupy zero bytes. This includes strings,
scalar values, nested empty containers and zero-size arrays. Empty top-level
dictionaries have no content to retain and acquire an anchor on the first owned
promotion/cache copy. Initial caches are included.

Why this: array byte accounting answers a different question from data presence.
Using `measured_bytes > 0` skipped supported scalar-only metadata and caches.
Initial inspection also stopped after the first positive array, skipping graph
validation of later containers. The lifecycle now validates every exact supported
auxiliary dictionary, then anchors nonempty ownership independently of bytes.

## Changed files and boundaries

```text
src/app/managed_data_lifecycle.py   supported graph validation and age anchoring
tests/test_auxiliary_retention.py  metadata/cache controls and native successor
docs/auxiliary-retention.md        correction, usage, commands and limits
docs/adr/ADR-0213-anchor-owned-metadata-age-independently-of-array-bytes.md
```

The app orchestrates existing measurement ports; it does not import adapters.
Exact NumPy graph validation stays in the adapter. No new public interface,
dependency, environment variable, scheduler, native equation or byte metric.
Original clocks, consent, budgets, gates, sample/event IDs and parameter influence
retain their existing meaning. See [retention expiry](retention-expiry.md) and
[owned copy bytes](retained-payload-budget.md) for policy and shutdown limits.

## Example (zero training, prediction or sleep)

```python
import numpy as np
from src.adapters.numpy_learners import BackpropLearner, make_managed_data_lifecycle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from src.core.retention_driver import RetentionDriverLimits
from src.core.serving_ports import ServingConfiguration
from hashlib import sha256
import pickle

wall = [0.0]
source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
actor = PromotableActor(source, version="actor-0", configuration=ServingConfiguration(3, 2), feature_digest=lambda value: sha256(pickle.dumps(value)).hexdigest(), metadata={"raw": "text", "empty": np.empty((0, 2))})
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: wall[0])
runtime = ActorShadowRuntime(source, actor_version="actor-0", candidate_version="candidate-0", clock=LogicalClock(), budget=budget, actor=actor)
gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
manager = ManagedExperienceOwner(ResourceSharedRuntime(runtime, gate), limits=LifecycleLimits(2, 2))
policy = DataRetentionPolicy(128, 10, PayloadOwnershipLimits(8, 16), owned_payload_copies=PayloadCopyLimits(0), max_retention_seconds=2.0)
cleanup = make_managed_data_lifecycle(manager, policy=policy)
assert cleanup.payload_byte_snapshot().charged_bytes == 0
gate.pause()
driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 100, 2.0))
driver.start()
try:
    wall[0] = 2.0
    driver.wake()
    assert driver.wait_for_purge(1.0)
    assert actor.serving_snapshot().metadata == {}
    assert cleanup.payload_byte_snapshot().charged_bytes == 0
    assert gate.snapshot().paused and budget.updates_completed == 0
finally:
    assert driver.stop()
assert not driver.snapshot().alive
```

## Age, failure and removal semantics

- Initial nonempty metadata/cache content anchors to the original budget clock.
  New copies, discard, failed/rejected preparation, GC and handoff do not renew
  an existing auxiliary age. A failed attempt can conservatively spend an age.
- Only actual all-holder cleanup resets the auxiliary anchor. A later cohort
  obtains a fresh first-copy anchor using the same original clocks and allowances.
- Expiry of any owned auxiliary content invokes the existing conservative
  all-holder purge, even if a sample's own deadline is later. It invalidates
  checkpoint/promotion/rollback copies and clears raw inbox/native replay buffers.
- Byte reservations still count supported numeric arrays. Zero measured array
  bytes does not claim zero Python/RSS memory or no private content. Scalar metadata
  is subject to retention and deletion despite exclusion from the array byte metric.
- The worker requires a running process, quiescence and responsive trusted ports.
  Busy shutdown reports unfinished and blocks access. No hard real-time/RAM
  overwrite, caller-copy deletion, arbitrary callback graph/clock attestation,
  durable restart or parameter unlearning is promised.
- Transient/audit-only categories remain refused. Existing lifetime record,
  holder, ingress, copied-array and work quotas are never refunded or renewed.

## Validation and extension

```powershell
python -m pytest tests/test_auxiliary_retention.py -q -k "not scalar_metadata_before_sample_age"
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
python -m ruff check src tests scripts
```

The fixed native successor uses its own single reservation after all correctness,
type, static, source and executable-guide gates. Evidence and exact selected
regression/format commands: `artifacts/runs/r36b2b-auxiliary-20261007/`.
Both type targets use the local Windows interpreter. Broader runtime matrices,
clean-clone/global formatter gates and historical scientific closures remain
separate. Do not rerun any spent capture without a new prospective scope.

Extend supported native ports and ownership discovery with deterministic denial,
quiescence, partial-failure and non-resurrection tests. Durable process recovery
or additional retention categories require separately specified authority,
accounting and crash semantics before enabling them.
