# Automatic owned-data retention expiry

Declare `DataRetentionPolicy.max_retention_seconds` on a fresh original manager,
then install one `RetentionExpiryDriver`. Each declaration is anchored to the
original budget clock and the original logical clock. Reaching either deadline
refuses raw access. Handoff and new cache entries do not renew existing ages.
Auxiliary metadata/cache arrays use a conservative first-copy deadline, reset
only after actual all-holder cleanup. Without the optional elapsed policy, the
existing logical-age behavior is unchanged.

Why this: refusing overdue reads alone leaves raw owned copies retained. The
opt-in worker attempts paused, quiescent cleanup across the original ownership
registry. It has an independent sharing hold, preserving the user's manual pause
state. Busy manager, candidate, serving, checkpoint or registry gates are retried
within the original poll and elapsed run allowances. No training budget, ingress
byte budget, copy quota, sample identity or declaration age is renewed.

## Structure and responsibilities

```text
src/core/retention_driver.py      immutable limits, results and observations
src/app/retention_expiry.py       bounded worker, sharing hold, stop and cleanup
tests/test_retention_expiry.py   deterministic controls and fixed native capture
docs/retention-expiry.md         usage, boundaries and extension guidance
docs/adr/ADR-0212-purge-expired-owned-data-under-original-clock-and-gates.md
```

Core has no IO, clocks, locks, threads or payload references. App uses the
existing lifecycle's native ports; NumPy composition stays in adapters. The
original clock must be finite, nonnegative and monotonic. Replacing it, moving
backwards, or returning an invalid value closes admission and retained-state
access. Payload-free snapshots retain only counters/state and exception type,
never error messages or tracebacks. Boundary logs include only counts and type.
No new dependency or environment variable.

## Example (zero training, prediction or sleep)

This uses an explicit deterministic original clock. Production callers should
supply their original monotonic budget clock and keep the process running.

```python
import numpy as np
from src.adapters.numpy_learners import BackpropLearner, make_managed_data_lifecycle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.retention_expiry import RetentionExpiryDriver
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration, LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from src.core.retention_driver import RetentionDriverLimits

wall = [0.0]
source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: wall[0])
runtime = ActorShadowRuntime(source, actor_version="actor-0", candidate_version="candidate-0", clock=LogicalClock(), budget=budget)
gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
manager = ManagedExperienceOwner(ResourceSharedRuntime(runtime, gate), limits=LifecycleLimits(2, 2, allow_synthetic=True))
policy = DataRetentionPolicy(128, 10, PayloadOwnershipLimits(8, 16), owned_payload_copies=PayloadCopyLimits(128), max_retention_seconds=2.0)
cleanup = make_managed_data_lifecycle(manager, policy=policy)
driver = RetentionExpiryDriver(cleanup, limits=RetentionDriverLimits(0.01, 100, 2.0))
manager.declare(LifecycleDeclaration(("episode", "sample"), DataProvenance("example", "subject", True, True), DataConsent(True, True), "replay"))
manager.record_experience(Experience("sample", "episode", 0, "actor-0", np.array([[0.3, -0.2]]), "train", ExperiencePermissions(True, True)))
driver.start()
try:
    wall[0] = 2.0
    driver.wake()
    assert driver.wait_for_purge(1.0)
    assert cleanup.payload_byte_snapshot().observed_retained_bytes == 0
    assert cleanup.payload_byte_snapshot().charged_bytes == 16
    assert cleanup.admitted_payload_bytes == 16 and budget.updates_completed == 0
finally:
    assert driver.stop()
assert not driver.snapshot().alive
```

## Limits and failure semantics

- `max_polls` and `max_run_seconds` are original lifetime allowances. The run
  allowance begins at construction, including any delay before `start()`.
  Cleanup attempts are capped at `max_polls + 2`. A second driver/restart is
  refused. Terminal states permanently close raw admission/access.
- Stop or allowance exhaustion attempts conservative cleanup of all grants and
  owned raw payloads, even before their deadlines. Existing all-holder deletion
  invalidates pending checkpoints, prepared promotions, rollback and caches.
  Parameters, original budgets/gates and consumed IDs remain.
- `stop()` joins for at most `join_timeout_seconds`. False means unfinished
  cleanup or a live callback; inspect `snapshot()`. Retain the process/owner,
  release contention, join, then use `finish_cleanup()` within the remaining
  attempt allowance. It never restarts admission or refunds capacity.
- `wait_for_purge()` waits for the first successful cleanup and is bounded by
  the declared join timeout. It does not attest a later cohort's cleanup.
- A partial erasure failure retains revocation and the sharing hold, stops
  candidates and blocks raw access. Retry may finish erasure, never reopen
  stopped work. Unsupported/incomplete native ownership fails closed.
- Physical purge needs a running process, quiescent owners and responsive trusted
  callbacks. This is no hard real-time deadline or process/RAM overwrite
  guarantee. A running callback cannot be preempted; overdue publication/access
  is refused at the next guarded boundary. Caller-returned copies, arbitrary
  callback graphs, learned influence and durable restart recovery are excluded.
- Retention categories `transient` and `audit_only` remain refused. Logical/elapsed
  ages, record/holder limits and conservative owned-array byte reservations are
  separate from RSS, parameters, temporary buffers and caller allocations.

## Validation and extension

```powershell
python -m pytest tests/test_retention_expiry.py -q -k "not automatically_expire_real_handoff"
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
python -m ruff check src tests scripts
```

The fixed native capture has its own one-shot reservation; consult
`artifacts/runs/r36b2b-expiry-20261007/` before any rerun. Both type targets run
under the local Windows interpreter; this does not claim a new Linux runtime
matrix. Scoped format files and full selected regression commands are recorded
there. Clean-clone/full matrix/global formatter debt remain separate gates.

Extend supported native measurement/erasure through adapter ports and add
deterministic quiescence/failure/non-resurrection controls first. Any durable
restart or new retention category requires its own declared acceptance and
evidence; existing drivers and scientific allowances cannot be renewed.


## Auxiliary age correction and R3.6 acceptance — 2026-10-07

[Auxiliary retention](auxiliary-retention.md) closes the historical scalar-only metadata gap:all supported nonempty metadata/cache dictionaries anchor independently of numeric array bytes,including zero-size arrays/nested empty values. Every initial graph validates;copies/discard/rejection never renew existing age,only actual all-holder purge resets. Original clocks/consent/IDs/work/byte quotas and measurement metric remain unchanged. Full original R3.6b2b/R3.6b2/R3.6b/R3.6 criteria now pass641 current cases,both669-file types/static/AST/source/guide/resource and11-group audit:r36b2b-auxiliary-20261007. Earlier unchecked/missing-anchor status is historical. Transient/audit-only remains refused;caller/RAM/unlearning/physical deadline/clock attestation and durable restart limitations remain. R3.7/R3.8/G3 and broader runtime/clean-clone/scientific work remain unfinished. Do not rerun spent native captures without a new prospective scope.
