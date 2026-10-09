# Managed payload cleanup

This opt-in same-process coordinator connects the original consent manager to
all enrolled actor, candidate, checkpoint and promotion owners. Legacy runtimes
retain their existing behavior until the coordinator is installed. Install on a
fresh manager before declaring or delivering any records.

## Responsibilities and boundaries

`DataRetentionPolicy` and `DataCleanupReport` are immutable core records.
`ManagedDataLifecycle` accepts explicit trusted measurement/erasure ports and
coordinates leases, revocation and deletion. The outer NumPy adapter's
`make_managed_data_lifecycle` supplies exact BackpropLearner/CircadianLearner ports;
the app never imports adapters. No new dependency or environment variable.

```text
src/core/data_retention.py       policy and payload-free result validation
src/app/managed_data_lifecycle.py original-authority cleanup orchestration
tests/test_managed_data_lifecycle.py deterministic controls and fixed native capture
docs/managed-data-lifecycle.md   usage, limits and extension guidance
docs/adr/ADR-0210-coordinate-conservative-payload-cleanup-with-original-authority.md
```

Why this: native replay snapshots do not carry managed episode/sample keys.
Deletion therefore conservatively clears **all delivered records** across every
enrolled inbox and all native replay buffers, and revokes all their delivered
identities as well as the requested keys. It also invalidates every pending
checkpoint and promotion ticket, drops rollback authority, and clears retained
serving metadata/cache dictionaries. Other subjects' delivered records can be
removed; the report identifies every revoked key. New undelivered grants that
were not requested remain eligible.

Deletion requires paused, quiescent original sharing authority plus nonblocking
manager and all-holder leases. Busy, incomplete, unsupported and invalid history
preflight fails before revocation or erasure. A native erasure failure after
revocation invalidates old tokens, stops all enrolled candidates and blocks
retained-state access. `retry_cleanup()` can finish erasure; it does not reopen
stopped training, refund work, reissue identifiers or restore old tokens.

Handoff retains the original logical and budget clocks, resource probe, progress,
sampler, policies, sharing gate, consent hooks, declaration ages and lifetime byte
ledger. Retired inboxes retain their own completed-work count, so cleaning them
after the live budget advances does not manufacture a fresh allowance.

## Example (no training, prediction or sleep)

Run from the repository root in its existing virtual environment:

```python
import numpy as np
from src.adapters.numpy_learners import BackpropLearner, make_managed_data_lifecycle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration, LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits

source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
clock = LogicalClock()
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
runtime = ActorShadowRuntime(source, actor_version="actor-0", candidate_version="candidate-0", clock=clock, budget=budget)
gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
shared = ResourceSharedRuntime(runtime, gate)
manager = ManagedExperienceOwner(shared, limits=LifecycleLimits(2, 2, allow_synthetic=True))
cleanup = make_managed_data_lifecycle(manager, policy=DataRetentionPolicy(128, 10, PayloadOwnershipLimits(8, 16)))
manager.declare(LifecycleDeclaration(("episode", "sample"), DataProvenance("example", "subject", True, True), DataConsent(True, True), "replay"))
manager.record_experience(Experience("sample", "episode", 0, "actor-0", np.array([[0.3, -0.2]]), "train", ExperiencePermissions(True, True)))
manager.record_label(LabelArrival("label", "sample", "episode", 0, "actor-0", np.array([[1.0]])))
gate.pause()
report = cleanup.delete((("episode", "sample"),))
assert report.revoked_keys == (("episode", "sample"),)
assert cleanup.admitted_payload_bytes == 24 and budget.updates_completed == 0
assert runtime._inbox._experiences == runtime._inbox._labels == {}
gate.resume()
assert shared.train_ready().updates == ()
```

`manager.opt_out(subject)` delegates to coordinated cleanup when installed;
without it, the existing admission-only behavior is unchanged. Unsupported
transient/audit-only categories remain refused. Synthetic/unverified provenance
still needs its explicit original manager permission; declared provenance is
not an attestation.

## Current limits and unfinished acceptance

- Authorized numeric array ingress bytes are charged before opaque copying.
  Failed copy attempts stay charged; deletion, expiry and handoff never refund.
- Holder live/lifetime enrollment limits and controller pending/preparation
  limits bound copy multiplicity. They **do not measure aggregate retained array
  bytes**, serving metadata graphs, temporary builder/callback data or RSS.
- Declaration-age expiry uses the original LogicalClock. At the deadline,
  training, admission, retained-state snapshots, checkpoint inspect/restore and
  rollback refuse access. The caller must pause and invoke `expire()` to purge.
  There is no automatic elapsed-time worker or physical retention guarantee.
- Cleanup drops owned raw buffer references; native parameters, RNG, counters,
  policies and already completed work remain. Learned influence is not removed.
- Caller source models, already returned inspection copies, callbacks and
  arbitrary externally held Python graphs are outside this ownership catalog.
  Memory overwrite, durable restart recovery and parameter unlearning are absent.

Full R3.6/R3.6b/R3.6b2/R3.6b2b remain unchecked until their original acceptance,
including retained-byte and time enforcement, is proven. Extend next by adding
aggregate owned-array accounting and pre-copy admission at every retained-copy
boundary, with failure/expiry controls under the original authority. Do not
enable new retention categories or algorithms to bypass that work.

## Validation commands

```powershell
python -m pytest tests/test_managed_data_lifecycle.py -q -k "not clean_real_owned_native"
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
python -m ruff check src tests scripts
python -m ruff format --check src/core/data_retention.py src/app/managed_data_lifecycle.py tests/test_managed_data_lifecycle.py
```

The two `clean_real_owned_native` cases belong to the single fixed six-wake
capture recorded in `artifacts/runs/r36b2b-managed-cleanup-20261007/`; repeating
that capture consumes a separate experiment allowance. These are engineering
correctness checks, not evidence of algorithmic superiority.


## Subsequent byte enforcement increment — 2026-10-07

The original manager can additionally declare owned_payload_copies=PayloadCopyLimits(...). [Retained payload budget](retained-payload-budget.md) now enforces a conservative lifetime copied-array allowance before owned copy/growth and observes all-holder retained raw/auxiliary arrays. Earlier absence of aggregate accounting is historical;the optional configured supported policy is implemented. Parameter/temporary/caller/Python/RSS memory remains excluded. Automatic elapsed-time purge still requires implementation;full original R3.6b2b stays unchecked. Use ManagedNumpyBuilder for supported NumPy preparations.


## Subsequent elapsed retention increment — 2026-10-07

[Bounded retention expiry](retention-expiry.md) now anchors original-clock declarations and owned auxiliary arrays,attempts automatic all-holder quiescent purge and reports bounded stop/join failures. Original age/byte/work/IDs survive handoff. Prior absence of elapsed scheduling is historical. Full all-holder time policy remains unfinished:nonempty scalar-only metadata with0 measured array bytes receives no anchor and survives its deadline poll until stop. Preserve original full R3.6 criteria;next anchor nonempty supported auxiliary data independently of byte measurement. Do not rerun the spent four-wake capture;consult r36b2b-expiry-20261007 acceptance audit.


## Auxiliary age correction and R3.6 acceptance — 2026-10-07

[Auxiliary retention](auxiliary-retention.md) closes the historical scalar-only metadata gap:all supported nonempty metadata/cache dictionaries anchor independently of numeric array bytes,including zero-size arrays/nested empty values. Every initial graph validates;copies/discard/rejection never renew existing age,only actual all-holder purge resets. Original clocks/consent/IDs/work/byte quotas and measurement metric remain unchanged. Full original R3.6b2b/R3.6b2/R3.6b/R3.6 criteria now pass641 current cases,both669-file types/static/AST/source/guide/resource and11-group audit:r36b2b-auxiliary-20261007. Earlier unchecked/missing-anchor status is historical. Transient/audit-only remains refused;caller/RAM/unlearning/physical deadline/clock attestation and durable restart limitations remain. R3.7/R3.8/G3 and broader runtime/clean-clone/scientific work remain unfinished. Do not rerun spent native captures without a new prospective scope.
