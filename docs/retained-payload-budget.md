# Retained payload copy byte allowance

The original manager can now declare `DataRetentionPolicy.owned_payload_copies`
with `PayloadCopyLimits(max_lifetime_owned_bytes)`. Without this optional policy,
the existing managed cleanup behavior is unchanged. Install it on the fresh
manager with all supported measurement ports; the NumPy factory supplies them.

Why this: ingress bytes alone miss copied checkpoint views, prepared/failed
models, retired inboxes, promotion/rollback metadata and cached arrays. One
monotonic reservation ledger is shared by the original lifecycle authority.
Reserve before owned copies and native replay growth. Failed copies, failed
preparations, discard, GC, handoff and deletion never refund this lifetime
allowance. This deliberately conservative quota can exhaust while retained
arrays occupy fewer bytes. A quiescent observation measures the current catalog
and verifies `observed_retained_bytes <= charged_bytes <= limit`.

## Structure and boundaries

```text
src/core/payload_bytes.py        immutable limits and payload-free observations
src/app/payload_copy_budget.py   nonblocking original monotonic reservation ledger
tests/test_retained_payload_budget.py controls and separate fixed native capture
docs/retained-payload-budget.md  usage, scope, commands and extension guidance
docs/adr/ADR-0211-bound-owned-payload-copies-with-monotonic-byte-reservations.md
```

The lifecycle orchestrates copy reservations around existing inbox, checkpoint,
promotion and cache operations. NumPy adapter ports inspect exact supported
arrays and replay/checkpoint fields without copying them. The app never imports
adapters. Learned parameters, RNG/counter arrays, temporary native/callback
buffers, already returned caller copies, Python/container/string overhead and
process RSS are outside this payload-array metric. It is not a physical memory
allocator or arbitrary graph attestation. No new dependency/environment variable.

Supported auxiliary graphs use exact numeric NumPy arrays, lists, tuples,
string-keyed dictionaries, scalar metadata and cache prediction records. Aliases
within one independent copied graph count once; independently copied graphs
count separately. Cycles, opaque objects, object arrays, non-string keys,
depth above32 and measurement traversal above4096 nodes are refused before
copying. The bounded iterative walk stays in one function to keep its active/
seen stack validation together. Unsupported nodes never invoke object hooks.

For checkpoint/promotion preparation, use `ManagedNumpyBuilder(source)` with the
NumPy policy. Its exact source and supplied snapshot expose a conservative raw
replay copy bound before invocation. Keep that trusted caller-owned source fixed
during preparation. Opaque builders are denied before copying/invocation. Custom
ports and builder bounds are trusted contracts; dishonest callbacks/private
object mutation are not sealed by this API.

## Example (no training, prediction or sleep)

```python
import numpy as np
from src.adapters.numpy_learners import BackpropLearner, ManagedNumpyBuilder, make_managed_data_lifecycle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration, LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from hashlib import sha256
import pickle

def digest(value):
    return sha256(pickle.dumps(value)).hexdigest()

source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
clock = LogicalClock()
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
runtime = ActorShadowRuntime(source, actor_version="actor-0", candidate_version="candidate-0", clock=clock, budget=budget)
gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
shared = ResourceSharedRuntime(runtime, gate)
manager = ManagedExperienceOwner(shared, limits=LifecycleLimits(2, 2, allow_synthetic=True))
policy = DataRetentionPolicy(128, 10, PayloadOwnershipLimits(8, 16), owned_payload_copies=PayloadCopyLimits(72))
cleanup = make_managed_data_lifecycle(manager, policy=policy)
manager.declare(LifecycleDeclaration(("episode", "sample"), DataProvenance("example", "subject", True, True), DataConsent(True, True), "replay"))
manager.record_experience(Experience("sample", "episode", 0, "actor-0", np.array([[0.3, -0.2]]), "train", ExperiencePermissions(True, True)))
manager.record_label(LabelArrival("label", "sample", "episode", 0, "actor-0", np.array([[1.0]])))
gate.pause()
controller = CandidateCheckpointController(shared, build_learner=ManagedNumpyBuilder(source), state_digest=digest, policy_digest=lambda learner: digest("fixed"))
token = controller.capture()
assert cleanup.payload_byte_snapshot().observed_retained_bytes == 48
replacement = controller.restore(token)
assert replacement._budget is budget and cleanup.payload_byte_snapshot().charged_bytes == 72
cleanup.delete((("episode", "sample"),))
assert cleanup.payload_byte_snapshot().observed_retained_bytes == 0
assert cleanup.payload_byte_snapshot().charged_bytes == 72 and cleanup.admitted_payload_bytes == 24
assert budget.updates_completed == 0
```

## Admission and failure semantics

- Ingress reserves owned bytes before its opaque copy; denied copies do not
  consume the ingress ledger, while copies that start and fail remain charged.
- Native replay growth is reserved before detaching update inputs or entering
  native training. Existing sharing admitted attempts remain charged on denial;
  no native work occurred and the candidate can remain unstopped.
- Checkpoint views reserve replay/inbox arrays before snapshot copying. Restore
  reserves source/prepared replay and inbox handoff arrays before builder/copy.
  Existing preparation attempts remain spent even when the byte gate refuses.
- Promotion reserves prepared-model and metadata copies before guard evaluation.
  Negative guards and failed preparations do not refund reservations.
- Cache misses reserve their native output bound before prediction; hits do not
  double charge. An output exceeding its trusted declared bound is refused before
  cache publication. Current/previous/pending bundles are measured under leases.
- The original ledger and consent/byte-growth hooks survive handoff. A checkpoint
  controller on an alternate sharing wrapper is refused, preserving original
  sharing authority. Zero-size budgets are valid for no-payload operations.

Full R3.6/R3.6b/R3.6b2/R3.6b2b remain unchecked. Automatic elapsed retention
purge still needs implementation and evidence; the existing logical-age gate
requires explicit `expire()`. Next define bounded purge scheduling with quiescent
retry and access denial while overdue data remains. Preserve original ages,
budgets, gates, consumed IDs and failed reservations. Keep transient/audit-only
categories disabled; learned influence, caller-copy erasure and RAM overwrite
remain excluded.

## Validation commands

```powershell
python -m pytest tests/test_retained_payload_budget.py -q -k "not bound_real_owned_copies"
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
python -m ruff check src tests scripts
python -m ruff format --check src/core/payload_bytes.py src/app/payload_copy_budget.py tests/test_retained_payload_budget.py
```

The separate `bound_real_owned_copies` capture has a single fixed allowance of
four new wakes and two feedforward predictions, with no sleep. Its artifacts
live under `artifacts/runs/r36b2b-bytes-20261007/`; a rerun requires a new
prospective scope. These engineering checks make no comparative research claim.


## Subsequent elapsed retention increment — 2026-10-07

[Bounded retention expiry](retention-expiry.md) now anchors original-clock declarations and owned auxiliary arrays,attempts automatic all-holder quiescent purge and reports bounded stop/join failures. Original age/byte/work/IDs survive handoff. Prior absence of elapsed scheduling is historical. Full all-holder time policy remains unfinished:nonempty scalar-only metadata with0 measured array bytes receives no anchor and survives its deadline poll until stop. Preserve original full R3.6 criteria;next anchor nonempty supported auxiliary data independently of byte measurement. Do not rerun the spent four-wake capture;consult r36b2b-expiry-20261007 acceptance audit.


## Auxiliary age correction and R3.6 acceptance — 2026-10-07

[Auxiliary retention](auxiliary-retention.md) closes the historical scalar-only metadata gap:all supported nonempty metadata/cache dictionaries anchor independently of numeric array bytes,including zero-size arrays/nested empty values. Every initial graph validates;copies/discard/rejection never renew existing age,only actual all-holder purge resets. Original clocks/consent/IDs/work/byte quotas and measurement metric remain unchanged. Full original R3.6b2b/R3.6b2/R3.6b/R3.6 criteria now pass641 current cases,both669-file types/static/AST/source/guide/resource and11-group audit:r36b2b-auxiliary-20261007. Earlier unchecked/missing-anchor status is historical. Transient/audit-only remains refused;caller/RAM/unlearning/physical deadline/clock attestation and durable restart limitations remain. R3.7/R3.8/G3 and broader runtime/clean-clone/scientific work remain unfinished. Do not rerun spent native captures without a new prospective scope.
