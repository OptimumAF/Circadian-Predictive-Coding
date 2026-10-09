# Complete lifecycle checkpoint bytes

## Structure and boundaries

```text
src/core/lifecycle_codec_policy.py           independent complete original policies
src/adapters/lifecycle_checkpoint_schema.py  explicit fields/raw preflight/metadata aliases
src/adapters/lifecycle_checkpoint_codec.py   bounded envelope and original bindings
tests/test_lifecycle_checkpoint_codec.py     full schema/refusal and fresh capture composition
docs/adr/ADR-0231-bind-complete-lifecycle-bytes-to-original-capture.md
```

`LifecycleCheckpointCodec` implements `CheckpointCodec[ManagedLifecycleCapture]`.
Inputs are complete validated captures or bounded bytes, independent original
owner/retention/driver policies and record/UTF8 bounds,source/policy/content SHA256
bindings,an original capture and an independently supplied authority digest.
Outputs are complete detached metadata and that original capture's authority tuple.
No disk IO,clock reads,measurement ports,native operations or restoration runs.

Why this: the original capture contains separate portable observations and live
authority. Encoding every observation preserves policy/epoch/charge/history;
keeping the original tuple prevents wire bytes from creating renewed authority.

## Full version1 envelope

| Field | Preserved meaning |
| --- | --- |
| codec_version/kind | Exact1/lifecycle_full_v1 |
| binding | Independently expected original source/policy SHA256 tags |
| policy | Complete original owner,retention,optional driver policies and capture bounds |
| authority_sha256 | Independent original authority tag |
| reference_manifest | All48known original paths,presence and first identity-alias index |
| metadata | All current catalog/provenance/consent/optout/revocation/order/declaration clocks/limits,ingress/last clock/auxiliary epoch/faults,retained enrollment/lifetime counts,optional copy limit/consumed charges,optional driver original limits/state/counters/created_at/held/pending/error/events/thread observations |
| metadata_aliases | First preorder identity index for every immutable record and tuple |

Source schema inventory remains67fields/48original live reference slots. Explicit
wire schemas cover17native record classes. Required holder/copy policy aliases and
shared-versus-distinct equal consent/provenance/key/anchor records round-trip;
finite native integer/float epochs,large supported integer seconds and signed zero
retain type and bits. Fixed enum spellings are schema constants;only caller
identifiers/error strings consume the independent UTF8 bound. None denotes actual
unconfigured policy or actual dead retained weak entry,not an omitted record.

Before typed construction:bound raw bytes,verify independently expected content
SHA256,reject duplicate JSON keys,require exact canonical envelope and original
policy/reference manifests,then validate every native field/flag/type/sequence,
aggregate record and UTF8 limit,consent/provenance/identity/anchor/quota/epoch/charge/
enrollment/driver relationships and complete alias table. Alias indices must name
the first node of the same record/tuple type and exact canonical value bits;shared
parent/child aliases must agree. Alias-node capacity is32*max_records+128 in
addition to the byte bound. All checks precede metadata materialization. The inner
native validator then independently rechecks the constructed complete graph.

No live object is placed in bytes. The manifest contains no address,callback,lock,
token,event or thread data. The supplied original tuple is retained unchanged.
Tags and manifests are integrity comparisons;they do not attest code closure or
prove that a supplied capture came from a quiescent owner. Use the actual capture
API and independently known original bindings. No model/clock/consent/budget/copy
ledger is replaced,reset,refunded or authorized by decode.

## Self-contained constructor-only example

The digest inputs below are demonstration original-owner tags. Real orchestration
must supply independently known original source/policy/authority bindings.

```python
from hashlib import sha256
from src.adapters.numpy_learners import BackpropLearner
from src.adapters.lifecycle_checkpoint_codec import LifecycleCheckpointCodec
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_lifecycle_capture import capture_managed_lifecycle
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
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
capture_limits = LifecycleCaptureLimits(64, 128)
original = capture_managed_lifecycle(owner, limits=capture_limits)
policy = LifecycleCodecPolicy(owner._limits, lifecycle._policy, None, capture_limits)
codec = LifecycleCheckpointCodec(policy, original=original, authority_sha256=sha256(b"original authority tag").hexdigest())
binding = CodecBinding(sha256(b"original source tag").hexdigest(), sha256(b"original policy tag").hexdigest())
limits = CodecLimits(32768, 1, 1, 1)
raw = codec.encode(original, binding=binding, limits=limits)
decoded = codec.decode(raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits)
assert decoded.metadata == original.metadata
assert decoded.authority is original.authority
assert decoded.metadata is not original.metadata
assert codec.encode(decoded, binding=binding, limits=limits) == raw
assert budget.updates_completed == 0
```

## Local verification and next extension

These constructor/admission-only fixtures require a NEW declared allowance and
unused basetemp. Original native/capture/worker scopes remain spent.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_lifecycle_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=<new-directory>
.\.venv\Scripts\python.exe -B -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/core/lifecycle_codec_policy.py src/adapters/lifecycle_checkpoint_schema.py src/adapters/lifecycle_checkpoint_codec.py tests/test_lifecycle_checkpoint_codec.py
```

Next:compose complete native/inbox/consolidation/lifecycle components under the
original exclusive owner with independent full source/policy/content/authority
bindings and alias/consent/revocation/retention/cumulative-charge invariants.
Keep all original full recovery/live/disk/native/model/coordinator-loss/scientific
acceptance open. This component does not prove portable authority or restore.
