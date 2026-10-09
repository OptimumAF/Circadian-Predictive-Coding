# Complete paired record bytes

`ManagedRecordCheckpointCodec` implements `CheckpointCodec[ManagedRecordCapture]`.
Its `managed_record_pair_v1` format preserves the complete consolidation and
lifecycle records from one original managed owner interval. Decode returns
detached immutable metadata and the exact original caller-supplied authority
tuple. It performs no native operation, clock read, callback, IO or restore.

## Modules

```text
src/core/managed_record_codec_policy.py             independent component policies
src/adapters/managed_record_checkpoint_schema.py   complete raw wire preflight
src/adapters/managed_record_checkpoint_codec.py    bound encode/decode port
tests/test_managed_record_checkpoint_codec.py      round-trip and refusal controls
```

The app's existing common-interval capture and core records remain unchanged.
Core imports only inner modules. No dependency, environment variable or runtime
configuration format was added.

## Complete version 1 envelope

| Field | Preserved meaning |
| --- | --- |
| codec_version/kind | Exact 1/managed_record_pair_v1 |
| binding | Independently expected original source and policy SHA256 tags |
| policy | Full original lifecycle, consolidation, driver, holder, copy and capture policies |
| authority_sha256 | Independently supplied original authority tag |
| reference_manifest | All 59 original paths, presence flags and first identity indices |
| metadata | All 13 runtime observations and every complete consolidation/lifecycle field |
| metadata_aliases | First identity index of every immutable metadata record and tuple, including cross-component aliases |

The explicit trusted schemas cover 24 native record classes, including all
policies. Complete records retain consumed consolidation IDs, receipt attempt
gaps, native diagnostics, versions, limits, revision/retired/stopped/ready flags,
catalog order, consent/provenance, optout/revocation, declaration and observed
clocks, lifetime quotas, driver state/events/epoch/error/thread observations,
retention faults, consumed copy charges and retained ownership enrollment.
Finite native diagnostic integers/floats, negative values, signed zero and
supported larger integers retain their types and bits. Fixed enum spellings
remain independent of caller identifier capacity.

## Original observation and authority

Why this: matching revision and budget counters alone would allow a lifecycle or
driver epoch from another observation to be spliced into the pair. The codec
requires an independently supplied original complete capture. Both the complete
canonical metadata projection and its immutable alias graph must match that
original before encoding or decoding. Structural validity alone is insufficient.

Original policies and source/policy/content/authority tags are supplied separately.
The reference manifest contains paths, presence and alias indices, with no object
addresses. Encoding requires every live value to be the original object by
identity. Decode returns that exact original tuple without copying live values.
Identical portable observations from different owners are not a provenance seal;
the separately supplied original authority remains necessary. No lock, port,
token, thread, consent, clock, budget or copy allowance is recreated or renewed.

Before typed wire construction: check wire capacity and independently expected
content digest; reject duplicate keys and noncanonical JSON; check complete
envelope, original full policies, bindings and all 59 manifest slots; validate
every raw type/field/UTF8/count, quota, ID/version/receipt, stop/enrollment/budget,
clock/epoch/driver/copy/catalog/consent/revocation relationship; then validate the
full alias graph and complete original observation. Shared parent/child aliases
must agree, with original holder/copy policy aliases retained. Aggregate capacity
counts one runtime observation and all eight history collections jointly. Alias
nodes have an additional `32 * max_records + 128` ceiling. All checks precede
materialization; native validation independently rechecks the resulting graph.

The shared lifecycle walkers accept trusted internal schema descriptors and a
path prefix, retaining their existing defaults. No wire data selects a schema,
callable, record constructor or authority. Managed arbitrary consolidation
transforms remain refused; richer histories are synthetic metadata controls.

## Constructor-only example

The original tags below are illustrative. Real orchestration must supply
independently known original source, policies, capture and authority.

```python
from hashlib import sha256
from unittest.mock import patch
from src.adapters.numpy_learners import BackpropLearner
from src.adapters.managed_record_checkpoint_codec import ManagedRecordCheckpointCodec
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_record_capture import capture_managed_records
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.consolidation_codec_policy import ConsolidationCodecPolicy
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.managed_record_codec_policy import ManagedRecordCodecPolicy
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits

class WallClock:
    blocked = False
    def __call__(self):
        assert not self.blocked, "codec read the clock"
        return 0.0

def forbidden(*args, **kwargs):
    raise AssertionError("capture or codec invoked an original port")

class Footprint:
    blocked = False
    calls = 0
    def __call__(self, model):
        assert not self.blocked, "capture or codec measured a native footprint"
        self.calls += 1
        return ReplayPayloadErasure(0, 0, 0)

wall = WallClock()
footprint = Footprint()
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
    measure_payload_bytes=forbidden, native_footprint=footprint, native_erase=forbidden,
)
capture_limits = LifecycleCaptureLimits(64, 128)
wall.blocked = True
assert footprint.calls == 2  # Original descriptor probes during installation.
footprint.blocked = True
with patch.object(LogicalClock, "now", forbidden), patch.object(BackpropLearner, "snapshot_state", forbidden):
    original = capture_managed_records(owner, limits=capture_limits)
    policy = ManagedRecordCodecPolicy(
        LifecycleCodecPolicy(owner._limits, lifecycle._policy, None, capture_limits),
        ConsolidationCodecPolicy(runtime._consolidation_limit, capture_limits.max_identifier_bytes),
    )
    codec = ManagedRecordCheckpointCodec(policy, original=original, authority_sha256=sha256(b"original authority tag").hexdigest())
    binding = CodecBinding(sha256(b"original source tag").hexdigest(), sha256(b"original policy tag").hexdigest())
    limits = CodecLimits(65536, 1, 1, 1)
    raw = codec.encode(original, binding=binding, limits=limits)
    decoded = codec.decode(raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits)
    assert decoded.metadata == original.metadata and decoded.metadata is not original.metadata
    assert decoded.authority is original.authority
    assert codec.encode(decoded, binding=binding, limits=limits) == raw
assert budget.updates_completed == 0
```

## Local verification and extension

Constructor/admission controls require a fresh declared numerical allowance and
an unused pytest directory. Prior native/capture/worker scopes remain spent.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_managed_record_checkpoint_codec.py tests/test_lifecycle_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=<new-directory>
.\.venv\Scripts\python.exe -B -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/core/managed_record_codec_policy.py src/adapters/lifecycle_checkpoint_schema.py src/adapters/managed_record_checkpoint_schema.py src/adapters/managed_record_checkpoint_codec.py tests/test_managed_record_checkpoint_codec.py
```

Next, inventory and implement the full original native/inbox/paired record
composition under the original owner, retaining array layout/aliases and all
consumed-budget/consent/revocation/retention/copy rules. Component bytes do not
satisfy model restoration, actor/sharing/ownership, live/disk/coordinator loss,
scientific or human gates. Preserve those original criteria and supply separate
current evidence before claiming recovery.
