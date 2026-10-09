# Durable component codecs

R3.5b2e1 implements a complete bounded **backprop model snapshot component**.
Full R3.5b2e checkpoint/recovery acceptance remains open. No disk restore or
completed native work is admitted by these functions.

## Structure and responsibilities

```text
src/core/checkpoint_codec.py              binding, resource limits, typed port
src/adapters/backprop_checkpoint_codec.py exact native backprop byte schema
tests/test_backprop_checkpoint_codec.py   round-trip and refusal controls
docs/adr/ADR-0223-encode-explicit-complete-native-components.md
```

Dependency direction: adapter -> existing NumPy snapshot adapter and core port.
Core imports neither adapters nor infrastructure. No dependency or environment
variable is added. The original algorithms and learner interfaces are unchanged.

Why this: the in-process CandidateCheckpoint retains live ownership identities,
not a portable restoration authority. Explicit component schemas make omissions
and unsupported graphs fail before publication rather than loading arbitrary
objects through pickle.

## Backprop byte contract

Canonical UTF-8 JSON uses `codec_version=1`, `kind=backprop_full_v1` and native
`snapshot_version=1`. Exactly ten native dictionary fields are covered:

| Fields | Representation / ownership |
| --- | --- |
| `input_dim`, `hidden_dims` | Exact positive integers / tuple topology |
| `_hidden_weights`, `_hidden_biases`, `_traffic_sums` | Complete ordered array lists |
| `weight_hidden_output`, `bias_output` | Complete output parameter arrays |
| `_traffic_steps` | Exact nonnegative integer below 2^63 |
| `weight_input_hidden`, `bias_hidden` | Explicit references to first hidden weight / bias |

Array frames declare exact shape and `<f8` dtype, with canonical base64 of every
little-endian float64 value in C traversal order. Finite values are required;
traffic is nonnegative. Signed zero survives. Strided input arrays preserve
logical values and required identity edges; decoded storage is independently
owned and contiguous. Additional shared-memory edges, missing/future native
fields, unknown frames and versions are refused. No RNG lives in BackpropMLP
after initialization; its constructor RNG is local and discarded.

Learning rate remains learner policy, outside BackpropSnapshot. The caller must
bind the complete original policy bytes and independently reviewed source to
`CodecBinding`. Those two digests are compared exactly with the envelope. Decode
also requires the independently expected complete encoded byte digest. Digests
are integrity comparisons, not signatures or proof of source graph closure.

Before encode creates array byte/base64 copies, topology and aggregate unique
array bytes are checked, and exact resulting wire length is computed. Before
decode, input length and expected digest are checked; then all frame shapes,
types and lengths are checked before any native array allocation. Base64 syntax,
padding, exact bytes, finiteness and traffic are validated during bounded decode.
Limits bound array/wire payload sizes; they are not a total process RSS limit or
an original cumulative copy-budget reservation. Concurrent caller mutation is
unsupported: encode must receive a detached snapshot under its original owner.

## Small byte round-trip example

This example creates an untrained fixture and performs no learner restore.
In a real owner, replace the fixture binding bytes with independently frozen
complete source and policy bytes and retain the encoded digest outside the blob.

```python
from hashlib import sha256

from src.adapters.backprop_checkpoint_codec import BackpropCheckpointCodec
from src.adapters.numpy_learners import BackpropLearner, BackpropSnapshot
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits

source = b"fixture-only source identity; not a recovery seal"
policy = b"fixture-only backprop learning_rate=0.1"
binding = CodecBinding(sha256(source).hexdigest(), sha256(policy).hexdigest())
limits = CodecLimits(32768, 8192, 8, 64)
learner = BackpropLearner(BackpropMLP(3, 4, seed=0), learning_rate=0.1)
saved = learner.snapshot_state()
codec: CheckpointCodec[BackpropSnapshot] = BackpropCheckpointCodec()
raw = codec.encode(saved, binding=binding, limits=limits)
expected = sha256(raw).hexdigest()
detached = codec.decode(raw, binding=binding, expected_sha256=expected, limits=limits)
assert codec.encode(detached, binding=binding, limits=limits) == raw
assert detached.state["weight_input_hidden"] is detached.state["_hidden_weights"][0]
assert detached.state["weight_input_hidden"] is not saved.state["weight_input_hidden"]
```

## Composite state inventory and unfinished work

The source-bound static census is in
`artifacts/runs/r35b2e1-backprop-codec-20261008/component-inventory.json`.
It records complete reads, declared record fields and assigned private attributes
across 31 relevant sources. It is an inventory aid, not a complete schema or
ownership certification. Dynamic values and alias edges require explicit tests.

| Component | Required durable content / independent live references | Status / next action |
| --- | --- | --- |
| Candidate checkpoint | All view fields: actor version/generation, learner version, native state, inbox, attempted IDs, consolidation receipts/limit, stopped, budget dictionary/progress, sharing, event tick. `_Pending` additionally retains revision, digests, builder/policy/probe callbacks, original runtime/budget/clocks/sampler/progress/resource/sharing/actor, sampler snapshot. | View alone is incomplete. Separate durable fields from independently re-established ports; forbid callback deserialization. |
| Circadian native | Full v2 dictionary and optional policy fields: all prehidden/adaptive weights/biases, chemistry/traffic/age/importance, IDs/parents/next ID, prune masks/TTLs/cooldowns, topology/config, clocks/counters, energy/reward histories, replay deque/maxlen and every batch/priority/fraction, RNG bit-generator state; optional retention/exposure/side-effect policy and identity sets/counters. | Next codec increment: enumerate exact supported variants, RNG and alias contracts; reject unknown fields. |
| Optional Torch classifiers | Complete CircadianClassifierSnapshot native tensors, topology/config, replay/structural state and native RNG/ownership semantics. | Unsupported by backprop codec; preserve requirement and explicitly scope subsequent codec work. |
| Inbox | v1/v2 learner/capacity, complete experience feature payloads, labels/targets, applied receipts, last tick, stopped, completed updates, erased tombstones; consumed event IDs and history must remain derivable/preserved. Registration/training guards, original clock and budget remain independent. | Add exact typed metadata and supported payload byte codecs; test histories and payload ownership. |
| Consolidation | Complete native result, TrainingDiagnostic definition/value, event/actor/learner/attempt receipt, attempted IDs, quota and spent attempts; original candidate revision/base identity. | Explicit versioned records; reject duplicates, omissions and refunded attempts. |
| Managed consent | Lifetime catalog declarations/provenance/consent/retention, original limits, permanent opted-out subjects and revoked keys, declaration ticks/seconds; issuing token and guards cannot become byte authority. | Define complete typed snapshot under original gates; preserve consumed identities. |
| Lifecycle / copy ledger | Original policy/budget/sharing limits, admitted bytes, last tick/seconds, failure/retention fault, auxiliary start, lifetime charged copies and limits, all observed payload groups. Resource/callback/clock/sampler/progress/locks and original lineages remain independent. | No durable lifecycle snapshot exists; introduce it before composite codec. |
| Holder registry / retention | Total lifetime enrollments, original limits, holder kind/readiness/enrollment and every model/inbox/snapshot/auxiliary reference edge. Driver original creation/start, limits/state/polls/purges/attempts/pending/error; live thread/events/leases are independent and cannot restart from counters alone. | Metadata snapshots omit references and original live authority. Explicit owner reconstruction/refusal controls required. |
| Stable / promotable actor | Stable version and complete independently owned native learner/policy. Promotable active/previous bundles: learner, version, configuration, cache payloads, metadata, last cache tick and generation. Read gates, registry and feature digest stay independently bound. | Full native and serving bundle codecs + single-owner/stable actor proof remain open. |
| Sharing / resources | Original limits, active requests/training flag, paused, admitted updates, every deferral; actual resource ownership and original serving/training gates cannot be serialized as capability. | Define quiescent supported durable state with refusal of active/unknown ownership. |
| Recovery authority | Original epoch/start/caps/manifest/owner/sequence, spent counters, stopped/uncertain history, independently retained anchor/worker identities and fresh time/RSS. | Existing metadata codecs remain unchanged. Composite bytes must not reset or manufacture these values. |

## Verification and extension

PowerShell commands (use a new pytest temp path each invocation):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_backprop_checkpoint_codec.py -q --basetemp=artifacts/runs/codec-example-unused
.\.venv\Scripts\python.exe -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/core/checkpoint_codec.py src/adapters/backprop_checkpoint_codec.py tests/test_backprop_checkpoint_codec.py
```

Add the next explicit native codec as a separate adapter implementing the same
inner port. Require complete field/variant/alias controls and bound all copies
before connecting it to disk, original lifetime copy reservations or live restore.
Keep R3.5b2e and native/model/coordinator-loss/scientific acceptance unchecked.


### Current bounded validation evidence - 2026-10-08

R3.5b2e1 component criteria passed in the independent
`artifacts/runs/r35b2e1-validation-20261008/` scope:current51codec cases and both
697file type targets with `--cache-dir nul`,static/source/AST/exact guide pass.
All inherited shared cache bytes/hashes/mtimes remained exact;newcache bytes0.
The installed Windows `os.devnull` string is lowercase `nul`;that exact spelling
is used for mypy's source-verified null-cache boundary. Earlier unproven cache
growth and13unit-update literal-exclusion failure remain historical failures.
This component acceptance does not complete R3.5b2e or admit live/disk restore.


## Complete current circadian native component

`CircadianCheckpointCodec` implements `CheckpointCodec[CircadianNetworkSnapshot]`
for native snapshotv2. Wire version2/kind `circadian_full_v2` contains exact
metadata plus every native dictionary field;it accepts all eight combinations
of base/content-hash/recent-FIFO/seeded-reservoir retention and optional wake-only
replay side effects supported by the current native configuration methods.
Nondefault retention carries its complete observed/duplicate/exposed ID sets and
occurrence/update counters. Default retention has no native exposure fields.

Structure/dependencies:

```text
src/adapters/numpy_checkpoint_frames.py     exact bounded f8/i8/i4/bool frames
src/adapters/circadian_checkpoint_schema.py frozen36base/73config variant schema
src/adapters/circadian_checkpoint_codec.py  complete byte orchestration
tests/test_circadian_checkpoint_codec.py    complete native/refusal controls
docs/adr/ADR-0224-encode-complete-circadian-native-variants.md
```

These adapters depend on existing inner native contracts and CheckpointCodec;
core imports neither adapter nor infrastructure. No dependency,environment
variable or algorithm equation changes. Why this:explicit variants prevent
unknown native fields from disappearing through implicit generic object loading.

Every current parameter/pre-hidden array,chemical/traffic/age/importance vector,
lineage ID/parent/next ID,pending prune mask/TTL/cooldowns,topology/configuration,
wake/replay/sleep/traffic counter,energy/reward history,replay deque/maxlen and
full ReplaySnapshot arrays/priority/fraction is preserved. Optional retention
budget/policy/exposure/side-effect fields are exact. RNG uses the current native
PCG64 implementation's full128bit state/increment and cached uint32 state. Other
generators or undeclared shared storage are explicitly unsupported;no existing
in-memory native snapshot API is narrowed. Full recovery of other supported
native types still requires their own complete explicit codec.

Source/policy/content comparisons remain independently provided. Initial layers,
current adaptive width and every native frame dtype/shape must agree. Aggregate
unique-array bytes and exact prospective wire length are checked before array
byte/base64 copies or decoded native allocation. Malformed collections/unknown
nested records fail before object copies. Metadata/configuration parsing and
bounded validation can allocate small temporary structures;limits are payload
bounds,not total processRSS or original lifetime copy reservations. All frames
are verified before the first decode;remaining semantic relations are verified
after bounded detached decode using existing native topology/config/replay
validators. Native byte copies never mutate the source snapshot.

Decoded arrays own contiguous C/F native storage;logical bytes including signed zero
are preserved. Deques,sets and generator are independently owned. Configuration
is immutable and represented by value. Input must be a detached snapshot under
its original owner;concurrent caller mutation and graph/consent/source attestation
are outside the codec. No callbacks,pickle,IO,work admission,disk/live restore or
completed native work.

### Circadian byte example (no training or restoration)

```python
from hashlib import sha256

from src.adapters.circadian_checkpoint_codec import CircadianCheckpointCodec
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot, CircadianPredictiveCodingNetwork

model = CircadianPredictiveCodingNetwork(2, 4, seed=23, min_hidden_dim=2, max_hidden_dim=8)
saved = model.snapshot_state()
binding = CodecBinding(sha256(b"fixture-only source").hexdigest(), sha256(b"fixture-only policy").hexdigest())
limits = CodecLimits(65536, 8192, 8, 64)
codec: CheckpointCodec[CircadianNetworkSnapshot] = CircadianCheckpointCodec()
raw = codec.encode(saved, binding=binding, limits=limits)
expected = sha256(raw).hexdigest()
detached = codec.decode(raw, binding=binding, expected_sha256=expected, limits=limits)
assert codec.encode(detached, binding=binding, limits=limits) == raw
assert detached.state["_rng"] is not saved.state["_rng"]
assert detached.state["weight_input_hidden"] is not saved.state["weight_input_hidden"]
```

Run the new controls only under a prospectively declared native fixture allowance:
fixed seed23/batch/rates,at most64native updates,16sleeps,32fixture-local restores
and4structural operations per complete invocation. The fixture records exact
wake/replay updates/sleeps/restores/structural counts beneath its new pytest temp
path. Current tests compare exact native continued state/energy,not
scientific superiority. No old spent fixture or worker replay is implied.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_circadian_checkpoint_codec.py tests/test_backprop_checkpoint_codec.py -q --basetemp=artifacts/runs/circadian-codec-unused
.\.venv\Scripts\python.exe -B -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -m ruff format --check src/adapters/numpy_checkpoint_frames.py src/adapters/circadian_checkpoint_schema.py src/adapters/circadian_checkpoint_codec.py tests/test_circadian_checkpoint_codec.py
```

Extension: add exact component schemas through the existing inner port. Full
R3.5b2e still needs inbox/consolidation/lifecycle/actor/sharing and original
copy-budget/source/single-owner admission before any composite durable restore.


### Retained earlier native layout failure

Current143controls pass and value/state continuity is verified for eight fixed
ordinary variants. A separate nontraining completeness probe preserves shape,
dtype and logical bytes but changes an independently owned Fortran-contiguous
native weight to C-contiguous storage. This is a retained negative result in
`artifacts/runs/r35b2e2-circadian-codec-20261008/native-layout-probe.json`.
At that earlier closure R3.5b2e2 remained unchecked. Its wire-v1 schema normalized storage order;
complete native preservation must retain supported C/F order and prove continued
native behavior,including dynamic structural state. Do not treat existing green
value tests as that broader proof. Full composite restoration stays closed.
Backprop's documented normalized-layout component also needs separate layout
evidence before broad full-native recovery;its original scoped task acceptance
remains preserved. No algorithm,seed,metric or baseline was changed to hide this.


### Storage-preserving circadian wire v2

Each exact array frame contains `dtype`, `shape`, `order`, and `data`. Encode and
decode traverse bytes in the declared C/F order. Vectors and singleton axes have
both contiguous flags and use the unique canonical C tag; F tags for those shapes
are refused. Unsupported strided/reversed arrays fail before serialization.
Unknown/missing/order-type/version frames fail before native allocation. C/F tags
have equal length;the complete wire preflight includes the tag before any payload
copy. Aggregate array/payload bounds remain distinct from process RSS/lifetime
copy charges. Old CPC wire-v1 bytes are explicitly unsupported;they are not
silently reinterpreted. Native snapshotv2 and learning APIs are unchanged.

New controls: `tests/test_circadian_checkpoint_layout.py` (34cases). Fixed seed23
source-native split/prune yields F weights for all8retention/wake-only variants.
Fixture-local decoded restore then original/decoded wake,sleep and replay compare
exact complete state,energy,prediction bytes and every array's logical bytes,
dtype,shape,C/F flags and independent owned storage. This is correctness evidence,
not scientific superiority or composite/disk/live recovery acceptance.

Fresh complete invocation budget includes existing tests plus new structural
controls:96total native updates/24sleeps/24restores/20structural operations. New
module separately enforces32updates/16sleeps/8restores/16structural. All old fixture
allowances remain spent. Use a newly declared envelope and unused basetemp for
any future invocation. To extend safely,add an explicit versioned component
schema and independent corruption/resource/ownership/continuation controls.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest tests/test_circadian_checkpoint_layout.py tests/test_circadian_checkpoint_codec.py tests/test_backprop_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=artifacts/runs/layout-validation-unused
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -X utf8 -m ruff format --check src/adapters/numpy_checkpoint_frames.py src/adapters/circadian_checkpoint_codec.py tests/test_circadian_checkpoint_layout.py
```

Backprop wire-v1 retains its documented C normalization. Separate R3.5b2e1a
layout/alias/continuation qualification is required before broad full-native
recovery. Full R3.5b2e/Torch/inbox/consolidation/lifecycle/actor/sharing/cumulative
copy/single-owner/model/coordinator-loss/scientific/human criteria remain open.


## Explicit storage-preserving Backprop component (R3.5b2e1a)

Select `LayoutBackpropCheckpointCodec` through `CheckpointCodec[BackpropSnapshot]`
for wire v2/kind `backprop_layout_v2`. Existing `BackpropCheckpointCodec` remains
wirev1 with its original logical-value/noncontiguous normalization. Explicit
selection preserves that working API and avoids implicit wire reinterpretation.
The v2 decoder refuses v1 and all unsupported version/kind/order/extra/missing
fields;the native BackpropSnapshot version remains1. Required legacy aliases
attach to the decoded first hidden weight/bias by object identity.

```text
src/adapters/backprop_layout_checkpoint_codec.py   explicit v2 storage codec
src/adapters/backprop_checkpoint_codec.py          original v1 native schema
src/adapters/numpy_checkpoint_frames.py            shared bounded C/F frames
src/core/checkpoint_codec.py                      inward binding/limits/port
tests/test_backprop_layout_checkpoint_codec.py     v2 native/refusal controls
docs/adr/ADR-0226-qualify-backprop-layout-with-explicit-codec.md
```

Why reuse:the two Backprop adapters share the existing exact native/topology/
alias schema and limits;only the explicit frame/wire contract changes. Existing
internal schema helpers are adapter implementation details,not a new public
learner interface. Newcodec responsibilities:complete current Backprop fields,
exact C/F arrays/canonical signed-zero bytes,required aliases,independent binding/
content checks and bounded detached snapshots. No IO,callbacks,pickle,ownership
attestation,policy construction,disk/live restoration or recovery admission.

All dtype/shape/contiguity/native graph checks occur before serialization. Exact
wire preflight includes C/F tags (same length),and unique-array bounds count the
required aliases once. On decode,all frame metadata and all bounded raw base64/
finite/traffic payloads validate before the first detached NumPy materialization.
Scalar `<d` unpacking performs finite validation without an early NumPy decode;
raw payload aggregate remains bounded by original unique-array limits. Then a
single detached native copy per array preserves C/F flags and alias edges.
Temporary raw/JSON/scalar allocations are bounded payload structures;these limits
do not establish total process RSS or original cumulative lifetime copy charges.
Strided/reversed layout is explicitly unsupported in v2. Vectors/singleton axes
with both contiguous flags use canonical C and reject ambiguous F tags.

### Backprop layout byte example (no updates or restoration)

```python
from hashlib import sha256
import numpy as np

from src.adapters.backprop_layout_checkpoint_codec import LayoutBackpropCheckpointCodec
from src.adapters.numpy_learners import BackpropLearner, BackpropSnapshot
from src.core.backprop_mlp import BackpropMLP
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits

native = BackpropMLP(3, 4, seed=23, hidden_dims=(4, 2))
native._hidden_weights[0] = np.asfortranarray(native._hidden_weights[0])
native.weight_input_hidden = native._hidden_weights[0]
native.bias_output[:] = -0.0
saved = BackpropLearner(native, learning_rate=0.03).snapshot_state()
binding = CodecBinding(sha256(b"fixture source").hexdigest(), sha256(b"fixture policy").hexdigest())
limits = CodecLimits(32768, 8192, 8, 64)
codec: CheckpointCodec[BackpropSnapshot] = LayoutBackpropCheckpointCodec()
raw = codec.encode(saved, binding=binding, limits=limits)
detached = codec.decode(raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits)
assert codec.encode(detached, binding=binding, limits=limits) == raw
assert detached.state["weight_input_hidden"].flags.f_contiguous
assert not detached.state["weight_input_hidden"].flags.c_contiguous
assert detached.state["weight_input_hidden"] is detached.state["_hidden_weights"][0]
assert not np.shares_memory(saved.state["weight_input_hidden"], detached.state["weight_input_hidden"])
```

The bindings in examples are fixture-only comparisons;they do not certify source
closure,consent,ownership or a composite restore. Learning rate remains an
independently provided learner policy. Native Backprop reports preupdate BCE loss,
not a separately defined energy. Continuation controls compare that exact native
metric,all state/prediction bytes and array storage/ownership/required aliases.
No new scientific metric or superior-model claim is introduced.

60v2controls plus51original v1controls run under a fresh declared fixture allowance.
Twelve fixed C/F/mixed and hidden-topology fixtures use seed23,original declared
4x3 batch,binary targets and learning rate0.03. Per complete invocation at most
36native updates/12fixture-local restores/0sleep/0structural. All prior spent
CPC/learner/worker/native fixtures remain spent;their unchanged source-bound
126CPC controls are reused as historical current-source evidence,not rerun.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest tests/test_backprop_layout_checkpoint_codec.py tests/test_backprop_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=artifacts/runs/backprop-layout-unused
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -X utf8 -m ruff format --check src/adapters/backprop_layout_checkpoint_codec.py tests/test_backprop_layout_checkpoint_codec.py
```

Extension:add the next complete explicit inbox/lifecycle/consolidation component
through the inner codec port after actual supported-state/ownership inventory.
No generic loader or scalar/ABSENT summary can satisfy full R3.5b2e authority.
Torch/composite/lifecycle/actor/sharing/cumulativecopy/singleowner/native/model/
coordinatorloss/scientific/human restoration criteria remain open.


## Complete supported NumPy inbox cursor codec (R3.5b2e3)

`NumpyInboxCheckpointCodec` selects wirev2/kind `numpy_inbox_v2` through the inner
`CheckpointCodec` port. It preserves native InboxCursor v1/v2 and every Experience,
permissions,candidate/action/reward,LabelArrival,AppliedExperience/diagnostic and
ErasedExperience field. Live paired/unmatched source/label history,consumed IDs,
erased event IDs,reasons/nullable source/label times,update numbering,learner/actor
versions,capacity,clock tick,stop/uncertain completed count remain exact. Decoded
cursors are detached observations;they grant no live/native/budget restore.

```text
src/core/inbox_codec_policy.py                original shape/dtype/record bounds
src/adapters/inbox_checkpoint_schema.py       complete metadata/role relations
src/adapters/inbox_checkpoint_payloads.py     bounded real numeric C/F frames
src/adapters/numpy_inbox_checkpoint_codec.py  typed byte orchestration
tests/test_numpy_inbox_checkpoint_codec.py    complete cursor/refusal controls
docs/adr/ADR-0227-encode-complete-supported-inbox-records.md
```

Why these boundaries:metadata validation precedes payload inspection,while payload
shape/dtype/ownership/byte validation stays outside the opaque inner records.
Existing domain/native/inbox APIs and all original source/test bytes are retained.
No dependency,environment variable or learning equation change.

### Original policy and complete payload scope

Caller supplies immutable `InboxCodecPolicy`:input width,batch rows,record capacity,
UTF8 identifier bytes,candidate count and allowed feature/target dtype tuples.
Encoded shape/dtype policy must exactly match that original policy;source/policy
and content digests are independently expected. They do not attest source closure,
physical provenance,consent,ownership or portability of live callbacks.

Actual current native batch validation accepts real integer/float arrays. The
initial f8-only195green tests did not establish this complete scope;their sources,
receipts and unaccepted wirev1 are retained. Wirev2 explicitly handles signed and
unsigned8/16/32/64 and float16/32/64 with both byte orders,including exact signed
zero bits and C/F storage. Current Windows NumPy2.4.6 longdouble maps f8;other
extended formats require separate explicit support. Object/boolean/complex/
foreign/strided/shared payload variants refuse explicitly. Source captured inboxes
own separate payloads per registration;undeclared additional ownership edges
are not normalized or dropped. Binary/soft labels preserve their native zero-to-
one range and both members of a pair must have the same row count.

Vectors are not inbox batches;payloads have exact N,input_dim features and N,1
labels. Singleton axes with both contiguous flags use canonical C. All metadata,
roles,training permissions,duplicate/event/pair/applied/tombstone/version/time/
count relations validate through existing inner records before payload inspection.
Candidate tuples and erased keys are bounded before list copies;strings check
character count before bounded UTF8 encoding. No unknown nested field is ignored.

Aggregate exact payload bytes (actual dtype widths) and complete prospective wire
length (order/dtype/schema/metadata/base64) precede serialization/materialization.
On decode,all metadata and all raw canonical/finite/range payload bytes validate
before any detached NumPy array. Scalar endian-aware unpacking avoids an early
array allocation on corrupt late data. Limits bound payload structures,not total
processRSS or original cumulative lifetime copy charges. Concurrent mutation of
caller snapshots requires the original owner lease;the codec grants none.

### NumPy inbox byte example (zero native work)

```python
from hashlib import sha256
from typing import Any
import numpy as np
from numpy.typing import NDArray

from src.adapters.numpy_inbox_checkpoint_codec import NumpyInboxCheckpointCodec
from src.core.checkpoint_codec import CheckpointCodec, CodecBinding, CodecLimits
from src.core.inbox_codec_policy import InboxCodecPolicy
from src.core.inbox_cursor import InboxCursor
from src.core.experience import Experience, ExperiencePermissions, LabelArrival

source = Experience("sample", "episode", 1, "actor", np.array([[0, 1, 2], [3, 4, 5]], dtype="<f4", order="F"), "train", ExperiencePermissions(training=True))
label = LabelArrival("event", "sample", "episode", 2, "actor", np.array([[1], [0]], dtype="|u1"))
saved: InboxCursor[NDArray[Any], NDArray[Any]] = InboxCursor(1, "learner", 8, (source,), (label,), (), 3, False, 0)
policy = InboxCodecPolicy(3, 4, 8, 128, 4, feature_dtypes=("<f4",), target_dtypes=("|u1",))
binding = CodecBinding(sha256(b"fixture source").hexdigest(), sha256(b"fixture policy").hexdigest())
limits = CodecLimits(65536, 8192, 8, 64)
codec: CheckpointCodec[InboxCursor[NDArray[Any], NDArray[Any]]] = NumpyInboxCheckpointCodec(policy)
raw = codec.encode(saved, binding=binding, limits=limits)
detached = codec.decode(raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits)
assert codec.encode(detached, binding=binding, limits=limits) == raw
assert detached.experiences[0].features.dtype.str == "<f4"
assert detached.experiences[0].features.flags.f_contiguous
assert detached.labels[0].targets.dtype.str == "|u1"
assert not np.shares_memory(source.features, detached.experiences[0].features)
```

Fixtures/guides perform0native updates/sleeps/restores/structural operations/model
predictions/workers. Actual ExperienceInbox integration records and captures only,
with a zero-update original local budget and spies refusing model operations.
Pure historical applied/uncertain/tombstone records test schema preservation;
they do not claim executed training or restored authority. Use a fresh declared
engineering envelope/newunused basetemp for any future verification.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest tests/test_numpy_inbox_checkpoint_codec.py tests/test_backprop_checkpoint_codec.py -q -o addopts= -p no:cacheprovider --basetemp=artifacts/runs/inbox-codec-unused
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform win32 --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m mypy --platform linux --no-incremental --cache-dir nul
.\.venv\Scripts\python.exe -B -X utf8 -m ruff check src tests scripts
.\.venv\Scripts\python.exe -B -X utf8 -m ruff format --check src/core/inbox_codec_policy.py src/adapters/inbox_checkpoint_schema.py src/adapters/inbox_checkpoint_payloads.py src/adapters/numpy_inbox_checkpoint_codec.py tests/test_numpy_inbox_checkpoint_codec.py
```

Extension:inventory and implement complete consolidation/lifecycle/consent/catalog/
revocation/retention/copy/owner records separately,then compose under the original
owner/resource/actor authority. Existing manager/lifecycle mutable state and live
ports are not in InboxCursor. Full R3.5b2e/Torch/actor/sharing/singleowner/native/
model/coordinatorloss/scientific/human criteria remain open;no scalar/ABSENT/pickle
summary can replace them. Earlier failed/spent/negative scopes remain immutable.
