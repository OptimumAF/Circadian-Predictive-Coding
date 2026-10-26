# Complete managed source capture (qualification in progress)

## One current update after restore

Successive single-update polls can now continue this same capture path, subject
to the original birth budgets. The latest paid success is verified before the
next poll and replaced only after original checks succeed; earlier charges are
never returned. The original retired publication stays fixed. The existing
section anchor is retained for links.

A deferred poll after a success also pays for a bounded observation of the
returned reason and its exact counter change. It preserves the native success
identity, receipt and completed count, so pause and resume can continue safely.
It grants no training allowance; manual counter changes still refuse admission.

For a trained, unerased retained publication, construct the original ledger with
`retain_inbox_origins=True` and the original manager with
`retain_capture_history=True`. After the original restore, admit the next update
through `ledger.train_ready()`. The existing capture API continues to use
`replay_origins=ledger, replay_checkpoints=manager`; no new call or authority is
required. The runnable `--restored-replay` example still demonstrates the earlier
unchanged-publication branch.

Each successful poll pays bounded metadata to the original admission, without a
new origin or update allowance. Reserve enough live-record capacity at original
construction for retained copy reservations and the next inbox/replay witnesses;
reservations remain spent after eviction. The two-update qualification uses 32
live records from birth and preserves exhaustion at 16 as a refusal control.
Capture checks retired native/inbox history and
the complete current success, including original receipts, consent and birth
inbox history after replay eviction. Manual state or policy changes, foreign
holder roles and late copy mutation refuse; late charges remain spent. Existing
normal clock/RSS telemetry can advance under original resource checks. Configure
the original lifetime metadata allowance for the intended number of paid
successes; each manager observation remains bounded to 128 KiB. Multi-update
polls, chained restores and erased/untrained continuation remain unfinished. See
[ADR-0306](adr/ADR-0306-seal-one-admitted-current-update-after-publication.md).

For exact coupled PC/CPC learners, see the separate reusable
[coupled managed capture API](coupled-managed-capture.md). Its focused and native
qualification does not close the broader historical capture/recovery criteria
below. It retains original authority separately from detached data.
The [coupled data codec](coupled-composite-codec.md) adds bounded graph bytes
without closing legacy/all-variant capture or managed recovery qualification.
The [live checkpoint mode](coupled-composite-files.md#save-from-the-live-managed-serving-demo)
now exercises coupled B/C capture/files and original continuation/cleanup. The
legacy example below adds actual populated pending/retired Backprop/CPC capture
with naturally empty replay. Original R3.5b2e5b all-variant and nonempty retained
replay lineage qualification remain open.

`capture_numpy_managed_composite(owner, limits=...)` assembles supported retained
source state under the installed owner's original nonblocking gates. It preserves
native policy and state, complete inbox storage, current and previous actor
bundles, retained holder records, sharing flags, budget/progress and clock state.
Live locks, owners, callbacks, identity tokens and sampler handles remain original
authority references outside the copied graph. Their paths are observations, not
restore permissions.

**R3.5b2e5b remains unfinished.** Existing controls qualify Backprop/CPC candidate
state, stable/promotable actors, inboxes, empty controller histories and synthetic
shared rollback/replay edges. R3.5b2e5b4 adds actual populated pending histories,
retired/current holders and original resource/copy accounting for the fixed
naturally-empty-replay composition below. Nonempty retained replay lineage,
broader native variants/concurrency, full legacy composite encoding, durable
recovery and model restore remain unfinished. Capture grants no recovery authority.

## Actual legacy populated-holder example

```powershell
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --describe
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --run
```

The bounded Backprop/CPC example uses real checkpoint capture and restore to
retire the original candidate, then retains an issued pending checkpoint and a
prepared promotion. It captures populated inboxes, cache/metadata and original
sampler/resource observations. All updates remain zero; the CPC replay buffer is
naturally empty. It is never cleared or disabled to pass admission.

Existing `ManagedNumpyBuilder` and native codecs bind policy/state; the original
owner, consent catalog, sharing gate, budget, logical clock, sampler and copy
ledger remain authoritative. Full capture reserves original retained raw bytes
before its two passes through one memo. Reported native snapshots are caller-owned
observations outside that raw ledger. The example uses injected constant RSS/time,
not physical-memory or latency measurements. Its two native objectives remain
unchanged; no learning benefit or comparative result is claimed.

Nonempty replay in copied holders still requires qualified original row lineage.
Captured fields or matching keys cannot issue that authority. Refusal occurs
before native projection/payload copying, and this example does not reconstruct
an owner or grant restore/process recovery permission. All original b/all-variant,
nonempty-replay, byte-format and durable/live recovery requirements remain open.

Source: `src/adapters/numpy_capture_example.py`; focused controls:
`tests/test_numpy_retained_capture.py`. Exact resource limits/commands/results:
`artifacts/runs/numpy-retained-capture-20261015/SCOPE.md` and `RESULTS.md`.

## Runnable issued snapshot example

```powershell
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --describe --issued-snapshot
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --run --issued-snapshot
```

This opt-in mode trains one fixed CPC batch with two requested inference
iterations and sleep disabled, then genuinely issues one still-pending checkpoint.
The original ledger and issuing manager qualify its nonempty replay for complete
capture. The report contains complete native and inbox component data and every
remaining issued-view field, plus original identity/resources/copy accounting.
Component codec bytes are data observations and grant no restore permission.
Joint alias/detachment checks apply to the captured graph separately.

The original copy allowance is declared at birth; retained raw bytes and each
paid charge are reported without refunds or resets. Logical time and budget time
are deterministic; this example measures neither host RSS nor comparative learning
benefit. It performs no restoration, handoff or promotion. Default CLI mode keeps
its two zero-training Backprop/CPC cases. Source:
`src/adapters/numpy_snapshot_capture_example.py`; bounded validation:
`artifacts/runs/numpy-snapshot-demo-20261015/`.

## Runnable successive replay example

```powershell
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --describe --successive-replay
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --run --successive-replay
```

The fixed CPC composition performs three updates with seed23 and sleep disabled.
It issues and actually restores one checkpoint after the first update, then
observes a paused poll after the second update and resumes for the third. Both
history flags are enabled at construction. The retired model remains at one
update; the current model reaches three, with two retained rows. Superseded
success evidence and evicted row references are released without refunding
payments. The original consumed checkpoint view remains unchanged.

Work is fixed before setup: three allowed updates, 32 live origin records and
256KiB lifetime metadata. Each observation stays bounded to128KiB; the original
copy and ingress allowances remain4096B. The report shows raw retained256B,
charge800B before capture and1056B after, separately paid original publication
and latest success metadata, full native/inbox components and joint aliases.
These are deterministic engineering observations, not host memory/performance
measurements or evidence of a learning advantage.

`--successive-replay`, `--restored-replay` and `--issued-snapshot` are mutually
exclusive and require `--describe` or `--run`. Output is bounded to131072 UTF8
bytes including physical newlines. Existing modes retain their original fixed
workloads. Complete data carries no portable authority or recovery permission.
Chained restores, other holder branches, erasure and full managed recovery remain
unfinished. Source reuses `src/adapters/numpy_restored_capture_example.py`; focused
checks: `tests/test_numpy_successive_demo.py`; native CLI composition:
`tests/test_native_numpy_successive_demo.py`. Budget/evidence:
`artifacts/runs/numpy-successive-demo-20261016/`.

## Runnable restored replay example

```powershell
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --describe --restored-replay
.venv/Scripts/python.exe -m scripts.run_numpy_managed_capture --run --restored-replay
```

This mode reuses the same fixed CPC batch, seed, objective and tiny original
budgets. Setup performs one update, issues one checkpoint and genuinely restores
it with history retention enabled at manager birth. It then captures the actual
retired and current replay holders through the original manager and ledger.
The report contains complete native/inbox component data, historical and current
update counts, the checkpoint/current learner alias, and original paid metadata
and raw-copy accounting. The consumed caller-held checkpoint view stays outside
pending snapshot authority and is checked unchanged.

Capture/report uses borrowed projections and existing codecs without native
methods. Output is bounded to 131072 UTF8 bytes including physical newlines.
`--restored-replay` and `--issued-snapshot` are mutually exclusive; either needs
an explicit action. Existing default and issued-snapshot modes remain available.
Injected time and resource observations are deterministic, not physical RSS or
performance measurements. No learning advantage or recovery authority is claimed.
This supports the unchanged restored pair; subsequent work, chained handoffs,
other roles, erasure, portable bytes and durable/live recovery remain open.
Source: `src/adapters/numpy_restored_capture_example.py`; validation:
`artifacts/runs/numpy-restored-demo-20261015/`.

Pure routing/report checks:
`.venv/Scripts/python.exe -m pytest -q tests/test_numpy_restored_demo.py`.
The separately budgeted actual CLI test is `tests/test_native_numpy_restored_demo.py`.

## Nonempty issued checkpoint snapshots

Pass the original manager together with its original ledger:

```python
captured = capture_numpy_managed_composite(
    owner,
    limits=limits,
    replay_origins=ledger,
    replay_checkpoints=manager,
)
```

The manager must be the actual issuer of a still-pending checkpoint. Its original
copies are selected internally; an explicit `replay_copies` must be that same
object. Every appearance of the snapshot must belong to the original checkpoint
snapshots role with the actual issued view. The reader checks original rows,
inbox partitions, receipts, consent, state/history and nonblocking gates before
and after copying. One original ledger admission and cumulative raw-copy charges
apply; a late refusal retains its charge. Matching contents or a constructor-copy
witness alone cannot qualify a snapshot.

This snapshot branch supports unchanged original runtime/history. The separate
publication branch below covers one restored pair. Later-current history, other
roles and durable recovery remain open.
See ADR-0304 and checkpoint-snapshot-capture-20261015 for bounded validation.

## Nonempty replay after one real handoff

Construct the original `ManagedReplayCheckpoints` with
`retain_capture_history=True` before issuing or restoring its checkpoint. After
one successful `manager.restore(controller, token)`, use the same complete
capture call above with the original manager and ledger. The flag defaults to
`False`; enabling it later cannot recover proofs from a past publication.

```python
manager = ManagedReplayCheckpoints(
    ledger,
    graph_ports,
    builder_source=builder_source,
    retain_capture_history=True,
)
```

The handoff retains weak original old/current runtime, learner, model and inbox
identities with their already paid row witnesses. Additional bounded logical
metadata pays its exact canonical UTF8 size plus 1024 bytes from the original
admission before publication. No allowance is renewed. Native replay and source
consent are checked through original producer records; matching bytes cannot
create lineage. Complete capture checks every holder role, including the shared
current learner in the checkpoint model list, before and after its two passes.
Late failure retains the raw-copy charge. Capture invokes no native learning,
snapshot or restore operation and grants no restoration authority.

This branch requires the unchanged original publication history. Further current
updates, new checkpoint preparations, chained handoffs, promotions, erasure and
recovery remain unfinished. Logical metadata and raw payload accounting do not
measure process RSS. Source: `src/app/replay_restored_capture.py`; decision:
ADR-0305; bounded validation: `artifacts/runs/restored-replay-capture-20261015/`.

## Nonempty retained checkpoint constructor copies

The optional `replay_copies` argument admits nonempty replay in an originally
witnessed checkpoint constructor-copy model. Supply the original live objects:

```python
captured = capture_numpy_managed_composite(
    owner,
    limits=limits,
    replay_origins=ledger,
    replay_copies=copies,
)
```

`copies` must be the exact `ManagedReplayCopies` bound to that original ledger.
It must have observed the actual constructor source/target/memo and recipient at
creation. New capture refuses old unbound witnesses and a replacement learner
wrapper even around the same model. Existing public `copies.origins()` callers
retain their prior behavior; they cannot be called inside capture's held leases.

The app reader uses the already held original leases and one original ledger
invocation/metadata admission. It issues no witness or restore permission. Every
inventory round checks original controller enrollment, learner/model/list/position,
builder/policy/attempt history, original paid rows, consent and integrity. The
same model in an actor, candidate, promotion or snapshot role refuses even while
the checkpoint controller still holds it. All scans use bounded reference ports;
no native fields are added. Late refusal keeps the original copy charge.

The native positive qualifies a model retained after a real observed constructor
copy followed by a policy-preparation failure. Its original pending snapshot is
discarded through the public API; replay remains nonempty and unchanged. That
snapshot remains unqualified without its original issuing-manager witness.
Constructor-copy witnesses alone do not qualify successful native restoration/
publication; the separate birth-opt-in publication proof above covers one
unchanged restored pair. Other copied holder roles, erase and compound variants
remain open under R3.5b2e5b3d2/d2b. See ADR-0303 and
`artifacts/runs/retained-replay-capture-20261015/RESULTS.md` for exact evidence.

## Capture modules and boundaries

```text
src/core/managed_composite_state.py       bounded records and inward ports
src/app/managed_composite_sources.py      explicit complete source-field roles
src/app/managed_composite_capture.py      original leases/admission and orchestration
src/app/replay_copy_capture.py            original retained checkpoint replay checks
src/app/replay_checkpoint_capture.py      original issued snapshot witness checks
src/app/replay_restored_capture.py        paid weak original publication checks
src/adapters/numpy_composite_capture.py   exact NumPy projection, validation and copying
src/adapters/numpy_snapshot_capture_example.py fixed trained issued-snapshot demo
src/adapters/numpy_restored_capture_example.py fixed trained restored-pair demo
scripts/run_numpy_managed_capture.py     explicit describe/run and opt-in modes
tests/test_managed_composite_capture.py   bounded constructor/refusal/copy controls
tests/test_checkpoint_snapshot_capture.py fake snapshot witness/consent/lease controls
tests/test_native_checkpoint_snapshot_capture.py actual one-graph snapshot gate
tests/test_numpy_snapshot_demo.py        pure CLI and output-bound controls
tests/test_native_numpy_snapshot_demo.py actual one-graph runnable CLI gate
```

The source table names each field as data, nested source state, native state or
an original reference. Unknown fields/classes refuse. Source records contain
qualified class names and ordered complete field tuples; input data cannot select
a class loader or callback. Native projections retain every supported native
model field through existing explicit Backprop/CPC validators. The app imports
only inward port definitions, not NumPy adapters.

Why this: separate snapshots lose aliases and can describe different source
epochs. Projection reads references without copying payload arrays. After original
consent/retention/elapsed-budget checks, quiescence, bounded full-graph preflight
and cumulative payload admission, one memo detaches the whole payload graph. The
copy port restores array writeability flags; dtype, shape and C/F storage survive.
Repeated array/record/container identities remain repeated; distinct overlapping
arrays and cycles refuse. Failure after admission retains original charges.

The original sampler gate is acquired without waiting before any RSS read.
Original RSS/wall/retention checks run before and after the payload copy. A second
pass through the same memo copies fresh final observation records, reusing every
already detached payload array. Budget and progress share the final RSS segment.
Terminal, unavailable, invalid or changed sampler sources refuse; the original
baseline, peak/count, elapsed origin and consumed copy charges are preserved.

Retained raw payload accounting uses the installed original measurement ports.
`PayloadCopyLimits` continues to exclude native parameters, Python overhead and
RSS. `CompositeCaptureLimits.max_array_bytes` independently bounds all unique
arrays, including parameters. The node/depth/dimension limits bound projection
validation. These are finite capture bounds, not a lifetime parameter allowance
or a hard process memory ceiling.

## Pure record example

```python
from src.core.managed_composite_state import (
    AuthorityPath, CompositeCaptureLimits, SourceRecord,
)
from src.core.managed_lifecycle_state import LifecycleCaptureLimits

limits = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)
limits.__post_init__()
clock = SourceRecord("src.core.experience.LogicalClock", (("_time", 0),))
assert dict(clock.fields)["_time"] == 0
original = AuthorityPath("clock.original", True)
assert original.present  # Presence alone is not original-owner provenance.
```

For an already installed, paused managed NumPy owner, compose the outer adapter
with these independent limits. It returns `ManagedCompositeCapture` and never
constructs a replacement model or transfers live ownership.

## Verification

Each pytest invocation needs an unused basetemp and a fresh declared fixture
allowance. The current complete-control invocation is retained under
`artifacts/runs/r35b2e5b-composite-capture-20261008/third.xml`.

```powershell
python -B -m pytest tests/test_managed_composite_bindings.py tests/test_managed_capture_resources.py tests/test_managed_composite_capture.py tests/test_managed_record_checkpoint_codec.py tests/test_lifecycle_checkpoint_codec.py tests/test_process_memory.py -q -o addopts= -p no:cacheprovider --basetemp=<new-path>
python -B -m mypy --platform win32 --no-incremental --cache-dir nul
python -B -m mypy --platform linux --no-incremental --cache-dir nul
python -B -m ruff check src tests scripts
python -B -m ruff format --check src/core/managed_composite_state.py src/app/managed_composite_sources.py src/app/managed_composite_capture.py src/adapters/numpy_composite_capture.py src/app/managed_data_lifecycle.py tests/test_managed_composite_capture.py
```

## Exact remaining work

Qualify populated checkpoint/promotion histories and all original reference
relationships, retired/tombstoned/stopped holders, native variants and sampler
concurrency before checking e5b. In particular, inspect RSS sampling/locking and
post-copy budget observations; schema coverage alone proves neither quiescence
nor current resource admission. Prove mixed-owner/epoch refusals and full field/
array/reference preservation against actual source inventory. Keep existing
successful components and negative results. Then implement explicit bounded
canonical composite bytes with independently supplied original bindings and all
aliases. Only after complete correctness gates declare tiny native restore work.

## Original sampler observation example

This pure local example starts no worker and constructs no model. For managed
capture, the app supplies this capability directly to the original budget while
all original gates remain held. RSS values here are injected test observations.

```python
from src.shared.process_memory import ProcessRssSampler

sampler = ProcessRssSampler(read_rss_bytes=lambda: 100)
sampler.sample()
with sampler._lease_observation() as observe:
    segment = observe()
    assert segment is not None
    assert segment.start_bytes == 100 and segment.sample_count == 2
try:
    observe()
except ValueError:
    pass
else:
    raise AssertionError("expired observation capability accepted")
assert sampler.sample_count == 2
```

R3.5b2e5b1 evidence is under
`artifacts/runs/r35b2e5b1-resource-admission-20261008/`. Its bounded controls cover
original admission and metadata refresh; the full e5b qualification listed above
is still required. Arbitrary reader callbacks must return without calling public
sampler methods that reacquire the held lock. The lease refuses recursive use of
its own observation capability. It does not impose a timeout on external code.

## Retained pending source bindings

The app's `managed_composite_bindings.py` validates exact original runtime,
controller, pending and token fields before holder payload ports or native
projection read them. Every pending runtime must belong to the original leased
registry. All retained runtimes use its actor, lineage, budget, clock and installed
consent/copy guards. Retired ledgers keep historical work without resetting the
current cumulative budget.

Complete graph bounds precede checkpoint integrity and pure promotion decision
checks. Checkpoint inbox payload shapes use the independent native input dimension;
their consent is checked even when the current inbox is empty. Captured budget
origin/policy and archived sampler chronology must describe original monotone
history. Original pending references and consent are checked again after copying;
admitted failures keep charges.

Promotion tickets include their portable fields in the detached graph. Their
original identity and the original rollback receipt remain authority references
outside it. Copied tokens grant no controller or recovery permission. A stale
checkpoint remains retained data; capture never invokes its restore guard.

### Pure bound-method example

```python
from src.app.managed_composite_bindings import require_bound_method

class Owner:
    def approve(self):
        raise AssertionError("capture must not invoke this guard")

owner = Owner()
installed = owner.approve
require_bound_method(installed, owner.approve)
try:
    require_bound_method(Owner().approve, owner.approve)
except ValueError:
    pass
else:
    raise AssertionError("foreign bound-method owner accepted")
```

Evidence: `artifacts/runs/r35b2e5b2-source-bindings-20261008/`. Bounded issued
checkpoint histories are distinguished from synthetic promotion/retired-history
fixtures. Fixture digest probes are deterministic stubs; they do not prove native
provenance. Actual promotion issuance/retirement, all native variants, replay
consent, complete provenance, canonical composite bytes and recovery remain open.
Extend this module through explicit original relationships and bounded refusal
controls. Do not reuse historical fixture allowances.


Original managed native update observation (ADR-0238):optional native_observer
ports in inbox/runtime/sharing/managed owner expose original source/label/learner,
actual detached inputs and committed receipt/spent count through synchronous
expiring access. See docs/native-update-origin.md. Original consent/admission,
owner instance fields,default calls and update order remain unchanged. Callback
faults preserve original failure/receipt/resources;returned references remain
caller-owned. No new configuration/dependency/environment variable. Core defines
the reference contract;app manages lifetime;neither imports adapters/infra.
Persistent replay origin and every storage/retention/dedup/eviction/fork/checkpoint/
promotion/restore/erase path remain open under R3.5b2e5b3. Complete compoundcapture,
canonical bytes and recovery gates remain unchecked. Do not infer row lineage
from content hashes or treat these observations as consent/restore permission.
Tests:fixed fake-only origin controls +five selected fake inbox controls +990
current composite/resource/codec controls;both742types/wholeRuff/check format.
For safe extension:add a bounded original replay-write/row port with weak or
owned-accounted payload references;preserve terminal failures and original gates.
