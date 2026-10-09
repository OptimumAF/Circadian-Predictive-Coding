# Complete managed source capture (qualification in progress)

`capture_numpy_managed_composite(owner, limits=...)` assembles supported retained
source state under the installed owner's original nonblocking gates. It preserves
native policy and state, complete inbox storage, current and previous actor
bundles, retained holder records, sharing flags, budget/progress and clock state.
Live locks, owners, callbacks, identity tokens and sampler handles remain original
authority references outside the copied graph. Their paths are observations, not
restore permissions.

**R3.5b2e5b remains unfinished.** Current positive controls qualify Backprop and
CPC candidate state, stable/promotable actors, inboxes, empty controller histories,
and synthetic shared rollback/replay edges. Populated pending histories, retired
holders, sampler concurrency/resource accounting and all native variants still
need complete qualification. Full composite encoding, durable recovery and model
restore are unfinished. Do not use this capture as a recovery authorization gate.

## Modules and boundaries

```text
src/core/managed_composite_state.py       bounded records and inward ports
src/app/managed_composite_sources.py      explicit complete source-field roles
src/app/managed_composite_capture.py      original leases/admission and orchestration
src/adapters/numpy_composite_capture.py   exact NumPy projection, validation and copying
tests/test_managed_composite_capture.py   bounded constructor/refusal/copy controls
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
