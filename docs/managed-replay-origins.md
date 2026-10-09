# Original candidate replay row origins

## Completed inbox history at original birth

Canonical native replay eviction can leave a trained source, label and receipt
in the inbox. Enroll their history before training when that complete capture is
needed:

```python
ledger = ManagedReplayOrigins(
    owner, ports, limits, window_limits, retain_inbox_origins=True
)
poll = ledger.train_ready()
```

The default is `False`. Existing or trained ledgers cannot enable the history
later. `core/inbox_origin.py` validates bounded metadata and original numeric
contents against the actual consumed inputs. `app/managed_inbox_origins.py`
retains weak pair witnesses, charges the same original admission before each
witness, attaches inserted receipts and qualifies only a successful final poll.
Neither module owns raw arrays or grants consent, copy or restoration authority.

History and copied pairs consume additional original metadata/live capacity;
choose limits at original birth. Original exhausted limits remain exhausted.
Tombstone pruning preserves cumulative charges and permanent copied slots.
Accounting counts metadata without requiring an expired weak payload to be live;
access still requires complete identity, content, consent and work verification.
See ADR-0243 and the current plan/log for qualified cases and open variants.

`ManagedReplayOrigins` is an opt-in ledger for one fresh managed candidate in the
current process. Enroll it before any update and with an empty native replay
buffer. Call its `train_ready()` instead of bypassing it through the owner. Read
`origins()` for bounded immutable metadata and `accounting()` for spent charges.
These records supply no checkpoint, fork, promotion, restore or erasure authority.

## Modules and composition

```text
src/core/replay_origin.py                 metadata, limits, monotone admission, ports
src/app/managed_replay_origins.py          original owner operation and weak row ledger
src/adapters/numpy_replay_origins.py        native NumPy reference and integrity ports
tests/test_managed_replay_origins.py        managed fake behavior and refusal controls
tests/test_replay_origin_ports.py           pure accounting and tiny array descriptors
tests/test_native_managed_replay_origins.py bounded actual CPC wake integration
```

The app depends on core ports and existing owner/runtime operations. The NumPy
adapter implements those ports; the app imports no adapter or infrastructure.
Existing native fields, snapshot schemas, copy behavior and retention policies are
unchanged. No environment variables, dependencies or configuration defaults change.

Construct `ReplayOriginPorts(replay_model_reference, retained_replay_references,
replay_copy_bytes, replay_payload_references, replay_payload_fingerprint)` using
the functions in `src.adapters.numpy_replay_origins`. Pass the original
`ManagedExperienceOwner`, explicit `ReplayOriginLimits` and `ReplayWriteLimits`
to `ManagedReplayOrigins`. These injected functions are trusted observational
ports: they must return actual original references, avoid mutation and retain no
raw payloads in closures. Keeping references elsewhere is caller-owned retention.

## Authority and failure behavior

The ledger binds original owner, runtime, actor, candidate, model, inbox, policies,
budget time origin, clock, consent guards and lineage. Actual producer callbacks
bind source, label and declaration identities. Before-copy observations charge a
row range. Copied observations bind the actual new snapshot and its actual arrays.
All raw references in persistent ledger records are weak. Identical content from
another subject does not inherit the previous subject's consent.

The original owner's operation gate encloses the shared update and final checks.
The runtime gate protects verification and final commit. A provisional completed
callback becomes usable only after original committed receipts, returned poll,
original work count, resource checks and retained inventory all agree. Native work
and receipts survive a post-update refusal. An exception after an admitted start
permanently makes the ledger uncertain; charges remain spent and origins refuse.
Pre-update refusals have no new row or invocation charge. Exceptions retain the
original managed observer behavior, including grouped secondary observer faults.

Reads recheck original source/label/declaration, current consent, revoked keys,
tombstones, original clock age, actual snapshot/array identity, integrity and
committed receipt identity. Unknown retained snapshots, foreign roots, weak-reference
expiry and changed metadata or payloads refuse. Clearing or evicting native rows
releases weak ledger entries during the next successful inventory operation and
never refunds lifetime charges. This module performs no payload erasure itself.

Why this: identity observations establish the producer; SHA256 detects subsequent
changes to an already bound object. A matching digest, portable record or copied
declaration cannot grant provenance. Metadata is not a consent certificate or an
unlearning claim. Trusted in-process code is not isolated from arbitrary private
attribute mutation; these checks do not form a security sandbox.

## Limits and accounting

Limits bound live rows, cumulative created rows, started invocations, cumulative
metadata accounting bytes, source age in the original logical clock, and bytes
examined per native payload fingerprint. Separate write-window limits bound input
rows, retained references and notification attempts. Charges occur before native
copy, including copies later evicted or rejected. They never renew or refund.

`metadata_bytes_charged` charges exact canonical UTF-8 JSON metadata plus **1024
accounting units** per row and per invocation. This deterministic admission measure
is not a measured Python heap ceiling or process RSS. JSON string rendering is
bounded before allocation. Fingerprints use existing contiguous array buffers and
include dtype, shape, strides, writeability and native scalar descriptors; they
allocate no raw array copy. Original `ToyBudgetSession` wall/work/RSS checks remain
active. Persistent references, temporary metadata, dictionaries and Python objects
still have physical overhead. No unmeasured physical byte guarantee is claimed.

## Pure accounting example

This creates metadata only: no model, payload array, worker, snapshot or update.

```python
from src.core.replay_origin import ReplayOriginAdmission, ReplayOriginData, ReplayOriginLimits

limits = ReplayOriginLimits(2, 4, 2, 16384, 120, 4096)
admission = ReplayOriginAdmission(limits)
data = ReplayOriginData(
    ("episode", "sample"), "label", "actor", "candidate", "subject", "source",
    0, 1, 1, 3, 1, 32, "0" * 64,
)
admission.start(0)
admission.reserve(data, 0)
spent = admission.accounting(1)
assert spent.records_created == spent.invocations_started == 1
assert admission.accounting(0).metadata_bytes_charged == spent.metadata_bytes_charged
print("bounded metadata charges survive eviction")
```

## Validation and next extension

The development log and stage receipts record exact pytest, Ruff, both platform
mypy and guide commands, failures, repaired fixtures, skips and actual native work.
The selected controls exercise managed source identity, duplicate replacement,
eviction, consent revocation, tombstones, expiry, corruption, reentry, admission and
post-update failure. Native integration has a separate fixed allowance after the
initial correctness, typing, static, cache, source and resource gates.

Next qualify retained copies/holders and ledger enrollment before capture. Fork,
actor, checkpoint, promotion, restore, erasure, compound alias histories, canonical
bytes and live/disk/model/coordinator-loss recovery remain unfinished. Extend ports
and explicit original holder contracts with refusal tests before enrolling another
path. Full R3.5b2e5b3/e5b/e5 and scientific gates remain unchecked.


## Subsequent capture admission

[Composite replay admission](replay-capture-origins.md) uses a ledger-only lease
inside original owner/runtime capture gates. Capture attempts now also spend
one original lifetime invocation and1024metadata accounting units;failure never
refunds. Expired callbacks release their own strong graph references. Original
training behavior and row schema unchanged. Other-holder lineage remains open.


Qualification status: native capture timed out in d1. The leased-consent source
correction is pending fresh regression/type/native qualification. d1 remains
unchecked; pre-timeout green evidence does not qualify current source.


## Leased lifecycle repair qualification

Capture uses the original lifecycle's leased retention accessor for both runtime
open checks and row declaration checks. The runtime check retains original
retired/stopped checks and requires original registry/lifecycle identity. Ordinary
public ledger operations retain the ordinary lifecycle consent path. Capture
already owns the time gate; using its leased accessor preserves elapsed retention.
An installed-lifecycle/fake-learner regression patches class ordinary elapsed to
fail immediately while holding that gate. It covers valid capture/public reads,
revocation/optout,logical and elapsed expiry,rewind,changed guards and cleanup
failure. Pre-open/guard refusals have no new capture invocation charge; admitted
source refusals preserve the spent invocation/metadata charge. No native defaults,
instance schemas,environment variables or dependencies change.

Why this:the first real-lifecycle regression showed an earlier reentry through
runtime._require_open,with an exact traceback. Both earlier source diagnosis and
native timeout remain preserved. Full retained-holder lineage and recovery remain
unfinished. See new repair-stage receipts for current qualification;earlier failed
stage evidence remains historical.


## Current qualification — mandatory replay admission accepted

R3.5b2e5b3d1 now passes1099unique controls,both757type targets,static/guide/source/
cache/resource gates. Native capture completes with original elapsed-retention
tripwire;missingledger and copiedactor rows refuse before projection/copy. Old
unvalidated/failed notes above are preserved historical handoffs,not current status.
Original55s timeout stays failed/countsunknown. Actual freshnative4wakes20copies;
ordinarypublic default behavior preserved. Original retained-holder copy issuance,
fullvariants/history/bytes/recovery/scientific gates remain unfinished. See latest
development log and repair-stage evidence;next d2 creates actual holder witnesses.
