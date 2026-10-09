# Original managed replay-copy witnesses

`ManagedReplayCopies` wraps the existing `CandidateCheckpointController.restore`
operation. Its constructor accepts the original `ManagedReplayOrigins` ledger and
a trusted builder-source port. The NumPy port supports the original exact
`ManagedNumpyBuilder` whose source is the original CPC candidate.

Why this: the controller owns a built learner in `_models` before policy checking
or native restore. Even a failed preparation retains real copied replay rows.
Observe that actual constructor copy rather than infer origin from matching data.

Before allocation, source row consent, tombstones, original receipts, budget,
clock, controller enrollment, builder and preparation history are verified under
the original owner/runtime and copy registry/time/resource leases. Additional
replay bytes are reserved in the original payload budget. Witness metadata and
invocations consume the same original ledger allowance used by training/capture.
Copied live metadata slots remain conservatively reserved after faults, target
collection or coordinator disposal. A new coordinator cannot renew them.

The copier's original memo binds new model/row/input/target identities. Persistent
witnesses hold weak references. `origins(controller, learner)` checks the actual
original controller model position, preparation attempt and enrollment before
returning copied row metadata. Modified, foreign, revoked, expired, unwitnessed
or restored-without-witness rows refuse. Integrity digests verify already bound
identities; they do not identify the original producer.

```python
from src.app.managed_replay_copies import ManagedReplayCopies
from src.adapters.numpy_replay_copies import replay_builder_source

# Construction only: no copies, model operations or new allowance are issued.
assert callable(replay_builder_source)
assert callable(ManagedReplayCopies.restore)
```

These witnesses certify copied row origins only. They do not certify native policy
acceptance, successful restoration, snapshot lineage, handoff history, whole model
integrity, full capture or scientific permission. A later restore can replace
constructor-copied rows and needs a separately observed original chain. Other
holders still refuse nonempty replay capture. Original checkpoint behavior and
results/errors remain owned by its existing controller. Native schemas, defaults
and retention policy are unchanged. Metadata units are not a heap/RSS ceiling;
the raw-copy budget covers replay storage, not all parameter storage.

## Current owned-copy consent qualification — October8

R3.5b2e5b3d2b2b2f accepted: continued originally admitted copied-row metadata
after legitimate canonical eviction; direct and final actual fingerprint consent
revocation refuse.314 CURRENT qualified unique controls:312 exact inherited
production/test/default-helper controls carried without rerunning fixtures+2new
N8 native controls,0skips. No production change. Both full789-file type targets,
full Ruff/format/independent source and actual evidence/source/cache/isolation
gates PASS. N7 failed2cases before consent due incorrect raw-charge equality
across legitimate SECOND training; immutable failure retained and SPENT. N8 was
separately declared before its fresh2graphs, within SAME600sec/24child/24MiB cap.
Correction asserts exact SECOND ingress32+consumed32 raw bytes; metadata reads
spend exactly1 original invocation+1024metadata units and0raw bytes/new records/
copy slots. Original birth16live/retention1row64B/work2/raw4096 preserved.
N8 actual2models8learners6forks4wakes8steps20trainingcopies2captures2preps/
2nativeRestores0handoffs16events3416sourceB96memoReads(max8),0cleanup/GC.
N7 has the same actual counts, failed evidence excluded from acceptance.
All full ancestors/mixed erased-live capture/other native families/general
cleanup/TTL/public release/disk/RSS/recovery/scientific/human gates remain open.
N7 FAILED/SPENT; N8 PASS/SPENT; unused allowances CLOSED at resource receipt.
Evidence: artifacts/runs/r35b2e5b3d2b2b2f-owned-consent-20261008/HANDOFF.md.
Exact next: Reconcile this stage terminal and resource closure with the checkout. Begin R3.5b2e5b3d2b2b2g under a separate prospective finite scope: implement an original-admitted payload-free erased receipt/tombstone witness during trusted erasure of a previously birth-observed committed pair, then preserve complete disjoint live-plus-erased applied history through actual memo capture/materialization and handoff. Start at ManagedInboxOrigins.prune, ExperienceInbox._prepare_erased_history/_commit_erased_history and ManagedReplayCheckpoints._Operation/_original_inbox/_verify_inbox. Do not simply relax receipt-count checks, drop erased history or reconstruct provenance from current keys/digests. Preserve original admission, work, raw-copy/metadata/live/retention limits and permanent charges; N7 FAILED/SPENT, N8 SPENT and all old scopes stay closed. First declare exactly one erased-trained plus one live-trained case and original-budgeted negative controls; broader variants remain unchecked.
