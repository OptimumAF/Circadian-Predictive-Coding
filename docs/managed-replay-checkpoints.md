# Original managed replay checkpoints

## Complete history after canonical eviction

With `retain_inbox_origins=True` at fresh ledger birth, complete verified weak
inbox histories may supply pair lineage after canonical native replay eviction.
Default-off retains the existing conservative canonical-row requirement.

The coordinator admits every copied inbox pair under the same original ledger,
in addition to native-row and raw-copy charges. Actual source/label/receipt/array
memo identities bind copies. Original complete pair stamps and declarations are
checked after opaque ports; trusted weak rebinding precedes publication. The
optional history transition commits only prepared plain assignments after the
unchanged controller publication statements. No charge or validator runs after
that publication boundary.

R3.5b2e5b3d2b2b2e qualifies two trained inbox pairs with one canonical replay row:
actual capture/restore/handoff, original16-live exhaustion, missing enrollment,
late evicted payload/consent refusal and receipt identity replacement refusal.
Its fresh positive ledger was born with32 live slots; no existing allowance was
enlarged. Four fresh default-off native controls also pass. Exact source and
commands are in `artifacts/runs/r35b2e5b3d2b2b2e-inbox-history-20261008/`.

Mixed erased/live complete capture remains a conservative refusal: payload-free
historical receipts do not authorize a reconstructed erased pair. Compound
cleanup/TTL/public release, all native families, continued owned-copy consent
after canonical eviction, disk/RSS/recovery and scientific acceptance remain
open. Earlier “complete evicted inbox capture unfinished” notes below describe
their historical source state; the current slice does not complete its parent.

`app/managed_replay_checkpoints.py` composes the original row ledger, native fork
admission and actual graph copy observation with the existing checkpoint
controller. `app/checkpoint_replay_handoff.py` supplies its narrow final
publication lease. `core/checkpoint_content.py` supplies fixed bounded local
content checks after all opaque callbacks.

## Responsibilities and boundaries

Inputs: original `ManagedReplayOrigins`, bounded `ReplayGraphPorts`, the original
builder source port and originally enrolled controller. Outputs: the existing
controller's actual checkpoint token or published runtime. The manager charges
original admission and retained payload budgets before copies and binds weak
native row/array and inbox source/label/receipt witnesses from the actual memo.

The controller owns preparation, failed retained models and publication. The
manager prepares weak replacement anchors before retirement and commits only
original ledger fields after the original publication body. Original consent,
history, resource and held gate checks must remain valid through publication.
Why this: copy observations precede the controller's later native/policy probes.

This module implements no algorithm, training policy, persistence, cleanup,
unlearning, scientific scoring or recovery. Content digests establish integrity
only after actual identity and original authority binding. Raw replay array byte
charges do not measure native weights, Python heap or process RSS.

## Usage

```python
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
from src.core.replay_graph_origin import ReplayGraphPorts
from src.adapters.numpy_replay_copies import replay_builder_source
from src.adapters.numpy_replay_graphs import replay_graph_rows, replay_graph_payload_bytes

# ledger and controller must already belong to the original managed owner.
manager = ManagedReplayCheckpoints(
    ledger,
    ReplayGraphPorts(replay_graph_rows, replay_graph_payload_bytes),
    builder_source=replay_builder_source,
)
token = manager.capture(controller)
prepared = manager.restore(controller, token)
origins = ledger.origins()
```

## Supported bounded slice

Every live inbox pair must correspond to an originally verified retained row.
The fixed content proof accepts exact supported CPC/core inbox records,
contiguous real numeric arrays and PCG64 state, within original metadata/array
ceilings and fixed node/depth bounds. Unsupported types, opaque collection
callbacks and larger graphs are refused. Full native variants, untracked/erased
histories, complete capture and compound retention/recovery remain unfinished.

## Validation and extension

Current status: the declared original checkpoint handoff and public opt-out
slices are accepted. Evidence qualifies340 unique controls:322 inherited pure,
16 actual handoff/late-refusal controls and2 actual cleanup cases, with zero
skips. Both783-file type targets, lint, formatting, independent source/evidence
review, source/default/cache and resource audits pass. Actual public opt-out
clears registered original and failed-copy replay arrays and inbox payloads,
adds original tombstones, preserves payload-free receipts and cumulative charges,
and invalidates old checkpoint/copy/capture authority. The original failed native
scope remains excluded. Full native variants, eviction, expired weak targets,
general cleanup failures and all broader recovery criteria remain unfinished.
See the current native-qualification stage's `HANDOFF.md`.

Run pure controls and both configured type platforms before any native fixture:

```powershell
python -B -m pytest tests/test_checkpoint_content.py tests/test_checkpoint_replay_handoff.py -q -o addopts= -p no:cacheprovider
python -B -m mypy --platform win32 --no-incremental --cache-dir nul
python -B -m mypy --platform linux --no-incremental --cache-dir nul
python -B -m ruff check --no-cache src tests scripts
```

The native tests require a prospectively declared local scope; current exact
commands, outcomes and limitations belong in `docs/development-log.md` and
`artifacts/runs/r35b2e5b3d2b2b2b-native-qualification-20261008/`. A code example does
not establish current acceptance. Extend supported records through fixed inward
checks and original budgeted negative controls; preserve the original builder,
authority and lifetime counters rather than constructing substitutes.


## Native eviction and weak-target qualification

Two actual native cases additionally qualify a capacity set at original model birth, canonical eviction/pruning and old-checkpoint refusal before preparation, plus copied-target collection after explicit private holder reference release. Original admission identities, cumulative charges and permanent copied slots stay spent. This is not a public release API or TTL-expiry test. Source inspection distinguishes canonical eviction from consent revocation of admitted copies; continued owned copied-row consent after eviction remains an unqualified parent variant. Full capture of legitimately evicted inbox pairs also remains unfinished: current _original_inbox requires all live pairs to correspond to surviving canonical rows. Preserve conservative refusal until original lineage and budgeted history are implemented. Evidence: artifacts/runs/r35b2e5b3d2b2b2d-native-expiry-20261008/acceptance-audit.json;342 controls includes340 inherited and2 new, both784 type targets and all scoped gates PASS.

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


## Original erased transition qualification

R3.5b2e5b3d2b2b2g1 accepted: trusted original lifecycle deletion transitions a
previously committed live inbox witness to payload-free weak receipt/tombstone
lineage, paid through the same original admission before allocation. Original
native erasure and weak expiry, later distinct live training, clone/content
tamper refusal and refusal of unobserved retrospective lineage are qualified.
348 CURRENT qualified unique controls:316 fresh (270 default scalar +32 new
pure +4 default native +6 history native +2 consent native +2 N10 erasure) and32
byte-identical inherited core controls carried without rerunning,0skips. Both
full793-file type platforms/current full Ruff/source/cache/isolation/AST and
independent source/actual evidence gates PASS. Format checked7files; N10 changes
only two same-width constructor counter literals in that exact formatter output.
N9 FAILED/SPENT: positive passed, negative setup stopped at attemptedfork3>cap2;
original limits/source/XML/work retained and excluded. Independent diagnosis
proved unchanged setup needs6learners/4forks for2graphs. Separate prospective N10
corrected only those engineering counters within SAME600sec/24child/24MiB.
N10 actual2models6learners4forks3wakes6steps3stores3predicts15trainingcopies,
2cleanup attempts;0capture/preparation/nativeRestore/handoff/graph/memo/sourceB/GC.
Original birth live32/work2/retention1row64B/raw4096 preserved; witness reserves
exactly1 additional record + original encoded metadata +1024overhead, raw0 and
invocation0. All cumulative/permanent charges stay spent. No science experiments.
Full g/g2/ancestors/mixed capture/other native families/pending untrained tombstones/
general cleanup/TTL/public release/disk/RSS/recovery/scientific/human remain OPEN.
N9 FAILED/SPENT; N10/E1/F1/D2/H2/C2 SPENT; remaining allowances CLOSED at receipt.
Evidence: artifacts/runs/r35b2e5b3d2b2b2g1-erased-history-20261008/HANDOFF.md.
Exact next: Reconcile this exact terminal and resource closure. Begin R3.5b2e5b3d2b2b2g2 under a separately declared finite scope: preserve the complete disjoint union of the existing original live inbox witness and original erased receipt/tombstone witness through actual memo capture, copied materialization and checkpoint handoff. Start at ManagedReplayCheckpoints._Operation/_original_inbox/_verify_inbox and ManagedInboxOrigins.rebind; preserve exact original receipt/tombstone identities and scalar history, same admission/work/raw/metadata/live/retention authorities and cumulative/permanent charges. Do not simply relax applied-count checks, discard erased lineage, restore deleted payloads or grant consent from keys/digests. Declare one erased-trained plus one live-trained actual positive and original-budgeted negative controls before fixtures. N9 FAILED/SPENT, N10 SPENT and all earlier scopes stay closed; pending/untrained tombstones, broader erasure families and full g remain unchecked.

See docs/erased-inbox-origins.md and ADR0244. Complete mixed capture count checks remain unchanged until g2.


## Current complete mixed history qualification

R3.5b2e5b3d2b2b2g2 and parent R3.5b2e5b3d2b2b2g accepted for the stated bounded one-erased-trained
plus one-live-trained case, subject to final exact resource/readback receipt.
Complete disjoint applied coverage now survives actual paid memo capture,
materialization and prepared handoff. Original immutable scalar data and exact
receipt/tombstone lineage persist; no erased arrays or revived consent. Same
original admission/work/raw/metadata/live/retention authorities and cumulative/
permanent charges. Pure helper owns integrity only; no authority or IO.
387 CURRENT unique controls:323 fresh (32 mixed scalar +270 default scalar +4
native default +6 history +2 copied consent +2 original erasure +7 mixed) and64
exact inherited pure controls carried without rerunning;0skips. Full796-file
win32/Linux types, full Ruff, seven changed format, source/AST/default schema/
isolated author/cache/independent source and actual evidence gates PASS.
Initial003 type failure retained; narrow typing corrections before any fixtures.
Projected795 corrected to actual796 (one new module plus two tests). The proposed
six-graph scope was prospectively split to seven before authoring/execution so
clone and invocation exhaustion remain independent controls; no original runtime
budget, criterion, seed or metric was changed. Every fixture scope ONCE/SPENT.
N11 actual7models23learners16forks13wakes26steps13stores13predicts65trainingcopies,
4captures2preparations2nativeRestores1handoff30events6042sourceB228memo(max13),
7cleanup; within fixed declared caps. Independent MW49 exact parent acceptance
assessment covers every criterion at the explicitly bounded starting case;
untrained/broader erasure was expressly separate. All higher ancestors remain
unchecked. No science experiment, dependencies, download, remote, commit/push.
Evidence: artifacts/runs/r35b2e5b3d2b2b2g2-mixed-history-20261008/HANDOFF.md; acceptance-result.json; XMLs and actual work JSONs.
Exact next: Reconcile the exact g2 terminal/resource closure, then prospectively scope R3.5b2e5b3d2b2b2g3: inspect original inbox erasure and cursor behavior for an untrained tombstone beside a trained live pair. Define complete payload-free untrained tombstone provenance separately from erased applied receipts before changing partition counts. Preserve original authority, consent, receipt/order/work, weak expiry and permanent charges; declare a tiny actual positive and mutation/default-off/unenrolled/exhaustion negatives before fixtures. Keep all spent scopes closed; broader erasure, TTL/public release, disk/RSS/recovery and scientific/human gates remain unchecked.
