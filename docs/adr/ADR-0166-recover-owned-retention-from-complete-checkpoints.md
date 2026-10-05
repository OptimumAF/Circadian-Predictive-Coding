# ADR-0166: Recover owned retention from complete original checkpoints

Status: accepted and implemented; complete correctness/publication/readback
acceptance passes for P6.10b1.

## Context

P6.10a preserves all original resource scopes, but 300 raw method projections
omit owned retention. The original complete 134,554,378-byte training result
already stores every initial/after-A/after-B retention view and canonical
owner state. Saved metadata inspection b438ffb1...4162 covers all 1,680
checkpoints without model/data/train/final access. The unchanged original
training reader independently validates every state, role, raw work, observed
resource and current request/source/input identity without constructing models.

Each checkpoint stage contains 290 baseline arms with no owned buffer and
70 zero-memory circadian arms with a null view and empty maxlen-zero deque.
The remaining 200 arms have a configured empty buffer initially and eight
retained rows after A/B. Sorted content IDs and ordered array fingerprints
are available; actual sample arrays are not stored in this derivative view.

## Decision

Add separate app checkpoint/projection/rendering and infra reader/reference/
binding/artifact modules with a fixed CLI. Freeze new sources and the complete
request before any new development fixture. Preserve the original 114-source
inventory/evidence set within the new source closure and all original artifact
identities, scientific settings, caps, seeds, baselines and metrics.

Read both complete original training bundles through their unchanged actual
reader, verify each decoded part against the whole bound file and compare
complete projections. Discard each large training graph before the next read.
Store every nullable retention view, original replay owner field, ordered
input/target array fingerprint and exact JSON pointer, alongside checkpoint
state/parameter and full original role identities. Do not reopen samples or
claim a fresh content-ID rehash from array fingerprints.

Derive owned input/target array bytes from independently validated <f8
geometry. Preserve null retention views. Derive zero bytes for a baseline
only after closed-state verification proves no owned replay buffer, and for a
disabled circadian arm only after its explicit empty deque is verified. Keep
configured-empty and retained states distinct. A baseline's zero owned arrays
does not imply zero external replay supply, process RAM or compute.

Keep one shared FIFO per original context. Initial shared zero is a protocol
derivation from the pinned empty constructor before the first wake; later
values/IDs come from the exact stage's last recorded boundary. Reconcile every
group and each stage independently. Never sum stage snapshots as simultaneous
memory or allocate shared storage by arm count. Checkpoint copies, Python
overhead and per-arm RSS remain unmeasured.

Declare a fixed 180-second local derivative budget before fixtures/publication.
Fresh complete training readers and decoded-byte validation are heavier than
the prior preservation-only inventory; original cost/report validation took
75/178 seconds. Original scientific 16,000-update/600-second/512-MiB caps stay
unchanged. Require exclusive deterministic publication/repetition and both
complete independent reader reconstructions before checking P6.10b1.

## Alternatives

- Leave the raw method gap unresolved: the full saved state already supplies
  the proof required for honest outcome-versus-memory presentation.
- Replace all null views with zero measurements: this would erase original
  applicability/configuration and overstate what was measured.
- Allocate group bytes to every arm or profile again: shared FIFO and owned
  arrays have distinct provenance, and this task needs no new measurement.

## Consequences

The original inventory and all historical receipts remain immutable. The new
ledger resolves its missing projection metadata while retaining original null
views and measurement limits. P6.10b still requires complete official report
reader/outcome presentation; P6.10 keeps its original acceptance and stays open.
Private fixtures and IO spies establish behavior only, never actual original
reader authority. No favorable scientific selection or new experiment follows.

## Acceptance evidence

The prospective 121-source/111-runtime-import freeze precedes development
fixtures. All 83 new/506 related tests and Ruff/thirteen-file format/mypy 475
pass. Both actual publications and both independent complete reader
reconstructions exit 0 within unchanged 180-second budgets; eight fresh
original training reader calls, all 24 scientific guards zero. Complete
JSON/Markdown repeat at 3338c308...b970/17,804,510 and cc9ba656...5afb/329,301
bytes. All 560 cells/1,680 checkpoints/sixty contexts/180 context-stage
records/3,200 array pairs and 300 resolved projection gaps remain.
Validation c40d6086...6b99/23,476 bytes and the resource document/log bind
full identities, commands/outcomes and original/current provenance. Source
and scientific bytes stay unchanged. P6.10b and P6.10 remain open.
