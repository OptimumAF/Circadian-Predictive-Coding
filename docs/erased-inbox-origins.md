# Original erased inbox receipt lineage

The optional history enrolled at `ManagedReplayOrigins` birth can observe
`ledger.delete(keys)`. This delegates to the same original lifecycle deletion.
The lifecycle decides which registered payloads are cleared and which keys are
revoked; its existing cleanup report and failure behavior remain authoritative.

Why this: deletion keeps applied receipts. Removing a live weak pair witness
without observing the actual transition loses the receipt's original lineage.
The new witness stores original admitted scalar metadata plus weak references
to the exact original receipt and actual tombstone. It contains no source,
label, declaration, array, or new consent authority.

Before deletion, all current live pairs must have original committed witnesses.
During the actual original inbox commit, preparation checks the saved history,
current original source/label/receipt identities and actual prepared tombstones.
It reserves another metadata record through the same original admission before
allocating the erased witness. Original record charges and permanent copied
slots remain spent. The conservative transient live limit is also checked.

An exact scope binds the original ledger deletion, thread and execution context.
Copied-context operations cannot close the original scope. No callback runs
after payload removal: publication consists of prepared map/storage and flag assignments.
All borrowed strong references are cleared on original scope exit.

Use the installed environment and prospectively bounded tests recorded in the
development log. This module has no settings, dependencies, IO, worker or native
algorithm of its own. Inputs are an original observed erasure operation; outputs
are integrity and lineage metadata. Unobserved deletion remains functional in
the lifecycle, but optional history cannot retrospectively grant erased lineage.
Receipt/tombstone replacement or content mutation refuses subsequent verification.

This prerequisite qualifies original observation. The subsequent g2 helper and
coordinator now qualify complete mixed erased/live capture and handoff for the
bounded starting case; see `mixed-inbox-checkpoints.md` and its actual evidence.
Pending/untrained tombstones, other cleanup families, TTL, public holder release,
heap/RSS/disk/recovery and scientific acceptance remain separate unfinished work.
Metadata units are not measured physical memory.


## Current g3a validation status

Original untrained observation remains unaccepted: N14 normal arrival stamp rechecks fail. Source inspection identified fresh nested tuple identities in the proof; actual trace does not isolate which id changed and does not prove numeric payload mutation. Preserve original identity/content criteria; fix temporary proof framing before a newly scoped run. Paid untrained memo/handoff g3b remains unsupported and unchecked. Audit024 also failed default-encoding document readback; exact failure retained, no rerun. See docs/development-log.md and artifact HANDOFF.md.
