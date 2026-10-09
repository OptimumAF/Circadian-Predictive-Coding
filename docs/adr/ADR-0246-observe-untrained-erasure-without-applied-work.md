# ADR-0246: Observe untrained erasure without applied work

## Context

The original inbox can erase source-only, label-only and paired arrivals before
training. They retain tombstones but no applied receipt. Existing trained erased
history cannot represent them without manufacturing native work.

## Decision

Keep a separate exact scalar data schema and payload-free weak tombstone witness.
Original birth-enrolled history validates current managed arrival and declaration
before revocation, rechecks temporary weak identity/content seals after native
eraser callbacks, pays original metadata admission, then publishes prebuilt maps
through the trusted inbox commit. Original work is an observed boundary only.

Why this: original observation, integrity and paid copying are distinct boundaries.
The witness primitive owns no authority. Completed erasure observation is g3a;
actual paid memo/materialization/handoff remains separately unfinished g3b.

## Alternatives

Reuse a trained receipt: rejected because no original receipt/update exists.
Keep raw sources or declarations: rejected because deletion must release arrays.
Accept keys alone: rejected because keys do not establish original provenance.
Change inbox/native schemas: unnecessary; existing tombstones already support
optional source/label metadata.

## Consequences

All original authorities and cumulative/permanent charges persist. Default-off
untracked untrained capture refuses explicitly. Witnesses do not authorize payload
access or revive consent, attest post-birth registration ordering, prove physical
memory release or qualify broader cleanup/TTL/recovery/scientific claims. No new
configuration or dependencies. Existing trained history requires fresh regression
evidence; all original criteria and unrelated changes remain preserved.


## Current g3a validation status

Original untrained observation remains unaccepted: N14 normal arrival stamp rechecks fail. Source inspection identified fresh nested tuple identities in the proof; actual trace does not isolate which id changed and does not prove numeric payload mutation. Preserve original identity/content criteria; fix temporary proof framing before a newly scoped run. Paid untrained memo/handoff g3b remains unsupported and unchecked. Audit024 also failed default-encoding document readback; exact failure retained, no rerun. See docs/development-log.md and artifact HANDOFF.md.


## Accepted g3a original observation, paid copying still unfinished

The preceding N14 failure remains historical evidence. Stable original
proof framing is now qualified by438unique controls, including seven
actual N15 cases and twelve deterministic stamp tests; all original
g3a criteria were independently reviewed. Original identities/content/
consent/chronology/work and all spent authorities/charges remain.
Paid untrained memo/materialization/handoff g3b and full g3 remain open.
See current development log and g3a2 artifact HANDOFF.md.
