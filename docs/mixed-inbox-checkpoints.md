# Complete live and erased inbox checkpoint lineage

The original birth-enrolled replay history keeps two disjoint inventories:
live committed source/label pairs and payload-free erased receipt/tombstone
witnesses. Mixed checkpoint capture requires complete applied receipt coverage
from their union. Unknown receipts, overlapping keys and untrained tombstones
cannot acquire checkpoint lineage.

Why this: deletion retains cumulative applied work after releasing arrays.
Counting only live pairs loses the erased receipt's original history. A copied
key or digest supplies integrity information; actual original admission and
memo bindings supply provenance.

`src/app/checkpoint_erased_inbox.py` is a pure integrity and memo binding helper.
It validates bounded exact records and scalar history, requires detached actual
memo receipts and tombstones, and rechecks originals after opaque lookups. It
retains only original admitted scalar data and weak copied references. It owns
no data access, consent, admission, cleanup, native state or filesystem IO.

The application coordinator admits every inbox copy through the same original
replay admission and permanent copied-slot accounting. Erased records incur
metadata charges and no erased payload raw bytes. Live arrays retain existing
raw-copy admission. All attempted charges remain spent on later refusal.

Capture and materialization bind the complete original receipt and tombstone
sequence through the actual copier memo. The final check occurs after opaque
fingerprint/resource ports. It rechecks original history, captured and actual
materialized metadata, current original revocation and exact prepared receipt/
tombstone identities. Prepared live and erased maps are built before handoff;
the trusted commit performs plain assignments alongside existing anchors/rows.

Existing default-off capture retains its original live-only provenance rule.
Only original-birth enrolled, observed erased history can support mixed capture.
The original lifecycle cleanup API and all fixed inbox/native schemas remain
unchanged. New dependencies and settings are unnecessary.

Run and acceptance commands belong to the prospectively bounded g2 scope in
`docs/development-log.md`. The bounded one-erased-trained plus one-live-trained case passes exact source,
full type/static, isolated author, cache and native evidence gates; final closure
is recorded in the g2 resource/readback receipts. Untrained tombstones, other cleanup/native families, TTL,
public holder release, disk/RSS/recovery and scientific claims remain unfinished.

Extend the fixed supported record schemas through explicit helpers and bounded
negative controls. Preserve actual memo order, original authorities and spent
charges when adding another erased-history variant.


## Current g3a validation status

Original untrained observation remains unaccepted: N14 normal arrival stamp rechecks fail. Source inspection identified fresh nested tuple identities in the proof; actual trace does not isolate which id changed and does not prove numeric payload mutation. Preserve original identity/content criteria; fix temporary proof framing before a newly scoped run. Paid untrained memo/handoff g3b remains unsupported and unchecked. Audit024 also failed default-encoding document readback; exact failure retained, no rerun. See docs/development-log.md and artifact HANDOFF.md.
