# ADR-0305: Retain paid weak publication history for complete capture

## Context

A successful managed checkpoint handoff replaces the ledger's live rows with
rebound prepared rows and clears pending checkpoints. Closing the operation then
drops original retired row proofs. Constructor-copy witnesses precede native
restore and cannot prove that lineage. Matching later contents cannot recover it.

## Decision

Add exact boolean `retain_capture_history=False` to the manager constructor.
Opt in at birth to retain bounded weak original old/new publication witnesses.
Reuse already validated original row metadata, receipt/source/consent witnesses,
content/history stamps, enrollments and paid authorities. Do not retain raw
payloads or owners strongly. Prepare and validate sealed replacement storage
before publication; the existing trusted transition commits plain assignments.

Count bounded canonical UTF-8 logical witness metadata plus the existing record
overhead, charging the original metadata admission before publication. Add no
invocation or row charge and refund no failed expense. This is logical metadata
accounting, not physical heap/RSS measurement.

Use the existing `replay_checkpoints` capture argument to read the proof under
held nonblocking leases. Prove retired rows against their original retired
inbox/receipt and current consent, plus current rebound rows and every old/new
holder appearance. Original builder witnesses remain tied to the old source;
the constructor-only current-source rule stays unchanged.

Why this: the existing producer has the proof at publication, while later capture
does not. A small paid weak record closes this specific gap without reconstructing
authority or changing default restore costs.

## Alternatives

- Infer old lineage from matching keys or bytes: does not prove original ownership.
- Retain raw snapshots: adds payload ownership and duplicated raw expenses.
- Always enable history: silently changes existing restore costs and retention.
- Validate through public manager methods: reenters held leases or performs native work.

## Consequences

The proof includes complete original old/current learner field identities, so
changing wrapper policy invalidates it even when the model and policy callback
stay equal. Preparation runs before the existing final publication validation:
opaque model-reference ports must finish before the pure commit barrier. These
requirements arose from independent source review and have refusal controls.

Default restore behavior and expenses remain. Opt-in metadata exhaustion refuses
before publication and keeps actual spent preparation/copy expenses. No later
opt-in can authenticate a past handoff. This increment supports one unchanged
published pair; later native work, chained handoffs, promotion, erasure/compound
paths, portable bytes and recovery remain open under the original parents.
Focused controls, scoped statics, independent review and one bounded actual
restored capture are required before acceptance.
