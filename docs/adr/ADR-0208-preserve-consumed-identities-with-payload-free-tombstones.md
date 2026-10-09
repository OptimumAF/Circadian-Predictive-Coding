# ADR-0208: Preserve consumed identities with payload-free tombstones

## Context

Inbox format1 requires full source/label payloads to reference applied receipts.
Removing those records would invalidate checkpoints or allow duplicate delivery.
Historical CPC replay stores whole batches without subject provenance indices.

## Decision

Add canonical format2 cursors with explicit payload-free erased identities, while
keeping format1 supported and emitted for unerased histories. Preserve applied work,
event IDs and lifetime identity capacity. Validate all prepared metadata before
dropping payload references. Add native whole-buffer replay erasure, preserving
all other native state, policy and counters.

## Alternatives

Keeping zero-shaped payload placeholders would hide erasure status. Dropping IDs
or refunding capacity permits replay of consumed records. Selective native removal
without a source-to-batch index cannot demonstrate subject deletion.

## Consequences

Metadata and learned influence remain. Native removal conservatively clears other
retained rows too. Reference erasure is not physical RAM zeroization or unlearning.
These are prerequisites: original-authority coordination, lifetime/byte policies,
checkpoint/retired-owner/promotion/rollback copies and non-resurrection remain
unfinished R3.6b2 work. No complete public deletion API is claimed yet.
