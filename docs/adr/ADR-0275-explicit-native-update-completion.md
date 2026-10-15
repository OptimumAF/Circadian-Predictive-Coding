# ADR-0275: Explicit native-only update completion

## Context

Existing native admissions remain uncertain even after healthy work. Ordinary
authority transitions and journal floor checks intentionally reject clearing that
uncertainty or changing component hashes. Accepted supervised writer recovery
therefore supported checkpoint/read-only work but no completed native update.

## Decision

Add a separate pure AuthorityCompletion bound to its original single-update
reservation and the same live registered worker. The coordinator validates a
complete parent-owned B checkpoint through a trusted callback, takes a fresh
pending host observation, then commits an explicit completion CAS. Only completed
count, uncertainty, report fields and the equal native/payload digest pair change;
all charges, ownership and other components including actor remain unchanged.
Require an original native-only payload; full managed bundles remain unsupported.

Keep generic floor validation and legacy execute behavior unchanged. Only this
journal instance's validated successful CAS adopts B as its live floor. Extend
terminal failure witnesses with the exact original completion attempt so an
ambiguous COMMIT can be reconciled to stopped pending or stopped completed state,
never automatically resumed. No metadata wire fields or schema migration needed.

Use a separate one-update native worker and runnable PC/neutral/active example.
Preserve the existing no-training read/stage worker. Actual parent validation
checks the complete one-update reference, distinct B file and reload; a deliberate
validation refusal after valid B creation demonstrates that bytes are insufficient.

## Alternatives

Relaxing generic floor rules would let disk contents assert completion. Clearing
uncertainty based only on a digest would not establish validated native work.
Reusing a new session would renew limits and lose original charges. Marking every
failed acknowledgment incomplete would contradict a completion already committed.

## Consequences

Healthy native-only candidates can complete under existing surviving-parent
authority while failed/interrupted work stays refused. Trusted callbacks remain
an internal composition boundary, not an external API for arbitrary manifests.
Actor promotion, complete managed component graphs, uncertain-work retirement
and coordinator-loss recovery remain separate unfinished requirements. No change
to scientific comparisons, native objectives, baselines, seeds or final isolation.
