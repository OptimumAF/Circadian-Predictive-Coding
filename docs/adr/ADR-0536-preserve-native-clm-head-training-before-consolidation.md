# ADR-0536: Preserve native CLM head training before consolidation

## Context

Existing local streams exercise causal weight and context feedback. Retrieval
matches the successful native memory endpoints and the online memory case has no
full-case accuracy gain over its linear control. R5.3 requires a matched native
fine-tuning baseline before adding consolidation. No such conventional CLM head
learner existed in the inspected checkout.

## Decision

Implement the conventional paired group-masked CLM loss and its AdamW, clip1 and
cosine OneCycle update over explicitly version-bound frozen embeddings. Keep core
metadata free of Torch; put native operations behind the existing learner port.
Reuse ExperienceInbox for delayed TRAIN labels. Keep projections fresh and expose
generation changes; do not introduce a projection cache or new orchestration.

Snapshots own the complete local continuation state. Updates/restores stage copies
and preserve physical work charges. Exact identity and fixed schedule admission
prevent an old snapshot from silently renewing the horizon or changing the recipe.
Trusted Python modules remain trusted; this is not an untrusted loader.

## Alternatives

- Add circadian head mechanisms first: would leave the conventional comparator
  undefined, preventing an interpretable matched comparison.
- Reuse the repository's supervised PC/BP adapter unchanged: its objective and
  parameter layout do not reproduce the native paired CLM update.
- Load the full released8B encoder immediately: separates poorly from diagnosing
  the small head update and exceeds this component's local resource allowance.

## Consequences

A tiny source-parity case can qualify the native update and state behavior. It
cannot qualify released encoder/head execution, feature provenance, arbitrary
serving requests, useful adaptation, consolidation or comparative benefit. Those
requirements remain explicit parents. The native scheduler's invalid horizon10
continues to fail loudly rather than being replaced by a more favorable recipe.
