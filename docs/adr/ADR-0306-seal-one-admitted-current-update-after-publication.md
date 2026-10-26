# ADR-0306: Seal one admitted current update after publication

## Context

The retained publication proof deliberately freezes retired and current native
state. Replay row fingerprints alone cannot authorize changing whole model state
after restore. Original inbox history can outlive an evicted native replay row.

## Decision

When both original birth flags `retain_inbox_origins=True` and
`retain_capture_history=True` are enabled, the validated restore prepares a paid,
weak current baseline. The existing trusted publication installs that binding.
The original ledger's next poll borrows the existing copies and manager gates
without blocking, validates the baseline, and observes its original update.

The first completed observation records provisional native, inbox, learner and
cumulative work state before opaque ports run. The original ledger's final
receipt, consent, resource, row and inbox-history checks must succeed before
bounded metadata is paid and the final success is sealed. Existing budget
normalization permits ordinary clock/RSS telemetry while binding cumulative work.
Complete capture checks the immutable retired proof and this exact success under
the original authorities. Native eviction does not release original consent.

Current runtime limits and consolidation history remain publication-bound; only
the revision advanced by original ingress and polling can change. The existing
manager's complete inbox check proves membership as well as known-record
identity. Started observation admits only the original committed prefix and the
actual new pair; final success excludes extra entries and erased partitions.

Why this: only the successful original operation can establish this transition;
loosening current-state comparisons would accept unobserved model mutation.

## Alternatives

Unpinning the current proof loses whole-state provenance. Rebuilding proof at
capture would grant authority retrospectively. A separate lock or new accounting
pool duplicates existing authorities and is unnecessary.

## Consequences

Defaults retain their existing behavior. Seed and success metadata spend the
original admission allowance, with no additional update or origin quota. A late
failure retains charges and existing poisoned-state semantics. Only one further
update with trained, unerased publication history is qualified by this increment;
chained publications, further updates, erased/untrained continuation, persistence
and recovery remain unfinished. Capture creates detached data, not restore rights.

## Amendment: successive single-update polls

The original qualification above covered one further update. Subsequent work
removes that artificial ceiling by validating and pinning the latest success
before each next original poll. Completion must advance its exact prior count by
one. The original retired publication remains immutable; only the latest weak
success is stored, while every earlier payment and reservation remains spent.
Prior metadata and explicit state must agree; a new seal cannot launder prior
corruption or late replacement. No record chain or new authority is introduced.

Why this: a retained owner must continue useful bounded training after capture.
The original lifetime metadata allowance may cover multiple paid observations;
the manager constructs each observation limit as min(lifetime,128KiB), matching
the existing normalizer domain. This preserves all earlier <=128KiB cases and
never mutates an admission after birth. Multi-update polls, chained publications,
erased/untrained continuation and recovery remain unfinished.

Ordinary deferral must also permit resume: the original sharing gate increments
a counter even when a poll completes no native work. A bounded paid observation
records that actual returned reason and exact sharing transition, anchored to
the previous native success and original gate. It preserves the success identity,
work count and receipt; subsequent admission validates the observed sharing
position. New work clears the observation without refunding its payment. Final
pure checks after opaque resource callbacks reject late state/counter mutation.
This specific observation is necessary to distinguish a genuine deferral from a
manual counter change; it is not a new native completion or authority.

## Runnable continuation composition

The existing NumPy restored example exposes `--successive-replay` with both
original history enrollments and sufficient limits fixed at birth. It exercises
three actual updates, one real restore and a paid paused poll before resume,
then complete data capture. Why this: demonstrate the accepted runtime behavior
without introducing another algorithm or authority format. Existing modes keep
their original workloads; the full parent/recovery/scientific gates remain open.
