# ADR-0309: Partition shared replay retention

## Context

The shared replay buffer already deduplicates and bounds FIFO or seeded bottom-k
retention. It cannot reserve separate recent and historical allocations. R2.7
observations give only a limited residual contrast, so reliable memory coverage
is an independent prerequisite for studying retention/adaptation controls.

## Decision

Add opt-in recent/reservoir row-and-byte quotas to that buffer, with one copied
array pair per distinct union ID. Pay overlap against both logical quotas and
report physical union storage separately. Refresh recent duplicates; use stable
distinct-content reservoir rank without an ever-growing seen ledger. Alternate
distinct recent/reservoir selection on each partition's turn, with fallback.

Reuse the existing arrived-training schedule and CLI. Bind opt-in configuration
to a distinct protocol/digest via the existing configuration function; preserve
default v9 identities and payload. Do not weaken native CPC/shared-buffer
equality checks: the legacy learner's native buffer does not support this mode,
so current CLI output is a proposal rather than executed learning.

Why this: explicit memory/exposure accounting and detached shared selections
provide a simpler retention control without a new runner or trigger policy.

## Alternatives and consequences

Separate full buffers would duplicate array storage for overlap. One undivided
FIFO can lose old data, while a reservoir alone does not reserve newest data.
Context-stratified retention would require additional available-label/context
assumptions. A bounded reservoir cannot guarantee arbitrary old-context coverage;
preserve zero-coverage outcomes without selecting favorable seeds.

Default behavior remains available. The new schedule requires native consumer
integration before any learning benefit can be evaluated; that is unfinished,
not a reason to bypass equality checks or report proposed updates as executed.

## Rotating exposure follow-up, 2026-10-16

The fixed rank-first schedule retained an old ID for seed17 but never proposed
it during B. Add a separately named rotating sampler, preserving that result.
Persist one last-scanned reservoir ID and next requested partition turn across
calls. Continue strictly after its rank, wrapping once; scan duplicates within
each turn and publish the cursor only after detached return construction.

Why this: memory presence does not ensure replay exposure. Persisting the turn
also prevents one-row requests from always spending their budget on recent
data. A stable R-row reservoir gets full proposal coverage within2R successful
nonempty calls. Do not extend this bound to changing membership or recent-only
rows. Trusted in-memory copies continue deterministically; no additional
checkpoint/admission framework is needed. Native learning integration and
comparative efficacy remain separate unfinished work.
