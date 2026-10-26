# ADR-0550 — Bind one finite memory view to live CLM typed inference

## Context

Native TRAIN memory and a generic evidence-first/frozen-fallback mode already
exist. The temporal CLM CLI still refused their composition. Completed phase
memory has sealed work counters and reports; silently treating more retrievals
as part of that old prefix would weaken its accounting and corruption guards.

## Decision

Allow the existing memory and typed options together for an all-head none-recipe
catalog stream. After completion, bind one copied36-event bounded serving view
to the existing frozen CLM predictor. Charge copied writes/bytes, keep the old
phase report/work unchanged, and cap this view at five requests. A paid failed
bind cannot retry. Existing source/binding/head/encoder guards remain in place.

Use four existing canonical known requests plus a prospectively fixed unlabeled
unknown request. The latter exercises actual head fallback and has no target or
accuracy metric. Validate the full observed source/selection/work prefix before
cached scoring; no final/probe data or new head updates occur.

Why this: separate bounded ownership is simpler than allowing ambiguous counter
changes on the sealed phase store. One explicit view supplies useful serving
behavior through existing components without another runtime or optimizer.

## Alternatives

Sharing the mutable source store would require admitting and tracking every
post-completion counter change. Always calling the head would discard the
existing memory-availability policy. A new runner duplicates current protected
binding/inference orchestration. Reconstructing trained heads from JSON cannot
recover unsaved weights.

## Consequences

The copy adds36writes and logical storage bytes plus Python/JSON copies. Source
and view retain at most72events total, with no unlimited success-record chain.
The encoder keeps its original five-forward birth limit; known-memory calls
consume none. Memory-versus-cached-head probability differences are interventions,
not numerical parity. Literal pipeline correctness does not establish native
unknown-task quality, calibrated memory confidence, recovery or CPC benefit.
