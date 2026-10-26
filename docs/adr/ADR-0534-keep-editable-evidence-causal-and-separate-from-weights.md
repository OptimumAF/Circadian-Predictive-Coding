# ADR-0534 — Keep editable evidence causal and separate from weights

## Context

R5.7 needs editable episodic decision context alongside existing native adapters.
Current lexical retrieval is stateless; numeric action memories solve another
problem. The native ticket case succeeds on invoice but fails parcel recovery.
A new memory mechanism must preserve that evidence and not repair model outputs
from targets during evaluation.

## Decision

Use a small bounded TRAIN-only in-memory revision store in core and an outer
typed predictor wrapper. Label edits preserve observation/source/policy/consent.
Every attempted view seals its cutoff; later writes cannot backfill that view.
Latest visible revision resolution precedes active-target filtering. Protected
instructions/candidate definitions stay outside rendered evidence.

Keep retrieval-only and actual context-model output distinct. Reuse the existing
ticket dispatcher and frozen conventional predictor for a controlled case, with
edits/retrieval costs reported separately from weight updates. Require fresh
protocols before native models or the four-arm comparative experiment.

## Alternatives

Overwriting a mutable dictionary loses historical truth. Automatic eviction can
change old views silently. Relabeling a context model's answer from retrieved
targets manufactures apparent competence. A new orchestration/storage framework
is unnecessary for this bounded local outcome.

## Consequences

Event and UTF-8 caps fail loudly; storage is finite and may exhaust. Future-cutoff
reads seal that future time, so callers must use their actual observation clock.
JSON field separation is a wire boundary, not semantic injection immunity.
Context editing can change retrieval decisions without changing weights; whether
a model can use that context remains a separately measured, possibly negative,
result. Full R5.7/native/release claims remain open.
