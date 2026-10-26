# ADR-0535: bind consented raw feedback to delayed stream receipts

## Context

The fixed four-arm comparison supports useful local memory-assisted recovery,
with retrieval matching its successful endpoints cheaply. It does not show
causal feedback publication while the existing temporal learner runs. That runner
uses projected array experiences with training-only consent; those permissions
do not authorize memory reuse of a raw typed observation.

## Decision

Observe new raw TRAIN requests through the existing prearrival hook and declare
their explicit training/reuse consent before labels. Keep the projected view's
original permissions unchanged. Publish each row's evidence only after its real
batch receipt, preserving original observation/arrival/application ticks and
validating request, allowed IDs, key/model version and consent at the app boundary.
Expose only detached committed receipts at a checked quiescent stream boundary.

Why this: the existing stream already owns scheduling, budgets and successful
updates. A small feedback boundary supplies the missing causal join without
creating another learner or queue. Fixed live-prefix readouts are cached and
scored only after TRAIN. Known missing evidence has an explicit unavailable value;
full accuracy, coverage and answered-only proper scores keep distinct denominators.
The original wrapper retains its sticky uncertainty/refusal behavior.

## Alternatives

- Read projected experiences alone: their consent is insufficient for raw-memory
  reuse, and their features no longer carry the full original typed request.
- A separate feedback scheduler: duplicates existing clock/receipt ownership.
- Score before deciding later work: expands evaluation access unnecessarily for
  this fixed-dose case; cached scoring supplies the requested measurements.

## Consequences

The new raw view and its memory writes have explicit additional observation,
lookup/storage and process costs. Version filtering can exclude an old stable
label after a policy change; availability loss must be reported without calling
it ordinary forgetting. The source and actor stay frozen while the candidate
learns; this is a local reference, not deployed promotion or a new native-model
qualification. No final/probe labels, optimizer replay or external actions enter.

[Usage](../online-temporal-feedback.md) ·
[Protocol](../../artifacts/runs/online-temporal-feedback-20261022/PROTOCOL.md).
