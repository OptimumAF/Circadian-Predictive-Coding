# ADR-0566 — Bind native typed decisions to the existing promotable actor

## Context

The fresh BP task recipe (#48) learns useful candidate states while its stable
actor remains initial. Native typed decisions previously used detached frozen
readouts. Existing promotion ports already provide measured refusals, full atomic
publication and complete one-step rollback, with accepted generic concurrency
tests. Duplicating them would add no useful learning behavior.

## Decision

Add optional TRAIN-only actor injection to AttributionStream, retaining original
base-actor identity in experience/label receipts. Bind typed decisions to one
atomic full actor snapshot per request using the existing raw-logit decoder.
Retain no readout cache or unlimited history. Rebuild explicitly and account for
copies and native initialization. Use existing categorical native models through
pluggable guard callbacks; a fixed BP96->104demo requires full new-template fit,
positive gain and no old-template accuracy loss before publication/rollback.

## Alternatives

Global probability-to-logit reconstruction can lose scores through underflow.
Detached episode readouts cannot follow live publication. A generation-keyed
model cache would require extra lifecycle/access and invalidation guarantees;
the tiny native head does not yet justify that complexity. A new promotion or
controller framework duplicates supported application transactions.

## Consequences

Version/generation evidence follows real serving swaps and rollback. Each request
pays native reconstruction and full snapshots. Template guards are development
checks, not independent generalization; action-ID membership does not replace
effect-time permission checks. Final/probe data remain excluded. Continuous
post-publication rebase/restart, broader failure/long-run requirements and
scientific/release claims remain separate and open.
