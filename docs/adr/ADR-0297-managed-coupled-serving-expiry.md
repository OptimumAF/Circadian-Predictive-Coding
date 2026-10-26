# ADR-0297: Managed coupled cached-serving expiry

## Context

Fixed coupled learners now support managed elapsed retention and owned-copy
accounting through handoff. The existing coupled cache demo uses structural
learners without a managed lifecycle. A serving cache and a prepared promotion
retain additional raw metadata and predictions beyond inbox/checkpoint copies.

## Decision

Add a separate fixed coupled integration using the existing PromotableActor,
ServingPromotionController, guard evaluator, lifecycle, copy ports and expiry
driver. Keep the structural example and older handoff modes intact.

Why this: composition exercises the missing serving ownership boundary while
leaving cleanup, single-point publication and worker lifetime in the app layer.

Use the same original clock, copy budget and resource objects through handoff.
Populate the cache, prepare but do not commit a promotion, and expire all owned
raw references/tickets at the original deadline. Full native state and receipts
must remain unchanged. Start/wake/wait and stop use the established bounded driver.

Guard data are fixed, disjoint inner-guard batches, separate from training and
final evaluation. Broad fixed thresholds test transaction plumbing; no benefit,
ranking or physical-latency conclusion follows. A full pending-ticket quota can
refuse preparation before the deadline check; report the actual refusal order.

## Alternatives

Extending structural ports would add an unrelated native-state contract. A new
cleanup mechanism would duplicate the lifecycle coordinator. Committing the
ticket before expiry would fail to test pending-promotion invalidation.

## Consequences

Lifetime raw-copy charges remain spent after cache/metadata/inbox/checkpoint
cleanup. Native parameters and CPC state remain; caller-owned arrays and reports
are outside deletion. Injected elapsed time and finite local threading do not
prove physical expiry latency, memory overwrite, unlearning, durable recovery or
continuous-service guarantees. Parent requirements remain open.
