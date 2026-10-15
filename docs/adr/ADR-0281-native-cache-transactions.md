# ADR-0281 — Native cache transaction integration

Date: 2026-10-15

## Context

R3.7 requires stale-cache and failure-path controls with stable serving. Existing
PromotableActor and ServingPromotionController already own complete bundles and
publish/restore them atomically. Earlier accepted native growth controls do not
exercise cache expiry/eviction through guarded promotion and rollback.

## Decision

Add a fixed three-family native adapter and thin CLI using existing ports. Keep
the actor at its warmed width2 while the candidate splits to4 and updates once.
Exercise FIFO eviction, exact TTL expiry and backward-tick refusal, then native
guard preparation/commit, fresh-cache serving and exact saved-bundle rollback.
Check full native/CPC/history state and complete configuration/cache/metadata/
version/generation/tick, including consumed-handle refusals.

Use separate inner guards and fixed broad engineering thresholds. Preserve
original candidate quotas and disjoint evaluation identities; report real native
measurements without a comparative benefit claim. Bound invocation/output/RSS
observations prospectively and reuse unchanged core/controller evidence.

## Alternatives

A new transaction or cache abstraction would duplicate existing behavior. A scalar
fixture alone would miss native state reconstruction and CPC/history transfer.
Utility-tuned guard bounds would confound transaction controls with learning
performance. A large sweep is unnecessary for this deterministic integration.

## Consequences

The runnable integration provides concrete native evidence under fixed finite
settings. It does not certify durable recovery, arbitrary external builders,
continuous hard memory bounds or long-running production service. Future cache
policies require separately specified behavior and focused tests.
