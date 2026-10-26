# ADR-0415: Give the deferred game boundary one proof owner

## Context

The deferred count-control game reader and its application both enclosed the
same resource callback with full registry proofs. The inline compatibility
adapter repeated that enclosure around supplied clock/native-state reads. The
existing native cases remain incomplete at their original20-second endpoint cap.

## Decision

Add explicit `DeferredProbeRollouts.run_qualified` through the existing
`EndpointBoundary` contract and `ProbeRollouts.observe_qualified`. The application
owns registry and original benchmark-witness proofs around each external group.
Successful read groups use proof/resources/proof/read/proof/resources/proof:
adjoining proofs can be shared only when no external call intervenes. Original
closing clock/RSS observations remain. Failed evidence blocks subsequent ports;
a read exception keeps its first cause while valid closing checks still run.

Keep `run` as the original no-argument compatibility enclosure. Share only the
execution loop; preserve all preparation sites, private paid-history checks,
independent phase entry/exit, original capture storage and budget authority.

Why this: use the already implemented contract to remove proven duplicate work
without adding a cache or a new orchestration system.

## Alternatives

Increasing the endpoint allowance, skipping state/history checks, admitting
unobserved progress, or caching trust across callbacks changes the guarantee.
Those approaches do not satisfy this repair.

## Consequences

Matched scripted profiling preserves external work and saved non-timing results.
Whole-case proof reduction8.7165% fails the original25% target, which remains open.
The actual deferred phase735→371 checks gives49.5238% reduction; this scoped
outcome is separately accepted after focused checks. Scripted wall change0.8115%
does not establish native speed or20s completion. Native games require a separate
frozen protocol. No learning, metric, seed, final-data access or benefit claim changes.

[Evidence and retained failures](../../artifacts/runs/deferred-qualified-boundary-20261018/RESULTS.md).
