# ADR-0262: Opt-in complete coupled topology state for existing runtime ports

## Context

Native pruning returns an independently validated PC/CPC model, but existing
CoupledLearner restore pins hidden geometry. ActorShadowRuntime consolidation
restores into its current candidate. ServingPromotionController already builds
independent guard/prepared models and publishes complete bundles with rollback.
Structural state must include geometry and CPC wake history as well as weights.

## Decision

Add a bounded core topology record and a separate TopologyCoupledLearner adapter.
Pin native family/input/precision/CPC policy and fixed update settings; explicitly
reconstruct hidden widths on restore/build/fork. Validate the complete private
replacement before assigning one payload containing learner/geometry/lineage.
Ordered coordinate records are self-consistent metadata and may shorten when
restoring earlier states; they grant no runtime authority.

Adapt the existing native damage gate to a ConsolidatedState callback. Refused
decisions raise with detached evidence before candidate restoration. Reuse
runtime attempt accounting/locking and serving guard/commit/rollback. Retain
managed-retention refusal. Callers declare selection IDs as well as wake IDs to
exclude all structural selection labels from independent inner guards.

## Alternatives

- Broadening CoupledLearner restore would silently change accepted defaults.
- Replacing runtime or serving publication would duplicate existing concurrency,
  revision, ownership, cache and rollback machinery.
- Weight-only snapshots would restart or misassociate CPC adaptation state and
  fail to describe changed geometry.

## Consequences

Existing behavior remains available. The opt-in path needs complete wrapper
digests and one shared bound builder object for guard/preparation. Native restore
failures are atomic within this adapter, while runtime conservatively stops on
any restore exception. Existing post-commit budget semantics remain observable.
No durable codec, managed-retention support, general damage guarantee, nonlinear
scaling result or deployment-quality claim follows. Later transfers must prove
complete state/lineage mapping and runtime behavior before acceptance.
