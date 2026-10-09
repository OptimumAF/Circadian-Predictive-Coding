# ADR-0199 — Bound local experience arrivals to native updates

## Context

Accepted native learner ports preserve separate objectives and layouts. Existing
benchmark role seals and prospective source declarations have scientific scopes
that a generic runtime must not weaken. Local delayed-label delivery also needs
explicit identity, clock, ownership and failure semantics before actor/promotion work.

## Decision

Add pure immutable experience/label/permission records and an explicit logical
clock, plus a bounded app inbox for declared train-role pairs. Permit label-first
delivery, enforce matching identities/roles/actor versions and chronological
eligibility, reject duplicate IDs and order each drain deterministically. Copy
trusted payloads at registration and again before native calls.

Reuse the existing native step, adding optional start/completion hooks. Commit
label identity before the post-update budget check. Leave pre-update refusals
pending; stop after uncertain native failure, including a native budget-typed
exception. Retain completed history within the same identity-count quota.

## Alternatives

- Shape- or loss-specific delivery would prevent the existing opaque ports from
  serving ordinary and CPC learners through the same orchestration.
- Last-write-wins duplicates could overwrite role/version/target facts or retrain.
- Sorting already completed work would require rollback and change real history.
- A completion-only hook cannot identify an unfinished native call's stop exception.
- Reusing scientific final-release records as runtime permission would grant
  authority their metadata alone cannot establish.

## Consequences

Local eligibility and failure behavior are testable without new learning rules or
experimental claims. Role/version/arrival fields are trusted declarations, not
physical proof. Record-count quotas do not certify payload bytes/RSS. Persistence,
global watermarks, actor concurrency, promotion, replay, privacy and scientific
source/native admission remain separate contracts and gates.
