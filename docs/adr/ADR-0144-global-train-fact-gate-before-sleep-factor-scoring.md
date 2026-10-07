# ADR-0144: Compare all train facts globally before sleep-factor scoring

## Context

P6.3c3 produced a byte-identical train-only artifact for nine no-replay arms on three development seeds. Scoring seed by seed would expose early outer outcomes before later cells finish, inviting a score-informed repair. The c3 source identity is frozen and cannot be refactored without invalidating its evidence.

## Decision

Build a separate c4 scored app that reuses c3's model, wake, guard, and fact helpers. Preserve after-A models with deep copies, complete all A→B training, and compare the entire JSON-represented c3 fact object to the saved, hash-bound reference before reading any outer-selection array. The public adapter verifies the c3 result bytes, request and audit, then binds its own source/manifest/adapter identities and budget. Score the full two-task accuracy matrix and only the three predeclared paired factor contrasts; publish every seed and null/negative outcome. Keep final sources sealed and reserved confirmation seeds unused.

## Alternatives

- Score each seed as soon as its training ends: rejected because a later cell could fail after earlier outer values are known.
- Retrain after calling the c3 preflight: rejected because it doubles optimizer work and weakens the exact-work comparison.
- Edit the c3 source to return checkpoint models: rejected because its already verified source and artifact hashes would no longer name the same run.
- Use c3 historical guard accuracy as an outcome: rejected because the inner guard is a training decision role, not outer selection.

## Consequences

The app retains a second small model set per seed during the global gate and depends on frozen c3 private helpers. That dependency is explicit and hash-bound. A mismatch aborts with no outer score. The resulting three-seed comparisons are development evidence only; they cannot establish a circadian benefit, close the full matrix, or license final-role access.
