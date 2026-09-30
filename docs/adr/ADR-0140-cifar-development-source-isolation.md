# ADR-0140: Omit the CIFAR final source during development writers

## Context

The older CPU feature-profile and local/pretrained validation CLIs replaced
the final-test loader with a raising iterator. Their default loader call still
constructed the CIFAR final dataset before any profile or validation work.
The saved selection records show no final-test iteration or score, but that
weaker check did not establish physical source isolation. The app already has
an `include_final_test=False` loader path and a validation-only
`development_only_source=True` tuning path.

## Decision

Use the existing development-only path in these three current writers. Their
wrappers reject a request to include the final source, and the profile and
selection calls explicitly request omission. Keep the historical seeds,
candidate grid, work budgets, metric names, artifact names, and saved results.
The local-CIFAR selection writer also saves a separate failure record with
available attempted trials; it preflights that path before source access.

## Alternatives

- Retain only an iteration sentinel. This would still construct the held-out
  dataset, leaving the observed gap unresolved.
- Rewrite the CIFAR splitter or tuning contract. The existing opt-in source
  path already provides the needed boundary and has focused tests.
- Rerun or replace historical selections. Their original access boundary is
  part of their provenance and their scores remain historical evidence.

## Consequences

Raising source sentinels now prove the current development writers never
request final-source construction. Tiny local archive and weight stubs exercise
the real writer contracts without a download or full-data rerun. A successful
selection remains validation-only and the manifest still freezes separate
confirmation seeds. Historical saved artifacts remain unchanged and retain
their earlier, weaker construction boundary; this correction does not upgrade
their provenance retroactively or assert score equality with a new full run.
