# ADR-0137: Publish a source-verified descriptive v14 table

## Context

The checked-in dashboard and several figures are historical snapshots with
missing or compromised source provenance. P5.1 already publishes a completed
fixed v14 bundle whose manifest binds raw training/outcome bytes, protocols,
source roles, seeds, and source commit. P5.2 projects raw seed rows but does
not aggregate them. P5.6 needs reports that can be regenerated from checked
artifacts without editing the fixed v14 result or choosing a favorable cell.

## Decision

Add a pure v14 summary transform and an exclusive `summary-report-v1`
directory beside a completed P5.1 bundle. The file boundary calls the
existing bundle verifier, checks the bytes it subsequently reads against
the verified manifest, and writes `summary.json`, `summary.csv`, and a
hash-bound report manifest atomically. Verification re-derives both table
files from the current source. Every configured arm and method appears in
manifest order. For four already recorded final metrics, each row reports
the number of configured seeds and their observed mean, minimum, maximum,
and range. The report carries the protocol, fixed NumPy synthetic continual
track, source run/commit/dirty identity, and explicit descriptive scope.

A completed v14 bundle verifies all declared seed/arm/method cells, so the
published bundle has zero failed or missing cells. It does not contain a
record of failed attempts outside that bundle; the report marks this history
`not_recorded_in_bundle` and limits the zero to
`published_completed_bundle_only`. It makes no winner or causal claim.

## Alternatives

- Aggregate the checked-in historical chart CSV: rejected because its source
  is missing for several charts and it is not the fixed validated v14 run.
- Edit v14 outcomes or the existing dashboard by hand: rejected because it
  would change frozen evidence or allow figures to drift from their source.
- Treat all failed experiments as zero: rejected because a completed bundle
  has no campaign-wide failure history.
- Sort cells by score or select a preferred seed: rejected because the table
  must preserve the fixed study grid and negative/null outcomes.

## Consequences

The table is reproducible from one verified bundle and supports a later
plot/dashboard renderer. It is descriptive across the existing unmatched
NumPy learning rules and has two seeds in the current fixed v14 artifact;
its range is a small-sample spread, not an uncertainty interval. The
report cannot establish a failure rate across unpublished attempts or
generalize to Torch/vision tracks. Those scopes require separately
validated sources and track-aware rendering. Fixed v14 JSON bytes and
historical dashboard files remain unchanged.
