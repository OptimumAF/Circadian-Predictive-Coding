# ADR-0486: Bind recorded readiness phases to existing residual readouts

## Context

Qualified TRAIN demonstrations already retain readiness phases, while the bounded
continuous learner passes `all` to its existing stage-aware residual helper.
Current native outcomes are negative; aggregate errors cannot locate phase fit.

## Decision

Add a small pure owned stage binding and an optional path through the existing
runner/robot CLI. Bind complete original episodes before deriving model views;
validate source metadata and original row partitions before more work and final
release. Reuse the existing diagnostic prediction and summary helper. Default
kwargs, output and all-stage behavior stay unchanged.

## Alternatives

A new diagnostic runner would duplicate orchestration. Time-derived phases would
mislabel a geometry-dependent teacher. Rerunning old cohorts or reconstructing
old parameters solely to fill missing reports would repeat baseline work. Neither
aggregate loss nor a source-level memory hypothesis proves task usefulness.

## Consequences

Selected grouping adds scalar summaries/index scans and bounded source copies,
but no additional model forward. Bindings are not qualifications or release
permissions. Old phase-specific scores remain unavailable; future substantive
learner experiments can record them prospectively. Useful control and scientific/
release parents remain open. No dependency or new orchestration framework.
