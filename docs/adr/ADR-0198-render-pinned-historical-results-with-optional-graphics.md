# ADR-0198: Render pinned historical results with optional graphics

## Context

P9.5 requires useful reports and figures with original uncertainty, costs,
negative results and limitations. The accepted P6.10 complete presentation is
available locally. Its original producers/readers have completed scopes and
spent budgets; reopened G0 and runtime correctness prevent new scientific
admission. Required project Python has no Matplotlib/ReportLab. The existing
Codex runtime has ReportLab 4.4.9 and Poppler, without installing dependencies.

## Decision

Use a small pure app projection and an outer write-once export CLI. Bind the
whole accepted source by fixed byte count and SHA256 before projection and
again at export completion. Preserve all cells, original analysis vectors,
interval methods, unknowns and separate resource scopes. Export with optional
ReportLab charts to SVG and PDF, and retain exact numerical JSON alongside the
readable report. Label every page historical; do not grant source/scientific
authority or claim new protocol reproduction.

Why this: rendering already accepted numerical records satisfies an independent
presentation increment without changing models, environments, baselines, metrics,
seeds or estimators. Whole identity binding avoids an exporter silently accepting
a convenient subset or new unvalidated artifact.

## Alternatives

- Install a plotting dependency in required Python: unnecessary environment
  change for this local presentation increment.
- Rerun original readers or experiments: outside this scope and their budgets.
- Replace or relabel old figures in place: would lose historical evidence before
  the remaining full P9.5 inventory is complete.
- Select attractive arms/seeds or recompute intervals: changes the scientific
  question and conceals negative or missing evidence.

## Consequences

Default tests/help/refusals require no optional plotting package. Actual export
requires an existing ReportLab runtime and the locally retained accepted source;
clean clones without that artifact refuse explicitly. New export directories
cannot overwrite old files. Rendered PNG inspection and complete metadata
readback complement numerical tests. This derivative is not an independent
scientific verifier. Parent P9.5 remains incomplete until all remaining report/
figure coverage is reconciled, and the R3/G0 dependency remains separate.
