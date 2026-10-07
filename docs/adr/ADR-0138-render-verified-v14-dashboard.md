# ADR-0138: Render the verified v14 table as a static dashboard

## Context

P5.6a publishes a checked, descriptive table from a completed fixed v14
bundle. P5.6 also calls for plots and dashboard content that cannot drift
from those source bytes. The existing `docs/index.html` is a historical
snapshot. Its provenance warning was in the README, but inspection found
that the dashboard page itself did not display that caveat. The fixed v14
study has only two seeds and no record of failures outside its published
bundle.

## Decision

Render a standalone HTML page and four final-metric PNGs from the verified
P5.6a summary, preserving its arm/method order and fixed metric names. A
plot dot is the observed seed mean; its bar spans the observed minimum and
maximum. The page prints the same nine rows with mean/minimum/maximum/range,
seed count, 18 completed cells, scoped failure facts, exact outcome protocol,
source commit/dirty state, and the fixed NumPy synthetic continual track.
It labels the spread descriptive and does not rank cells or infer an
uncertainty interval.

The file boundary calls the report verifier before reading, rechecks its
source bytes, and atomically publishes an exclusive `dashboard-v1` directory
with a manifest binding the source report and every output SHA-256. Its
verifier re-derives the HTML and PNG bytes and refuses stale, removed,
added, or hand-edited files. Use the existing Pillow dependency for static
plots and self-contained CSS for the page. Do not replace or correct the
historical dashboard as part of this derived view.
Add the existing historical caveat and a link to the provenance notes to
`docs/index.html` itself, without changing its historical charts or data.

## Alternatives

- Replace or reinterpret the historical dashboard: rejected because its
  existing evidence and provenance limits must remain visible.
- Use a remote chart library or service: unnecessary for four fixed static
  metrics and adds a new dependency and availability boundary.
- Recompute plot values from raw outcomes separately: rejected because the
  verified report is the single presentation source for this milestone.
- Sort or highlight a winning cell: rejected because the study has mixed,
  null, and negative findings and the renderer must not select outcomes.

## Consequences

The new page and images can be regenerated and audited from one completed
local bundle without training. Their exact PNG bytes depend on the local
Pillow renderer and are checked within that environment; P5.7 will define
broader reproducibility tolerances. The page makes no claim about failure
rates beyond the published bundle, causal advantage, or other benchmark
tracks. Source v14 files and the historical dashboard's chart data remain
unchanged; its page gains an explicit provenance warning.
