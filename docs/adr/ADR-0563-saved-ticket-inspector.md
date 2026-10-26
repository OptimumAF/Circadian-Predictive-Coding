# ADR-0563 — Present native ticket tasks in the existing inspector

## Context

Saved BP demonstrates a new input-context acquisition with old-task retention;
the matched-exposure native PC/CPC controls fail full tasks. Their raw JSON is
available, but the shared inspector cannot route these report definitions.
Another model run cannot repair this presentation gap.

## Decision

Add a pure bounded projection for the two existing ticket report families, using
the existing source-cell contract, table renderer, source dialog and exclusive
HTML exporter. Preserve all original bytes, negative/partial outcomes, counters,
prior charges and missing/null distinctions. Every observed value resolves to
its original pointer. Show environment/action/effect steps and native metadata
when recorded. No model import, state codec, training or metric recomputation.

Why this: reuse gives useful end-to-end inspection without another orchestration
or provenance system. The existing bounds and inert JSON/text publication cover
this small report family; source claims are not authenticated by the viewer.

## Alternatives

- New dashboard/runner: duplicates working publication and source inspection.
- Repeat a successful native case: adds cost without improving the reader.
- Plot interpolated learning curves: fabricates unobserved intermediate scores.
- Hide failed/inactive controls: misrepresents the comparison and missing evidence.

## Consequences

The same command reads BP or any peer report. Recorded completion, learning and
task success remain distinct. Equal exposure is not equal compute/optimization;
one synthetic case does not establish CPC benefit. This scoped route contributes
to R4.7/R4.8; it does not complete all environments, packaging or release gates.
