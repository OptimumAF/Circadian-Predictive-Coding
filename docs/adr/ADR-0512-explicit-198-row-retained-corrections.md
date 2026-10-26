# ADR-0512: Explicit198-row retained correction scope

## Context

Accepted current120 labels cover49 inputs outside original112 correction inputs.
Keeping both50-row batches with18expert+80old rows requires198 total rows.
Existing loader only admits112 and both packets share observed IDs; existing
mixed data/weight/receipt caps148 cannot fit both without dropping old data.

## Decision

Add exact explicit maximum_rows=198 alongside the148 default in existing mixed
facades/continuation. Preserve head-GN130/default schemas/calls/update ceilings;
validate cap before forks, pin it across forks/weights/sinks/aggregation, and bind
extended report metadata. Mathematical objective and update implementation stay.
Share existing parser/core qualification; new120 loader validates complete original
packet/bytes/history/receipts before deriving a unique target ID with source120
provenance. No original episode, clocks, labels, consent or budget is rewritten.

## Alternatives

Dropping original50 rows or overwriting by shared ID loses required retained
supervision. Broadly raising the default cap changes existing refusal behavior.
A new learner/runner/source-admission framework is unnecessary.

## Consequences

Qualify real198-row updates and actual two-packet input use before task-level
fitting. This prerequisite is distinct from learning a useful task; the original
all198-row learning parent remains open pending a frozen small fit experiment.
Native/update/diagnostic costs and all failed attempts must remain recorded.

[Qualification contract](../../artifacts/runs/expanded-corrective-continuation-20261021/PROTOCOL.md).
