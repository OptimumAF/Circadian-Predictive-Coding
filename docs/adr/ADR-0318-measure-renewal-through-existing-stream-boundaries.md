# ADR-0318: Measure renewal through existing stream boundaries

## Context

R2.6 needs actual capacity/work alongside bounded renewal. Complete renewable
NumPy measurement already exists, while the stream initially excludes multiple
helpers. Renewal is fixed-size and owns a permanent replacement allowance;
capacity's scheduled structural split uses another native snapshot format.

## Decision

Permit observation-only capacity alongside the existing renewal helper. Measure
wakes before renewal and the candidate after each reached scheduled event.
Post-event samples receive no additional wake weight. Preserve original helper
bindings and recorded charges; retain per-event time separately from wake work.
Read available late-failure state without admitting more work or masking the
primary error. Identify the event before opaque callbacks can affect its schedule.

## Alternatives

- Another stream orchestrator duplicates delay, ownership, receipt and scoring
  behavior that already works.
- A generic arbitrary transform callback weakens the explicit fixed-size boundary
  and is unnecessary for the existing renewal helper.
- Coercing ordinary renewal into raw structural state changes native assumptions
  and could silently make the later comparison incomparable.

## Consequences

Actual reset work and native array costs can accompany existing candidate
renewal while preserving measured/unmeasured learning trajectories. Split/replay
helper combinations and renewal/capacity continuation remain unsupported.
Complete R2.6 cohort binding and a frozen comparison still remain; this
implementation provides no scientific-benefit or release claim.
