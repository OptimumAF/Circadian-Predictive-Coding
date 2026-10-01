# ADR-0145: Isolate scheduling with guarded matched replay at fixed width

## Context

The full v14 periodic/current-adaptive/no-sleep comparison bundles replay and structural changes. P6.3 requires schedule attribution and matched baselines, and the completed c4 component factor cannot choose a schedule from its outer scores. The existing core has explicit adaptive clocks and the app has a guard rollback helper that commits baseline replay only after the neutral controller accepts.

## Decision

Freeze a new train-only three-policy factor on seeds 79/83/89. For each policy, use width-eight backprop, ordinary PC and neutral circadian PC with exact parameter parity between the latter pair and identical conditional replay rows. Add planned width-12 no-replay references. Keep periodic interval four, the existing adaptive thresholds and no-sleep control; retain the same FIFO supply at every wake opportunity. Disable topology, gating, difficulty modulation, homeostasis and chemical reset effects. Guard due sleeps on current-task inner roles, count rejected replay executions as cost, and preserve inactive adaptive decisions. Bind all 33 cells and bounded public repeat artifacts before a separate scored gate.

## Alternatives

- Reuse the v14 final outcomes as a new schedule factor: their roles are already opened and replay/topology differ together.
- Give each baseline its own decision or replay rows: this loses matched exposure.
- Lower the adaptive threshold to force events: this would select a new heuristic from earlier negative results.
- Ignore rejected replay because model clocks roll back: this understates executed work.

## Consequences

The adaptive signal is scoped to the neutral head's chemistry and wake history. This factor can show schedule feasibility, applied and rejected work and matched replay behavior, but cannot establish a full circadian benefit. Different accepted event counts mean different total optimizer work. Full-minus-one, independent confirmation, and complete resource attribution remain open.
