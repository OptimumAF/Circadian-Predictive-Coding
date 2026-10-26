# ADR-0511: Label the current balanced student's observed states

## Context

The accepted fixed balanced120 fit improves first/expert fit but still fails
Push; old correction inputs match only reset. A fixed velocity-clearing control
slightly improves correction mean while worsening first/expert/old-group means.
Neither fit nor historical support supplies labels at new student observations.

## Decision

Reuse the native TRAIN collector through an explicit route in the existing
study. Reconstruct the inherited balanced120 source, fixed148 rows/weights/eight
updates, and require exact prior continuation and full private readout before
collection. Execute only the student; query provisional teacher labels at actual
visited states. Bind reset and every transition to accepted balanced120 native
history before releasing a separate finite50-row packet. Preserve original112
labels, original clocks/receipts/budgets and negative physical outcomes.

## Alternatives

Off-state nearest/time labels would fabricate counterfactual supervision.
Another optimizer knob lacks evidence after the fixed velocity result. A new
runner/collector is unnecessary: existing collector admits native TRAIN120.

## Consequences

No new learning math, collector, framework or dependencies. Reconstruct80SGD and
original diagnostics with full costs; one separately budgeted native collection.
Packets share context IDs, so source artifact/count provenance must distinguish
112 and120 before any future combination. No new-label fit in this scope.
Released supervision is useful data, not demonstrated task acquisition, retention,
CPC benefit or release readiness. Full learning parents remain open.

[Protocol and gates](../../artifacts/runs/balanced-student-corrections-20261021/PROTOCOL.md).
