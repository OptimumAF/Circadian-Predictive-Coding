# ADR-0036: Restart adaptive history after a width change

## Context

The circadian training diagnostic includes half the mean squared hidden
residual. That term divides by hidden width. Even with the same summed
residual and output prediction, widening from four to five units changes
the diagnostic. A plateau window spanning split/prune widths can
therefore report a trend caused by the denominator rather than learning.
Both adaptive trigger and adaptive structural budget read this window.
Previously they retained it after topology changes.

## Decision

In opt-in `components` mode, an **actual width change** clears the
adaptive diagnostic history after a successful sleep event. NumPy also
clears it after a finalized gradual prune on a successful wake update
and after an external `apply_neuron_proposals` width change. Merely
scheduling a prune or performing chemical/homeostatic/replay work with
unchanged width does not clear it. Torch's immediate split/prune path
uses the sleep-event boundary. The existing training diagnostic formula
and its machine-readable metric ID are unchanged.

Until a full current-width window exists, component-mode adaptive
budgeting uses its configured **minimum** scale, including at startup.
This is a conservative uncertainty rule, not a tuned choice to favor
the circadian model. Adaptive triggering requires a full window and
therefore cannot fire from a mixed-width or incomplete history. The
`legacy` route retains its history and historical insufficient-window
budget scale of 1.0. Torch in-memory snapshots already copy history,
so a guard rollback restores the pre-attempt window. Full NumPy
transaction rollback remains P3.3/P3.7.

## Alternatives and consequences

Keeping mixed-width values under a versioned metric would preserve
legacy behavior but not make a width-spanning plateau interpretable.
Rescaling old diagnostics assumes the hidden residual sum remains
comparable after topology changes, which is not guaranteed. A separate
per-width history is more stateful than needed for this local gate.
Clearing the window delays the next adaptive trigger and temporarily
restricts structural budget; fixed periodic forcing can still attempt
sleep. This changes only the opt-in component route. Other sleep actions
may also shift diagnostic values without changing width; replay and
homeostasis side effects remain under P4.3 and the broader transaction
and retry work under P3.7/P3.8.

## Evidence

The deterministic two-width fixture holds summed residual and output
prediction fixed and measures a 0.002 diagnostic difference solely from
the width divisor. Initial component split and gradual-prune tests
failed because history remained populated; the corrected route now
clears it while legacy retains it. With the adaptive minimum set to
zero for the fixture, legacy can immediately report another trigger
from its old window; components wait for current-width wake data.
NumPy external topology proposals and unchanged-width consolidation
are covered separately. The development log records the full gate.
