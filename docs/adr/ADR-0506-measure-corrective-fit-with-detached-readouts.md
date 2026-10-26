# ADR-0506: measure corrective fit with existing detached diagnostics

## Context

All148-row corrective fitting reaches120 but fails all four fixed native TRAIN
scenes and loses112 Pick17 success. The accepted112 readout measures original98
rows only. We do not know whether120 fits the correction labels or damages the
old groups. More optimizer updates cannot answer that uncertainty alone.

## Decision

Add explicit corrective=True to the existing TRAIN readout, selecting five ordered
distinct-ID groups and148 total rows. Preserve the four-group130-row default,
detached PhaseReadouts64KiB boundary, numerical diagnostics and failure accounting.
Use existing study composition to reconstruct accepted120 exactly, measuring the
same five groups at112 and120. Require the accepted whole corrective scope and
input receipt to match before the120 readout. Native scenes remain historical.

The model itself is not serialized in the accepted report. Reconstructing it pays
the original work again; no new optimizer dose follows120. Account separately for
all new gradient/residual readouts, copies, memory and time. Keep all groups/axes,
native objective and physical residual definitions; no favorable subset selection.

## Alternatives

- More training immediately: changes the learner before diagnosing the observed
  failure, and adds an uncontrolled optimization choice.
- New checkpoint loader/export: unnecessary infrastructure for this fixed question.
- Replace existing diagnostics: duplicates working detached isolation/paid work.

## Consequences

One explicit fixed five-group mode and one diagnostics-only study path; no new
optimizer, dependency or orchestration framework. Corrective losses can inform a
later conventional learner repair; they cannot establish physical success,
reliable adaptation or a CPC advantage. Every negative result remains preserved.
