# ADR-0640: Separate nuisance TRAIN validation from chooser fitting

## Context

The existing MiniGrid validation recipe changes chronology but reuses the
chooser-fitting episodes and grey A3/A4 endpoint games. It cannot establish
transfer to an input domain excluded from fitting. Reusing its fitting baseline
for a different validation domain would give the wrong before/after comparison.

## Decision

Add the fixed Python recipe `nuisance_redwall_delay2`. Fit the original chooser
on grey A3/A4; freeze it before collecting a red-wall A3 TRAIN pair. Keep the
old grey A3 pair and new red A3 pair in a separate validated study. Both domains
use branch3, so geometry and task semantics stay fixed. Reserved F layouts and
final/probe sources are unavailable.

Use the existing stream, bank, policies, budgets and evidence guards. Defer a
fresh grey/red A3 warm baseline and all endpoints until every validation window
finishes. Bind collection and readout to the same fixed layouts/seed/cues/roles.
Reuse complete-state readouts only within that domain/invocation; reused utility
is unavailable. Preserve the legacy recipe and fitting baseline unchanged.

Move the existing G/A4 native collector into an adapter, retaining its script
wrapper. The validation path shares its teacher, cleanup and provenance behavior
without creating a cycle among script orchestrators. Extend that collector only
for the fixed red A3 profile, with packet/native-record equality checks.

## Alternatives

Reusing fitting episodes tests schedule changes only. Reusing the fitting
baseline mixes A4 geometry with the A3 nuisance domain. Accepting arbitrary
sources/layout callbacks would broaden admission and source-selection scope
unnecessarily. A new runner or cache would duplicate existing components.

## Consequences

One new collection pair is paid under the original cumulative owner. Its records
and charges enter the original bounded evidence store, including failure-prefix
recovery. Frozen chooser/source and RNG checks continue during validation.
The public API gains one explicit recipe; CLI defaults and scientific budgets
do not change. Literal checks establish software boundaries, not native nuisance
performance, generalization, controller benefit or release readiness. Any native
observation needs a separate frozen protocol, preserving earlier spent studies.
