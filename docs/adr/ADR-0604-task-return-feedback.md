# ADR-0604: Separate complete task returns from classifier feedback

## Context

Fixed references previously saw classifier error. A correctly classified set of
TRAIN rows is not a complete workflow return.87added private task-trial readouts,
but its sole fit failed saving evidence; no fitted task policy is qualified.

## Decision

Add an explicit immutable controller_train task scalar and a fixed-drift-only
runtime opt-in through the existing reference schedule. Reuse the private six-
task reader/typed actors, source checks, finite budget and paid-record guard.
Pin live actor version and complete candidate state at every private-read boundary.
Reject task-unit transfer into classification-trained contextual policies.
Keep original packet clocks distinct from current diagnostic availability.

## Alternatives

Reusing the classifier type would conceal changed units. Treating task success
as packet-label correctness would permit unjustified acknowledgements. A new
runtime or controller fit would duplicate existing components and violate the
spent87scientific quota. None is required for this fixed-reference input mode.

## Consequences

Private diagnostics cost additional typed reads/copies/time; they do not identify
individual update credit or delayed utility, and may leave correct pending work.
Bounds apply to the original14/24horizon; failure/prefixes and negative outcomes
remain. One known-template integration qualifies the mechanism, not scientific
benefit, generalization, efficiency or release readiness. Matched withheld-stream
evidence needs its own prospective protocol; classifier defaults stay unchanged.
