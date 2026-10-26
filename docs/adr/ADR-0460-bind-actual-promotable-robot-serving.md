# ADR-0460: Bind actual promotable serving at the robot boundary

## Context

The accepted fixed visual context case learned B in a private nearest candidate,
but actual A-only serving failed B. Existing serving promotion and guard
evaluation already implement publication. The frozen robot smoke previously
accepted only exact StableActor and expected its private `_learner` field;
PromotableActor instead owns a generation and complete serving slot.

## Decision

Add a narrow adapter and admit exact nearest-BC PromotableActor through the
existing smoke argument. Use actual public predict/serving_snapshot, private
type/identity admission only, and complete typed generation/version/model/
configuration/cache/metadata/tick checks around controls and cleanup. Compare
bounded plain auxiliary values; retain existing learner validation. Refuse
changes, clear stale flags, preserve partial work, and require fresh admission
after publication. Existing controller alone authorizes publication.

## Alternatives

Restoring a copied learner would bypass actual serving. Broad subclass admission
would assume incompatible ownership. A new publication runner or cache codec
duplicates existing responsibilities. Ignoring cache/generation changes would
attribute controls to an unobserved bundle. None satisfies this boundary.

## Consequences

Fake robot tests exercise real app commit/rollback and actual controls. Public
snapshot deepcopies remain trusted and outside comparator allocation bounds;
checks do not lock native commands against concurrent publication. No new native
experiment or scientific benefit is established. Native publication requires
legitimate disjoint arrived inner_guard data; accepted TRAIN17/19 trajectories
cannot be relabeled. Full research and release requirements stay open.
