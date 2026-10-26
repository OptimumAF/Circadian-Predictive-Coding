# ADR-0456: Route continuous TRAIN pairs through the existing runtime

## Context

Direct nearest-prototype ingestion now reconstructs two qualified TRAIN reset
contexts and retains the first. That stage-only append does not apply the existing
arrival, budget and stable-serving rules. The generic native learner/runtime
ports already accept the nearest learner's opaque complete state; the categorical
example's hardcoded shapes and snapshot version reporting are the actual gap.

## Decision

Add a continuous recipe beside the existing native stream examples, reusing
ActorShadowRuntime, ExperienceInbox and ToyBudgetSession without modifying them.
Warm from A, then submit B as one array-pair packet with labels at tick2 under a
one-update/one-identity budget. Keep source/actor warm-exact, append only to the
private candidate and do not promote automatically. Preserve all native state,
frozen statistics, physical units and explicit native-call/SGD distinctions.

The existing CLI owns bounded qualified TRAIN17/19 receipt IO. It preserves the
four old scenarios and default23. It reports experiment schema separately from
the complete in-memory nearest snapshot type, with no invented persistence format.

## Alternatives

A new stream framework would duplicate working generic ports. Changing the old
categorical example's tensor/state assumptions globally would increase unrelated
risk. Cold nearest serving is undefined, so an explicitly owned warm model is
required. Automatically promoting a candidate would bypass independent utility,
retention and operational checks.

## Consequences

Saved continuous labels can exercise real logical arrival and budget boundaries
while serving stays stable. The report accounts for ingestion, memory and query
work; actor distance work is explicitly derived because the stable actor returns
only actions. This is offline runtime integration, not live robot adaptation or
scientific benefit. Native serving/promotion, visual features, broader learning
and release requirements remain open.
