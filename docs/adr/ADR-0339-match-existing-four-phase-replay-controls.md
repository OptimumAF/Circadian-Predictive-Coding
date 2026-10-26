# ADR-0339: Match existing four-phase replay controls

## Context

The accepted four-phase policy learns C then forgets it after A returns.
StreamRehearsal already supports recent-only and partitioned complete-episode
replay. A comparison needs matched update/storage ceilings before attributing
any effect to another mechanism. Original replay accounting only admits192wakes.

## Decision

Add explicit256-wake accounting while retaining the old192 default and full
guards. Compose existing recent/partitioned policies through the same runner,
384nativeupdates/16retainedtransitions each, with final data after training only.
Use separate serial CLI processes and frozen source/protocol; first-arm results
never enter the other arm's training. Preserve all five endpoints and failures.

## Alternatives

Comparing only256no-replay to384replay confounds update count. Another runner
duplicates working ports. Changing recent selection/cue order after observing a
result would invalidate the declared control. Treating more cached accuracy as
useful live retention omits actual failed behavior.

## Consequences

The frozen primary is tied at zero retained B/C live success/return. Both controls
add50% training work over the earlier no-replay reference. Content/cue coverage
differs (blue-only recent under existing period/order), so no pure reservoir/
memory-age causal claim follows. No circadian mechanism is compared. A future
declared-context coverage control must expose its additional metadata and match
ordinary controls at the same capacity/update ceilings; old negatives remain.
