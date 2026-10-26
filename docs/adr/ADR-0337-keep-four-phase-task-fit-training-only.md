# ADR-0337: Keep four-phase task fit training-only

## Context

The accepted long return recipe has A64/B64/A64, while R4.2 calls for a third
changed context before return. Historical C/D final outcomes must remain final
evidence. The existing recurrent learner/AttributionStream already supplies
ordered episode updates, delayed labels, owned scoring forks and permanent costs.

## Decision

Add an explicit A64/B64/C_train64/A64 train-only route using those boundaries.
Declare C_train rotation2/branch3/swappedkeys/appearance−.75 separately. Retain
all17 fixed train-fit points and five endpoints, with explicit signed metrics,
one continuous optimizer and no final/probe/replay/live/promotion/selection.
Keep all old routes unchanged. Freeze/review a single bounded native case.

## Alternatives

Repurposing historical final C would confuse the data boundary. A new runner
would duplicate working orchestration. More viewer/checkpoint increments do not
exercise the missing task sequence. Automatically increasing updates or selecting
seeds from failures would invalidate the declared evidence.

## Consequences

The route measures task fitting and forgetting at known costs; it cannot certify
live skill, generalization, acquired-skill recovery or circadian benefit. C fits
completely then is forgotten in the observed case. An explicit post-training
cached/live four-phase evaluation is the next implementation, with independent
final data and a separately frozen protocol. All historical negative results
and full parent requirements remain open.
