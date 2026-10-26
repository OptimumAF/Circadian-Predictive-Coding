# ADR-0466: Keep corrective labels separate from executed student actions

## Context

Existing visual MLP TRAIN fits do not produce native task success. The matched
binary objective improves some errors and worsens others and both forgetting
costs; a data-coverage cause remains unproven. Existing successful teacher
demonstrations execute their own commands, so replacing student actions in an
episode with queried teacher commands would falsify the actual trajectory.

## Decision

Add an explicit correction option to the existing frozen learner rollout. Store
bounded provisional labels separately, paired with the exact public observation
and executed transition. Use immutable core records with local logical arrival
gates, existing teacher/geometry machinery within infra, actual reset-seed/owner
binding and existing full learner state guards. Preserve failed-work evidence,
without releasing a usable batch after late state or cleanup failure.

## Alternatives

Synthetic teacher episodes would misstate executed actions and rewards. A new
runner or supervision framework is unnecessary. Native retries, seed selection
and stronger-expert claims would exceed this implementation contract. The
default frozen rollout and accepted matched evidence need no repetition.

## Consequences

A later budgeted private learner can consume corrective pairs while retaining
the actual student trace. Fixed nominal schedule labels at off-reference states
are provisional; physical correction quality and learning benefit require a
separate frozen experiment. Serving actors/clocks are initially excluded from
the option. No new dependency, algorithm, native success or CPC claim is added.
