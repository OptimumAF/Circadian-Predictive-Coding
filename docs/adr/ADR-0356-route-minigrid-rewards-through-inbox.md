# ADR-0356: Route native MiniGrid rewards through the existing inbox

## Context

The native148/K7 sampled collector and recurrent reward objective are accepted.
Their one-update engineering qualification does not exercise the generic runtime,
delayed releases, returning contexts or the original serving actor over a stream.
The existing custom8/K5 recipe uses different physics and cannot demonstrate this
MiniGrid integration. Useful acquisition remains unproven.

## Decision

Compose the accepted collector with ActorShadowRuntime, Experience, LabelArrival
and RewardUpdate in one fixed TRAIN-only A/B/C_train/A recipe with both cues.
Collect from an owned fork of the current candidate checkpoint. Verify that no
update occurs before a matching label, then release its exact ID one tick later.
Validate the original receipt, full native state and permanent budget before
collecting the next episode. Source and serving actor remain unchanged.

The existing native observer retains committed receipts before late resource or
validation checks. Failed prefixes keep observed collection/native charges and
stop further admission. Keep initial/current adapter checkpoints and disclose
the existing bounded eight-payload inbox memory. Add a small CLI with pure
description and exclusive bounded JSON output. No new orchestration framework.

## Alternatives

Direct reward calls omit delayed release and serving isolation. Reusing the
custom game collection changes native inputs and action semantics. A new stream
framework duplicates working runtime authority and receipt handling.

## Consequences

The fixed small recipe exercises returning contexts and on-policy continuity.
No teacher, held-out/probe data, promotion, replay or evaluation enters it.
Eight updates and zero rewards are valid engineering outcomes. This establishes
no new-task acquisition, old-task retention, recovery or comparative advantage.
Those requirements need separately frozen scientific protocols after integration.
