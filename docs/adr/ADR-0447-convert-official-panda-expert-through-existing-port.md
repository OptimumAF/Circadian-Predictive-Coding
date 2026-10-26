# ADR-0447: Convert the official Panda expert through the existing port

## Context

R4.4 now has an executed CPU robot port and continuous BC learner, but no
qualified robot expert data. The official expert uses joint-position commands;
the existing deployed port uses four-axis end-effector deltas and a50-step limit.
The official converter may add target steps and clip teacher commands.

## Decision

Use one bounded TRAIN joint-reference scene and the official converter, with
every converted target step recorded by the existing continuous port. Preserve
the deployed controller/horizon and actual quantized labels. Require independent
observed collection/replay/target success plus complete valid target data before
the result exposes a qualified training episode. Report reset/control costs and
internal unmeasured planning explicitly. Stop upstream loops at observed terminal
flags; preserve failures rather than extending horizons or tuning the teacher.

## Alternatives

Joint commands mislabeled as end-effector deltas would corrupt BC targets.
Changing the deployed controller or extending the horizon would change the task
before qualifying the accepted interface. A replacement scripted teacher would
add an unverified policy. A separate data/orchestration framework is unnecessary.

## Consequences

Existing continuous records/NativeLearner remain reusable. Two real scenes and
reference replay have explicit costs. The official expert may fail within the
fixed task or lose success under conversion; that negative result is valid.
Compatibility alone and successful demonstration collection do not prove strong
BC, retained skill, useful adaptation or a circadian advantage.
