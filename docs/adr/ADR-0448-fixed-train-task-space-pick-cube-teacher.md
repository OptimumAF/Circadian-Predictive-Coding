# ADR-0448: Fixed TRAIN task-space teacher for the accepted robot controller

## Context

The accepted CPU robot port and BC learner lack qualified task demonstrations.
The official joint-space teacher runs after compatibility repair but produced no
sparse success within the accepted50-step contract. Its failure is preserved.

## Decision

Add a pure fixed50-call align/descend/grasp/lift/settle teacher using TRAIN-only
privileged geometry in the existing controller's root frame. Bind through the
existing continuous port, retain actual normalized commands and require complete
data with final sparse success before training release. Freeze the schedule
before execution and report any-step success and costs separately.

## Alternatives

Extending the horizon, selecting seeds or changing the deployed controller would
change the task to evade the observed failure. Further SDK compatibility work
does not address teacher task quality. A learned teacher or new runner is not
needed to test this data prerequisite. Official teacher quality remains open.

## Consequences

The baseline data producer changes explicitly; physics, learner interfaces,
task/controller and scientific comparison criteria remain unchanged. Privileged
TRAIN actor reads are bounded and isolated from deployed/final/probe learners.
A negative native result is valid and does not authorize tuning in the spent
protocol. Successful data collection alone does not establish BC quality,
retention/recovery or a circadian advantage.
