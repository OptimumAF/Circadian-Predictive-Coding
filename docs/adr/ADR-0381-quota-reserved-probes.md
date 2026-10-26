# ADR-0381: Reserved probes at owned quota checkpoints

## Context

The acquired quota route has completed TRAIN prefixes and physical collection,
but its full paired final readout remains incomplete under the original240s
limit. Existing R2.1/return helpers already measure E16 learning on disposable
forks. The quota runner owns local checkpoints rather than AttributionStream
state, and its typed episode learner is incompatible with array probe training.

## Decision

Add one explicit TRAIN/probe-only mode to the existing quota runner, with pure
initial/end checkpoint callbacks and no final construction. Restore a complete
owned recurrent snapshot into the existing array learner and use the original
`run_checkpoint_probe` / `run_fresh_task_probe` helpers. Keep independent probe
charges and genuine native indices; guard initial success before subsequent
training, data/record identity and values, original runtime/budget/clock state,
and terminal runtime readback. Default routes retain their current behavior.

Why this: it supplies a runnable learning-capacity diagnostic with existing
learners and execution tools. Probe updates cannot change the benchmark or
serving learner; benchmark typed role admission remains intact.

## Alternatives

- Another full-readout qualifier: does not directly answer remaining E capacity
  and would repeat the incomplete experiment without a new implementation need.
- Widen typed benchmark training to arrays/probe roles: obscures the release and
  role boundaries already enforced by its native episode path.
- A new probe runner or fake AttributionStream: duplicates existing behavior and
  gives local quota state a misleading owner.

## Consequences

Initial0/endactual compares both prefix and reward history. It cannot isolate a
causal reward/quota effect, prove old-task retention/recovery, generalization or
matched trained-control/CPC benefit. All costs and failed/incomplete outcomes
remain visible; failed reserved-data construction has unknown physical work.
Full paired native acceptance and the broader research/release goals stay open.
