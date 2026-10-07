# ADR-0033: Select sleep components independently

## Context

Both circadian backends previously returned from `sleep_event` when the
split and prune budgets were zero. This also suppressed chemical reset,
homeostasis, and NumPy replay. A zero-topology event could not be
distinguished from disabled sleep by its result. Existing callers and
reports rely on the old budget-gated behavior.

## Decision

`legacy` remains the default and retains the budget gate and existing
structural/report semantics. `components` enables independent switches for
chemical reset, homeostasis, split, and prune; NumPy additionally enables
replay. Its nonstructural work can run with zero structural budgets. Torch
has no replay implementation or replay switch. `disabled` makes forced and
adaptive sleep calls return without changing model state or the sleep
clock. A component switch set to false under `legacy` is rejected, so a
run cannot silently ignore a requested ablation. A `components` event
with all switches off still executes and resets the sleep clock; use
`disabled` for a true no-op.

The existing operation order stays in place: NumPy selects topology,
applies split/prune, homeostasis, replay, chemical reset, then resets its
clock; Torch selects/applies topology, homeostasis, chemical reset, then
resets its clock. Warmup and unmet adaptive triggers still skip the entire
event. `SleepEventResult.performed` is true only after this sequence
finishes and false for skipped or disabled calls. It is an execution
signal, not a rollback acceptance or detailed trigger record.

Toy and continual reports count `performed` events in `components` mode;
`legacy` keeps its topology-change count. Guard evaluation still runs
around a performed, no-topology vision event. Report text identifies the
selected mode without changing existing protocol IDs or selecting a new
metric. Vision CLI flags expose the mode and available Torch switches.

## Evidence and limits

The first zero-budget NumPy and Torch component fixtures failed because
the configs lacked a sleep mode. Tests now isolate every available
component, disabled and legacy zero-budget calls, no-topology event
reporting, vision guard evaluation, CLI mapping, and unchanged seeded
wake steps. The development log records the full quality gate. These
switches do not make sleep atomic: full-state snapshots, rollback,
trigger-clock semantics, and detailed telemetry remain P3.2–P3.10.
