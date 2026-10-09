# ADR-0214: Bound repeated runtime faults before crash recovery

## Context

Current individual controls cover many runtime refusals. Original R3.7 also asks
for repeated failures, runaway growth, stable actor availability and recovery.
Same-process checkpoints do not establish authentic recovery after process loss.

## Decision

Add finite deterministic fault sequences using original budgets and authority,
then one separately reserved eight-cycle stream for both existing native adapters
after correctness/type/static/source/guide gates. Capture owned-array bounds,
sampled Windows RSS, traced allocations, counters and immutable source identity.
Keep full R3.7 unchecked until actual crash/recovery and remaining long-run scope
are proved. Add no production retry loop or new dependency.

Why this: repetition can expose retained growth or quota renewal without a large
sweep. Separate process/trace/array metrics avoid pretending one measures every
allocation. The fixed native fixture keeps cleanup and original authority in one
scope; repeated fake scenarios remain focused standalone tests.

## Alternatives

Counting isolated green tests as crash recovery weakens the original criterion.
Unbounded fuzzing/sweeps before correctness gates spends uncontrolled resources.
Replacing accepted label-first behavior with candidate-base version refusal would
break existing replay/version-neutral semantics. The first new test made that
assumption; restore production source and test mismatch against a known experience.

## Consequences

Repeated supported in-process behavior receives current evidence, with explicit
finite probes and preserved spent work/IDs/quotas. Native observations use original
seed/configuration and no comparative tuning. Source/metric/scientific closures
remain unchanged. Exceptions do not prove process-crash recovery, allocator
preemption, arbitrary callback safety, durable ownership or sustained production
memory stability. R3.5b2 authority policy must precede durable restore work.

Evidence: `artifacts/runs/r37a-inprocess-20261007/`; usage and limits:
`docs/runtime-failure-sequences.md`.
