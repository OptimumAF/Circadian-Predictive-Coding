# ADR-0206: Retain all requests and verify actual native overlap

## Context

Isolated learner timing does not measure serving during training. A phase named
training can contain requests after native work ends. A readiness barrier or wait
inside a timed native call can falsely imply computation/serving overlap.

## Decision

Time actual shared actor requests and original native calls on one monotonic
nanosecond clock. Keep readiness/wait/join synchronization outside native intervals.
Retain every request and report both full phase populations and prespecified
timestamp overlap/containment populations. Use fixed nearest-rank quantiles and
explicit coverage/resource/count gates. A reserved finite two-method fixture runs
only after correctness/source binding; no favorable request/seed/run selection.

## Alternatives

Timing only native learning omits serving latency. Labeling all phase requests as
overlapping training hides scheduling gaps. Filtering slow requests or adapting
the protocol after seeing relative performance invalidates matched comparison.

## Consequences

Negative/no-coverage/no-advantage results remain valid retained evidence. One small
fixed-order run describes this local fixture only. Timeout errors carry raw partial
observations/live-worker status rather than silently rerunning. Scientific utility,
hard preemption/RSS limits and durable restart recovery are separate unfinished work.
