# ADR-0271: Admit and qualify complete inference iteration granularity

## Context

The original equal10ms allocated-time PC comparison retained an adaptive window
above its frozen11ms maximum. Native updates are indivisible; stopping checks,
buffer copies and bookkeeping contribute to timing. Accepted correctness does
not make that failed case qualified. The original comparison remains incomplete.

## Decision

Add an optional complete-iteration ceiling to the existing timed application
runner, strict matching qualification in its summary and an explicit CLI flag
for a distinct30ms allocated/3ms iteration protocol. Derive those constants only
from all prior timing fields: max-call plus all noncall overhead yields a maximum
retained envelope2.32390001474414ms; round upward to3ms and allocate ten times it.
Keep the10% observed window allowance and all scientific controls unchanged.

Observe every complete interval, including the initial prefix, checks and final
terminal boundary. Failed or unknown intervals retain known committed work and
no scores. No retries, preemption, online calibration or changed default policy.
The old CLI default and failed artifacts remain. A new experiment needs its own
frozen budget within the existing local ceilings and independent source gates.

## Alternatives

Relabeling the old overrun or widening its tolerance would erase a valid failure.
Choosing new learner thresholds from final scores would compromise isolation.
Latent-step or gradient counters omit stopping arithmetic and orchestration;
they cannot replace the enclosing wall-time measurement.

## Consequences

Both methods receive the same prospective timing basis and observed ceiling.
New scheduling stalls may still invalidate cases; negative/incomplete results
remain valid. Three seeds on fixed data cannot establish general speed, compute,
stability or learning benefit. Original R1.5 acceptance depends on actual reviewed
comparison evidence, not this engineering change or a favorable-looking case.
