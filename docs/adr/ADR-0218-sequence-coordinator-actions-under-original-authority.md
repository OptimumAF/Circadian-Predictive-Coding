# ADR-0218: Sequence coordinator actions under original authority

## Context

The bounded journal commits metadata reservations but does not own retained
process registrations or sequence callbacks. The existing in-process candidate
checkpoint controller has its own native/lifecycle ownership protocol; injecting
disk recovery into it before complete codecs and live authority would change
its working guarantees. R3.5b2d also needs a publication lease absent from the
current journal read/advance port.

## Decision

Add a small application coordinator using inward authority/observation ports,
bounded exact costs and a closable registration protocol. Retain independent
original state and process probes; acquire a nonblocking local lane; observe and
commit before trusted preparation, then recheck authority around callbacks.
Native admissions remain uncertain without a completion protocol. Handoff
requires the registered predecessor ended and a different live successor.
Terminal close releases all owned probes; failure never refunds charges.

Why this: this prerequisite tests orchestration without changing existing native
checkpoint behavior or pretending caller checkpoint state is authentic authority.
Keep full R3.5b2d acceptance and add scoped R3.5b2d1 evidence for this boundary.

## Alternatives

- Importing Windows/SQLite into app would bypass the existing inward ports.
- Publishing native results by clearing an uncertain flag would invent completion.
- Treating a local lock as an external-writer lease would overstate ownership.
- Reworking the working candidate controller now would precede its codec gates.

## Consequences

Ordering, failure, liveness, contention and cleanup are testable with deterministic
fake ports and real local SQLite. This does not supply arbitrary callback work
attestation, preemption, durable failure/high-water state, coordinator restart
or atomic publication against another writer. Post-publication checks cannot
undo callback side effects. Live composition/publication lease and separately
budgeted process-crash evidence remain part of full R3.5b2d;complete component
and lifecycle codecs remain part of native recovery acceptance.
