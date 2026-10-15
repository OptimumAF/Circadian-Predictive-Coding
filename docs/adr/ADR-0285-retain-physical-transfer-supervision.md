# ADR-0285: Retain physical supervision through coordinator replacement

## Context

R3.7n validates explicit transfer and SQLite CAS but does not establish physical
process or clock continuity. Windows observations identify their actual current
process; a parent cannot manufacture a replacement process's observation by
anchoring its local observer to that process. The supervisor's closing time/RSS
also occurs after the remote observation and must be retained separately.

## Decision

Retain the supervisor and original live Windows handles in one single-use outer
adapter. Own replacement coordinator and worker as separate direct children;
pin their full identities while alive. Use the fixed trusted child's own native
observation over its private bounded channel, bracketed by supervisor facts.
Add optional `supervisor_after` to the pure observation, using its later time and
all measured peaks while preserving the original absolute start and quotas.
Commit the existing explicit CAS before private-channel adoption.

Why this: genuine retained process references and actual observations establish
the physical boundary without teaching a stale journal to trust new disk bytes.
The fixed example binds a real original coordinator before a deliberate crash;
replacement continuation uses existing coordinator ports and original limits.

## Alternatives

An idle observer alone would not demonstrate loss of an app coordinator.
A grandchild worker could outlive the bounded transport's owned cleanup.
Discarding the final supervisor sample would lose real time/RSS; rewriting it as
the remote fact would invent evidence. Relaxing the generic floor would allow
disk bootstrap. None of these is required.

## Consequences

No new dependency, schema, generic floor or native objective. Existing three-arg
pure callers remain supported. Failed or ambiguous transfers cannot retry or
admit work; only exact witnessed states reconcile to stopped metadata. A child
that has already advanced authority requires a later explicit protocol, not an
overwrite from the old transfer witness. Native checkpoint restoration and full
coordinator-loss recovery remain subsequent integration gates.
