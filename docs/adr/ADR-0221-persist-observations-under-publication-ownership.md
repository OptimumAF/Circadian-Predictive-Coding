# ADR-0221: Commit observations while retaining publication ownership

## Context

The d3a guard holds the authority database writer transaction across publication.
Original d2 acceptance requires the latest validated clock/RSS facts to be durable
before the callback. Committing that transaction would release writer exclusion.

## Decision

Add a permanent one-byte nonblocking native writer lock on the canonical private
journal path. All supported journal writers, terminal reconciliation and the
legacy guard acquire it. Add an inward lease port permitting only observation
reports while ownership remains held. The surviving coordinator checks its
retained original probes and commits fresh observations before/after publication
through the lease, then checks again after release.

Why this: separate serialization lets existing SQLite FULL-synchronous bounded
transactions remain durable without weakening the original pre-publication fact
assertion. Existing legacy guard behavior remains available. No new dependency.

## Alternatives

Reporting only before lease acquisition misses the freshest lease-entry facts.
An uncommitted report inside the old guard is not durable. Commit/reacquire leaves
a competing-writer interval. Atomic rollback of a model callback is not provided
by this authority journal. A separate SQLite lock database would introduce another
schema and transaction recovery boundary.

## Consequences

The permanent lock file must stay in place; aliases/hostile replacement and remote
filesystems are outside the trusted private path contract. Writer contention
remains immediate. A restricted lease cannot admit native work, stop/reopen,
refund spent counters or complete updates. Lease handles cannot be reused after
release or from another thread. Failure disables the adapter and retains exact
terminal/commit-ambiguity witnesses for explicit stop-only reconciliation.

The current evidence uses trusted fake host facts, real local SQLite and native
thread/descriptor exclusion. Live retained Windows composition, actual journal/
worker crash capture, native model/codec completion, stable actor ownership and
coordinator-loss recovery remain separate unfinished requirements. Callback
preemption, generic authentication and rollback of external side effects are not
claimed.
