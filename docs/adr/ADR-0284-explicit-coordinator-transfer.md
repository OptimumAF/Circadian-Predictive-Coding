# ADR-0284: Explicit witnessed coordinator transfer

## Context

The Windows recovery composition retains the original live coordinator as its
clock anchor. Generic authority-floor checks correctly reject a changed anchor,
clock tag or checkpoint manifest. A replacement coordinator needs a separate
trusted transition; accepting arbitrary new disk state would bypass that fence.

## Decision

Add pure typed transfer records and validation in `core/coordinator_transfer.py`.
Require an independent surviving supervisor, ended old coordinator and worker,
and trusted replacement coordinator/worker observations on the same absolute
clock domain. Preserve the original start, full manifest and cumulative quotas;
charge one attempt and advance the owner fence. Add an explicit SQLite CAS method
requiring the exact independent floor. Advance that instance only after COMMIT.

An ambiguous commit has a separate witness-based, terminal-only reconciliation
path, accepting only the exact two possible outcomes and their stopped records.
Either branch charges the transfer once. No failed adapter is reopened.

Why this: isolate the intentional anchor change while preserving the existing
generic floor and fail-closed work admission rules.

## Alternatives

Relaxing generic floor checks would let disk bytes establish new authority.
Resetting elapsed time on the new anchor would renew the original allowance.
Retrying an ambiguous transition could acknowledge work without knowing which
fence committed. Each is rejected.

## Consequences

Core remains free of IO; SQLite owns transactions and writer locking. No schema,
codec, dependency or generic authority behavior changes. Real SQLite tests and
synthetic coordinator ports can validate this boundary independently of native
work. A physical surviving-supervisor adapter and actual coordinator replacement
remain required before claiming full process recovery.
