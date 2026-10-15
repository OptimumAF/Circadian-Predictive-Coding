# ADR-0287: Stop interrupted replacement work at independently known facts

## Context

R3.7p demonstrates acknowledged native completion across coordinator loss. A
replacement may instead mutate its native model before its new checkpoint is
saved or completion is acknowledged. Its completed B and spent pending admission
must survive without allowing diagnostic C to authorize further work.

## Decision

Add an optional fixed worker wrapper that delegates native training once and
pauses after mutation, before returning to the session. Send full bounded C
telemetry and actual pending authority over the existing owned channel. The
parent independently checks its exact original reservation and reference C,
terminates only that owned replacement, and records genuine ended facts.

Use the existing FailureWitness(pending) terminal reconciliation with no new
child observation. It preserves the last known child stamp/RSS and every spent
counter, changes only sequence/stopped state, and retains completed B. Keep later
supervisor observations separate. Refuse pending/stopped adoption or transfer,
guessed C floors and stale transfer witnesses. The default successful mode stays
available.

Why this: unknown work cannot be refunded or silently completed, and an ended
child cannot provide a fresh observation. Existing terminal-only persistence is
sufficient; the fault control needs no broader authority or recovery mechanism.

## Alternatives

Treating emitted C as completed ignores validation/COMMIT acknowledgment. Rolling
back to clean B would erase spent work. Relabeling parent facts as child facts
would invent an observation. Reusing the old transfer witness could overwrite a
newer admission. None is needed to retain safe stopped evidence.

## Consequences

No learner objective, baseline, seed, metric, native session/core/SQLite contract
or dependency change. This validates one explicit interruption boundary under a
surviving supervisor. Full managed recovery, arbitrary crashes, supervisor loss,
power loss, long-run behavior and comparative benefit remain separate gates.
