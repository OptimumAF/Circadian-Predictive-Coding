# ADR-0220: Serialize private journal publication with a held SQLite transaction

## Context

R3.5b2d2 persists terminal facts, observations and spent charges. A local coordinator
lock and an exact journal CAS still allow a second legitimate writer to alter
authority between validation and publication. Original R3.5b2d3 acceptance requires
cross-writer serialization and subsequent live composition/crash evidence.

## Decision

Add an inward context-managed publication port. Its private SQLite adapter holds
`BEGIN IMMEDIATE` across exact independent authority comparison, fresh observation
validation, trusted callback execution and fresh exit validation. Every supported
writer already uses that transaction kind with zero timeout. The guard changes no
authority rows. Exceptions poison its adapter and close the connection.

Why this: the existing database provides writer exclusion without another
dependency or a platform-specific lock. Validate the port/adapter as R3.5b2d3a
before coordinator integration, keeping the original full task unchecked.

## Alternatives

Standalone read/CAS leaves the publication interval exposed. A one-instance lock
does not exclude other adapters. A second external file lock introduces another
ownership/recovery protocol. A SQLite transaction covering authority writes and
publication cannot atomically roll back an external model or arbitrary callback.

## Consequences

Reservations, reports and terminal reconciliation are excluded during guarded
publication on the supported private database. Callbacks cannot write the guarded
journal. Spent authority stays exact; observations inside this guard are checked,
not independently persisted. Coordinator integration must persist observation and
terminal facts around acquisition/release and preserve failed-commit witnesses.

This guard cannot authenticate arbitrary host facts, stop hostile file changes,
preempt callbacks, undo publication, grant native completion, prove live Windows
composition or recover after coordinator loss. These remain separate requirements.
