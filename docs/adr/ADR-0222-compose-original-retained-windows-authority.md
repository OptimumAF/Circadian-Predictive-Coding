# ADR-0222: Compose recovery with original retained Windows registrations

## Context

The coordinator now has inward leased publication/report ports and bounded private
SQLite persistence. Windows registration and observation adapters exist separately.
Composition must preserve original authority and retained physical identity without
minting new budget, repinning stale workers or interpreting checkpoint metadata as
coordinator authority.

## Decision

Add one outer infrastructure factory accepting the independent original record,
existing journal path and concrete original retained anchor/worker registrations.
Verify platform, current physical coordinator identity, exact PID/creation, shared
API instance, liveness, terminal/uncertain flags and exact persisted state. Compose
the existing native observer and application coordinator through inner ports.

Why this: a small composition boundary is easier to audit than new registration,
launch, bootstrap or model restoration behavior. Invalid initial record/types leave
ownership with the caller; accepted types transfer handle ownership, with explicit
deduplicated failure cleanup and preserved primary/close errors.

## Alternatives

Repinning from stored PIDs loses the original registration witness. Creating a new
journal from worker metadata changes original authority. Accepting arbitrary probe
implementations at this Windows boundary conceals unsupported native composition.
Importing Windows/SQLite into app would bypass inward dependencies.

## Consequences

The trusted dispatcher retains original state/process ownership and supplies live
registrations; this factory only transfers their handles. The current validation
uses concrete native adapters with private deterministic test API hooks and real
local SQLite. Actual physical worker/crash capture remains separate after gates.
Cleanup errors are reported, not asserted successful; uncertain handles cannot be
reused. No new dependency, algorithm, budget refund, native completion, side-effect
rollback, arbitrary authentication or coordinator-loss recovery is introduced.
