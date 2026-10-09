# ADR-0212: Purge expired owned data under original clock and gates

## Context

Coordinated deletion and conservative copy quotas exist, but logical-age refusal
requires an explicit cleanup call. Overdue raw owned copies can otherwise remain.
Checkpoint and serving callbacks may cross a deadline while their gates are held.

## Decision

Add optional elapsed seconds anchored to the original budget clock plus logical
declaration ages. Add one opt-in bounded app worker under the original ownership
and sharing authority. Use a separate sharing hold, nonblocking quiescent retries,
payload-free observations and bounded stop/join. Stop/exhaustion attempts all-data
purge and closes admission. Invalid/replaced/backwards clocks fail closed.
Revalidate elapsed access before checkpoint/cache/serving publication after
trusted callbacks. Never renew work, bytes, IDs, ages or driver allowances.

Why this: composing the existing erasure ports preserves the inward dependency
boundary and avoids a new scheduler authority. Separate sharing hold preserves
manual pause state. One small driver keeps bounded lifecycle state together;
long constructor validation/initialization was reviewed and kept in one scope.

## Alternatives

Caller-only expiry cannot demonstrate automatic cleanup. Unbounded daemon retries
hide shutdown and allowance failures. Blocking acquisition risks deadlock or a
false deletion claim. Resetting age on handoff or copy weakens retention.

## Consequences

Supported owned payloads are purged when owners become quiescent and trusted ports
respond. Overdue raw access/publication is refused while cleanup is pending.
Non-daemon worker shutdown is explicit; blocked callbacks can outlive join bounds
and are reported unfinished. This is no hard real-time/OS erasure or caller-copy,
parameter-unlearning, arbitrary callback graph or durable restart certificate.
Transient/audit-only admission remains disabled. No dependency/environment change.

Tests include deterministic deadlines, real bounded workers, quiescent contention,
partial failure, shutdown/attempt exhaustion, late publication probes and one
reserved fixed native handoff capture. Evidence: r36b2b-expiry-20261007.
