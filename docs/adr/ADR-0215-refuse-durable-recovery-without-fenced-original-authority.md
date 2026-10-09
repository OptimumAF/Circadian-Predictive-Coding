# ADR-0215: Refuse durable recovery without fenced original authority

## Context

Same-process candidate handoff preserves live clocks, budgets, RSS samplers,
consent managers and ownership gates. Those objects do not establish authentic
authority after process loss. Original R3.5b2 and R3.7 require cumulative resource
accounting, complete state and exactly one owner through real crash recovery.

## Decision

Introduce a pure bounded metadata admission validator before creating a durable
payload API. Require a separately trusted authoritative expected record and next
owner fence. Restrict the first proposed adapter to the same authenticated OS
boot epoch and cross-process monotonic nanoseconds, with original start/caps,
spent reservations and observed absolute RSS peak. Reject unsupported epochs,
rollback, uncertain partial work, stopped candidates and invalid owner facts.

Why this: counting downtime and retaining original reservations prevents a new
process from silently renewing elapsed/work/copy limits. Component bindings must
include consent, consumed IDs, revocations, tombstones and retention anchors.
An equality check is useful only when authoritative facts are independent.

## Alternatives

Trusting a caller-supplied digest or pickled checkpoint confers no authenticity.
New process clocks or wall-time deltas do not preserve a verified monotonic epoch.
Replacing RSS with cumulative allocation would change the original metric.
Automatic retry after ambiguous native work can repeat work and renew quotas.
Timeout-based owner takeover can permit two live owners.

## Consequences

The new core module has no outer-layer imports, IO, callbacks, native work or data
copy. Exact bounded records are revalidated and return accounting observations
only. Existing runtime interfaces remain unchanged. No dependency is added.

Durable storage/codec, transactional authority, OS epoch/RSS adapter, live lease
rechecks, failure charging and actual crash tests remain unfinished. A synthetic
`RecoveryFence` proves no real ownership or authentication. Full R3.5b2/R3.7
acceptance is preserved. See `docs/recovery-admission.md` and local evidence in
`artifacts/runs/r35b2a-admission-20261007/`.
