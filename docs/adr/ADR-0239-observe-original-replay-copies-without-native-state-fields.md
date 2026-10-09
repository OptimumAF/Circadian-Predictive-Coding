# ADR-0239: Observe original replay copies without native state fields

## Context

Legacy replay stores batch snapshots; bounded retention replaces individual rows
by content ID and evicts under three policies. Current model dictionaries and
complete codecs do not carry original row consent. Content matching loses source
identity. The original managed update port now exposes the actual detached inputs.

## Decision

Add an inward bounded scoped write port at native copy/retention boundaries and
compose it with the original managed producer. Use an explicit ContextVar token
owner with thread, reentry, lifetime and input identity checks. Report exact native
snapshot references and ranges without copying arrays or adding model fields.
Reset the original token before releasing references. Keep default storage rules.

Why this: the actual copy is the earliest point where producer input and retained
object identity can be bound without content inference. An explicit scope survives
refused foreign/reentrant close; a suspended generator could lose its token owner.

## Alternatives

Persisting callbacks in model dictionaries changes legacy snapshots and introduces
raw retention. Inferring origin from hashes merges equal inputs from different
producers. Updating storage, persistent lineage and all checkpoint paths together
would hide independent failures.

## Consequences

This increment observes actual writes but supplies no persistent row ledger or
consent/restore certificate. Reports are provisional; caller-retained references
need ownership/accounting. Native parameters or replay storage may already have
changed when an observer fails. Original managed failure/work semantics remain.
Persistent origin and full retained native variants/copy/recovery paths remain open.
