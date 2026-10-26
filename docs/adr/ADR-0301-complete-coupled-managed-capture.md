# ADR-0301: Complete coupled managed capture through existing leases

## Context

Coupled learners already support complete native checkpoints and managed lifecycle
ports. The complete managed capture adapter admitted only legacy NumPy learners.
Native snapshots alone omit learner update policy, explicit model geometry and
managed records needed to observe the whole retained state.

## Decision

Add exact coupled learner/model projections and a coupled graph preflight subclass,
then compose the unchanged app capture coordinator. Reuse its original owner,
holder, resource, consent and copy leases, and the existing shared graph walker
and two-pass deepcopy memo. Extract only borrowed structural metadata validation
from the native codec; no capture-time encoding or model method is needed.

Keep all stored learner/model fields and typed pending snapshots. Validate the
empty replay inventory against every supported native holder. Keep original
authority references outside detached state and retain native-only recovery
completion restrictions.

## Alternatives

Model snapshots would omit explicit stored geometry and learner settings. A second
coordinator or walker would duplicate subtle admission and final-observation
rules. Mutable global type registration would widen legacy acceptance. A new
integrity/hash framework is unnecessary: bounded existing temporary integrity
encoding preserves corruption refusal before original raw reservation.

## Consequences

The reusable API covers exact fixed coupled PC/CPC managed state with detached
payload aliases. Independent graph-array bounds include learned arrays; the
original logical retained-raw ledger does not. No new restore authority, disk
format, algorithm, process restart or learning-benefit claim is introduced.
Future state format changes require explicit schemas and behavioral tests.

The fixed evaluator helper uses its return annotation for typing instead of a
runtime generic subscription. This avoids typing's incidental `__orig_class__`
field while preserving exact source schemas; real integration exposed the need.
