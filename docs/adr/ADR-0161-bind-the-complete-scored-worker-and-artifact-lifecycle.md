# ADR-0161: Bind the complete scored worker and artifact lifecycle

Date: 2026-10-01
Status: Accepted and verified for P6.7c1c2 correctness and subsequent c2 execution.

## Context

Complete train references and strict scored JSON/final execution observation
are implemented, but neither establishes a bound scored process or completed
artifact lifecycle. Both original training results are 134 MB. The full scored
worker must preserve all original algorithms, settings, seeds, metrics, null
failures and 16,000-update/600-s/5-ms observed 512-MiB limits.

## Decision

Add separate pure execution declarations, infra current bindings, a bounded
worker, exclusive artifact storage/readback and a thin CLI. Reuse both unchanged
complete training readers in the parent and keep only their small reference
report. Bind that exact independently verified pre-final report (72,050 bytes,
SHA cc1c1deb...26b001) to the closed new request; rederive it through both readers
on each actual parent execution/readback. The child rechecks all six file bytes
and markers without retaining a decoded training result beside its live models.

Extend and prospectively freeze the full static local source closure and exact
request/command/environment before worker fixtures. Fixed old source pins stay
exact, including the historical reference-inspection producer even though the
new CLI calls its reader directly. The conservative closure is the union of
both entrypoints; preserve all 92 prior pins rather than dropping provenance.
The new binding module and CLI bind their own bytes in each request to
avoid self-referential source constants. No seed/metric/baseline/cap override or
partial scientific scope. Pure metadata verification does not grant IO authority.

The child samples from initial binding through unchanged training, held copies,
global whole-fact/full-state proofs, actual final observations, independent JSON
validation and scientific serialization. Retain views until post-serialization
state/content checks. Compare actual optimizer work including rejected replay
with the exact verified reference. Require every release/field/prediction/count
link and resource fact. Framing stdout and parent publication stay outside child
RSS sampling, matching the original predeclared training contract.

Compute intended request/result/audit byte identities before exclusive writes.
Recheck actual bytes/current bindings after publication; a late failure prevents
completed readback. Cooperative claims prevent writer interference; preserve
foreign occupied bytes and do not attach our failure to a foreign request.
If failure-marker publication fails after our verified completion audit, revoke
only that still-identical owned audit, preserving request/result evidence and
foreign/changed audit bytes. A failing V1 fixture demonstrated readable success
without this rule; the linked V2 freeze precedes repaired fixture reruns.
Require complete request/result/audit, no failure/claim, exact canonical decoded
bytes, current references/sources/environment/command and every pure worker/audit
link on independent readback. Contract/state/source/resource errors abort; only
the already frozen numerical prediction policy produces explicit null cells.

Why this: existing verified contracts can be composed without changing science,
decoding another 134-MB graph in the child, or trusting producer-local counters.
Separate modules keep pure declarations, live execution and filesystem ownership
reviewable. No actual reserved final before every c1 gate passes.

## Alternatives

- Decode complete references beside live models: unnecessary memory pressure.
- Trust saved app totals: cannot independently verify actual executed work.
- Modify old worker/algorithm pins: invalidates already reviewed evidence.
- Widen resource caps or replace failed results: changes the declared experiment.

## Consequences

Five new production modules and isolated boundary/development/fabricated tests;
no dependency or environment/configuration change. Source freezes retain every
repair and must precede scored worker fixture reruns. All original correctness
criteria now have evidence: 203 new/984 related tests, actual two-reader metadata
preflight and a bounded genuine development child with fabricated final fields,
original observers/caps and complete state preservation. No reserved final was
opened. The worker report/log record all source/request/artifact identities and
private fixture limits. C1c2/c1c/c1 correctness can close; c2 still owns both
complete actual scored processes/repeats. Those subsequently pass both public
readbacks and exact scientific result-byte equality; the actual results report
records evidence. P6.11b owns every actual seed and predeclared interval;
scientific/resource/reporting parents remain unchecked.
