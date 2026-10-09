# ADR-0242: Hold original replay authority through checkpoint publication

## Context

Actual native, checkpoint and inbox copy observations precede the controller's
final native and policy probes. A copied event cannot authorize publication.
Probes may mutate payload values or replace authority fields after an earlier
identity or fingerprint check. Failed preparations remain enrolled and charged.

## Decision

Compose `ManagedReplayCheckpoints` with the existing original replay ledger and
constructor copy coordinator. Charge original invocation, metadata, permanent
live-slot and retained payload allowances before every observed graph copy. Bind
weak witnesses through the actual copier memo rather than matching contents.

Add a narrow application lease around the controller's existing four publication
statements. It prepares weak ledger anchors under original enrolled holder,
consent, time, budget, sampler, actor and sharing guards. After all opaque ports,
fixed core code checks bounded exact native/inbox contents and current held gate
identities. The controller then performs its original publication and two plain
ledger assignments. Cleanup releases trusted guards; no new probe or validation
is added after publication.

Exclusive contexts release the lexical lock they actually acquired. Why this:
a callback can replace a gate field; rereading it during cleanup masks the
refusal and leaks the acquired lock. The replacement field is never restored.

The content stamp is local integrity evidence for an already admitted graph. It
uses exact supported records, bounded scalar metadata and borrowed contiguous
numeric buffers, including shape, strides, writeability and RNG state. It grants
neither provenance nor codec/recovery authority. The default controller path
has no replay transition and retains its original behavior.

## Alternatives

- Grant authority in the last copy callback: rejected because later probes run.
- Add another injected fingerprint loop: rejected because its final callback
  can mutate an earlier row or native weight.
- Hold secondary guards across the entire controller operation: rejected because
  existing lifecycle and copy operations already acquire those guards.
- Allocate a replacement ledger or allowance: rejected because it renews quota
  and loses the original consent, work and holder histories.

## Consequences

The bounded vertical slice refuses unsupported graphs and untracked inbox pairs.
It preserves failed preparations and spent charges. Larger native variants,
compound retention/erase, complete copied-holder capture and scientific/recovery
gates retain their original unfinished acceptance criteria. Exact source review
and budgeted current-tree tests are required before accepting this implementation.
