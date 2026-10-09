# ADR-0207: Bind inbox training to the original consent authority

## Context

CPC retains native replay payloads independently of inbox permission metadata.
Public registration and training remain accessible through the underlying owner,
and supported checkpoints replace the inbox without replacing the budget or clock.

## Decision

Install an optional permanent registration and ready-selection fence on a fresh
inbox. Keep a bounded lifetime metadata catalog in the original manager; transfer
exact hook identities during supported handoff. Charge grants permanently and
invalidate old checkpoint authority when declarations or opt-out change.
Require training/replay consent and permissions, with explicit synthetic/unverified
policy. Refuse transient/audit-only retention until actual purge semantics pass.

## Alternatives

A caller-only wrapper permits public runtime bypass. Serialized consent embedded
in checkpoint copies can resurrect revoked authority. Claiming transient storage
without modifying native replay would misstate the actual retention behavior.

## Consequences

Legacy uninstalled behavior remains intact; installed public arrival/train paths
stay consent-bound across same-process handoff. Record quotas do not bound native
bytes. Opt-out prevents future arrived updates but does not erase existing buffers,
checkpoints, caller copies or parameter influence. Full R3.6 remains unfinished.
Arbitrary native consolidation, durable restoration and authenticated provenance
need separate acceptance controls.
