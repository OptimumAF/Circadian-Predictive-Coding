# ADR-0229: Model complete lifecycle state and original references

## Context

Full R3.5b2e4b requires original lifecycle, retention, copy and ownership state.
Current source has 66 fields across five owners. Scalar public snapshots omit
catalog/clock history, driver epoch, held token, events, thread and live ports.
Several driver mutations occur outside its operation gate, so acquiring that
gate alone does not prove a coherent capture.

## Decision

Implement the complete inward typed metadata and authority-reference contract
before extending synchronization and actual capture. Preserve original catalog
order and policies, provenance/consent/revocation, declaration clocks, ingress
charges, fault/auxiliary epochs, full driver counters/events/thread observations,
retained enrollment metadata and lifetime copy charges. Keep original policy
aliases in the detached metadata graph.

Retain all original root, lock, callback, weak-reference mapping, token/event/
thread and clock/budget/progress/lineage relationships in explicit named slots.
Validate identities without foreign equality or callback invocation. Copy only
validated immutable metadata, never live authority. Use exact complete native
source-field guards to reject future schemas rather than omit unknown fields.
Independently supplied aggregate entry and UTF8 bounds apply before metadata
traversal and copying.

Split only the implementation increments: R3.5b2e4b1 covers this complete contract;
R3.5b2e4b2 must implement coherent capture and original driver mutator coverage.
Original R3.5b2e4b/e4/full-parent criteria and later encoding/recovery stay open.
No native algorithm, metric, seed, baseline, dependency or experiment change.

## Alternatives

- Reuse scalar driver/copy/ownership observations: omits state and original ports.
- Deepcopy the live authority graph: creates invalid locks, threads and clocks.
- Serialize callback or lock identities as new authority: renews ownership.
- Capture under only the driver operation gate: concurrent stop/wake/terminal
  changes can produce a record that never existed as a whole.
- Omit an active driver or configured copy budget: falsely replaces actual state
  with absence and can reopen spent allowances.

## Consequences

The contract makes every current stored field and reference explicit and testable.
It supplies validated metadata detachment and source-schema refusal. It does not
prove source provenance, lease ownership, atomic live capture or durable restore.
The next required implementation must coordinate all actual driver mutators and
original owner/holder/time/copy/sharing leases while preserving existing behavior.
Full lifecycle and composite/live/disk/native/model/coordinator loss, scientific
and human acceptance remain unfinished.
