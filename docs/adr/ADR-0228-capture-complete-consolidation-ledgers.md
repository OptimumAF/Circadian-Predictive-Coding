# ADR-0228: Capture complete consolidation ledgers under the original lease

## Context

R3.5b2e4 requires complete consolidation and lifecycle/retention/copy records for
eventual recovery. Current source inspection shows these records have distinct
owners. Candidate consolidation uses an original nonblocking exclusive lease,
consumed attempted-ID set and ordered applied receipts. Failed transforms consume
IDs before copying; uncertain restore failure stops the owner. Same-process
handoff retires it. The existing receipt property omits failed attempt identities,
quota and owner flags. Managed lifecycle and driver records have separate live
ports and capture gaps that an inbox or consolidation codec cannot fill.

## Decision

Add a complete immutable ten-field `ConsolidationCursor` and read-only leased
capture on the original candidate. Retain all consumed IDs, complete successful
receipts and native diagnostics, original versions/limit, stopped/retired/revision/
ready fields. Observe closed owners without reopening or touching a model,
inbox, clock, budget, payload or callback. Validate exact supported records before
detaching them; preserve shared diagnostic identity in the copy.

Encode the explicit wirev1 schema through the existing inward checkpoint port.
Require independently known original policy/source/content/authority references,
bounded counts/UTF8/wire fragments and exact canonical input. Preserve shared
diagnostic identity with first-occurrence references and verify complete value
and type at each reference. Preserve native finite Python integers wider than
64 bits as well as float signed zero. Initial narrow integer handling was found
insufficient before acceptance and expanded according to actual native rules.

Split R3.5b2e4a from remaining R3.5b2e4b because their original leases and complete
state are separate; preserve R3.5b2e4 and every full parent criterion unfinished.
No dependency, experiment, model equation, metric, seed or baseline change.

## Alternatives

- Serialize only successful receipt counts: loses consumed attempts and stop flags.
- Reuse candidate native snapshots: calls a model and still omits full ledger state.
- Normalize diagnostics to floats: changes integer type/value and can lose precision.
- Flatten equal diagnostic values: loses original shared versus distinct identities.
- Serialize live owner locks/callbacks or restore a new budget: invents authority.
- Claim complete lifecycle using current scalar snapshots: omits original mutable
  clocks, faults, driver tokens/events, copy charges and owner relationships.

## Consequences

The component provides complete supported consolidation observations and bounded
canonical bytes. Decoding grants no live authority or restore operation. Original
methods, model behavior and unrelated dirty checkout changes stay preserved.
Focused controls cover complete records, failed/gapped attempt history, exact
types/bits/aliases, refusal boundaries and actual nonblocking capture without
native operations. Full lifecycle capture and all composite/live/disk/native/model/
coordinator loss, scientific and human acceptance remain required.
