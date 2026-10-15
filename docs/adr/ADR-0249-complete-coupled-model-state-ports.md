# ADR-0249: Restore complete coupled model state through existing ports

## Context

R1.2/R1.3 provide working coupled PC/CPC wake learners and a genuine neutral
control. The adaptation runtime requires owned forks and complete native state.
The existing parameter-only snapshots omit CPC chemistry, gates and wake counts,
so using them as complete state would change subsequent training.

## Decision

Add typed format-1 `CoupledSnapshot` and explicit `CoupledWakeState` to core.
Require matching topology, precision and chemistry policy; validate array domains,
native counters, initial-state invariants and gate consistency before copying
into committed state. Keep parameter-only restore unchanged.

Add one outer `CoupledLearner` adapter for both exact native families. Reconstruct
owned copies through native snapshot/restore, preserve fixed adapter rates and
inference count on fork, and use existing application orchestration unchanged.
The existing demo gains an explicit coupled-neutral scenario.

## Alternatives

- Restoring only weights would silently restart CPC's adaptive state.
- Copying arbitrary private dictionaries would hide the supported state contract.
- Adding another runtime or source-loading framework is unnecessary for these
  ordinary local model ports.

## Consequences

Full native continuation is testable independently from runtime ownership,
arrival and budget rules. Model restore cannot renew those external authorities.
These snapshots are trusted local in-memory data, not durable or untrusted-file
decoders. Sleep, replay lifecycle and recovery support remain separate work.
