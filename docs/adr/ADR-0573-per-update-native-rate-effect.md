# ADR-0573 — Stage a per-update rate effect without changing snapshot policy

## Context

R7.1 includes bounded learning-rate actions. Complete Torch snapshots pin native
optimizer LR and configuration to an immutable policy; changing policy or
weakening decoder validation would invalidate original state binding.

## Decision

Add an optional scalar rate port with implementation ceiling4 and require a
finite positive effective rate. Unity retains the established public training
method. Nonunit BP effects modify only staged SGD LR for the call, restore base
LR before state validation/publication and retain genuine momentum/gradients.
PC/CPC effects scale parameter learning only, retaining inference/configuration.
Forward the scalar through existing causal update/admission methods. Support a
finite rate+wake ID; expose the requested value in detached polls. Keep replay+
rate/gate/sleep combinations unsupported. Do not change durable receipt/state
schemas or grant final/probe/promotion access.

## Alternatives

Policy mutation contradicts snapshot validation. Rebuilding an optimizer without
momentum changes native semantics. A new runner duplicates the original causal
and budget boundary. Treating a zero-wake rate choice as a persistent setting
adds hidden state and makes later action provenance ambiguous.

## Consequences

Controllers can choose a bounded per-update effect through original authority.
Default math is explicitly rechecked against native references; failures/late
receipts remain charged and lost returns stop further app work. Fixed replay
remains compatible with observed wake receipts. This verifies implementation,
not useful rate tuning or scientific advantage. Review is self-review; gate,
sleep, combined effects and full repository readiness remain unfinished.
