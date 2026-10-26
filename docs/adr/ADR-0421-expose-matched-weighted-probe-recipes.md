# ADR-0421: Expose matched weighted stopping recipes in the existing pilot

## Context

The isolated bounded source works, but MiniGrid's existing F16/readout/A16 pilot
still selected ordinary unweighted sources. Comparing that objective directly
against a weighted bounded probe would confound objective and stopping changes.

## Decision

Add explicit fixed-weighted and bounded-weighted probe options to the existing
pilot and CLI. Both use the same admitted episode-weighted objective, native cap,
full warm state and shared unsafe-proposal fallback. Only stopping differs.
DefaultNone and original BP/PC/neutral/active controls remain unchanged. Existing
weighted/unweighted benchmark catalogue choice stays explicit and must be held
fixed within a comparison. Configuration is frozen before execution, not fitted.

## Alternatives

Replacing existing defaults would discard controls. A new runner would duplicate
existing isolation/budgets and return-phase logic. Repeating the warm benchmark
for each stopping recipe would waste work and weaken exact shared-state pairing.

## Consequences

Native32/outer7 costs flow through original paid probes and an owned scalar work
summary; construction/copies/validation add wall/RSS cost. A small local experiment
adapter invokes the original probe twice per existing warm checkpoint, retaining
prior guards and separate budgets. It runs each benchmark once because earlier
saved reports contain no restorable warm snapshots. No final data or automated
selection enters this route. Scientific benefit remains separate from correctness.
