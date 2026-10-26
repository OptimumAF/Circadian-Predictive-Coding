# ADR-0579: Display final FIFO36 observations separately from probe fitting

## Context

The accepted fixed benchmark128 D readout is negative. Its saved native report
contains two final games, no TRAIN score arrays and no probe learning. The
existing FIFO36 inspector schema specifically describes disposable F/A probes.
Reusing that schema would mislabel final data as fit or imply unrecorded scores.

## Decision

Add a pure final projection to the existing saved-FIFO app module and dispatch it
through the existing offline CLI, exclusive writer and MiniGrid player. The common
episode decoder accepts an explicit expected D layout; its old defaults and F/A
validation remain unchanged. Keep source history, zero new probe work and final
outcomes in separate sections. Every recorded value and derived transition count
links to its source field or input array; missing evidence has no invented path.

Validate fixed source/model identity, zero new work, original spent history,
finite budget clocks, complete preservation proofs, raw argmax, observation
continuity and bounded paid prefixes. Partial failures retain false/unknown
proofs, recorded calls and planned unknown slots. A complete claim with missing
or false evidence is rejected. No model or environment is imported or invoked.

## Alternatives

- Reuse probe curves: rejected because final observations do not measure fitting.
- Build a new dashboard or source-admission framework: unnecessary; the player,
  bounded JSON handling, source dialog and exclusive file writer already exist.
- Rerun final games: unnecessary; accepted source bytes and assumptions persist.

## Consequences

The saved negative result becomes inspectable offline without new scientific
work. Internal consistency validation does not authenticate external records.
Native index validity and applied calls are not safety or progress. Final results
must never enter training, tuning, promotion or probe selection. This child does
not complete the shared inspector, statistical claims or release qualification.

Evidence and command receipts: issue63 and
`artifacts/runs/fifo-heldout-inspector-20261023/RESULTS.md`.
