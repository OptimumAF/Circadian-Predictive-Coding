# ADR-0621 — Read the genuine retained after-B state in known A/B games

## Context

Issue109's chronological joint fits retain TRAIN class accuracy on A and B.
Useful native behavior requires complete closed-loop games. Issue108's different
six-current-phase attempt exhausted its budget without saving outcomes; its
failure remains spent and open.

## Decision

Add one exact opt-in retained-state binding to the existing evaluator and CLI.
Reuse the original game loop, full snapshot copier, source/budget guards and
raw initial/current history. Admit only the qualified109state4,4fits54rows,
two18-row heads; play A blue/red then B blue/red with no new fits. Preserve old
defaults. Keep full source6fit90row costs distinct from selected prefix and
new readout work. No new runner, math, replay service or provenance framework.

## Alternatives

Native success cannot follow from imitation accuracy alone. The offline joint
model already knew future tasks. Retrying108would renew a spent protocol. A
single retained state answers a different bounded functional question without
changing the original guards or admitting future fitted state6 to the scorer.

## Consequences

All four known TRAIN games succeeded in nine steps each,36physics steps total,
zero new fits. Original-source and model guards pass. This establishes after-B
task usefulness, not an isolated retention/replay benefit comparison: before-B
native performance is unmeasured. Independent acceptance, native return after
further learning, generalization, reliability, CPC benefit and release readiness
remain open. Next inspect bounded capacity when C_train arrives; do not extend
the checkpoint/capture infrastructure merely because this readout succeeded.
