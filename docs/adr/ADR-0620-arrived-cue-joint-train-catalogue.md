# ADR-0620 — Jointly fit current and previously arrived TRAIN rows

## Context

The chronological affine baseline learns the current task but leaves the other
at13/18. Existing sequential rehearsal would replace each head again. The
unchanged learner already accepts a homogeneous batch of up to128rows, and the
existing catalogue composer can combine independent rows with source spans.

## Decision

Add an opt-in bounded recipe to the original delayed stream. Reuse one previous
TRAIN packet per public initial cue, only after its label arrival. Compose old
and current9row packets into one joint fit, without fabricated trajectories,
new orchestration, optimizer or native-owner changes. Keep the default baseline.

Count all joint rows as native training exposure and report extra replay rows
explicitly. They are not separate consolidation events, so do not invent event
charges or describe the unchanged event counter as zero replay. Report shared
bank numeric equivalents, materialized inputs and full sampled process costs.
Preserve partial receipts and mark failed/unobserved replay as an upper bound.

## Alternatives

Sequential old/current fits overwrite each other. An unlimited bank hides
storage costs. Starting from joint future-task labels invalidates acquisition.
A new runner or live retention service is unnecessary for this finite question.

## Consequences

The sole fixed TRAIN case retains both A/B at18/18 through return, using90rows
instead of54. Joint squared error stays about0.5/head and current-task CE worsens
slightly. No matched-compute, CPC, native-game, reliability or generalization
claim follows. Existing native108failure stays open and spent. A new retained
model's known old/new native task usefulness is the next distinct outcome gate.
