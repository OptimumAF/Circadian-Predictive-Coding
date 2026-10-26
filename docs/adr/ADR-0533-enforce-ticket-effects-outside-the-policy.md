# ADR-0533 — Enforce local ticket effects outside the policy

## Context

The accepted temporal decision streams rank one workflow per request. R5.6 also
needs complete multi-step outcomes, recovery, candidate-construction cost and
permission enforcement. Existing learning permissions govern TRAIN/replay/data,
and the runtime guard deliberately refuses complete external admission. Neither
is a ready ticket tool executor. Existing key-door episodes address navigation.

## Decision

Add a pure in-memory ticket transition kernel and bounded application dispatcher,
using existing typed decisions. The policy chooses open, workflow and commit
against every currently active/permitted candidate at every phase. Independent
capability/revision/active-tool/phase checks precede each state replacement.
Ground truth scores committed state and never repairs an authorized wrong action.

Use one fixed conventional six-action policy learned through the existing
perceptron and training-only AttributionStream. Prior four-class coefficients
cannot choose open/commit, so reuse the learner/runner rather than those weights.
All fixed TRAIN completes before unchanged/learned development episodes.

Why this: candidate masking to one phase-correct action would let orchestration
solve the sequence. Independent effect admission lets the policy make observable
mistakes while keeping permitted local effects and task correctness separate.

## Alternatives

One-step scores cannot verify full completion or recovery. A scripted open/commit
controller would leave those actions unlearned. General tool registries or new
learner frameworks are unnecessary. Real filesystem/network tools would expand
the task and permission surface before the local functional boundary works.

## Consequences

Explicit environment invalidation/revocation occurs after prediction; stale and
denied actions pay their costs and cannot mutate the ticket. A stale ticket can
be observed/reopened; permanent commit revocation stops without regranting.
Capabilities are an owned initial grant plus explicit environment changes, not
an arbitrary mutable caller object. No host isolation of injected Python is
claimed. Known concepts/synthetic traces limit generalization; native-family,
science and release claims require their own evidence. The full goal stays open.

[Frozen protocol](../../artifacts/runs/local-ticket-workflow-20261022/PROTOCOL.md).
