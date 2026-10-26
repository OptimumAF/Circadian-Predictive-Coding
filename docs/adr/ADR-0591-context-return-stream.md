# ADR-0591: true contextual return through the existing ticket stream

## Context

Joint perceptron qualification74 solves all four contextual tasks; contextual
BP73does not. Joint fitting proves neither sequential acquisition nor retention.
The older policy-change recipe returns invoice under changed v2 truth, so its
failure cannot be called loss/recovery of the original v1 skill.

## Decision

Add explicit fresh-zero `--context-return` to the existing recurring runner and
TRAIN packet builder. Restore both observation context and task truth on return;
read all four same-policy tasks at0/4/8/12. Reuse accepted public-context teacher
validation, original learner, delayed stream, budget and sandbox. Preserve older
recipes. Frozen zero outcomes provide an unchanged control. Only declare recovery
after an acquired same-policy task was observed lost. No new learning rule or
orchestration framework; refuse unsupported native/replay combinations early.

## Alternatives

Using joint-pretrained source hides fresh acquisition. Returning only the domain
under v2 asks a different question. Increasing BP dose or adding CPC does not
address the unqualified base task. A new stream/runner duplicates working code.

## Consequences

The sole12update72exposure case exhibits acquisition and v1invoice loss/recovery,
but return loses v2invoice: retention remains open. No algorithm advantage follows.
Future interference repair should reuse existing bounded TRAIN rehearsal; its
current D7/single-ticket codec needs a concrete contextual extension. Preserve
all negative results and charge replay exposure/memory explicitly. Exact evidence
and commands are linked from docs/context-return-stream.md and issue75.
