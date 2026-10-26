# ADR-0515: Type allowed-candidate reference decisions

## Context

R5.2 needs an ordinary routing reference and candidate-order/recall controls.
Existing game evaluation identity is method-specific; no typed text/candidate
reference exists in the inspected checkout. Modern model native semantics and
terms remain separate R5.1 prerequisites.

## Decision

Use immutable exact candidate/request/score/result records for the categorical
reference. A pure lexical Jaccard plus fixed softmax control predicts without
labels or fitted statistics. App evaluation admits nonfinal roles, private copies,
disjoint source groups and reverse-order controls; reports all errors, Brier over
the union with the target, unconditional NLL status and descriptive five-bin ECE.
A bounded partial report preserves attempted/returned/validated calls and unknown
work after ordinary prediction failure. IO stays in a thin exclusive-output CLI.

## Alternatives

Extending the game-specific identity adds unrelated method/domain branches.
Starting with a generic agent or model loader would not supply a verified routing
reference. Pretending an untrained model is a released family violates R5.1.

## Consequences

The reference is deterministic and limited; missing candidates and lexical misses
remain errors. Injected Python callbacks are unsandboxed and their behavior/cost
claims remain unverified. No learning, modern-model compatibility, calibration
quality or agent success is implied. This categorical score contract must not
silently normalize or replace a released model's independent scalar-head
semantics; inspect the native interface before any adapter/comparison. A learned
linear-head reference and released-model comparison remain unfinished.
