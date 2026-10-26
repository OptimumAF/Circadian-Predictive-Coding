# ADR-0437: Play known TRAIN games from genuine H32 checkpoints

## Context

The fixed H32 quartet completed its original TRAIN fits without full acquisition.
Trace scores are insufficient for R4.2 live-game behavior. The existing evaluator
admits older H8 policies/clocks, while the new fit runner already owns a genuine
H32 endpoint and complete original stream/budget/source evidence.

## Decision

Add explicit H32 method admission at128updates/6912rows with source0/0, preserving
all existing recipes. Reuse the original two target-free A games, frozen reader,
public-history encoder and zero-update helper. Invoke after all fixed fit readouts,
inside the original parent time/RSS budget. Keep the readonly budget binding
through final publication and preserve paid partial reports before later guards.

Why this: it supplies actual episode behavior without a new environment, runner,
checkpoint format, training rule or label authority. Both cue slots are fixed.

## Alternatives and consequences

Another width/dose trial would repeat the capacity investigation. Reconstructing
weights from reports cannot supply a genuine owned checkpoint. Relaxing the old
counter or policy checks would broaden authority. A separate game orchestrator
would duplicate established isolation and failure accounting.

The new route requires the current completed stream; an external structural
snapshot is not proof of its learning history. Unknown helper work stays unknown.
Both budgets remain independently owned and original training time never renews.
Known-TRAIN success does not establish heldout, RL, retention/recovery or CPC
benefit. Comparative use needs a small frozen protocol and all negative results.
See [usage/structure/checks](../wide-neural-train-rollout.md).
