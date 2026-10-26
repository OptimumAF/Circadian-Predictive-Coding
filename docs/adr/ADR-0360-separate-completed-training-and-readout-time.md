# ADR-0360: Separate completed training and diagnostic time

## Context

The warm MiniGrid runtime already completes eight delayed reward updates, and
its TRAIN evaluator already accepts the actual variable-row checkpoint. The
remaining recipe composition must evaluate owned before/after checkpoints
without charging later diagnostic time to the completed training phase.

## Decision

Add an explicit opt-in to the existing recipe and CLI. Complete all eight
training updates, final memory check and full ownership/receipt validation
before freezing the original training budget's identity, configuration, clocks,
charges and row evidence. Keep validating these exact completed facts while
readouts run. Any later change, including advancement of the completed clock,
is refused. No training allowance is renewed.

Readouts use a distinct zero-update budget with at most 60 seconds and the
remaining original whole-case deadline. They share the original RSS sampler;
all diagnostic forwards, rows, sessions, physics and partial failures count.
The original 180-second whole limit, 45-second reward phase and 768 MiB warm
arm remain. Evaluate only the two owned checkpoints and six original TRAIN
cue packets through the existing frozen recurrent evaluator. All added
readouts occur after all training. Existing default paths remain.

## Alternatives

Continuing to tick the completed training clock incorrectly charges evaluation
to training. Renewing training time or loosening counter checks changes the
accepted budget and evidence. Recreating another learner from JSON cannot
recover the absent native optimizer and parameters.

## Consequences

The recipe can report known-task endpoint changes with explicit cost. It does
not give diagnostics training, tuning, promotion, final-data or probe authority.
Two endpoints cannot establish within-stream recovery or a causal reward
advantage. Runtime and scientific acceptance remain separate.
