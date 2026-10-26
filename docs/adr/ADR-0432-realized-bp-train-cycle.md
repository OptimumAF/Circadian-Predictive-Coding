# ADR0432: observe actual BP momentum on a detached TRAIN cycle

## Context

Fixed old-A rehearsal worsened both warm BP F gain and old-A fit. Existing prior
CE gradient geometry omits realized momentum; the PC update readout has different
math. Neither existing helper establishes a cause for the BP result.

## Decision

Add a bounded diagnostic using the unchanged native BP train_batch implementation
for one B TRAIN wake and two A TRAIN replays on a detached owned learner. Report
actual parameter displacement, prior-gradient and carried-momentum geometry,
old-A CE change and explicit attempted/returned/paid work. Preserve momentum
between calls. Require separate fresh3-update/18-replay-row authority and pin the
original snapshot before callbacks; retain failed prefixes. A small saved-TRAIN
CLI example proves runnable composition without another training framework.

Why this: actual native steps and an independent NumPy oracle avoid inferring
the realized direction from prior gradients alone. No optimizer tuning, reserved
data relabeling, deployed update or returned diagnostic model is required.

## Alternatives and consequences

Changing momentum or increasing doses would change the scientific question.
Replaying reserved F as TRAIN would violate role isolation. This readout supplies
descriptive counterfactual evidence only; it adds real native/diagnostic costs.
The one-warmup example does not reproduce historical joint128 state. Existing
negative results and acquisition/retention/recovery/full-goal requirements remain
open; integrate with existing TRAIN checkpoint readouts before scientific use.
