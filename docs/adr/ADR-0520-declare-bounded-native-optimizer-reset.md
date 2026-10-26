# ADR-0520 — Declare one bounded native optimizer-history reset

## Context

The fixed native A6→B6 stream retains A but fails B. The accepted cold B6
control learns B. Learned weights and AdamW history both differ, so the cold
control alone cannot identify an optimizer-history contribution.

## Decision

Add `max_optimizer_resets=0` to the existing native policy; continuous AdamW
remains the default. Explicit allowance1 admits one empty AdamW replacement
with the same live parameters and recipe. Weights, gradients, CPU RNG states
and model mode must stay unchanged. Attempts/completions are separately paid,
permanent and absent from restorable snapshots. Uncertain mutation poisons the
owner. Existing training counters and update allowances retain their meaning.

The outer control composes the existing inbox/phase/readout helpers. It binds
all initial and after-A trainable hashes before reusing accepted development
readouts, then performs the declared reset and six B updates. The reset is
selected before outcomes; final/probe labels and replay never enter this run.

Why this: a single intervention can test the concrete history uncertainty
without replacing the native baseline, introducing a new runtime or tuning a
sequence to obtain a favorable result.

## Alternatives

- Another cold B run repeats accepted evidence and leaves the uncertainty.
- More updates or changed learning rates alter multiple causal factors.
- Automatically resetting from endpoint metrics creates an untested controller.

## Consequences

Correctness establishes bounded state behavior, not adaptation quality. The
small fixed development study may fail acquisition or retention. Its twelve
TRAIN exposures match the warm prefix, but total computation differs because
readouts are reused and reset construction/checking adds work. Recovery,
generalization, calibration, PC/CPC comparisons and release readiness stay open.
