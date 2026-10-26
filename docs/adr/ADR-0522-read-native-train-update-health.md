# ADR-0522 — Read native TRAIN update health without changing the update

## Context

The native continuous/reset/rehearsal studies show different acquisition and
retention failures. Loss and global gradient norm do not identify which native
parameter groups receive gradients or move. Saved phase norms relative to the
original model cannot recover individual B updates. Another dose or mechanism
would leave this concrete uncertainty unresolved.

## Decision

Add an exact-boolean, default-off `record_update_health` policy field and the
existing control CLI's `--update-health` option. A small adapter helper reads
already arrived TRAIN gradients before clipping and actual parameter changes
after AdamW. It reports head/LoRA-A/LoRA-B group L2 norms and counts, using
float64 reductions over float32 coordinates. No extra model call or randomness.

Why this: measure what the current native optimizer actually did, including
moments and weight decay, through the existing `last_update` receipt transport.
This preserves the native objective and avoids a new runner or learner policy.

Only bounded trainable copies live until the update finishes. Reports contain
scalars, never tensors/base copies/success chains. A separate detached permanent
ledger charges attempted source elements before clone/reduction work. Native
five-field counters and reset counters keep their meanings; snapshot restore
cannot renew diagnostic charges. Failed diagnostics poison the owner, preserve
paid work and leave the previous successful report as previous evidence.

## Alternatives

- Learning-rate times gradient omits AdamW's moments and decay.
- Endpoint parameter norms cannot reconstruct each step or missing gradients.
- A new comparative run before instrumenting the learner repeats an opaque failure.

## Consequences

Disabled training is unchanged; enabled training costs extra copies/reductions.
Norms distinguish missing, zero and nonzero gradients but do not establish a
causal mechanism, alignment, learnability or adaptation benefit. Diagnostics
are TRAIN-only; final/probe information and outcome-based control are excluded.
The implementation is verified on literal CPU graphs. Released-model behavior
requires its own prospective bounded diagnostic protocol.
