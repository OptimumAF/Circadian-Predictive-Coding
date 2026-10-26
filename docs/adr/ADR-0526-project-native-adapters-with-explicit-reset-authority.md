# ADR-0526 — Project native adapters with explicit reset authority

## Context

R5.4-L pure factor composition/projection is accepted, but native owners could
not install it. Reparameterized factors invalidate AdamW coordinates even if
the full-rank represented operator remains the same. Native update/reset/fork
limits and independent source ownership must remain intact. Prior native
controls show failures/zero-shot results, not evidence rank projection helps.

## Decision

Add default-off native policy/method with bound live vanilla Linear adapters,
existing Parameter identities/allocated rank, pure core projection and guarded
float32 installation. Stage all modules and validate protected state first;
consume exactly one existing explicit optimizer reset before copy publication.
Clear old LoRA gradients, preserve head/base/buffers/modes/RNG. Charge projection
attempts/copies/reserved algebra/casts/writes/RNG permanently; poison paid
uncertain failures and retain any reset cost. Snapshots do not refund limits;
forks inherit projection history. Keep one detached historical receipt.

Why this: native training stays runnable with explicit optimizer semantics and
unchanged default behavior. Inactive B is zero while original inactive A rows
remain, allowing future B gradients without new random draws. Allocated memory
is unchanged and effective rank may regrow during later native training.

## Alternatives

- Retaining old moments after QR/SVD mixes incompatible coordinates.
- Replacing Parameters/rank invalidates optimizer/owner/loader identities.
- Zeroing both inactive A and B blocks both factor gradients in those directions.
- Silent reset/refund or rollback claims erase permanent paid interventions.
- A new orchestration framework duplicates existing native owner/stream ports.

## Consequences

Supported metadata is deliberately explicit and checked only for opted-in
owners; existing literal/default paths remain available. This does not establish
released-PEFT compatibility, task benefit, calibration or rank deficiency.
Float32 casting and internal library costs remain qualified. Stream exposure
and any actual native comparison need a separate frozen protocol, preserving
original A/B/C controls, evaluation isolation and all negative results.
