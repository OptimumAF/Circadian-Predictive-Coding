# ADR-0086: Restore CUDA streams for unmatched vision checkpoints

## Context

The trusted unmatched-vision checkpoint already preserves three complete
model outcomes, a circadian classifier/backbone snapshot, loader cursor,
guard retry state, and CPU process streams. It rejected CUDA before loading
data. The circadian head snapshot includes its local CUDA split generator,
but the runner did not save the process CUDA stream. Seeded v3 also trains
each variant inside `torch.random.fork_rng`: its active stream and the outer
stream restored when the fork exits are different continuation boundaries.
Earlier v1/v2 use a shared CPU loader generator without that fork.

## Decision

Keep the version-one trusted payload and add optional canonical CUDA device
and selected-device process RNG fields. An active circadian progress record
also stores the outer entry CUDA stream beside its existing outer CPU stream.
The full classifier snapshot continues to own the head-local split generator.
The runner resolves a CUDA alias to an explicit index before capture and
preflight. For seeded v3, resume sets the saved outer CPU/CUDA streams before
entering the fork, then restores the saved active process streams after the
fresh classifier and loader are reconstructed. Exiting the fork restores
the outer streams. For v1/v2, the saved active process stream and shared
loader generator continue directly.

Preflight checks the saved device, CPU byte-tensor type and validity of both
CUDA streams with temporary generators. It validates completed models, the
active full classifier, loader cursor, and retry/report state before live
restoration. A failed preflight restores the caller's Python, NumPy, Torch
CPU, and selected-device CUDA streams. CPU checkpoints retain empty optional
CUDA fields; an older CPU pickle without those fields still validates.
Only the configured benchmark device is in the saved CUDA process scope.

## Alternatives

- Re-seed CUDA at resume: loses draws between the seed and checkpoint.
- Restore only the active CUDA stream in seeded v3: the fork would restore
  the resumed process's unrelated outer stream after training.
- Use the head-local split generator as the process stream: it does not
  describe random draws in the backbone or other CUDA operations.
- Change loader seeding or model order: would alter the established v1/v2/v3
  protocols instead of restoring their state.

## Consequences and evidence

On the local RTX 3080, a tiny but real CUDA classifier consumes process
CUDA draws in its backbone. Six bounded fixed-fixture cases compare an
uninterrupted checkpointed control with wake and accepted/rejected
post-sleep resume across v1/v2/v3. Trained model hashes, non-timing reports,
and next Python/NumPy/Torch CPU/CUDA draws match. One wake checkpoint also
resumes in a second Python process with matching hashes and draws. Wrong
device, missing/malformed active and outer CUDA streams, and incompatible
head-local generator device reject before training while caller streams stay
unchanged. An older CPU payload remains valid. The CUDA focused set uses a
120-second external cap and 35-second child limits; no large sweep or new
algorithm comparison was run.

The fixture uses eight synthetic examples per role and a small backbone, so
it verifies the checkpoint contract rather than a full ResNet performance
result. The validated CUDA stream is for the selected benchmark device;
independent random work on other devices is outside this protocol.
