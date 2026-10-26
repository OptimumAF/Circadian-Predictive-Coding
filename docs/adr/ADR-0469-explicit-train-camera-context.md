# ADR-0469: One explicit TRAIN camera context

## Context

The accepted nearest-memory visual control acquires TRAIN19 and retains/returns
TRAIN17 on actual known trajectories. Several neural offline improvements have
failed native task validation, most recently the fixed-range learner. The next
substantial environment capability is a declared appearance context for R4.5.
Current camera validation, native creation, teacher and frozen rollout all pin
one exact default recipe; silently editing it would invalidate old evidence.

## Decision

Add a small pure immutable context and one known alternate recipe. Freeze physical
seed17 and camera eye(.3,.05,.6), with the old target and every other field
unchanged, before any new camera observation. Use a distinct context and recipe
ID, exact native type/value validation and explicit opt-in arguments through the
existing port, TRAIN teacher, frozen rollout and CLIs. Preserve the strict default
validator, default calls and CLI JSON schema. No new runner or dependency.

Bind visual TRAIN17 and metadata before creation. Own context records and teacher
outputs; recheck native bindings before/after controls, at extraction and cleanup.
Preserve paid native work and permanently stop corrupted interactions. Qualify
camera teacher data only with matching identity/role/recipe/spec and actual sparse
success. Refuse corrective queries because their current context schema differs.

## Alternatives

- Widen the old recipe validator to arbitrary camera fields: loses default
  evidence binding and admits unbounded pose selection.
- Select a camera pose after observing success: violates prospective scientific
  allocation.
- Add another loss/epoch variant or capture increment: does not add the missing
  environment capability.
- Create another simulator/stream orchestration layer: existing paths suffice.

## Consequences

The new path is independently reviewed and covered by zero-SGD fake-IO tests plus
affected boundary regressions. Actual rendered pixel effects and native camera
adaptation remain unexecuted and require a separate frozen small protocol. The
existing simple nearest control can support that precursor without claiming a
qualified neural R4.4 baseline, CPC advantage or completion of full R4.5. Default
range-normalization/old file-loader contracts stay strict. No other camera pose,
lighting, object or dynamics configuration is admitted by this change.

## Follow-up, 2026-10-19

A separately frozen one-use native camera acquisition/return case now completes.
The warm actor fails, private memory adaptation succeeds and original-camera
return is retained, with exact teacher-trajectory reconstruction. This does not
change the original scope or establish fullR4.5/generalization/CPC benefit.
[Evidence and costs](../../artifacts/runs/camera-context-return-20261019/RESULTS.md).
