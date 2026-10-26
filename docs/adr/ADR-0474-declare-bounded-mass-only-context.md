# ADR-0474: Declare one bounded mass-only TRAIN context

## Context

R4.4's working small state/visual policy baseline is accepted; full R4.5 still
needs object/dynamics drift and recurrence beyond the existing camera fixture.
The pinned official CPU Actor exposes set_mass and mass; its mass setter does
not expose a whole density/inertia update contract. Defaults and held-out roles
must keep their existing meanings.

## Decision

Expose one immutable declared TRAIN17 context through the existing episode port.
After its explicit seeded reset, read one positive finite float32 mass <=0.5kg,
set exactly twice that mass once, and require exact native float32 readback.
Before and after external mass access, retain actor/context identity. Verify
mass before controls and after returned output, and before episode extraction;
changed evidence permanently stops interaction. Keep constant-size original
readback and attempt/validated-return counters. Preserve original controls,
observations, clocks and default factory calls.

The semantics field describes API intent: only the mass setter is called; no
inertia/center-of-mass setter is called. Their actual native effects are
unmeasured, so this is neither uniform density scaling nor observed inertia
invariance. Do not combine mass and camera fixtures implicitly.

## Alternatives

A custom task subclass or generic physics randomization framework adds unnecessary
surface area. Silent mutable SDK settings lose role/context/readback binding.
Density scaling would need an explicit inertia/COM contract and new evidence.

## Consequences

The port can collect a distinctly tagged mass-intervention episode without
training or final-evaluation authority. Fake IO verifies binding, ownership,
readback, lifecycle and costs only. Actual mass/dynamics/learning/recurrence need
a separately frozen small native protocol. Teacher/CLI/stream integration is
unfinished; no simulator, scientific comparison or release claim follows here.
No dependency or environment-variable change.

## Follow-up: existing teacher and frozen-rollout interfaces

The optional declaration/readback/work now propagates through existing teacher,
frozen-rollout and CLI boundaries. Default CLI fields remain unchanged; raw
TaskTeacherResult dataclass gains optional None fields. Strict qualification and
original readback retention precede usable training data. A trusted pure port
metadata check follows foreign getters and owner snapshots, closing a demonstrated
callback-order hole without another native read. Scoped synthetic checks establish
integration correctness only; actual acquisition/return and full R4.5 remain open.
