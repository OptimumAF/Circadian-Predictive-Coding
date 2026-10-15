# ADR-0276: Pause an admitted native worker before checkpoint delivery

## Context

R3.7e supplies an explicit healthy native completion transition. Existing fake
failure controls do not demonstrate a real native process dying after admitted
computation but before returning a complete checkpoint to the surviving parent.

## Decision

Add an explicit engineering CLI flag to the existing fixed one-update worker.
After training and bounded encoding it sends computed-only telemetry and waits
for exact stop. The parent demo kills only its owned registered child inside
the preparation callback, so the coordinator's existing live-process check
refuses before checkpoint validation. Persist original uncertainty and all spent
charges; prove actor A remains available and unchanged with actual native files,
snapshots, predictions and retained process observations.

## Alternatives

- A sleep and timed kill is nondeterministic about whether computation occurred.
- Killing after the operation returns misses the pending update guard.
- Trusting telemetry or a hash as completion would weaken original admission.

## Consequences

Default healthy protocol is unchanged. Opt-in pause never delivers B, even on
stop; strict stop/error behavior comes from the existing bounded request helper.
No model, core authority, completion or journal semantics change. Native-only
engineering interruption is demonstrated; uncertain retry, full managed-bundle
and coordinator-loss recovery are not provided. Terminal authority may retain its
last valid coordinator observation; separate retained ended evidence is explicit.
