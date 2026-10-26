# ADR-0369: Bind bounded inference to the existing recurring stream

## Context

The kernel and coupled learner already support an owned per-call inference
policy. The shared stream factory binds only fixed4steps. A separate matched-time
runner already exists, and its10ms/30ms qualifications failed. Previous handoff
overstated this factory gap as a missing comparison prerequisite.

## Decision

Add one optional factory keyword and one small adapter selected by the existing
bounded-inference CLI's --stream flag. Reuse the original delayed A/B/A runner,
data, fixed engineering policy and resource sessions. Preserve None/default calls
and BP. Refuse invalid settings before construction. Expose unknown generic
receipt inference costs as null; instrument them only in the frozen native case.
Keep source learners unchanged, final scores after wakes, and partial paid work.

## Alternatives

Rebuilding a timing runner duplicates existing functionality. Adding per-update
inference telemetry throughout generic receipts changes unrelated public formats.
Expanding repeated-update capture does not address this integration gap.

## Consequences

The bounded policy is runnable in a recurring adaptation stream without changing
the numerical objective or default controls. A small fixed native demonstration
can establish integration correctness. R1.5b matching and scientific efficacy
remain open; no failed protocol is replaced and no final score selects settings.
