# ADR-0194 — Measure installed continuous guard coverage before enforcement

Status: accepted for measured capability; complete runtime guard/admission remains open.

## Context

Five prior admission counterexamples require continuous/source/native enforcement.
Monitoring/tracing coverage is an installed capability question; callback/internal
execution and caught guard errors cannot be assumed safe from an API name or proposal.

## Decision

Preserve production and measure the installed process under real V2/native ownership.
Save full actual observations/raw/native public values/disassembly and all events from
ordinary, native and secondary-callback intervals. Measure actual trace loss separately,
retain existing audit denials, verify actual callback cleanup and independently read
every artifact. Store primitive callback arguments to avoid changing object lifetime.
Use public borrowed pointers with correct ownership, never guessed private layouts.
Keep all failed probes/source versions/budgets and current source pins.

## Alternatives

- Assuming cross-tool callback visibility from the proposal contradicts the local result.
- Promoting CALL/C_RETURN to native instruction/source proof omits internal execution.
- Assuming a caught trace exception leaves enforcement active contradicts actual tracing.
- Inferring empty callbacks from tool names/masks misses an unmeasured cleanup criterion.

## Consequences

Seven controls pass, including two measured guard weaknesses. Primary monitoring
misses the second tool's callback execution; caught trace denial deactivates tracing.
The complete guard must reject unsupported paths before admission and maintain denial
after failures. All five prior gaps and native/build/private/source/omitted/nested/
arrival/ledger/prior/b3 gates remain mandatory. No scientific feature or policy changes.
