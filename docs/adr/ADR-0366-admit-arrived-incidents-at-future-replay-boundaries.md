# ADR-0366: Admit arrived incidents at future replay boundaries

## Context

Existing shift observers describe incidents without affecting adaptation. Rehearsal
runs before checkpoint observation in the stream. Alarm levels can be continuation
or carryover, and existing negative studies do not establish detector utility.

## Decision

Use one opt-in scalar gate on the existing helper. Validate exact sequential
incident records and completed wake receipts. BoundaryN consumes only an active
pending onset observed throughN-1; clear expires it and continuation cannot
retrigger. Consume before native work, preserve paid failure, and log zero-work
skips. Existing pause state owns the complete gate; original budgets/clocks stay
binding. Refuse missing observations before another wake. Sensor CLI declares
feature channel and upper costs explicitly; final data never enters admission.

## Alternatives

Reordering callbacks changes old scoring semantics. Acting on the current
unobserved incident fabricates information. Acting on every alarm manufactures
new triggers from continuation; renewal would bypass the permanent allowance.
A separate runner would duplicate working stream/retention/receipt mechanisms.

## Consequences

Opt-in feedback becomes runnable without changing periodic defaults or native
objectives. One successful small case qualifies chronology/state/costs, with
external launcher/trainer memory distinction documented. It establishes no
statistical FPR, causal recovery, compute-matched benefit or CPC advantage.
Stationary/recurring controls on the existing shift pilot are a separate next
integration; freeze a new protocol before further native work.
