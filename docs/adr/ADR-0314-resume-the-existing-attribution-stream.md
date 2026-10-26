# ADR-0314: Resume the existing attribution stream

## Context

Owned native Torch head and buffer component continuation already passed, but
the ordinary attribution function held delayed-label position, runtime and
checkpoint states in local variables. Component copies could not demonstrate
whole labeled-stream continuation or protect its original spent authority.

## Decision

Expose the existing loop as `AttributionStream`; retain the original one-call
facade. Issue one owner-local pause token with complete detached payload state.
Resume validates unchanged live state/authority, stages complete independent
candidate/inbox/replay data, verifies the restored candidate and cursor, then
publishes under the original runtime lease. Use existing inbox materialization;
retain original clocks, actor, budgets and receipts. Retire the old inbox and
clear the consumed token. Recheck authority after opaque scoring too.

## Alternatives

- Live pause without reconstruction does not establish snapshot/resume.
- Model-only resume drops pending labels, replay rotation or cumulative work.
- Fresh runtime/helper constructors renew admission and discard live history.
- A new disk/capture framework adds unrelated authority and persistence work.

## Consequences

One loop serves default, rehearsal and renewal execution. Supported labeled
streams have actual local continuation; no model objective or replay rule changes.
Staging and comparisons add bounded copying/work. Single-owner tokens do not
provide durable/process recovery, managed authority or renewal continuation.
Trajectory replay remains outside the binary fixed-feature port. Full R2.4 and
repository/scientific release claims are not inferred from this increment.
