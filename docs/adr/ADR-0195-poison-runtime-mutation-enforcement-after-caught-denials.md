# ADR-0195 — Poison runtime mutation enforcement after caught denials

Status: implemented mechanism; complete runtime admission remains unavailable.

## Context

Installed primary monitoring cannot observe another tool's callbacks, while a trace
error removes tracing. Boundary captures also miss transient creation/default changes.
The complete guard additionally needs source/native/private/lifetime correspondence.

## Decision

Prepare all actual GC functions/recursive code and immutable instruction maps. Use
global before-call/instruction callbacks, reject foreign tools/tracing/profile/threads
at entry, block mutation and unsupported native calls before execution, retain poison
and actual tool state after caught denial. Restrict release to the prepared owning
with boundary using actual3.14 exception regions; reacquire/read each callback slot.
Save complete records/raw/public fields/disassembly/every event and independently read
all content. Preserve all failures, source differences and budgets. No admission grant.

## Alternatives

- Trace-only denial loses protection after a caught trace error.
- Allowing foreign tools admits callback execution invisible to the primary tool.
- Checking a target subset or only boundary hashes misses created/restored bindings.
- Treating explicit C-call events as implicit/native/source proof omits required coverage.
- Resetting poison or accepting early manual release lets the same consumer continue.

## Consequences

Actual mutation controls and cleanup pass with complete historical V2/native records.
The module is optional and requires measured CPython3.14; old working observers remain
unchanged. A narrow verified stdlib typing omission is documented with exact whole
executable parity and differing physical source retained. Full integration/source/
native/private/opaque/membership/arrival/ledger/prior/b3 requirements stay mandatory.
Original120s failure gates remain unchecked; only measured current correctness successor
can complete. Scientific models/seeds/metrics/baselines and dispatch remain untouched.
