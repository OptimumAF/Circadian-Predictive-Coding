# ADR-0334: Reuse reserved predictions for bounded decision diagnostics

## Context

Reserved fresh-task probes expose aggregate scores. Earlier E accuracy matched
a majority rule; the matched warm-start control reduced CE while reducing
accuracy. Neither result demonstrates useful adaptation or circadian benefit.

## Decision

Add an exact boolean opt-in to the existing probe and a pure bounded core helper.
Reuse returned reserved predictions for class support, confusion, per-class loss
and row decisions. Preserve original scores, binary/categorical semantics,
checkpoints, native calls, callback order, original budgets and complete-point
failure behavior. Absent classes report undefined metrics. Return detached Python
primitives and explicit row/episode provenance.

## Alternatives

Additional learner forwards increase native cost without new information.
A separate runner duplicates existing isolation. Changing aggregate metrics or
choosing easier probes after observing final results would change the question.

## Consequences

Defaults remain unchanged. Opt-in reports require bounded additional CPU and
memory, excluded from the existing input-array byte count but included in total
invocation time and sampled process RSS. Engineering equivalence does not prove
learnability, old-task retention or return recovery. Any new comparative protocol
must still be frozen, reviewed and budgeted before execution.

## Existing runner integration

The key/door helper and learnability runner expose the same exact-bool opt-in.
The existing CLI accepts it only in the learnability scenario. Defaults retain
old callback keyword shapes and protocol fields. A requested report that omits
its enabled flag or reserved context maps fails with charges/reports retained.
This structural admission check delegates numeric semantics to the accepted core;
it creates no second diagnostic runner or independent scoring implementation.
