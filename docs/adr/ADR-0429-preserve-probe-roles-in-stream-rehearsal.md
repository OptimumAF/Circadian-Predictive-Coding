# ADR0429: preserve probe roles while reusing the existing rehearsal stream

## Context

The accepted fixed TRAIN bank supports old-task replay, but the existing stream
and inbox admitted only TRAIN while fresh-task cohorts carry probe_train. Renaming
reserved data or inventing wake receipts would violate isolation/accounting.

## Decision

Add an explicit bounded probe_train training scope to the existing experience,
inbox, runtime and attribution stream. Keep original role metadata all the way
through events, cursor data and same-owner resume. Only a fixed historical TRAIN
bank can rehearse alongside probe wakes. Final/heldout roles remain ineligible.

FreshTaskProbe optionally constructs the helper after restoring its disposable
fork and drives this existing stream with zero label delay. Scores observe its
candidate; the original A-return loop then continues on that same candidate
without replay. Keep F/replay/return budgets, native receipts and reports separate.
Pin original F authority before the factory callback. Guard paid F reports before
return, retain failed attempts and refuse unreadable replay evidence explicitly.

Why this: one existing update/consolidation path avoids a second replay runner.
Default direct probes/reports remain unchanged. Optional runtime/inbox role
properties keep ordinary TRAIN vars/capture schemas and bytes unchanged; explicit
probe owners have additional fields and cannot enroll in managed admission.
Cursor/codec data alone never grant runtime training scope.

## Alternatives

Relabeling F to TRAIN loses cohort semantics. Direct additional updates in a new
replay loop would duplicate consolidation and receipts. Widening managed schemas
would change unrelated capture payloads/charges and imply unsupported authority.

## Consequences

All four existing native methods can rehearse old TRAIN data during isolated F
learning and continue into A return. Replay adds real computation and exposure;
payload counts exclude copies/reports/native state and paid-report checking cost.
Public CLI aggregation and exact matched comparative protocol remain open. Scoped
correctness does not establish acquisition, retention, recovery, circadian benefit
or release readiness. Initial source-review NO-GOs/failures and repairs are kept.
