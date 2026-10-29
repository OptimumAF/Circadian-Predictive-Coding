# ADR-0641: End the original catalog window before using a published policy

Status: accepted for scoped software; native publication unqualified.
Task: R7.7-CAT.P / issue252. Single-agent self-review, not independent acceptance.

## Context

The native251case learned the new task with6updates36rows while retaining both
old tasks, but the original served actor still failed legacy. The frozen chooser
spent all9decisions; serving contention left2old-version TRAIN packets pending.
The existing drain publisher requires every label consumed. Extending that
horizon or pretending the packets were consumed would change the protocol.

## Decision

Add an explicit `publish_after_window=True` only to the existing frozen catalog
live API. A dedicated one-phase helper freezes the bounded paid prefix, pauses
the original gate with an observed counter-preserving transition, and invokes
the original `_publish_candidate`/inner guards/controller prepare+commit.

After the attempt, stop the old chooser and mark the original runtime retired
under its existing lease before retiring its inbox ledger. Keep all experiences,
labels, applied receipts, native state, source versions, budgets and charges;
pending packets remain unapplied. Create no successor owner, new budgets,
rebound labels or additional TRAIN decisions. On ledger-retirement failure the
runtime is already closed, the gate stays paused and the report is incomplete;
do not repeat a known or uncertain commit.

Bind known accepted/rejected outcomes to the original controller receipt identity,
its preserved values and full actor snapshot. Post-window typed readouts use that
exact publication plus the unchanged source, retired ledger and original resource
authorities. They are serving observations, never controller feedback or final/
probe selection. Distinguish known publication from failed cleanup and unknown
publication. Preserve the original training-window sharing snapshot and record
the final snapshot separately.

The new report format is refused by the old private utility/cost consumers:
guard work and post-window service cannot be silently omitted, including when a
guard rejects promotion and the actor remains unchanged. Native experiment
observers must account for all new guard/prediction/copy/IO work.

## Alternatives

- Force remaining updates or extend the decision horizon: violates frozen
  caps/chronology and changes the studied policy.
- Rebind pending labels to the new actor or renew budgets: changes source
  authority and can create unpaid learning.
- Reconstruct a live candidate owner from the saved251report: its model/readout
  evidence is not a capability to manufacture the original runtime ledger.
- Keep the candidate private indefinitely: cannot demonstrate actual use of the
  learned task policy during serving.

## Consequences

Default private, original drain, reference, CLI and followup behavior remains.
This opt-in is terminal for its training owner; continuous learning after
publication requires its own explicit lifetime-authority design and evidence.
Known guard rejection and terminal task failures are valid outcomes. Lost,
forged, changed or replaced evidence stops readout without rollback/retry or
claiming the whole case completed. No generic runner, policy change, core
promotion change or new dependency is introduced.

[Scoped checks and limitations](../reports/catalog-window-publication-20261027.md)
cover actual promotion machinery with literal learners/public ports. A distinct
native publication qualification is still required before task-usefulness or
broader deployment claims. The251/248scientific allowances remain spent.
