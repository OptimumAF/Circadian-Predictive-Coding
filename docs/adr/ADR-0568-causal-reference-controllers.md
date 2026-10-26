# ADR-0568 — Causal fixed references through existing admission

## Context

R7.1-A supplies finite wait/wake IDs but the task update schedule is scripted.
R7.2 requires no-learning, periodic, drift-triggered and budget-aware controls with
fair opportunities/overhead. Existing arrival, shared-admission, guard and native
task ports already supply the necessary execution boundaries. A no-learning
observation schedule cannot satisfy AttributionStream.finish's required full
training completion; that contract must remain intact.

## Decision

Add pure scalar reference policies and causal TRAIN feedback records in core.
An optional branch in the existing BP serving script reuses the original arrival
helper and actor/shared runtime over a fixed observation horizon. All references
receive the same schedule, original-actor feedback and hard budget. Pending
unapplied training is explicit; no forced updates. Native targets/task/guard/
final/probe outcomes are unavailable to selection. The original default path and
full-stream finish remain unchanged.

Separate candidate endpoint tasks from deployed actor outcomes. Freeze four
simple policy rules before observation; record choice/monitor/admission overhead,
all attempted/completed work and failed prefixes. Retain the original guarded
publication/rollback mechanics. Reuse the qualified source and saved fixed-eight
result without another warmup/baseline fit.

## Alternatives

Forcing all pending updates would erase no-learning/budget-aware behavior.
Weakening full-stream completion would alter existing correctness/recovery
semantics. Reading candidate task scores before selection leaks endpoint evidence
into dose tuning. A new orchestration framework duplicates working arrival and
admission ports. Skipping common feedback computation only for some policies
would change the monitored opportunities/cost definition in this protocol.

## Consequences

Four references execute through actual native task serving with a finite ledger.
Their updates differ, so their useful endpoint contrast is an exposure tradeoff.
Identical four-update states and drift matching the saved fixed-eight state
establish no isolated drift benefit. Monitoring remains paid even for no-learning.
The stable feedback actor cannot reflect shadow improvement during the schedule;
this may waste updates. Broader objective/chronology, generalization, advanced
actions, continuous rebase and learned-controller/release claims remain open.
