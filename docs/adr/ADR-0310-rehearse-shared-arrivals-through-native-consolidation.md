# ADR-0310: Rehearse shared arrivals through native consolidation

## Context

R2.3 shared recent/reservoir selection is qualified, including bounded rotation,
but those schedules do not train a learner. The coherent four-method attribution
stream already owns delayed training and deferred final scoring. Its runtime
supports transformations of detached candidate state. The separate legacy CPC
matched runner requires equality with a single-policy native buffer.

## Decision

Add an opt-in single-use rehearsal helper to the existing coherent stream.
Observe only the batch whose labels have actually produced a committed wake
update. At fixed boundaries select shared rotating rows, train a detached native
fork and commit through the existing consolidation hook before checkpoint capture.
Do not change native objectives, replay selection according to scores, or expose
heldout data to the helper. Preserve the no-replay schema and runtime gate.

Use separately planned wake and finite replay ToyBudgetSessions. Why this:
ActorShadowRuntime refuses replay limits it does not enforce; explicit composition
keeps that authority boundary unchanged. Check both sessions around replay, and
preflight the whole event. Charge before native admission because a failed call
can have performed work. Report charged/attempted/returned/committed separately,
including postcommit failures and all not-run events. No budget renewal/refund.

## Alternatives

- Modify the legacy native CPC buffer or waive equality: broadens scope and
  risks invalidating a separate matched-memory contract.
- Implement another runner or generic reservation framework: duplicates existing
  orchestration without a demonstrated requirement.
- Charge only successful commits: hides attempted work after native or restore
  faults and can enable repeated uncharged attempts.

## Consequences

Four matched native learners can actually rehearse identical arrived content.
Native CPC clocks/gates advance on replay calls; wake and replay counts remain
separate in the report. The legacy counter named completed is explicitly an
admitted exposure charge in this mode. No durable helper resume, managed-retention
integration, general context-survival guarantee or scientific benefit follows.
R2.3 remains open for its full long-stream criteria. Prior failed and negative
results remain unchanged. See docs/stream-rehearsal.md and the frozen scope in
artifacts/runs/r23-native-rehearsal-20261016/SCOPE.md.

## Reuse with learnability diagnostics

An opt-in CLI mode composes the existing reserved sixteen-step checkpoint
observer with post-rehearsal state copies. A small public facade on the existing
learnability adapter exposes its fixed data and observer without importing its
private helpers elsewhere. Separate templates/disposable probe forks and the
same complete six-point C curves preserve ADR-0307's evaluation isolation.
Why this: assess fresh-task ability beside retention/recovery using implemented
tools. No new objective, horizon selection, orchestration framework or benchmark
training on probe labels is introduced. Its separate376-update protocol includes
all72wake/48replay/256probe calls and keeps the default120-update case intact.

## Matched recent FIFO control

An explicit RecentStreamReplayPolicy contains only the period, event rows and
retention budget. The same helper uses the existing recent_fifo buffer and
select_recent chronology. Why separate records: silently ignoring a reservoir
allocation or seed would misdescribe the control and change old protocol schemas.
The old rotating policy/report remain unchanged; the new report identifies its
control and has no partition/cursor state. Shared budgeting, consolidation and
failure paths prevent the baseline from receiving different update semantics.

Compare equal declared array-memory caps and per-method native exposure under
a separately frozen protocol. Actual occupancy, metadata and latency differ.
Retention composition and row ordering change together; observed differences
do not isolate a reservoir effect. No control tuning, equivalence or universal
winner claim follows from one fixed local stream.

## Longer interference stream, 2026-10-16

The existing CLI selects a fixed64-update adapter using unchanged retention and
replay rules. Existing event records already contain bounded retained IDs; a tiny
read-only helper port exposes the detached endpoint after final non-event wake64.
Old reports/method bodies remain unchanged. A report-only train manifest enables
arrived-prefix verification and separate retention/actual-exposure measurements,
without giving context or future labels to the selector. Original R2.3 bounded
implementation criteria are accepted; exposure gaps and learning losses remain.
Evidence: r23-long-stream-20261016/RESULTS.md. No universal retention or replay
benefit is claimed; original negative runs remain unchanged.
