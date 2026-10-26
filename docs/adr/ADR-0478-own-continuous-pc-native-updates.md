# ADR-0478: Own continuous PC updates through the existing native port

## Context

The continuous PC energy and bounded inference are accepted, but do not change
parameters or own a learner. The robot BP adapter already supplies matched
initialization, frozen normalization, one-hidden geometry and full state
validation. Existing actor/shadow and delayed-label runners use NativeLearner.
An inference exception can occur before its normal diagnostic return, so an
update boundary needs observed partial work rather than a guessed cost cap.

## Decision

Compose the current BC policy/initializer/snapshot validation with accepted
continuous energy and the existing private inference loop. Keep vector-bias
native snapshots; convert only at the coupled math boundary. Own momentum and
publish one fully validated candidate after inference, parameter update and a
strict target-free forward. There is no task-improvement admission criterion.

Add one private per-call progress record to the shared inference loop, updating
actual counters before operations/after returns. The adapter counts transforms,
initialization, proposal/forward/publication work and wraps numeric failures in
the established TrainingUpdateFailed. Raw admission failures precede the math;
finite-input normalization overflow remains inside counted failed work.
Public binary/continuous inference APIs and guard equations are preserved.

Expose generic train_batch plus a richer work-returning method. Do not retain
last-success history. Restore/fork are detached and RNG-free. Match train_episode
roles, while raw arrays retain the existing caller-owned release responsibility.

## Alternatives

- A second stream runner: duplicates already accepted runtime boundaries.
- A separately reimplemented optimizer or initializer: risks unmatched BP controls.
- Treat failed inference as zero or a maximum guessed cost: hides actual work.
- Require each parameter update to improve task loss: changes the algorithm and
  selects favorable outcomes, undermining the planned matched comparison.

## Consequences

The one-hidden adapter matches the current robot BP architecture, and can run
through the existing actor/shadow boundary. Multiple-depth core primitives remain
available without claiming that this native state format covers them. Active and
neutral CPC, repeated-task retention/recovery and scientific benefit are still
separate unfinished requirements. Accounting adds small runtime overhead;
historical time comparisons are not requalified. No dependency, external data,
native experiment or final-evaluation access is added.

## Follow-up: optional continuous CPC modes on the same state path

Add a last optional circadian policy field and wake snapshot field to the existing
learner. Pure core helpers reuse the existing coupled CPC configuration/records
and chemistry/gate equations. One shared optimizer gates current partials before
adding unchanged old momentum; one complete candidate publication includes wake
state. Neutral min-plasticity1 performs actual chemistry and unit products, with
exact common PC state/diagnostic identity and separately retained CPC-only costs.
This avoids a second optimizer or inheritance-based partial publication path.
The [mode guide](../continuous-cpc.md) documents trusted-local schema scope and
validation/copy cost exclusions. Existing stream selection and scientific benefit
remain open, requiring a frozen comparison protocol before execution.
