# ADR-0093: Preserve failed arrived sleep attempts at a retryable epoch

## Context

The arrived v6 runner restores the circadian model and raises when inner-guard
scoring or core sleep fails. Its active checkpoint stays at `before_sleep`,
so a later resume retries that epoch without repeating wake updates. The
initial typed history allowed one event per epoch and could not retain the
failed attempt. A post-guard failure can occur after the core returns a real
split/replay proposal, while a core exception has no returned proposal.

## Decision

Snapshot before the first guard score. On pre-guard, core, or post-guard
failure, restore that snapshot, emit a typed `error` attempt with its runner
trigger, role hash, known guard scores, reason, and elapsed attempt time,
then raise the original exception. Missing post accuracy and delta remain
`null`; no score is invented. Count only completed guard score passes. If
core sleep returned, keep its entire proposal and measured core duration
while clearing applied work. If core raised, record zero proposal and zero
core seconds as an explicit unmeasured-core sentinel; attempt seconds still
include elapsed core work.

Checkpointed v6 saves the error event with the same `before_sleep` model
cursor before re-raising. A later call with `resume_from_checkpoint=True`
retries the same epoch. Ordinary v6 retains fail-loud behavior by default;
the opt-in `sleep_error_retries` parameter allows a bounded number of local
retries when a typed error was emitted. Checkpointed runs use explicit resume
and reject that parameter. The successful per-seed report keeps every failed
attempt followed by its final decision.

The ordered event tuple is the attempt cursor: each completed epoch has zero
or more `error` attempts followed by exactly one final decision, while a
pending `before_sleep` epoch can have only errors. Validation binds every
attempt to the same phase-local schedule and guard role, checks partial
score/reason consistency, and rejects a missing or extra final decision
before model restoration or final scoring. This extends the existing
version-one history without changing its fields, existing v6 checkpoint
format, role-event digest, v7 trial digest, or evaluation metric.

## Alternatives

- Retry every error silently: hides failures and changes default execution.
- Advance the checkpoint to `after_sleep` on error: skips an uncommitted
  decision and would change the training trajectory.
- Fill absent guard scores or core proposals with fabricated values:
  misrepresents measured work and a returned core result.
- Add an attempt-number field: redundant with the validated event order and
  would invalidate existing complete version-one histories.

## Consequences and evidence

The first direct error fixture failed because no event was emitted; the
checkpoint fixture then failed because the validator rejected an error on
the pending epoch. Focused cases cover core, pre-guard, and post-guard
exceptions/nonfinite scores; retained post-core split/replay proposals;
phase A/B and both model orders; ordinary bounded retry; one or two
checkpoint resumes at the same epoch; terminal reload; and tamper rejection
before model restoration. The existing role ledger, baseline metrics,
final-test release, and model-order behavior remain under regression tests.
