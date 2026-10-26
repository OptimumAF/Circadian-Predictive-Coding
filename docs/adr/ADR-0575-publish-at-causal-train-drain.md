# ADR-0575 — Publish through existing guards at the causal TRAIN drain

## Context

The live-serving trace learns a useful candidate by tick10 but keeps the old actor
through tick23. Legacy requests still fail after learning. Moving publication
earlier must preserve guard authority, receipts/budgets and learner-version binding.

## Decision

Factor original guards/prepare/commit without changing policy or data. An opt-in
mode attempts that callback once after all original labels/updates are consumed,
pending predictions are empty and the joined service worker leaves the gate idle.
Validate exact generation/version metadata and original receipt; preserve failed/
unknown publication prefixes. After verified swap, retain then invalidate old
scalar feedback rather than rebinding it. Continue the same serving window with
the original exhausted training envelope and complete original rollback.

## Alternatives

Publishing after a favorable live result leaks outcome selection. Repeated tiny
guard checks invite tuning and additional unrecorded cost. Retaining old feedback
under a new actor version violates causal binding. Replacing the runtime or
resetting budgets would create fresh authority. A new runner duplicates working
ports; none is needed.

## Consequences

Later requests can use an already learned candidate while earlier failures remain
visible. Genuine guard refusal still works, attempts do not repeat, and future
training beyond the consumed envelope is not granted. The known-template trace
demonstrates deployment functionality, not long-duration/controller/CPC benefit.
Noise/poisoning/generalization/release criteria remain open; review is self-review.
