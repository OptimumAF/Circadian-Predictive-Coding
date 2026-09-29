# ADR-0076: Resume v6 after each arrived wake and sleep transaction

## Context

ADR-0075 stores only completed v6 seeds. An interruption during Phase A
or B otherwise restarts the active seed, repeating updates and potentially
changing guard decisions. The v6 result also records release and guard
events, so a model-only cursor cannot reproduce its report.

## Decision

Extend format 6's active fields with the arrived development role IDs and
hashes, detached baseline and circadian model state, Phase A frozen state
when available, the observed access/guard/task ledger and its digest, and a
phase-local transaction cursor. Save after each nonterminal model update,
before sleep, after sleep, and at Phase B arrival. A model update is not
repeated after a durable wake transaction; a sleep decision is not repeated
after its durable after-sleep transaction.

On resume, validate the run header before source access and all earlier
unscored seeds before an active update. Regenerate only the arrived phase
roles. Check their content identity, the event prefix and guard role,
baseline step counts, circadian wake position, frozen Phase A state, and
replay membership in arrived train rows. Restore only after these checks.
Keep final source fields and all scores outside every training checkpoint;
release them after all declared seeds finish. The file remains a trusted
local pickle with checksum, not an authenticated untrusted format.

## Alternatives

- Restart an interrupted seed. That repeats work and can change the
  observed guard and event ledger.
- Save only models and infer the event cursor. That cannot distinguish a
  completed guard decision from a pending before-sleep transaction.
- Reuse the v5 active checkpoint. Its one development split and missing
  four-role audit do not represent the v6 protocol.

## Consequences

The fixed two-seed run resumes from A/B wake, before-sleep, after-sleep,
A/B arrival, later-seed, and terminal boundaries in both model orders.
Reports, saved model fields, replay rows, and guard/access records match an
uninterrupted checkpointed run. Raising source sentinels keep Phase B
unavailable until all A models finish and final input/labels unavailable
until all seeds train. Forged active A/B role content, future or nontraining
replay, model progress, position, or event ledger rejects before an update.
The extra per-model file writes are a correctness boundary for tiny local
experiments, not a performance claim. Outer setting selection and the
parent strict-online comparison remain open.
