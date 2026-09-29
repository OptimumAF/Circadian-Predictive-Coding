# ADR-0095: Bind v7 sleep history independently of selection scores

## Context

ADR-0094 exposed every candidate's typed sleep attempts in the public v7
result. The original `ArrivedOuterTrial` and frozen choice still lacked that
history, even though the v6 unscored checkpoint records retained it. Adding
timing-rich events directly to the original trial digest would change the
identity used for score selection and make equivalent runs differ by elapsed
time. Old format-7 checkpoints did not assert trial/freeze sleep provenance.

## Decision

Attach typed events and a role-ledger-bound sleep digest to each circadian
trial; baseline trials explicitly carry empty sleep fields. Compute the
historical trial digest from the original score/work fields only. Separately
digest the ordered candidate/seed sleep digests in each completed candidate
and in the frozen selection. Check both the typed trial events and all
independent digests against rehydrated v6 records before updates or final
release. Format 8 rejects older v7 checkpoint files lacking this binding;
the public v7 protocol ID, score objective, and choice digest stay fixed.

Ordinary v7 may opt into v6's bounded retry of a typed, restored sleep
error. Its default raises. Checkpointed v7 always raises, stores the failed
attempt in its nested v6 cursor, and retries only on explicit resume. A
completed guard decision retains two guard passes in the existing trial
work counter; a failed attempt's partial scored examples remain visible in
typed telemetry without silently changing that historical counter.

## Alternatives

- Include events in the original trial/choice digest. This would change
  selection identity when only elapsed time changes.
- Reuse only the v6 unscored digest without putting history in trials and
  the freeze. That would leave the public selection artifacts incomplete.
- Retry checkpointed failures inside one call. This would hide the durable
  failed attempt and bypass the explicit resume boundary.

## Consequences

Two candidates and two seeds in both model orders now expose phase A/B
accepted, rejected, skipped, and failed attempts. The interrupted path
resumes the same epoch and matches ordinary non-timing history, exact
candidate model fields, baseline scores, and final metrics. Tampered active,
completed-trial, candidate, and frozen provenance reject before resumed
updates or final release, including a forged reason with its digest
recomputed. Strict smoke JSON includes trial histories and freeze digest.
The old trial/choice digest calculation remains unchanged. Existing
format-7 checkpoints must be regenerated under format 8.
