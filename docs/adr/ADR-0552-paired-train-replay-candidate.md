# ADR-0552 — A separate paired TRAIN replay candidate

## Context

The paired CLM head has accepted native loss/gradient and temporal-stream evidence.
Catalog rehearsal and disposable probes have different objectives or authority.
They cannot silently stand in for paired TRAIN consolidation or renew the source
learner's completed allowance. A reusable candidate also needs actual original
receipt provenance and current temporal-policy consent.

## Decision

Copy warm weights from an exact successfully trained paired CLMHeadLearner into
one owned candidate with fresh native AdamW/OneCycle, cleared gradients and
separate RNG/work limits. Preserve the paired objective, encoder, dimensions and
native settings. No install/export/promotion or private probe conversion.

Use the existing ExperienceInbox and separate update/replay budget sessions.
Bind finite original observations/labels/receipts and explicit current-policy
semantics. Charge attempted replay rows before native work. One bounded source
snapshot protects original state without retaining a success-record chain.
Readouts recompute projections and carry no selection labels.

The default-off temporal CLI obtains two original v2 records through the existing
native observer, then runs a fixed six-update alternating replay only after the
ordinary source stream succeeds. Consent is granted at source creation.

Why this: it enables real runtime TRAIN replay with preserved source authority,
without a new runner, algorithm, dependency or ambiguous budget continuation.

## Alternatives

Continuing an exhausted source would renew its authority. Converting a probe
would cross role boundaries. Catalog training substitutes a different loss.
Reconstructing receipts from report JSON loses their original runtime linkage.
Fresh native optimizer state is explicit here; true optimizer continuation would
be a separate behavior with different budget and state requirements.

## Consequences

Replay adds updates/exposure and source capture/storage costs. Software parity
does not demonstrate useful native retention, recovery or a circadian advantage.
External library callers remain responsible for authentic provenance; structural
validation is not certification. Native comparison and optional internal gate/
renewal stay separate unfinished R5.3 requirements.
