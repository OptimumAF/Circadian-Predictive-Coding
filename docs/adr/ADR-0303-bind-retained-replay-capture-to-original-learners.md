# ADR-0303: Bind retained replay capture to original learner identity

## Context

Complete capture holds original owner, lifecycle, holder and resource leases.
Calling public copied-row origins() would reenter them. Existing constructor-copy
witnesses already retain weak original model/row/payload/controller references,
but do not identify the learner wrapper that received the model. Looking up the
current wrapper later would also accept a replacement around the same model.

## Decision

Extend the existing synchronous model-copy observation with an optional borrowed
recipient. CPC passes its actual constructor self at the copy boundary; no native
state fields are added. The observer retains only a weak recipient reference and
clears borrowed callback roots on exit or failure. The partially initialized
recipient is not inspected then; completed learner identity is checked on use.

Add an optional lease-aware app reader for original ManagedReplayCopies during
complete capture. Reuse original witness and consent/history checks, nonblocking
gates and the existing single ledger capture admission. Revalidate every global
inventory round before/after copying. Check every advertised holder occurrence:
an original checkpoint witness cannot authorize an actor, candidate, promotion or
snapshot role containing the same model. Existing inventory ports identify native
roots; the app does not inspect NumPy implementation fields.

## Alternatives

- Public origins(): fails because original leases are already held.
- Infer the recipient after the operation unwinds: a callback could have replaced
  the mutable model-list entry, so it would not establish constructor identity.
- Match payload contents or keys: does not establish original copy authority.
- Add a new observation framework: unnecessary; the existing copy window already
  supplies the synchronous bounded source/target/memo boundary.

## Consequences

Existing callers may omit recipient and keep previous public witness behavior.
New capture requires qualified original recipient identity and refuses absent,
expired or replaced wrappers. All reads use existing original allowances; late
failure retains charges. No restore permission, portable authority or new origin
is issued. Copied snapshots, successful publication/restoration, other holder
families and compound cases remain open under original R3.5b2e5b3d2 acceptance.
