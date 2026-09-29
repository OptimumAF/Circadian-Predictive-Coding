# ADR-0078: Bind the candidate manifest before checkpointed final release

## Context

ADR-0077 freezes outer-selected settings in an ordinary run. A v6 format-6
checkpoint binds one training configuration, so independent per-candidate
files would not prove that every declared setting and seed completed before
final-test access. A restart also needs the completed outer trials and
their exposure ledger to reproduce the same three choices.

## Decision

Use a distinct trusted local format-7 checkpoint with one atomic file.
Its run header stores the full ordered candidate IDs/configurations,
ordered seeds, and a digest that includes the fixed outer objective. The
cursor stores completed candidates as detached v6 unscored seed records
plus their full outer trial/exposure rows and digest. During an active
candidate, each format-6 wake/sleep/seed transaction is nested in the
format-7 file through an app store adapter. A completed-candidate save
advances the outer cursor; a final save stores the independently selected
choices and a digest of the manifest and all trials. None of these
transactions contains a final-test role value, hash, score, or released
source object.

Resume validates the run header and any nested v6 headers before source
access. It regenerates each completed candidate's arrived development
roles, validates the v6 model/replay/access cursor, and recomputes every
outer trial from its saved models and outer roles before another update.
The active candidate then uses v6's phase-local preflight and exact
transaction resume. At a frozen checkpoint, selection is recomputed from
all validated trials and must match the saved choice before any final
release. The public v7 API accepts the same checkpoint port for a fresh
run and `resume_from_checkpoint=True` for recovery; the ordinary API and
v1–v6 stores retain their identities.

## Alternatives

- Keep one v6 file per candidate. This does not atomically bind the full
  manifest, completed trial ledger, and final selection choice.
- Save scored final results in the checkpoint. This would move final data
  into the development transaction and complicate leakage preflight.
- Retrain completed candidates after interruption. This repeats work and
  can change observed guard and exposure records.

## Consequences

The two-candidate/two-seed run resumes from active A and B transactions,
the boundary between candidates, a later active candidate, and the frozen
choice in both model orders. Reports and field-wise trained states equal
ordinary and uninterrupted checkpointed controls, with exactly the
remaining model updates. Changed requested or stored manifest, seeds,
outer role content, recomputed trial/exposure rows, completed or active
event cursor, and forged frozen choice reject before an update or final
release. A terminal resume with changed final labels leaves trained state,
trials, and choice unchanged; the checkpoint bytes are unchanged while
the released final hash changes. The format remains a trusted local pickle
with a checksum, not an authenticated untrusted-data format. It retains
all candidate models and development trials, so it is intentionally
bounded to the declared small local candidate budget. Final scores remain
descriptive until the separate bounded strict-online confirmation gate.

ADR-0095 later advances this trusted checkpoint to format 8 to bind typed
sleep history in completed trials and the frozen selection. The original
score/work trial digest and selection objective remain unchanged.
