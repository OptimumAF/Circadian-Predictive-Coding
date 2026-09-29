# ADR-0066: Version continual checkpoints at the Phase B arrival boundary

## Context

The default offline continual checkpoint computes one digest from both
phases before Phase A training. That is useful for v1 recovery but makes
future Phase B data available to checkpoint orchestration too early for a
phase-arrival study. P1.3a's canary demonstrated the access difference.

## Decision

Add opt-in protocol `continual_phase_arrival_v2` and checkpoint format 2.
On a fresh seed or Phase A resume, construct only Phase A roles and bind
its train/validation development roles with a namespaced Phase A digest.
The Phase A checkpoint carries no Phase B split hash. After all Phase A
models finish, construct Phase B and save the first Phase B checkpoint
with the existing combined development-role digest and the four
train/validation role hashes. A Phase B resume rebuilds both arrived
phases before restoring model state. Final-test hashes enter the report
and completed-seed checkpoint only after both training phases finish.
Completed earlier seeds may rebuild both phases and final-test hashes for
audit. The checkpoint training context contains only the current phase's
training role and checkpoint identity, so final-test objects are not
available through the training helper's input.

Keep v1 protocol, checkpoint format, default behavior, and recovery path
unchanged. Both formats reject wrong config, role digest, format, model
order, and progress before a training update. File IO stays in `infra`.

## Alternatives

- Change the v1 digest in place. Old trusted checkpoints would become
  ambiguous or fail without an explicit format transition.
- Hash Phase B seed/config without its realized data. A resumed run could
  silently use different Phase B examples.
- Store future Phase B labels in a sealed wrapper during Phase A. The
  checkpoint orchestration would still possess future data.

## Consequences

The new path can resume after Phase A or Phase B interruptions and across
completed seeds without future Phase B source access during Phase A.
Source-construction, tamper, state, and final-test timing tests cover both
model orders where applicable. The generator still materializes Phase A
test data when it builds the source, but v2 defers test-role validation and
hashing until both training phases end. Final-test values are not passed to
training or used for decisions, and scoring remains after Phase B.

This remains an offline training schedule with the configured full A+B
horizon. It has no declared retained-memory budget, guard provenance, or
label-arrival ledger, so it is not a full strict-online result.
