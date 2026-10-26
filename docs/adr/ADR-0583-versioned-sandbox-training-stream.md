# ADR-0583: Bind policy-change TRAIN labels to existing sandbox outcomes

## Context

The temporal core already models v1→v2 and old-domain return under v2. The actual
ticket sandbox also supports both versions, but its teacher and controller packet
builders previously used v2 alone. Changing a packet label would be experimental
noise unless the changed target matches actual task truth.

## Decision

Add an explicit validated policy_version to the existing TRAIN teacher; v2 is
the byte-compatible default. Generate all traces through existing requests and
dispatch; require applied effects and actual workflow_success before returning.
Add one fixed recipe to the existing stream builder and opt it into the existing
recurring runner. Use original learner/AttributionStream/clock/budget/full-state
readouts and sandbox episodes. No new simulator, controller or runner framework.

Three four-pass/six-row phases are invoice+password v1, parcel+password v2, then
invoice+password under current v2. Roles, groups and IDs stay explicit and finite.
Observe unchanged-password retention, parcel acquisition/loss, and current-rule
invoice adaptation separately. Retaining a still-permitted obsolete route is a
task error, not desired retention. Stable actor/source are preserved; candidate
readouts do not imply deployed promotion.

Why this: a truthful policy stream is the concrete missing data prerequisite for
testing frozen controllers beyond repeated v2 receipt/noise schedules. It builds
on accepted temporal rules and protects the original final/probe chronology.

## Alternatives and consequences

- Reuse v2 packets and merely flip targets: rejected; sandbox truth would disagree.
- Replace the task simulator or training runner: unnecessary; compatible ports exist.
- Treat invoice failure under changed truth as catastrophic forgetting: rejected;
  password retention and unchanged-v2 parcel loss are reported separately.
- Immediately sweep controllers/native methods: deferred. Existing #54negative
  results stand; new comparative data/metrics/limits must be frozen separately.

The sole fixed scalar case acquires parcel/retains password/adapts invoice but
loses parcel on return. All failures/costs stay recorded. Native method ports are
reused; their new-policy outcomes are unexecuted. Single-agent self-review is not
independent/scientific/release acceptance. Full R7.5/#26/#18 remain open.
Evidence: issue67 and `artifacts/runs/versioned-policy-stream-20261023/RESULTS.md`.
