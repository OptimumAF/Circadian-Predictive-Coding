# ADR-0153: Compose unscored family trajectories before joint artifacts

Date: 2026-09-30. Status: implemented and verified for b1 fixtures; b2 pending.

## Context

P6.7a freezes six families, fifty distinct reserved sources, 560 model cells
and joint resource limits. Gating/replay runners score outer roles between
A and B. The four later preflights already return unscored facts, but their
public validators accept exactly three development seeds. Sleep/schedule
facts check rejected parameters; combined/parent also check complete state.
Editing those files would invalidate their scientific source identities.

## Decision

Split P6.7b into trajectory/checkpoint correctness (b1), then independent
JSON/artifact/resource validation and two complete reserved runs (b2).
Neither split closes the parent or reduces the original acceptance.

Compose unchanged family model, wake, replay, guard and selector helpers.
Retain original unscored role/cost/decision schemas. Gating/replay omit only
the scored metric fields, which are forbidden at this gate. Add separate
full checkpoint fingerprints, including baseline traffic state and every
circadian snapshot field, both RNGs and selector state. Capture complete
before/after witnesses around old sleep/schedule guard calls; require full
equality for rejected/skipped attempts. Do not attach observers to models.

Hold independent after-A copies. Complete and check all family/seed A work
before constructing any B source, then recheck every held A/B checkpoint.
Within-cell trajectories use local RNG streams, so this stronger arrival
barrier must match the original training exactly. Test each family with
its first original development seed, chosen by manifest order before any
fixture outcome, and raising outer/final fields. Use no reserved source
before b2 correctness/resource/artifact gates pass. The production entry
requires the complete unchanged confirmation manifest before any source;
private phase adapters support development fixtures, without an opt-out flag.

## Alternatives

Calling scored runners would open outer data before the joint gate. Changing
the old public seed validators or adding telemetry observers inside their
models would change pinned behavior/state. A replacement learning algorithm
or one giant conditional runner would duplicate core rules. Per-family
adapters keep scientific helpers and explicit boundaries intact.

## Consequences

Checkpoint and raw fact schemas are not interchangeable across families;
b2 must independently derive and validate their work, roles and links.
Passing b1 fixtures establishes composition correctness only. It is not
confirmation performance, bounded resource evidence or final release.
All original settings, arms, pairs, seeds and limits remain unchanged.

## Evidence

All 56 model cells on the first existing development seed per family match
legacy unscored role/cost/decision facts and complete initial/A/B fingerprints.
Raising outer/final and all-A-before-B, late state/role corruption, forced
rollback with observed rejected replay, independent copies, baseline aliases,
tensor shapes and nonfinite fixtures pass. **53 new/122 related tests**, zero
skips, Ruff/mypy (371 files)/format/diff pass. Twelve existing development
bundles/twenty usage files and the exact P6.7a scope record revalidate unchanged.
Zero-memory gating/sleep models have no optional retention budget; capture
records absence rather than configuring a new memory privilege. No reserved
source/model, new scientific score or experiment bundle was produced. B2
must still independently validate all JSON/work/checkpoint/role links, enforce
runtime/artifact gates, and execute/repeat the complete frozen 560-cell scope.
