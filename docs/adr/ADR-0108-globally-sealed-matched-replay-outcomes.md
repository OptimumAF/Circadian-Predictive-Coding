# ADR-0108: Freeze matched replay trials before common final outcomes

## Context

ADR-0107 established actual train-only replay parity for circadian, PC,
and backprop, but its artifact had no final outcomes. Scoring one
policy/seed immediately after training would allow final labels to be
available while later trials still train. A published comparison also
needs evidence that applied rows and update budgets stayed matched after
guarded sleep, even when a run is rejected or skipped.

## Decision

Add a separate `continual_matched_replay_outcomes_v9` manifest and app
runner. The manifest binds one arrived-role configuration, ordered seeds,
FIFO and seeded bottom-k policies, the shared newest-retained sampler,
and per-sleep replay/inference budgets before source access. The runner
trains every policy/seed through the P4.4b path. It then rebuilds the
train-only schedule for each unscored trial and checks the manifest
digest, A→B role IDs/hashes, retained and selected boundary IDs/order,
sleep event replay counts, actual per-method row/update/inference work,
circadian exposure IDs, and wake/replay clocks. A failed check raises
before any final source is opened.

Only after all trials pass does it release every A and B final role.
It compares final role IDs and content hashes across policies for each
seed before scoring any model. It then calls the existing NumPy
continual-shift scoring calculation for all three methods on those
matched roles and retains every policy/seed outcome and aggregate. The
local artifact includes resolved manifest, role identities, scores,
retention, exposure, applied work, and typed sleep facts. It excludes
only measured execution durations from its deterministic JSON; those
durations are not used for selection or scoring. No winner is chosen.

## Alternatives

- Score each seed immediately after training. That breaks the global
  final-test seal while later trials still train.
- Assume the P4.4b trace is enough for a model comparison. It has no
  final outcomes and cannot establish matched evaluation identities.
- Force a circadian advantage by changing seeds, guard tolerance,
  metrics, or baseline replay. That would invalidate the comparison.

## Consequences

The fixed two-seed/two-policy artifact has four scored rows and 16
matched boundaries. Tests make final source access fail until all four
trials finish, reject forged roles/work/exposure before final release,
reject cross-policy final-role mismatch before scoring, and show that
changing final labels affects final scores without changing replay or
development-role facts. This is a small, fixed NumPy comparison; it
does not support a broad ranking or replace the later P4.3 replay
side-effect ablations.
