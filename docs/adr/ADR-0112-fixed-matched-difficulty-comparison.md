# ADR-0112: Keep the difficulty comparison fixed and separate from core history

## Context

The P4.6a four-row signal probe shows that one bad label or feature can
raise the historical supervised-error scale, while same-step loss
improvement is unavailable to the current update. It cannot measure
learning, forgetting, or held-out effects. Existing checkpoint and
reward-named config identities must remain compatible.

## Decision

Use a separate `difficulty_matched_modulation_v11` app wrapper and local
JSON adapter. The fixed A/B synthetic source supplies arrived training
fields and a final source read only at the explicit release boundary.
Reuse the existing four-role splitter. Perturb one copied B train row
*after* clean partitioning and hash the effective train content separately,
so all development and final role identities stay matched. Train every
NumPy and CPU Torch modulated/control arm with equal within-backend
initialization and work. Save A state before B. Require all 24 train-only
trials and their role, content, A/total clock, and diagnostic preflight to
pass before releasing any final field. Report every score and signed
forgetting value without selecting settings.

The actual model scale remains its historical relaxed-state error factor.
Feedforward clipped error and prior-step BCE improvement are read-only
train diagnostics; their values do not enter training. Both backend
configs retain the reward-named switch and existing serialization.

## Alternatives

- Add clipped-error or improvement scaling now. The causal probe had no
  held-out learning result and improvement would need future state.
- Reuse the four-row probe as an outcome benchmark. It has no independent
  decision/final roles or A→B forgetting measure.
- Retune the fixed run after seeing poor B learning. That would change
  the predeclared budget and make this result selectively reported.

## Consequences

The v11 result is null for the modulation switch at its 40-row accuracy
resolution. Actual scales differ, yet no matched final accuracy or
forgetting pair differs. Torch's one-label-flip condition harms B
accuracy for seed 17 in both arms; NumPy B and seed 19 outcomes expose
underlearning at this short budget. The fixed artifact and limitations
are recorded in `docs/difficulty-modulation-comparison.md`. A more
powerful comparison needs a new version and prospective manifest, not a
revision of v11 or the historical algorithm.
