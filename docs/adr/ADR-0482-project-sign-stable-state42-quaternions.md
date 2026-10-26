# ADR-0482: Project sign-stable state42 quaternions explicitly

## Context

The accepted state42 BP192-update TRAIN17 return improved imitation/reset fit but
still failed task success. Saved diagnostics found a large normalized-state jump
coincident with a TCP quaternion sign change. Pinned SDK source binds the full42
layout and the TCP22:26/object32:36 quaternion slices. This suggests a representation
hypothesis; it does not establish the cause of failure or physical equivalence of
the observed adjacent states.

## Decision

Add a fixed, opt-in pure core projection selecting the sign for which the first
largest-absolute quaternion component is positive. Apply it to both quaternion
slices before fitting and frozen prediction. Validate full finite42 input and unit
norm within1e-4; preserve magnitude, other coordinates, labels, IDs and raw traces.
Canonicalize quaternion signed zeros, break ties by first component. No clipping,
renormalizing, new learner/optimizer, learned threshold or global continuity claim.

Reuse the existing frozen observation-projection path and its metadata, paid work,
full source/fork guards. The learned-robot API/CLI explicitly enables this recipe;
raw42/RGB/public21 defaults remain. Derived feature metadata binds training and
serving. The saved raw42 diagnostic refuses projected reports to avoid silently
interpreting raw trace coordinates under feature-trained statistics.

Dependency direction: existing CLI/adapter → pure state-quaternion core and
continuous records. Existing native port keeps recording original SDK state.
No new runtime framework, dependencies, environment variables or global settings.

## Alternatives

- Silently change official state encoding: rejected because old receipts,
  statistics and accepted raw semantics must remain valid.
- Normalize/refit or clip observations: changes another hypothesis and can hide
  actual departure. No such change is included.
- More epochs/seeds: earlier larger-dose negative does not justify an automatic
  sweep. Test the concrete input representation separately.
- Positive-first-component rule: the largest-component rule avoids choosing a
  near-zero component for this local sign chart. It still has chart boundaries.

## Consequences and verification

Equivalent q/-q inputs map identically with deterministic ties, but the feature
chart is not globally continuous and does not guarantee useful control. Correctness
requires pure invariance/validation/nonmutation tests plus actual prefit and frozen
serving composition, raw trace preservation and late-callback refusal. Native
performance requires a separate frozen budgeted experiment. All previous negative
results, costs and open acquisition/reliability/generalization requirements remain.
