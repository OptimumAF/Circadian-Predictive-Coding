# ADR-0471: Explicit public21 learnability CLI

## Context

The accepted nearest control succeeds with A-only public21 input in the known
camera fixture. Neural task competence remains missing. The existing MLP
learnability runner already supports matching arbitrary-dimensional TRAIN
episodes; the stream CLI lacks an explicit route from raw qualified RGB files
to that model input.

## Decision

Add `--public-vector-only` to visual continuous-mlp-learnability, projecting via
the accepted pure helper before fitting and reporting projection work separately.
Preserve defaults, source files and controller/role contracts; refuse incompatible
RGB-range normalization. Compose the existing fixed24-update result API for one
separately frozen offline arm and retain complete final parameters/momentum with
its existing typed policy/counters, avoiding a later reproduction solely for
native validation.

Why this: expose a working existing learner path with a sufficient simpler input
before inventing a learner or modifying loss/optimizer/epochs. The saved raw
eight-array state uses existing snapshot semantics, without a new codec/runner.

## Alternatives

A global encoder change would alter defaults and hide raw input costs. Official
42D state changes the input contract. A new orchestration layer duplicates the
existing runner. Reusing exact789-statistics prefix would change this arm's
existing one-A-fit recipe; its21D fit is instead explicit, with rounding differences
reported and frozen before outcomes.

## Consequences

The CLI is reusable and independently testable from scientific benefit. The
smaller network has a different shape and RNG draw layout despite unchanged
seed/settings, so initial parity, equal compute and pixel-causality claims are
invalid. All prior RGB/binary/range/corrective negatives remain. Strong neural,
generalization, CPC and release requirements remain open. See the public21 MLP
guide and its linked protocol/results/review for actual executed scope.

## Follow-up evidence, 2026-10-19

61focused checks pass on Windows and Linux; independent source/gate/actual review
accepts the single offline arm. Final A/B half-MSE worsens versus unchanged RGB
despite lower arithmetic, with mixed retention/axis tradeoffs. Complete final
parameters/momentum are retained for prospective native validation with0new SGD.
The initial filepath-import failure and reviewed module correction are preserved.
No native reward, neural generalization or circadian benefit is established.

## Saved-state native follow-up, 2026-10-19

Existing typed restoration and frozen public21 rollout reproduce all42source scalar diagnostics without retraining. Both fixed TRAIN17/19 native episodes return0/no success, unchanged full state and six zero RGB endpoint differences.76focused checks pass per platform with independent source/gate/actual acceptance. No native/generalization/CPC benefit follows; see [guide](../public-vector-mlp-native.md).
