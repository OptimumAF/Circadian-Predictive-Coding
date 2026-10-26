# ADR-0322: Evaluate recorded weights through live policy episodes

## Context

Accepted R2.4 stores actual complete demonstrations and native parameter
checkpoints. Cached action scores alone do not show an agent acting through a
complete environment episode. R4.1 requires explicit identities and recorded/live
origins, training/evaluation state and learning activity.

## Decision

Add a small callback-based app evaluator over actual LineNavigation. Preserve
every legal transition and illegal decision, with complete or owned partial
reports and bounded attempted/returned prediction counts. The adapter imports
validated logged weights into fresh native prediction copies and qualifies every
cached score before any live episode. It then evaluates the entire frozen set
through the existing recipe CLI, with zero training/feedback/promotion.

Why this: reusing accepted weights avoids another training comparison and tests
the actual native prediction/environment seam. A local `eval:` identity is
explicitly different from historical actor publication. Parameter import does
not claim historical optimizer, adaptive state or RNG recovery. Original
training receipts, charges and publication evidence remain untouched.

## Alternatives

- Retraining duplicates accepted work and changes the source evidence.
- Reimplementing the native forward pass risks evaluating a different learner.
- Inferring episode success from cached action accuracy misses off-trajectory
  states and illegal actions.
- A new orchestration framework is unnecessary for this bounded callback seam.

## Consequences

Original R4.1 is accepted after focused checks and independent actual review.
All final B episodes fail; correctness does not establish useful adaptation or a
circadian advantage. Broader MiniGrid/action schema, partial observation, recurrent
policy, held-out layouts and release requirements remain open. The next concrete
dependency is categorical-action support through existing native/episode ports,
preserving binary behavior. No new dependencies or environment settings are added.
