# ADR-0149: Extend explicit parent ranking with transactional state

## Context

The current proposal API supplies add counts and explicit prune indices;
the NumPy core still chooses parents by usage-related split scores. Its
eligibility and topology machinery is already tested and frozen by earlier
protocols. Complete snapshots copy all model fields. Direct proposal
application has no transaction because its old ranker draws no randomness
before width checks; random/cyclic selection introduces earlier mutable state.

## Decision

Add a dedicated core subclass overriding the explicit proposal ranking
seam. Usage delegates the old ranker; cyclic stable-ID and seeded random
selection retain its eligibility and chemical-preference tiers. Keep a
separate PCG64 selector stream and model-owned decision/cursor state.
Validate compatible selector snapshots and make the new direct proposal
operation atomic. Require explicit policy for split-capable sleeps through
the extension. Verify fixtures first (c9a), then independently freeze and
repeat arrived train-only controls (c9b) before separate scoring (c9c).

## Alternatives

- Relabel count proposals as random: does not choose random parents.
- Patch the pinned core ranker/config: breaks older source gates.
- Copy tensor split/prune implementations: duplicates correctness-sensitive
  behavior and risks drift.
- Mix selector draws into split-noise RNG: confounds parent and noise streams.
- Compare built-in strict thresholds to explicit scheduled counts without
  labeling their eligibility difference: changes more than parent choice.

## Consequences

The new extension supports explicit-policy comparisons; existing built-in
models retain their original behavior and source identity. The controlled
route has explicit mutable selector state and snapshot compatibility rules.
It needs prospective count/guard semantics and role/resource gates before
matrix claims. Core fixture correctness alone does not close c9, choose
parameters from c8 scores, or authorize confirmation/final release.
