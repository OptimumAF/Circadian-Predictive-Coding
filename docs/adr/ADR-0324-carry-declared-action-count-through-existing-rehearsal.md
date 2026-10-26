# ADR-0324: Carry declared action count through existing rehearsal

## Context

The native feature learner supports categorical actions, but complete packets,
stream target validation and final metrics assume binary actions. Key/door
imitation requires a deliberate class count across those boundaries without
losing role isolation, whole-episode storage or existing continuation authority.

## Decision

Append default2 class counts to current transition/episode/replay-schema/batch
records. Validate exact count agreement before training. Keep binary encoded
arrays unchanged; categorical packets carry one extra class-count header. Include
the count in original schema identity/content binding and normal continuation.
Use unchanged shared replay selection and permanent whole-packet exposure charges.

Add a small pure categorical metric module for hard indices and full probability
matrices. Choose binary or categorical scores from the declared batch contract,
only after all training on detached checkpoint copies. Keep native objectives and
binary metrics unchanged. A fixed small CLI recipe composes the existing runner
and explicitly labels its packets synthetic, with no environment-return claim.

Why this: explicit counts cannot silently change with observed labels, schema
replacement or prediction shape. One bounded codec header prevents cross-count
replay while preserving existing binary bytes; another buffer or trainer is
unnecessary. Separate readback errors preserve primary failures and incurred costs.

## Alternatives

- Treating action2 as a binary soft target violates the native class contract.
- Inferring count from packet labels misses absent legal classes.
- Reusing binary BCE on categorical columns changes metric meaning silently.
- Replacing the orchestration/buffer would duplicate accepted isolation and budget
  behavior before a demonstrated need.

## Consequences

Categorical demonstrations now use the existing adaptation stream boundaries.
Synthetic classification correctness is distinct from real environment behavior,
useful retention/recovery or a circadian advantage. Original R4.2 remains open for
actual partial-observation key/door, hidden cue, recurring/held-out layouts and
separately evaluated recurrent RL. No durable historical object migration,
publication authority or budget renewal is introduced.
