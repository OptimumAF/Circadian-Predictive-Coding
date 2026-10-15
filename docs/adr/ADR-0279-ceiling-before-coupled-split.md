# ADR-0279: Check an owner's width ceiling before coupled split allocation

## Context

Uniform raw coupled splitting already validates the globally supported expanded
configuration, precision mapping and rooted history. It cannot express an owner's
smaller width allowance, so an accepted sequence could outgrow the intended local
budget before a caller checks its result.

## Decision

Add optional keyword-only `max_width` to the existing split consolidator. Validate
an exact bounded positive integer or None at construction and expose a read-only
property. After current-state and pure target metadata validation, reject a target
above the ceiling before invoking the expanded split kernel. Preserve defaultNone
behavior, global limits and all existing mathematical/state contracts.

## Alternatives

- Checking the result would allocate the oversized candidate first.
- Reusing ToyBudgetSession's unsupported generic width option would falsely imply
  it measures/enforces every native shape and would weaken its explicit refusal.
- A new growth algorithm or general policy framework is unnecessary.

## Consequences

Owners can opt into a narrower uniform-width ceiling. Current-state validation
may still allocate at existing geometry; the guarantee concerns expanded arrays.
The runnable native control accepts2->4->8 and refuses16, preserving consumed
attempt IDs, complete candidate/CPC state and actor/cache; one subsequent update
demonstrates original-budget continuation. No new dependency or learning claim.
The configured consolidator is trusted local policy, not durable owner authority.
