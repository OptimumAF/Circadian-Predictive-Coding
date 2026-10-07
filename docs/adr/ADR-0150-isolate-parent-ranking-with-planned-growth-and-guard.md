# ADR-0150: Isolate parent ranking with planned growth and guard

## Context

C9a implements and verifies explicit-proposal usage, cyclic and random parent
selection. Counts alone previously could not select these parents. The existing
guarded schedule helper hardcodes a built-in policy and returns telemetry after
rollback, losing the proposed selector state. C8's combined splits were inactive;
its development scores cannot select a new threshold/count/seed.

## Decision

Freeze a separate no-score growth-only factor with the existing c3 structure-only
settings, explicit common planned counts and fresh consecutive-prime role seeds.
Disable pruning to isolate parent selection. Keep original phase budgets: five
one-add attempts and a final zero-add guarded event. Set prospective width ceiling
13 and include fixed-width ordinary/backprop/neutral and planned-width references.

Compose existing scheduling, core transactions, NumPy guarded telemetry and the
same inner-guard accuracy/tolerance rule in a focused app around explicit policy
sleep. Capture proposed selector/full state before rejection, then verify restored
state and independently rederive selection and applied work. Keep all outer/final
values sealed; pin and repeat complete train facts before separate c9c scoring.

## Alternatives

- Patch earlier frozen helpers: invalidates prior source identity gates.
- Inject callbacks/policies into model dictionaries: adds hidden state and makes
  complete snapshot/rollback evidence harder to interpret.
- Copy topology/replay algorithms: duplicates correctness-sensitive core work.
- Force a last-epoch add by changing phase budgets: changes the existing constraint.
- Choose counts from observed c8 widths or favorable development outcomes: creates
  a retrospective capacity/selection comparison.

## Consequences

The new app owns only explicit policy/guard composition and train facts. Core
splits and metrics remain unchanged. Parent cells share attempted counts, wake
exposure and available memory; their independent guards may give unequal applied
counts/capacity. Those differences must be published. This factor has no replay,
pruning, gating, reset or homeostasis claim and does not close the complete matrix
or independent confirmation. The exact prospective contract is in
`docs/p63-parent-factor-preflight.md`.
