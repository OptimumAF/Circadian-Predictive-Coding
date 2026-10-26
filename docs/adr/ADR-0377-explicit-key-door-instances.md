# ADR-0377 — Explicit key/door environment instances

## Context

R4.3 requires repeated environment instances and equal interaction allowances.
The accepted mapped quota route supplies96physical steps but fixes one geometry.
Its negative B readouts do not justify choosing new instances from final scores.

## Decision

Add a pure core factory enumerating the24existing rotation/branch/key assignments.
Expose exact distinct native seed indices0..23 as an optional pair on the existing
mapped quota recipe/CLI. A/B/C use the training instance with B controls remapped;
D uses the distinct final-only instance. Keep learner17 and private sampler29.
Freeze both role indices across collection, updates, scoring and final readback.
Omitted inputs preserve legacy defaults/descriptors. No new runner or dependency.

Why this: an exhaustive, transparent finite catalogue avoids accidental geometry
collisions or a new random stream. The index is configuration, not a claim of
statistical independence. Existing training/final boundaries remain responsible
for access to observations; constructing configuration is not observing a task.

## Alternatives

Arbitrary seeded random layouts require collision handling and broaden the
environment design. Changing learner/sampler seeds does not supply distinct
physical instances. Both add unnecessary ambiguity for this bounded prerequisite.

## Consequences

Catalogue coverage is finite and strongly related by rotation/key exchange.
Full R4.3, useful joint adaptation and generalization remain open. Future
comparisons must freeze indices and matched own-policy controls before execution,
preserve failed cases and report variable native/compute/reset costs. No new
native or comparative experiment is part of this implementation increment.
