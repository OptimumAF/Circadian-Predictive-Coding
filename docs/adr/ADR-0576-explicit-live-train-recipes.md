# ADR-0576 — Compose explicit live TRAIN recipes without changing comparison isolation

## Context

Live modes accept only original packets. Existing comparative recipes require
deferred evaluation; old-only label noise changes row1, which belongs to new data
in a mixed packet. R7.7 needs explicit delayed/poisoned-feedback fixtures.

## Decision

Add a small live wrapper around the existing recipe and reuse its owned TRAIN
builder. An optional retained-half fixture validates then changes copied row7
invoice inspect->defer; preserve original source/new rows/features/task truth.
Record corruption as deliberate experimental noise, not verified teacher output.
Provide a separate live argument/CLI flag; do not relax the comparative deferred
rule or change frozen recipe schemas. Pin main/worker recipe values. Keep native
rules, drift controller, original budgets and clean guards unchanged.

## Alternatives

Changing original noisy row1 semantics would rewrite frozen evidence. Reusing a
comparative argument for live evaluation obscures isolation. Relabelling guard/
final data as TRAIN leaks outcome truth. A new runner duplicates working ports.
Treating a knowingly false target as validated supervision misstates the experiment.

## Consequences

The same live schedule can consume bounded noisy TRAIN while guards/task truth
remain clean. One observed candidate learns legacy but loses password; unchanged
guards refuse it. This is a scoped useful refusal, not general poison detection or
an augmentation/CPC benefit. A matched-dose clean control is needed before causal
attribution. Full long-duration/scientific/release claims remain open; self-review.
