# ADR-0481: Preserve distinct readiness corrections

## Context

The fixed correction schema derives stages/gripper from elapsed steps. New
readiness phases are geometric and cannot be truthfully represented by those
records. Self-executed teacher success on two TRAIN resets does not qualify all
student-visited labels. Query proposals also are not executed closing commands.

## Decision

Add distinct owned provisional label/batch types with original public-observation
pairing/arrival clocks, and explicitly select the mode in the existing query and
private-update pipeline. Project through the existing raw/public21 helper without
changing family. Query passes previous actual student gripper to an optional core
feedback argument; defaultNone retains self-executedteacher behavior. Preserve
source/actor, budgets/receipts/fullnativecontinuation and oldfixed defaults.
Pin mode/teacher/port/paid history and recheck direct TRAIN authority after opaque
metadata readers; poisoned report remains available for paid-failure accounting.

## Alternatives

Relabeling fixed batches would erase provenance and silently weaken phase
validation. Counting proposed close labels can advance lift without the student
closing at all. A new collector/runtime would duplicate existing ownership/work
boundaries. There is no reason to add replay, a new objective or circadian feature.

## Consequences

A conventionalBP intervention can test the new signal locally before larger
comparisons. It remains provisional; negativeactualcommands express closing
intention, not contact or attachment. Different owntrajectories/information and
extraeightupdates do not establish matchedcompute/querynecessity. All previous
negative results/failed budgets remain; student retention/recovery/generalization/
safety/CPCbenefit/release stay separate requirements.
