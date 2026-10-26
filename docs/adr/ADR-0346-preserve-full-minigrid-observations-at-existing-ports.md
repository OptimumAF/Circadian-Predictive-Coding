# ADR-0346: Preserve full MiniGrid observations at existing ports

## Context

Full R4.2 names MiniGrid; the accepted local key/door task is custom. Actual
MiniGrid Memory supplies a public7x7x3 partial image, direction and constant
mission. Existing episode and native policy feature bounds are1..16. Cropping
the view to fit could hide its cue and change the intended task.

## Decision

Add a lazy optional infra boundary for pinned MemoryS7 and preserve all148
normalized numeric inputs. Extend existing DTO/codec/native policy admission
with that exact geometry, keeping previous widths/normalized ranges and all
other bounds unchanged. Use the existing stream and learner implementations.
Native actions and ending flags remain exact; no private environment fields.
Only terminated/nontruncated traces can become complete demonstration packets.

## Alternatives

Cropping or compressed semantic features would introduce an observation change.
An unbounded feature/episode expansion would enlarge every experiment budget.
A separate trainer would duplicate existing delayed-label/role/receipt behavior.

## Consequences

Optional dependencies remain outside base requirements. Larger packets incur
their actual storage/exposure costs. Exact version and observation checks reject
unsupported changes early. Memory is an actual environment prerequisite; the
full key/door recurrence/retention/held-out and recurrent RL requirements remain
open. Scripted boundary success is separate from learned useful adaptation.

## R2.1 reserved probe extension, 2026-10-16

The existing fresh-task probe still rejected148-wide packets even after the
episode/native learner ports admitted them. Add exact148 with its original65536
aggregate entry cap and other limits. Declare E-blue/E-red layouts separately
from the original benchmark tuple, with matching cue and probe-only role checks
before environment construction. Distinct contexts preserve one complete episode
per recurrent call and unique evaluation score keys; no extra context feature.

Why this: reuse the existing disposable fork and metric path for actual public
MiniGrid inputs without cropping or joining18step sequences. A new orchestration
framework is unnecessary. Fixed E geometry is declared independently of final D;
same deterministic traces and a component smoke do not establish generalization,
retention/recovery, preserved plasticity or circadian benefit. Original R2.1
acceptance and both failed R4.2 native cases remain preserved.
