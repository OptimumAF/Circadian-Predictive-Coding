# ADR-0507: positive row weights in the existing continuous BC loss

## Context

The accepted fixed112/120 corrective diagnosis improves mean correction fit but
worsens the first action and original expert Push fit. Equal-row native updates
cannot express a prefix/group-aware repair while retaining all148 observations.
The physical failure and all simpler controls remain valid negative evidence.

## Decision

Add optional positive float64 row weights to existing continuous BC derivatives
and native batch/episode updates. Normalize robustly by maximum then total; refuse
unrepresentable positive masses before numerical work. Uniform weights use the
original arithmetic branch exactly. Explicit weights have named objectives even
when all ones. Keep role checks, frozen normalization, momentum, atomic source
validation and permanent real-row counters. Weights are copied batch supervision,
not policy state. No new dependency, optimizer or orchestration framework.

## Alternatives

- Duplicate or remove rows: changes exposure or discards evidence.
- Add further uniform training immediately: leaves the diagnosed weighting
  question unresolved and introduces an unqualified dose choice.
- Replace the learner or runner: unnecessary for this conventional capability.

## Consequences

Qualify independent scalar loss, every parameter derivative, exact unweighted
equivalence and actual native failure isolation. Recorded mixed continuation
composition remains unfinished and must retain weight provenance and costs before
a separate frozen study. Implementation correctness establishes neither physical
adaptation nor a CPC advantage; final/probe isolation remains unchanged.
