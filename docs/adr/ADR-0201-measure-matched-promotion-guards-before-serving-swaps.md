# ADR-0201 — Measure matched promotion guards before serving swaps

## Context

The stable actor/shadow runtime is accepted. Its actor version is fixed and it
has no complete serving cache/state transaction API. R3.4 requires both meaningful
candidate rejection and atomic promotion/rollback. Introducing actor mutation
before matched guard measurement would make an incomplete safety boundary usable.

## Decision

Split R3.4 into R3.4a matched guard measurement/decision and R3.4b complete atomic
serving transactions. Preserve the original parent criteria and checkbox. Declare
one utility and policy before assessment, restore independent native copies from
frozen snapshots, score the same new/old inner guards, and retain complete
numerical/latency/explicit-byte/action evidence with rejection reasons. Deny
unauthorized roles, future labels, overlaps and stale base versions before payload
or callback access. A report is advisory and contains no mutable model handle or
serving authority.

## Alternatives

- Supplied score-only DTOs could bypass actual matched model/input measurements.
- Native training loss comparison would conflate different CPC/backprop objectives.
- Reusing final/outer labels for repeated promotion would contaminate assessment.
- Exposing an actor setter now would bypass still-unimplemented state binding,
  atomic cache/state transitions and rollback.

## Consequences

Guard correctness and negative outcomes are testable before actor mutation. A
trusted byte probe's definition remains explicit; serialized state size is not
RSS. Version/digest/role declarations do not replace physical/global scientific
authority. Input construction requires quiescent snapshots and conforming owned
builders. R3.4b must verify exact candidate/actor/policy binding and complete
concurrent transitions/rollback; no parent completion follows from this leaf.
