# ADR-0332: Preserve native state across demonstration and reward phases

## Context

The native recurrent learner has separate supervised and on-policy reward APIs.
The reward-only runtime port cannot admit a demonstration packet. The accepted
from-scratch RL case supplied no reward signal; original R4.2 calls for
demonstration-first development. Its negative outcomes remain unchanged.

## Decision

Add an exact train-role DemonstrationUpdate and a separate dual-objective adapter
owning one existing recurrent learner. Matching ID releases dispatch typed records
to unchanged native objectives. Reuse the existing recipe/runtime for a fixed
eight-demonstration then eight-reward schedule with one original budget. Preserve
full native snapshot/fork/optimizer/gradient continuation and per-phase costs.
Keep pure-RL default behavior and defer final scoring until every commit.

## Alternatives

Interpreting teacher classes as rewards loses objective provenance. Replacing the
learner at the phase boundary loses momentum and gradient continuation. A new
stream framework duplicates existing opaque runtime ports and budget semantics.

## Consequences

Phase scheduling stays in the recipe; native snapshots gain no parallel phase
counter or new codec. Exact packet/role/class checks guard training boundaries.
Zero-reward SGD can move warm-started weights via momentum. Implementation and
train-only engineering checks establish no comparative learning benefit: a future
frozen protocol needs a matched BC-only warm start and momentum accounting.
