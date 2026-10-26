# ADR-0436: Expose one fixed width-32 neural TRAIN recipe

## Context

Original width-8 neural acquisition is unproven; linear and polynomial-kernel
controls also lack a complete original TRAIN fit. This does not prove finite
model incapacity. Native Torch policies and invariant wrappers already support
width32 and all four methods; the constructive TRAIN runner hardcodes width8 BP.

## Decision

Add an explicit `wide_method` family and mutually exclusive `--wide-neural` CLI
selection. Reuse the existing four native algorithms at D296/H32/K7/seed17, original
optimizer/inference settings and unweighted objectives. Keep all original recipes
and defaults. Validate complete owned wide snapshots/method/policy/counters before
admitting checkpoints and readers. Reuse ordinary TRAIN roles, budget, clocks,
receipts and detached scoring. Distinguish observed native prefix from derived
operation counts; parameter payload does not stand for total native memory.

Why this: one existing capacity can test useful neural acquisition without a new
algorithm, optimizer search, classifier family or orchestration layer.

## Alternatives and consequences

Extending kernel doses would not answer neural acquisition. Mutating original
width-8 baselines would lose their frozen scientific meaning. A free width/seed
interface would add unnecessary tuning choices. Changing every learnability or
rollout path before a TRAIN witness would broaden scope without evidence.

Implementation checks only establish composition correctness. Real acquisition,
retention/recovery, heldout/RL and scientific benefit remain open. Equal width/
data/exposure does not match native computation. New comparative cases require a
small frozen reviewed protocol with all negative results retained. See
[usage/structure/checks](../wide-neural-train-fit.md).
