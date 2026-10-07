# ADR-0042: Preflight Torch post-split pruning on a detached head

## Context

Torch sleep selected split sources, mutated the live head with noisy
splits, then selected and applied prunes. A malformed post-split prune
index could fail after weights and the split generator had changed.
Selecting prune on the original width would be cheaper, but would alter
the existing rule: the parent or newly appended child can become a prune
candidate after its chemical value is halved by splitting. At the
minimum width, Torch could also split then prune back to that minimum.

## Decision

Validate the selected split tuple against the original width, resolved
and configured budgets, max width, threshold, and cooldown. When both
split and prune can run and a split is selected, copy all head tensors
used by structural work and clone the model-owned split generator into
a detached candidate. Apply the noisy split there, select prune on that
post-split candidate, validate its indices, budget, resulting minimum
width, threshold, age, and cooldown, then verify aligned candidate
topology. Only after this preflight does the live head perform the same
split and prune. One-action routes validate directly without making a
candidate copy.

Why retain post-split selection: changing it would silently change
legacy structural behavior and existing benchmark comparisons. The
parent or child may be pruned in the same event when eligible; the
reported prune index refers to the post-split topology. P3.4c will make
those identities and lineage explicit.

## Alternatives and consequences

Immediate mutation with per-action checks would leave the live split
committed when post-split prune validation fails. Selecting both actions
before split would remove the parent/child behavior. The detached plan
adds one tensor copy and one simulated split when both actions are
possible. It does not copy the ResNet backbone or change the head-only
sleep guard API. A bounded local CPU measurement with Torch 2.14.0,
feature/hidden/class dimensions 2048/256/10, one split and one prune,
three warmups and 20 planning calls observed 1.663 ms median and 1.889
ms 95th sample; the 12 cloned tensors totaled 2,116,648 bytes. This is
planning cost on one machine, not an end-to-end benchmark or CUDA
estimate. The live split generator was unchanged after each plan.

This preflight does not make all later sleep effects atomic; P3.7 still
owns rollback for errors or rejected acceptance after a valid plan.
Durable checkpoint/resume remains P3.9.

## Evidence

`tests/test_torch_builtin_proposal_preflight.py` first failed because an
invalid post-split prune reached live mutation. It now checks detached
rejection, preserved parent/child selection, a child actually pruned,
cooldown protection, minimum/maximum width, budgets/fraction, age,
index type/range/uniqueness, and exact split-generator continuation
against a control. The development log records the full quality gate.
