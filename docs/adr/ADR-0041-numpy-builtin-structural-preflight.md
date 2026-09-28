# ADR-0041: Validate combined NumPy sleep proposals on original indices

## Context

NumPy built-in sleep selected split and prune candidates from the original
width, then applied split before prune. With overlapping thresholds, both
selectors could choose the same original neuron. Gradual-prune selection
also counted marked neurons as available minimum-width capacity. Invalid
selector output could reach mutation after split noise had advanced the
local random generator.

## Decision

The built-in route selects prune candidates first, then selects split
sources from original-width neurons excluding those prunes. This makes an
eligible prune take priority, as in the external proposal rule from
ADR-0040. Both selected tuples are checked together before split noise or
tensor mutation: type, uniqueness, original-width range, resolved and
configured count/fraction budgets, max/min width, thresholds, cooldowns,
prune age, and pending marks. Existing gradual-prune marks reserve
minimum-width capacity. A malformed selection raises `ValueError` and
leaves model-owned state, including the local RNG, untouched.

Why keep original-width indices: NumPy already selects both actions
before changing width. This avoids a selector's result changing when a
child is appended. The split-then-prune tensor update order remains.

## Alternatives and consequences

Allowing a split parent to be pruned in the same event hides the source
of the retained child and makes later lineage ambiguous. Clamping a
second gradual prune at finalization reports more scheduled removals than
can fit above the minimum. The new overlap and pending-capacity rules can
change outcomes for those invalid edge cases. Ordinary disjoint
selection, including the historical legacy mode, keeps its indices.
Torch still selects prune after split and retains that rule pending
detached preflight in P3.4b2. Stable neuron IDs and parent lineage remain
P3.4c; atomic rollback of errors after a valid proposal remains P3.7.

## Evidence

`tests/test_numpy_builtin_proposal_preflight.py` first failed because a
split and prune both selected index 3. It now checks prune priority,
pending gradual-prune capacity, unchanged disjoint decisions, and
malformed index/budget/width/age/cooldown proposals rejected before RNG
or tensor mutation. The development log records the full quality gate.
