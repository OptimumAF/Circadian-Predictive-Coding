# ADR-0386: bounded FIFO action-memory control

## Context

The accepted exact count learner acquires and retains known MiniGrid TRAIN tasks.
Its isolated mature E16 copy cannot admit the next nine-key batch at the original
64-key limit: it stops at63 keys. The negative result is preserved. It does not
show neural plasticity loss, nor whether replacing old memories would help or
harm retention. The existing exact learner deliberately has no replacement law.

## Decision

Add a separate opt-in pure FIFO count model and native adapter. Keep the original
exact-memory learner, identity, defaults, refusal behavior and evidence intact.
Admit the same16/5 and296/7 geometries and retained-table capacities. Store entries
in insertion order; hits increment counts without moving the entry. A new key at
capacity retires the oldest entry and all its counts, then joins the queue tail.
An evicted key starts fresh if seen again. Process batch rows in order, but publish
only a complete valid batch. More unique rows than capacity can retire entries
within a single batch; report every exposure and eviction.

A distinct complete snapshot holds lifetime update/example counters, the ordered
retained table and two bounded scalar totals: evictions and retired examples.
Retained counts plus retired examples equal lifetime examples. There is no growing
eviction log. The extra two logical int64 counters cost16 bytes outside the declared
retained-table allowance; total logical cost and measured process overhead are not
identical to the refusal control. Insertion order needs no additional per-key ID
or timestamp beyond the existing ordered entries.

Explicit conversion from a fully validated exact-memory snapshot detaches state,
preserves original table/order and lifetime counters, and starts retirement totals
at zero under the new identity. This is a new experimental control, not a resume
or renewal of the source runtime's budget or serving authority. Restore across
the two model identities is refused.

## Alternatives

Increasing capacity changes the original question and erases neither failure nor
cost. LRU, learned replacement, context-aware protection and policy selection add
mechanisms before a simple fixed control is measured. Editing the old learner's
law would invalidate accepted evidence. A new runner, codec or provenance system
is unnecessary: the existing native and isolated probe ports already suffice.

## Consequences

The learner can keep acquiring under a fixed retained-table cap by forgetting
older entries. Whether this improves useful adaptation requires acquisition,
retention and returning-task measurements together. Component array fixtures can
verify deterministic behavior and budget isolation; they cannot establish game
performance or a comparative benefit. A later small old/new/old-TRAIN assay must be
frozen before execution, include replacement/retired exposure and memory/compute
costs, and preserve the unchanged matched neural and refusal results. Previously
inspected E tasks yield exploratory engineering evidence, not independent
confirmation. No final data, deployment, promotion, replay or baseline tuning.

## Matched E8/A8 return composition

Use the existing saved-memory adapter/script and helper for four isolated source/
control cohorts. Original A TRAIN DTOs and raw source reports bind through one
parsed source; mechanically renamed helper-role copies are explicitly known-task
retention data. Matched planned E8/A8 inputs and readouts preserve missing exact-
memory endpoints after capacity refusal. A validated unchanged failed prefix can
precede an independent cohort; every subsequent boundary checks its full paid state,
raw report and original source/budget. Other failure/unknown state stops work.

Why: stopping all methods after a known atomic refusal prevents measuring the
separate control. This is a prospective rule for this comparison, not a retroactive
change to E16. Overall failed status remains incomplete/CLI2. Flat preceding checks
and exact typed-core validation before native full-value equality remove observed
recursive/dictionary work while retaining every type/field/counter criterion.
The result measures deterministic fitting/relearning and its forgetting cost;
actual raw-action gameplay remains a separate unproved outcome.

## Saved endpoint raw-action readout

Add a separate fixed return-memory admission to the existing frozen MiniGrid
runner. Known A blue/red stay TRAIN, E-blue/blue and E-red/red stay probe_test;
no arbitrary-layout/final role release or label scoring is added. Restore only
the actual initial/end records, with the mature refusal prefix explicitly failed.
No missing checkpoint8 state is reconstructed and no original budget is resumed.

Why: teacher-forced fitting cannot establish actual action success. Reusing the
runner tests that behavior with zero training rather than another framework.
Eight bounded input/report seals plus private paid copies close demonstrated
callback corruption windows. Unknown lost-report costs remain unknown; resource
exit cannot relabel a failed guard or erase permanent charge observations.
