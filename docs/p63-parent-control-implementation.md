# P6.3c9a explicit split-parent control contract

## Inspection and scope

The existing NumPy proposal path validates add/remove counts, maximum and
minimum widths, change-fraction budgets, pending prunes and cooldowns before
`_rank_proposal_split_sources`. That ranker orders chemical-preferred then
fallback candidates by existing split scores. A count-only policy cannot
choose scheduled/random parents. Built-in sleep separately uses a strict
chemical threshold and is not the same proposal route.

Add a separate `src/core/controlled_parent_selection.py` extension for
**explicit proposals only**. Reuse the current parser, eligibility, tensor
split/prune, telemetry and complete snapshot machinery. Change no pinned
base/core/app source, historical configuration or result. This is a selector
implementation/correctness increment, with no benchmark, outer/final score,
winner, confirmation or claim that the matrix row is complete.

## Frozen selector behavior before implementation/training fixtures

An immutable setting specifies `usage`, `scheduled` or `random`, a separate
nonnegative integer RNG seed and an initial nonnegative stable-ID cursor.

- `usage` delegates the unchanged original proposal ranker exactly.
- `scheduled` preserves chemical-preferred/fallback tiers and orders each
  tier by stable ID, starting at/after the cursor and wrapping. The cursor
  advances to one past the last selected parent ID only on selection.
- `random` preserves those same tiers and uses a dedicated **PCG64** stream
  to permute each tier without replacement. It does not consume the core
  split-noise RNG. It has no usage-score ranking inside either tier.

All candidates/counts come from the original validated proposal path;
selectors cannot add parents, duplicate a parent or override an explicit
prune. When preferred candidates are insufficient, the remaining count
comes from eligible below-threshold candidates, matching existing proposal
semantics. Record mode, eligible/preferred/selected **stable parent IDs**,
cursor before/after and RNG before/after fingerprints. Zero additions make
no selector change. Require an explicit adaptation policy for split-capable
sleep through this extension, preventing silently labeled built-in growth.
Prune-only/disabled/skipped operations retain their existing semantics.

Store settings, cursor, selection count, RNG and last decision in model-owned
fields so complete snapshot/rollback captures them. Reject incompatible
settings or malformed selector snapshots before mutation. New direct
proposal application must also be atomic: selector RNG/cursor can change
before a later transient-width check, unlike the unchanged old ranker.
Wrap that extension operation in the existing complete snapshot/restore.
Retain core sleep exception rollback and externally guarded rejection.

## Acceptance and bounded fixture evidence

Before completing c9a, tests must demonstrate:

1. Settings fail early; no unsupported mode/seed/cursor reaches model setup.
2. Usage mode has exact original selections and every original core state
   value through split/prune/wake continuation, excluding the new fields.
3. Scheduled selection follows stable IDs through index shifts, wraps and
   advances only on successful selection. Random selection repeats under
   the same seed, draws from permitted tiers without replacement, and
   leaves core noise RNG consumption matched.
4. All modes respect explicit prune exclusion, pending/cooldown eligibility,
   phase/change/width/count limits and zero additions. Predictive function
   and stable child lineage remain preserved after existing splits.
5. Invalid/transient-cap and injected post-selection failures restore every
   field, selector RNG/cursor/decision and future retry behavior. A simulated
   guard rejection restores a committed proposal; rejected proposed IDs are
   retained separately by the caller. Snapshot replay reproduces subsequent
   selections/parameters, and incompatible/malformed snapshots fail atomically.
6. Earlier c5/c6/c7/c8 source/result identities and focused proposal/lineage/
   sleep/rollback tests still pass. Run Ruff, mypy, format and diff checks.

Fixtures are fixed tiny two-dimensional arrays with at most four labeled
rows, model seed 521 and selector seed 547 unless a test explicitly varies
seed/settings to test compatibility or support. A fixed **0..31** selector
seed range is a unit coverage check of allowed parents, not experimental
replication or seed selection. Individual initial widths are at most eight,
absolute test width ceiling twelve, at most two additions/removals per call.
Use at most three wake batches/cell and two latent steps, no outer/final
accuracy, no benchmark data source and no external work. Do not increase
these bounds to make an outcome favorable.

**Why this split:** the parent-ranking seam permits control without editing
frozen earlier sources or replacing adaptation policy. Correctness and
snapshot compatibility should precede arrived-role training and selection
claims. c9b must prospectively freeze source/development/confirmation roles,
scheduled counts, capacity/work/memory, guard and wall/RSS limits, then run
and repeat complete unscored paired controls. c9c separately gates scored
development against those saved all-seed facts. The original c9/P6.3c/P6.3
matrix and independent confirmation remain unchecked until their original
criteria pass; c9a alone cannot satisfy them.

## Usage and extension boundary

```python
from src.core.controlled_parent_selection import (
    ParentControlledCircadianNetwork,
    ParentSelectionSettings,
)
from src.core.neuron_adaptation import NeuronChangeProposal

model = ParentControlledCircadianNetwork(
    input_dim=2, hidden_dim=4, seed=521,
    min_hidden_dim=1, max_hidden_dim=12,
    parent_selection=ParentSelectionSettings("random", seed=547),
)
model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=1)])
decision = model.get_parent_selection_state().last_decision
saved = model.snapshot_state()
model.restore_state(saved)
```

Use `usage` to reproduce the old **explicit-proposal** ranker, or `scheduled`
with an optional `initial_cursor_id` for cyclic ordering. For sleep, pass an
existing `NeuronAdaptationPolicy` through `sleep_event(adaptation_policy=...)`;
its counts and prune indices still go through the original constraints.
Snapshot restoration requires the same selector settings and base model
configuration. The decision/state views are immutable and detached; copy
the proposed decision before external rejection restores the previous one.
New experiment orchestration belongs in `app` and artifact IO in adapters/
infra. Do not bypass eligibility or replace the inherited tensor operations.
No dependencies, environment variables, datasets or CLI defaults were added.

## Verified c9a evidence — 2026-09-30

`tests/test_controlled_parent_selection.py` contains **83 passing cases**.
They verify every original snapshot value through usage-mode sleep, pruning
and three wake batches; stable-ID cursor/index-shift/wrap behavior; exact
random continuation and fixed unit-seed allowed-parent support; preferred/
fallback and unique selection; original constraints and phase/no-op behavior;
function-preserving predictions and child lineage; equal core-noise draw
counts; direct/sleep transient-cap and injected post-split recovery; external
guard rejection with separately retained proposed IDs; detached snapshot
continuation; and incompatible/corrupt selector state refusal before mutation.
The settings, cursor, RNG, decision and all other fields participate in full
snapshot comparisons. No new scientific benchmark or scored result was produced.

The combined selector/proposal/lineage/sleep/clock gate passed **157 tests
in 2.62 s**. Existing c5–c8 app/CLI regression files passed **38 tests in
38.29 s**. Original atomic-sleep NumPy and Torch CPU cases passed **6 tests
in 1.40 s**. Total **201 passed, zero skipped**. Repository Ruff, mypy
(**351 files**), two-file format check and `git diff --check` passed.
The first new-fixture run had six failures from two incorrectly named
result/telemetry fields; corrected to existing `performed`/`changes`. Its
first mypy pass found 14 errors, corrected with explicit validated snapshot
types and fixture annotations. No existing interface was changed to make
these tests pass. See the development log for exact commands and outcomes.

Read-only finite-JSON readback verified both prior bundles for each c5/c6/
c7/c8 protocol, their exact saved result bytes, source maps, request/adapter/
manifest and audit/RSS identities, and complete payload validators. Selected
map digests remain `2e45a620...a1dcfd`, `3e7d8b80...9c69`,
`457df802...6c737f`, `ee0e2c8c...4d6b5`; their exact full values and result
hashes are recorded in the session log. This increment adds only a separate
core extension and its tests/docs; earlier pinned implementation bytes and
historical results remain unchanged.

Full CPU suite, CUDA, large sweeps, new scientific training/scoring,
independent confirmation and final release were skipped. **Next: c9b**
must freeze a new arrived-role train-only count/guard/work/capacity contract,
including matched ordinary/backprop/neutral and planned-width references,
then implement and repeat every cell under source/wall/RSS gates. C9/c9b/
c9c and original matrix/confirmation criteria remain unchecked. There is
no external blocker; c9a is implementation evidence, not a result about
accuracy, forgetting, capacity benefit or model superiority.
