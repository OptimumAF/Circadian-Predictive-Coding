# ADR-0523 — Own disposable native learners without renewing paid work

## Context

Native A/B/reset/rehearsal results show a task tradeoff. The seven-update
diagnosis confirms nonzero head/LoRA updates, without explaining task failure.
A reserved text-task probe must own mutable learner state separately. Existing
native snapshots are owner-specific and the array probe runner accepts numeric
batches. Temporarily training/restoring the source would violate probe isolation.

## Decision

Implement `.fork()` through the existing ForkableLearner port with opt-in final
policy field `max_forks=1` (default0). Copy trusted CPU model module structure,
trainables/gradients/registered buffers, optimizer moments, RNG, policy and full
paid history by value. Memo-share frozen Parameters with no gradients; do not
call constructors, loading, checkpoint/input-gradient setup or model execution.
Share only the stateless compiler callable. Refuse unsupported registered
storage aliases and excessive buffers before allocation.

Why this: sharing large frozen weights makes an independent mutable owner
practical on the existing host. Copying modules separates modes and hooks while
preserving the existing native objective and optimizer history.

Native/reset/health limits and counters are inherited, never reset. Fork costs
are separate permanent attempted-element charges, copied as already spent to
the child. One allowed copy cannot recursively generate more copies; snapshots
cannot refund copy attempts. Child new work is measured relative to its inherited
birth baseline, with no duplicate credit for parent work. Failure after admission
poisons the source; paid copy attempts remain and no success is recorded.

## Alternatives

- Whole-model weight cloning duplicates billions of immutable coordinates.
- Training the original and restoring a snapshot does not create isolated probe
  ownership or independent permanent work counters.
- Fresh AdamW or renewed native allowances would change the question/recipe.
- Converting text into fake numeric AttributionBatch arrays loses native inputs.

## Consequences

No automatic probe/data access or new scientific allowance. Snapshot owners
remain distinct. Normal frozen-weight version/identity guards apply to both
owners; hostile `.data` or arbitrary copy/hook/compiler code is outside the
trusted local port. Tensor source-element counts are not complete Python/
tokenizer-copy/allocator/FLOP or memory-peak accounting. Unregistered trusted
model metadata and hooks still depend on upstream copy behavior. Literal tests
do not establish an executed released-model fork; a small separately budgeted
reserved-text protocol must exercise that boundary before any scientific claim.
