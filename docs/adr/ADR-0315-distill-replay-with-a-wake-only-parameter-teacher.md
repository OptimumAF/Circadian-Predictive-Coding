# ADR-0315: Distill replay with a wake-only parameter teacher

## Context

Arrived-row replay and complete labeled-stream continuation are implemented.
R2.5 requires a versioned slow target ablation, with replay-only comparison and
teacher costs. The coherent NumPy objective already accepts soft binary labels;
the legacy Torch PC diagnostic and hard-label port have different contracts.

## Decision

Wrap exact ordinary coherent BP/PC/CPC learners with an owned student/parameter
teacher. Update the teacher after successful wake calls only. Recompute detached
probabilities for the existing arrived replay selection; bind policy, schemas,
revisions and original content IDs before native replay. Use the affine BCE
target mixture with explicit diagnostic IDs and unchanged native hidden penalties.
Keep complete native/teacher snapshots and stage updates before publication.

The existing helper accepts an optional core policy and structural teacher port.
Record teacher attempts before forwards and native replay charges before native
updates. Report parameter and temporary payload bytes separately from replay
retention and RSS. Compare one fixed four-method family with accepted replay-only
evidence after execution, with equal hard-row exposure and explicit extra work.

## Alternatives

- A new orchestration framework duplicates the existing isolated stream runner.
- Weakening the Torch hard-label contract silently substitutes an objective.
- Averaging chemistry or updating the teacher on replay changes the proposed
  wake-only mechanism and its clocks.
- Caching all historical teacher targets adds storage/version requirements that
  current arrived-row recomputation does not need.

## Consequences

The default native and rehearsal paths persist. Positive distillation adds
teacher weights, forward/EMA work and staged copies. Zero mixture omits teacher
storage/work and preserves native targets/diagnostics. Numerical correctness
does not imply learning benefit. Teacher continuation, Torch distillation and
trajectory semantics remain unsupported; no new persistence framework is added.
