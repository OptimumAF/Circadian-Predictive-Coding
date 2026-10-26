# ADR-0626: Inspect saved affine learning through the existing MiniGrid player

## Context

The portable affine command trains from bundled examples and executes complete
known B/C games, but the shared inspector has no route for its result. Its JSON
contains useful TRAIN learning curves, causal receipts and replay costs alongside
native public observations. Repeating training or building another viewer would
not improve this evidence.

## Decision

Add one explicit recorded projection for portable_affine_train_and_play_v1 to
the existing dispatcher. Reuse the original native episode decoder with explicit
B/C geometry and the existing MiniGrid HTML player, source dialogs and curves.
Only the new format gains affine labels, receipt/catalogue tables and scoped
cost fields. Preserve legacy formats and all original raw source data.

Why this: a small projection makes working learned behavior inspectable without
creating another orchestration, source-admission or rendering framework. Exact
JSON pointers connect observed values to their source. Missing receipts after
failed capture preserve paid counters and an unknown prefix, rather than an
invented zero-work result.

## Alternatives

A separate dashboard duplicates playback and source handling. Shipping fitted
models or rerunning games would add work unrelated to displaying already paid
evidence. Weakening the common native decoder would risk earlier formats; its
existing explicit geometry option already suffices.

## Consequences

The viewer imports no learner or native runtime. Existing dispatch transitively
imports base NumPy for unchanged legacy line evaluation; no model work occurs.
The source format does not include whole-process time/RSS or caller RNG proof,
so those remain unavailable. Scoped presentation checks are self-reviewed and
do not establish independent acceptance, scientific advantage or release readiness.
