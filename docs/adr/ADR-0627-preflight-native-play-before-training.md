# ADR-0627: Check native play dependencies before TRAIN work

## Context

The normal affine-play workflow previously reached optional MiniGrid validation
only in the native constructor, after six TRAIN fits. A missing-extra setup
could spend work before discovering it could not play. Base-only users still
need command descriptions and saved-result inspection.

## Decision

Call the existing infra `_require_minigrid` before source decoding or learner
creation in normal `run_workflow`. Keep its exact pins and actionable errors;
reuse the existing failed-report/exit contract. Descriptions and scientific
readback modes keep their current behavior. Add no checker or dependencies.

Why this: fail early at the existing external-runtime boundary, before paying
for work that cannot finish its requested task.

## Consequences

Pure setup tests and a fresh cache-only base installation can verify refusal
without new model or simulator work. Prior native evidence remains scoped and
unchanged. Base-only installation does not qualify a clean optional-native
installation, the whole repository or scientific/release readiness.
