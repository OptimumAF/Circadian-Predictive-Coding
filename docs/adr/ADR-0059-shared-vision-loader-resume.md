# ADR-0059: Resume the older unmatched vision protocols without reseeding

## Context

The v1 and v2 unmatched image protocols use one shuffled training DataLoader
through backprop, predictive coding, and circadian training. Its generator
advances across models and epochs. Unlike v3, model and augmentation streams
are not reset per variant. v1 also uses validation examples as its sleep and
stopping guard; v2 has a separate guard role. Changing these semantics would
make a resumed run differ from its ordinary reproduction protocol.

## Decision

The trusted vision checkpoint includes the shared train-loader generator at
each completed-model boundary. A resumed CPU runner restores it and the saved
process streams after validating config, data roles, model reports, and detached
model states. During circadian training, `app.shared_vision_loader` records the
epoch-entry generator/Torch streams, next logical batch, current sampler and
process streams, and loader configuration. It does not reseed or fork RNG on
ordinary iterations. A fresh loader replays consumed batches from the saved
entry states, checks the sampler state at the cursor, restores the saved
process streams, and only then yields remaining training work. Rejected replay
restores the caller's streams and loader generator.

The v1 checkpoint binds the validation content twice, as validation and its
aliased guard role. The v2 checkpoint binds its distinct guard. Neither path
scores final test until all three models finish. The loader requires an
ordered, resettable map-style random sampler with its own generator; unsupported
worker configurations fail before training starts.

## Alternatives and consequences

Reusing the v3 seeded loader would change the original shared shuffle and
augmentation trajectory. Serializing worker queues would rely on private
DataLoader internals. Replaying a prefix costs local CPU and transform work,
and checkpoint files can be large for a full ResNet backbone. The shared
loader is a separate small module because its no-reseed/no-fork behavior is a
scientific protocol boundary. The implementation does not alter baseline
budgets, learning rules, seed selection, metrics, or ordinary v1/v2 training.

This covers CPU continuation only. CUDA process and model-local streams need
verification on a CUDA host under P3.9c2b2b. The original parent checkpoint
tasks remain open for that evidence and the fixed-feature memory/capacity work.

## Evidence

`tests/test_vision_checkpoint_resume.py` compares ordinary v1/v2 trajectories
with model-boundary interrupted runs, seals final test until training finishes,
and compares checkpointed uninterrupted versus resumed mid-wake, pre-sleep,
accepted-sleep, and rejected-sleep runs. Zero-worker Torch/NumPy/Python
stochastic views and two-worker views preserve model hashes, non-timing
reports, and next process draws. Corrupt file, changed role data, malformed
cursor, mismatched shared generator, and a valid but wrong replay entry state
reject before a training update and preserve caller RNG. The development log
records exact commands, test counts, and the full quality gate.

## Subsequent decision

ADR-0086 verifies actual-device v1/v2 CUDA continuation alongside v3. The
CPU-only scope statement above records the evidence available when this
decision was made.
