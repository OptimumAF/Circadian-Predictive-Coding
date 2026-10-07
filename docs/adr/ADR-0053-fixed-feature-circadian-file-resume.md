# ADR-0053: Resume CPU fixed-feature guarded training from a local file

## Context

The fixed-feature benchmark materializes train, guard, and validation
features before training its matched heads. Its circadian head can split
or roll back during an epoch, while sleep retry policy and report counters
live in the runner. The P3.9a model/process checkpoint alone cannot say
which cached batch or sleep stage should execute next. The file must not
read final-test features during an interrupted training run.

## Decision

The ordinary CPU, fixed-epoch three-head route accepts a checkpoint store
and an explicit resume flag. The actual circadian trainer saves after each
successful wake batch, immediately before the sleep decision, and after
the accepted or rejected decision. A cursor identifies the next batch or
whether to execute or skip sleep. The payload includes all runner report
counters, P3.8 retry state, the P3.9a combined snapshot, the initial
head hash, a digest of the full benchmark configuration, materialized
train/guard/validation feature hashes, and split hashes. It never hashes
or materializes final-test features for checkpoint identity.

`TrustedLocalCircadianCheckpointStore` is an infra adapter. It writes a
checksummed pickle to a temporary file in the destination directory,
flushes it, and replaces the target file. Loading verifies the header,
checksum, and payload type before the runner validates identity and
counters. The combined checkpoint then validates a detached candidate
head and process RNG before live restore. Pickle files must come from a
trusted local run; the checksum is for accidental corruption, not for
authenticating a source.

## Alternatives and consequences

Recomputing the feature bank on resume is safe only when its hashes agree
with the checkpoint; the runner refuses a different bank. Backprop and
ordinary PC heads train afresh on each invocation. Their initial tensors
and cached features remain matched, and final-test materialization stays
after all included heads finish. Timing from an interrupted or
checkpointed run includes different persistence work and must not be
used as an equal-time head comparison. Learning metrics and report work
counters retain their existing definitions.

The initial route explicitly rejects checkpointing on CUDA or with a
wall-time deadline or process-memory measurement. ADR-0054 extends it
to CPU wall-time training with a cumulative active deadline. Memory and
CUDA remain P3.9b2b; P3.9b and P3.9 stay open. Whole-image loader order and
full-classifier state remain P3.9c. This does not select seeds, tune
baselines, or change the guard acceptance metric.

## Evidence

`tests/test_fixed_feature_checkpoint_resume.py` interrupts the actual
head loop during wake, before sleep, after accepted sleep, and after
rejected sleep. Fresh-head resume matches uninterrupted state and report
counters after a real split or rollback. Mismatched config/protocol,
features, corrupt bytes, and counter corruption reject before training.
A public-route fixture checks identical final head hashes and that the
final-test loader stays sealed through interruption. The development log
records the full quality gate.
