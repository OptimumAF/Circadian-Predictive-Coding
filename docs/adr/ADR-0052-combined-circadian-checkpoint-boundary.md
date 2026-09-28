# ADR-0052: Validate combined circadian continuation state before restore

## Context

P3.3 snapshots own model state, including local split generators and
replay buffers. P3.8 keeps guarded-sleep retry state outside that model
state. Neither snapshot alone records the runner's completed epoch,
protocol/data identity, or process random streams. A restore from only
the model snapshot could resume from the wrong sleep stage or repeat a
rejected attempt on a different schedule.

## Decision

`src/app/circadian_checkpoint.py` provides a versioned, in-memory
`CircadianRunCheckpoint` for a NumPy circadian network or CPU Torch
circadian head. The caller supplies a stable protocol ID, its model
configuration, a SHA-256 data digest, and a typed pre/post-sleep
position. The checkpoint deep-copies the model snapshot, optional
`SleepRollbackCooldownState`, and Python/NumPy process RNG states. A
Torch head also captures the process CPU Torch RNG; the head's local
split generator remains in its model snapshot.

Restore checks identity, version, stage, retry compatibility, and random
states, then restores the model snapshot into a temporary candidate to
validate topology and wake-batch progress. Only after every validation
passes does it restore the live model, retry state, and process streams.
The runner remains responsible for using the returned position to
resume the correct action. The primitive does not write a file or infer
a data digest from an arbitrary dataset object.

## Alternatives and consequences

Putting retry state inside the model snapshot would cause a rejected
guard to roll back operational rejection history. Trusting a supplied
configuration digest without checking the live model would allow an
incompatible caller to mislabel a checkpoint. Keeping identity and
position explicit makes fixed-feature runner integration testable next.

This boundary is CPU-only for Torch because it captures neither CUDA
process generators nor DataLoader/augmentation order. It also does not
capture the full ResNet classifier, runner counters, or durable file
status. P3.9b/c retain those requirements; P5.3 owns the broader
run-artifact write/status contract. No algorithm, metric, baseline,
or seed policy changes here.

## Evidence

`tests/test_combined_circadian_checkpoint.py` compares uninterrupted
and fresh-model continuation across pre-sleep, accepted-sleep, and
rejected-sleep checkpoints on both backends. It checks future updates,
split draws, retry suppression, process RNG draws, and unchanged live
state after incompatible protocol/config/data/stage/progress/model/retry
or random state. The development log records the full quality gate.
