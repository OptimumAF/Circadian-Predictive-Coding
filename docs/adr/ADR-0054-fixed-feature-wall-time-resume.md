# ADR-0054: Carry the active wall-time deadline across CPU head resume

## Context

The fixed-feature wall-time benchmark gives each head the same training
deadline after shared feature extraction. P3.9b1 records elapsed time in
the circadian runner checkpoint, but a resumed segment previously would
receive a fresh full deadline. Saving a file also takes time that the
uncheckpointed baseline heads do not spend.

## Decision

The CPU wall-time circadian route stores the wall-time protocol and its
per-head budget in checkpoint identity. A resumed segment receives the
original budget minus previously recorded active training seconds.
Checkpoint snapshot/serialization/write time is paused out of this
active deadline and reported head `train_seconds`; resumed-process setup
and feature rematerialization are outside the same head training scope.
The runner saves the post-sleep state before checking whether the
deadline was reached during a guarded sleep event, so a restart cannot
repeat a committed or rejected event from a pre-sleep file.

Why this: the baseline heads spend their full budget on training work.
Charging only the circadian head for opt-in checkpoint I/O would change
its learning allowance. The report remains a per-head active training
measurement, not elapsed calendar time across two invocations.

## Alternatives and consequences

Resetting the deadline on resume would give the circadian head extra
training. Charging checkpoint I/O to its training budget would shorten
only that head's wake work. The cumulative active-time rule preserves
the declared learning budget, while actual wall-clock completion time
can be longer. Checkpointed elapsed times must not be pooled with
uncheckpointed end-to-end throughput measurements.

Process RSS and CUDA allocator peaks do not inherit this clock policy;
they need a separate cross-process definition. Memory-enabled and
capacity routes still reject or lack checkpoint entry points, and CUDA
resume remains unverified on this CPU-only host. P3.9b2b and parent
P3.9b stay open. No seed, guard metric, or baseline hyperparameter was
changed.

## Evidence

`tests/test_fixed_feature_checkpoint_resume.py` uses a clock advanced by
successful wake updates and by each file write. It compares
uninterrupted and resumed deadline, work counters, head state, and
reported active seconds after pre-sleep, accepted-sleep, and
rejected-sleep interruptions. A different budget rejects before head
mutation. A public-route fixture confirms checkpoint forwarding and
sealed final-test access. The development log records the full gate.
