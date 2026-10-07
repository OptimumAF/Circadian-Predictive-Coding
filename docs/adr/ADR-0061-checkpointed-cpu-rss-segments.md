# ADR-0061: Report observed CPU RSS by checkpointed process segment

## Context

The older fixed-feature memory protocols report one process's sampled RSS
start and peak for each head. After a checkpoint is resumed in another
process, a single start value no longer exists and the final process's peak
omits earlier observations. The checkpoint previously held learning state but
no memory observations. A CPU-only route can measure and validate RSS across
actual process restarts; this host cannot validate CUDA allocator state.

## Decision

CPU checkpoint runs with memory enabled use distinct protocol IDs for fixed
epoch, wall time, and fixed-width capacity. The capacity route requires the
explicit `checkpoint_memory=True` opt-in; its existing default checkpoint
route remains memory-free. The generic three-head routes use their existing
`measure_memory=True` option with a checkpoint store. Legacy runs without a
store retain their prior protocol IDs and report fields.

`ProcessRssSegment` records PID, process RSS at the start of one head training
invocation, maximum *observed* absolute RSS, sample count, and the 5 ms
periodic interval. The sampler also reads at entry, each checkpoint boundary,
and exit. Its scope begins after feature-bank/head setup and includes wake,
guard, validation, and checkpoint persistence activity. It includes other
objects resident in the process; it is not a model-attributed allocation.

The circadian checkpoint stores all prior segments plus the current
invocation's observation immediately before each file save. A stopped
invocation contributes observations only through its last saved boundary;
work and RSS after that boundary are not claimed. On successful completion,
the report appends the final invocation's exit observation. Each baseline
head has one segment from the completing invocation, because baseline heads
retrain when the runner is resumed. For the circadian head,
`process_rss_start_bytes=None` represents the absence of a common process
baseline, `process_rss_peak_observed_bytes` is the maximum absolute peak of
its segments, and `process_rss_samples` is their sum. The typed segment tuple
preserves each process baseline and sample count. All CPU CUDA allocator
fields remain empty.

The checkpoint validator rejects missing, malformed, negative, or
wrong-interval segments before restoring the saved head or process RNG.
Protocol identity prevents capacity-only and checkpoint-memory files from
being exchanged. The wall-time route still excludes file I/O from its active
training deadline; this RSS observation may include persistence activity.

## Alternatives and consequences

Using only the last process's RSS would erase earlier observations. Summing
absolute RSS peaks would double-count unrelated process baselines. Subtracting
each segment's start would imply model-attributed bytes despite shared caches
and allocator reuse. The maximum absolute observed RSS plus explicit segments
is a descriptive process observation, not an equal-resource ranking. The
5 ms interval can miss brief peaks, and the save that commits an interrupted
segment can allocate after its captured observation. Callers must not compare
these values with legacy no-checkpoint memory runs to claim an algorithmic
memory gain. CUDA allocator semantics and actual device continuation remain
P3.9b2b2b.

## Evidence

`tests/test_checkpoint_memory_resume.py` runs interruption and resume in two
independent Python processes and checks both PIDs, persisted observations,
aggregation, baseline segments, and capacity sleep. Same-process cases
compare wake/accepted/rejected continuation with an uninterrupted run,
preserve final-test ordering, reject corrupt and cross-protocol segments
before restoration, and verify fixed-epoch and wall-time protocol routes. A
controlled-clock case verifies the cumulative active wall-time budget while
two RSS segments are collected. `tests/test_process_memory.py` checks the
sampler snapshot contract. The development log records exact commands and
quality-gate outcomes.

## Subsequent decision

ADR-0085 adds CUDA allocator segments beside this RSS contract. It does not
change the CPU protocol IDs or the meaning of the RSS observations.
