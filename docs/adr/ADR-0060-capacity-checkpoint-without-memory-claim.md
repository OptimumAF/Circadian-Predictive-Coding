# ADR-0060: Resume fixed-width capacity control without a memory claim

## Context

The original fixed-width capacity route always samples process RSS while each
head trains. A checkpointed circadian run can stop in one process and continue
in another. The current checkpoint records learning state and active time, but
does not record RSS samples from each segment. Reporting the last process's
peak as the whole resumed run would change the meaning of the existing memory
protocol. Absolute RSS also includes shared caches and other resident objects.

## Decision

Opting into a trusted local checkpoint on
`run_three_head_fixed_width_capacity_benchmark` selects a distinct
`vision_three_head_fixed_width_capacity_checkpoint_v1` protocol. It keeps
the same equal-width configuration gate, shared feature bank, learning rules,
forced guarded sleep, and pre-final-test capacity invariant. Its result has
`memory_telemetry_enabled=False`; RSS and CUDA allocator fields are absent.
The checkpoint binds the distinct protocol ID, so an ordinary fixed-feature
file cannot be resumed as a capacity-controlled run.

Without a checkpoint store, the existing
`vision_three_head_fixed_width_capacity_memory_v1` route and its sampled
memory reports are unchanged. The default checkpointed route remains CPU-only
and uses the existing fixed-feature head/retry/report continuation contract.
An explicit `checkpoint_memory=True` opt-in uses the separate per-process RSS
protocol specified later in ADR-0061; it does not alter this default.

## Alternatives and consequences

Taking the maximum of peaks from unknown process segments would hide missing
samples and different process baselines. Copying only the terminal process's
RSS into the old report would imply a whole-run observation that was never
made. A separate capacity-only protocol lets the structural and guarded-sleep
invariants be checked while the memory contract is specified separately.
ADR-0061 adds typed segment observations and actual process-boundary evidence
before exposing a resumed CPU RSS report. CUDA allocator and random-state
continuation remain unverified.

## Evidence

`tests/test_matched_head_capacity.py` interrupts the public route after a
wake batch, before sleep, and after accepted or rejected sleep. Resumed runs
match uninterrupted trained-head hashes, non-timing reports, capacity
metadata, sleep counters, and next process draws. A sealed final-test loader
rejects early access until capacity verification in the resumed run. Ordinary
fixed-feature and capacity-only files reject in either cross-protocol direction
before restoring a head. The existing memory-enabled capacity
test continues to check its original protocol and RSS fields. The development
log records the full test and static-check commands.
