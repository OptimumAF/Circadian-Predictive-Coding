# ADR-0085: Report CUDA allocator peaks by checkpointed process segment

## Context

ADR-0061 defines process RSS observations for checkpointed fixed-feature
training. A CUDA restart also creates a new PyTorch allocator instance and
peak counter. Reusing the last process's peak would discard earlier work;
combining allocator starts or summing absolute peaks would imply a common
baseline that does not exist. ADR-0084 verifies CUDA learning and random
state continuation separately.

## Decision

CUDA checkpoint-memory runs use device-specific protocol IDs for fixed
epoch, cumulative wall time, and fixed-width capacity. They retain the
5 ms RSS observations from ADR-0061 and add one typed `CudaAllocatorSegment`
per head-training invocation. The segment records PID, canonical CUDA
device, allocated and reserved bytes at entry, and the allocator's absolute
allocated and reserved peaks. After feature-bank and head setup, the runner
synchronizes the device, reads both starts, and resets PyTorch's peak
statistics once. It synchronizes again before each peak read. Checkpoint
observations are taken immediately before file save; the last successful
save commits the interrupted process's observation. A completing invocation
adds a final observation. Baselines retrain on resume and therefore each
have one segment from the completing process.

The circadian report retains every process's segment, sets the aggregate
allocated start to `None`, and reports the maximum absolute allocated and
reserved peaks across its segments. The RSS tuple, maximum absolute observed
RSS, and sum of RSS sample counts remain separate. A CUDA checkpoint must
contain nonempty RSS and allocator tuples with matching lengths and PIDs.
The validator checks types, nonnegative starts, monotone peaks, reserved at
least allocated, and the expected device before restoring the head, retry
state, or process random streams. CPU and non-memory checkpoints retain
empty CUDA tuples. The old memory and CPU checkpoint protocol IDs keep their
existing meanings.

## Alternatives

- Sum absolute peaks: double-counts process baselines and shared state.
- Subtract each process's start: suggests head-attributed growth despite
  allocator reuse and shared cached features.
- Save only the final process's peak: loses interrupted work.
- Reset at every checkpoint: erases the high-water mark within one process.

## Consequences

The numbers describe PyTorch allocator observations for the selected device,
not total device memory, head-only use, or equal-resource algorithm rankings.
Shared objects coexist in each process. A file save or snapshot may allocate
after its pre-save observation; that work is included only if a later
boundary observes it. RSS polling can miss brief host-memory peaks. Wall-time
checkpoint I/O remains outside the active deadline even though memory
observations can include its allocations.

`tests/test_cuda_allocator_contract.py` checks synchronize/start/reset/peak
call order with a fake device. `tests/test_cuda_checkpoint_memory_resume.py`
uses the actual RTX 3080 under an external 120-second cap: fixed epoch,
fixed-width capacity, and deadline routes restart in a second Python
process; fixed epoch also compares an uninterrupted control's learning
hashes, outcomes, and non-timing counters. Malformed CUDA segments reject
before live head restoration. The existing ADR-0084 tests cover process
and local CUDA random-stream continuation across accepted and rejected
sleep. No baseline, seed, metric, or model setting changed for this gate.
