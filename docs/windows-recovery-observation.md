# Windows recovery host observations

## Boundary and structure

```text
src/core/recovery_observation.py        bounded values and observation port
src/infra/windows_process_handles.py   documented API bindings and registered handles
src/infra/windows_recovery_observer.py time/RSS/owner observations under live anchor
tests/test_windows_recovery_observer.py deterministic API faults
tests/test_windows_recovery_process.py  one bounded real worker exit
```

Inputs are a trusted coordinator's retained process handle and, optionally, a
previously registered worker handle. Outputs are exact bounded process identity,
epoch, interrupt time in nanoseconds, current/observed peak absolute RSS and a
registered worker's alive/ended observation. There is no payload reader, native
model work, transactional storage, owner lease, `RecoveryFence` construction or
automatic recovery/retry. Core imports no outer layer; infra implements the inner
`RecoveryObservationPort`.

## Supported anchor policy

The coordinator registers its own live process as an epoch anchor and retains
that kernel handle across worker loss/restart. Each worker must also be registered
while alive, before work begins. Registration records PID and creation FILETIME,
opens a noninherited query/synchronize handle and verifies it is not signaled.
An expected identity mismatch refuses registration and closes the new handle.

Why this: a retained kernel handle refers to the registered process object even
after its exit. Looking up a dead worker by PID or assuming a timeout proves death
can inspect a reused PID or fail to find the original object. Missing/inaccessible
registration is an error, never an ended-owner observation. [Microsoft process
handles](https://learn.microsoft.com/en-us/windows/win32/procthread/process-handles-and-identifiers),
[OpenProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess),
[GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes).

The epoch identifier binds the supported clock policy and original anchor
identity. A live original anchor establishes the supported same-boot interval;
it is **not an OS boot GUID** or a certificate constructed from caller metadata.
This policy supports worker restart while that original coordinator survives.
Coordinator loss, reboot and recovery without retained registration remain
unfinished; the original R3.5b2 process-restart acceptance is preserved.

The composition root must trust the actual coordinator, original registration
and platform API bindings. The private injected kernel and RSS reader used by
tests confer no production provenance. Raw public handle construction and handle
subclasses are rejected. Python private-field mutation, malicious same-user code,
OS privilege compromise, cloned storage or forged process facts are not attested.

## Clock, RSS and failure behavior

`QueryInterruptTimePrecise` returns the OS interrupt-time count in 100ns units;
the adapter multiplies by100 without resetting its origin or using wall time.
The biased interrupt timeline includes sleep/hibernation; the unbiased API is not
used. This is distinct from the existing arbitrary `ToyBudgetSession` clocks:
future durable session creation must bind this policy from its original start,
not convert an unsupported clock retrospectively. [Microsoft precise interrupt
time](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryinterrupttimeprecise),
[interrupt-time policy](https://learn.microsoft.com/en-us/windows/win32/sysinfo/interrupt-time).

Bind the precise function through the documented realtime contract
`api-ms-win-core-realtime-l1-1-1.dll`. The first real guide found it absent from
this host's `kernel32` export table; that failure and a bounded export diagnosis
are preserved. This keeps the same precise function and100ns units;it introduces
no lower-precision/alternate-clock fallback. [Microsoft API-set table](https://learn.microsoft.com/en-us/uwp/win32-and-com/win32-apis).

The observer checks the anchor before and after time/RSS/owner queries. Time must
be positive, below2**63 nanoseconds and nondecreasing within the observer. RSS
must be a positive exact integer below2**63; unavailable observations never
become zero. It retains the larger sampled absolute RSS through its lifetime.
Durable carry of original start/peak/ceilings remains the coordinator's job.
Observed RSS is not hard allocator preemption or cumulative allocated bytes.

`WaitForSingleObject(handle,0)` is nonblocking. Only the process object's signaled
and nonsignaled states produce ended/alive respectively. Failure, abandoned or
unknown values are errors. The observer does not terminate processes.
[Microsoft zero-wait states](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject).

Bad time/RSS/anchor/API/owner facts disable that observer permanently. No automatic
retry clears the failure. Concurrent/nested observations refuse; per-handle locks
coordinate close with queries. Every failed partial registration closes its owned
handle. Close failure is reported and the uncertain handle cannot be reused; a
failed native close is not falsely certified as successful cleanup.

Every process identity and returned value is exact and bounded. The absence of a
registered prior worker returns no death fact. An alive/ended observation alone
does not prove a single live replacement: independent transactional compare-and-
swap fencing and live lease rechecks through publication remain required.

## Current-process example

This opens and closes only query/synchronize handles to the current process.
It makes no disk, model or authority changes and registers no replacement owner.

```python
import os
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver

api = WindowsRecoveryApi()
with WindowsProcessHandle.pin(os.getpid(), api=api) as anchor:
    observer = WindowsRecoveryObserver(anchor)
    first = observer.observe()
    second = observer.observe()
    assert first.clock_epoch == second.clock_epoch
    assert first.now_ns <= second.now_ns
    assert second.peak_rss_bytes >= second.rss_bytes > 0
    assert second.previous_owner is None and second.previous_owner_ended is None
```

## Verification and safe extension

From the existing environment and repository root:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_windows_recovery_observer.py tests/test_recovery_admission.py -q
.venv\Scripts\python.exe -m mypy --platform win32 --no-incremental
.venv\Scripts\python.exe -m mypy --platform linux --no-incremental
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m ruff format --check src/core/recovery_observation.py src/infra/windows_process_handles.py src/infra/windows_recovery_observer.py tests/test_windows_recovery_observer.py tests/test_windows_recovery_process.py
```

The separate real process test starts exactly one owned Windows child, waits for
a bounded JSON observation, registers it while alive, brackets its interrupt
time with parent observations, requests intentional `os._exit(17)` and verifies
the retained handle has ended. Pipe reads/communication/reader join have5second
bounds, parent/child observed RSS caps512MiB. It cleans its owned child, pipes,
reader and adapter handles. It performs zero native training or predictions.
`CIRCADIAN_RECOVERY_PROCESS_RECEIPT` is an optional test telemetry destination,
opened exclusively; it is not runtime configuration or recovery authority.

The one-shot local capture is reserved only after current fake/type/static/source/
guide gates. The first capture failed:the venv launcher PID differed from the
physical Python worker PID. The strict identity assertion is preserved;the
harness now chooses the existing base interpreter directly and checks its exact
Python version before reporting an observation. These observation modules import
only repository code and standard-library dependencies;this does not substitute
base-environment model execution for the existing venv. No package installation
or interpreter upgrade occurs. Selection/refusal controls pass. The separately
declared direct-worker successor now passes strict identity,time,exit,RSS and
cleanup checks;R3.5b2b is accepted within its unchanged scope. Evidence:
`artifacts/runs/r35b2b-direct-worker-20261007/`. Full durable model recovery and
transactional ownership remain unfinished.
Both known launcher/payload PIDs were absent after cleanup. No second worker
capture runs under the spent original scope. General test invocation is
`python -m pytest tests/test_windows_recovery_process.py -q`;
the Windows case is skipped on unsupported hosts. The configured Linux type
target runs on the local Windows interpreter and is not a Linux runtime result.

Next implement independent transactional coordinator state, original resource and
attempt charging, registered owner/anchor bindings and compare-and-swap fencing.
Keep a live authority/fence through future byte verification, complete lifecycle
codec and atomic publication. Actual durable model crash/corruption/recovery,
coordinator-loss recovery, stable actor availability and consumed-ID/quota
preservation remain unfinished. No observations substitute for full R3.5b2/R3.7.


### Direct-worker observation successor — 2026-10-07

R3.5b2b scoped validation now passes strict actual worker identity/time/exit/RSS/cleanup under original coordinator anchor,with unchanged production/test source and280current cases. Prior failed launcher scope/costs retained. No interfaces/dependencies change;transactional independent authority/CAS/live lease/native codec/recovery remain unfinished. See [guide](../windows-recovery-observation.md) and successor evidence.
