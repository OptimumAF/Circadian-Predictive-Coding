# Native checkpoint continuation after coordinator loss

The fixed Windows example completes a native candidate update A → B, loses that
coordinator, restores B in a replacement, and completes B → C. Complete native
state, energy and predictions must equal an uninterrupted reference for each of
PC, neutral CPC and active CPC. The surviving actor remains at A.

## Run

```powershell
.\.venv\Scripts\python.exe -m scripts.run_native_transfer --describe
New-Item -ItemType Directory -Path .\artifacts\my-native-transfer
.\.venv\Scripts\python.exe -m scripts.run_native_transfer --run --directory .\artifacts\my-native-transfer
```

Use a new private directory. Existing family directories are refused. Windows,
the existing base Python interpreter and installed repository dependencies are
required. No environment variables are required; numerical CPU thread counts
are fixed to one. This runs three fixed families, 18 native training calls,
72 energy evaluations and 12 directly owned child processes. It performs no
downloads, tuning or seed selection. Run output is compact JSON containing full
reference payloads, completion proofs and authority transitions.

Each family retains three checkpoint files, one SQLite authority database and
its writer lock. Original per-family limits are two admitted updates, four
checkpoint attempts, 655,360 charged copy bytes, 60 seconds and 512 MiB observed
RSS. The transfer and restore consume the same original allowance.

## Ownership and transitions

1. Warm the actor twice to A. A separate reference restores A, then trains B and
   C using seed 23, dimensions 2 → 3 → 2 → 1 and fixed train-only data.
2. The original physical coordinator adopts independently known A. Persist its
   update/copy/attempt reservation before checkpoint IO, model construction or
   native learning. Save and independently reload complete B.
3. The parent validates the actual acknowledged completion proof against its
   retained pre-update authority and complete reference B. Retain B and the live
   original process handles in the existing Windows supervisor before crash17.
4. After the original coordinator and worker end, observe the directly owned
   replacement pair. Commit the existing explicit transfer CAS before adoption.
5. Charge restore's copy/attempt cost before loading B or constructing its model.
   Restore every parameter, precision and CPC policy/chemistry/gate/wake field
   before prediction or further training. Perform and validate B → C.
6. Refuse another update before callbacks at the original quota. Persist stopped,
   clean authority with two admitted/completed updates and unchanged original
   start, limits and other manifest fields. Reap all owned processes.

PC and neutral CPC parameter frames must match exactly at A/B/C. Neutral CPC
still has genuine chemistry; CPC wake counts progress 2 → 3 → 4. Actor A's full
bytes and predictions remain unchanged at B, coordinator loss and C.

## Module boundaries

```text
scripts/run_native_transfer.py                 CLI
src/adapters/native_transfer_example.py        fixed protocol and owned lifetimes
src/adapters/native_transfer_worker.py         bounded trusted child commands
src/adapters/native_transfer_session.py        charged native IO/update/restore
src/adapters/native_transfer_protocol.py       constants and strict completion wire
tests/test_native_transfer_session.py          DI boundary and refusal controls
```

The session composes the existing application coordinator and infrastructure
codec/store/journal/Windows ports. It creates no journal, launches no processes
and publishes no actor. The journal subclass only retains an actual completion
after the existing SQLite implementation acknowledges it. Pure transition rules
remain in core. Constructor reports may advance sequence/time/RSS; adoption
retains that actual record as the independent floor for the next admission.

Why this: a completed file and a newer database record cannot establish restart
authority. The continuously surviving supervisor independently retains validated
completion and physical registrations. JSON validates data; trusted local code
and the private owned channel supply its explicit trust boundary.

Failures permanently stop the session under existing uncertainty rules. A missing
completion acknowledgment never becomes success from disk contents. Ambiguous
transfer can use only the existing terminal reconciliation; its old witness
cannot overwrite a newer child state. There is no automatic resend or renewal.

## Validation and extension

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_native_transfer_session.py tests/test_windows_transfer_supervisor.py tests/test_recovery_completion.py
$nativeTransferPaths = @(
  'src/adapters/native_transfer_protocol.py',
  'src/adapters/native_transfer_session.py',
  'src/adapters/native_transfer_worker.py',
  'src/adapters/native_transfer_example.py',
  'src/adapters/native_transfer_pause.py',
  'src/adapters/native_transfer_interruption.py',
  'scripts/run_native_transfer.py',
  'tests/test_native_transfer_session.py',
  'tests/test_native_transfer_interruption.py'
)
.\.venv\Scripts\python.exe -m ruff check @nativeTransferPaths
.\.venv\Scripts\python.exe -m ruff format --check @nativeTransferPaths
.\.venv\Scripts\python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @nativeTransferPaths
```

Repeat the type command with `--platform linux` for import/type portability;
physical execution remains Windows-only. The 194 focused tests passed using injected fake models
and real bounded files/SQLite, with zero actual native or process work. The one
physical run passed in 6.6346 seconds. Commands, counters, raw results and review
are in `artifacts/runs/native-coordinator-transfer-20261015/`.

This proves fixed native checkpoint continuation under a continuously surviving
supervisor. Full managed inbox/lifecycle/actor authority restoration, supervisor
loss, power loss, long runs and comparative learning benefit remain unverified.
An optional replacement interruption control is described below. See ADR-0286.

## Replacement interruption before completion

```powershell
.\.venv\Scripts\python.exe -m scripts.run_native_transfer --describe --interrupt-after-train
New-Item -ItemType Directory -Path .\artifacts\my-interrupted-transfer
.\.venv\Scripts\python.exe -m scripts.run_native_transfer --run --interrupt-after-train --directory .\artifacts\my-interrupted-transfer
```

This fixed mode follows the same A → B transfer and charged B restore. A trusted
model wrapper pauses immediately after the replacement's one real native train
returns. It emits complete C telemetry and the actual pending authority, then
waits on its bounded private channel. It can never return a training result to
the session. The parent validates the expected full C and original reservation,
checks the owned replacement is live, terminates and reaps it, then records its
genuine ended observation from the surviving supervisor.

C is diagnostic data: the session has neither saved it nor completed the update.
The existing terminal reconciliation stops only the independently received
pending record. Its last child time/RSS are retained exactly. Later supervisor
observations remain separate, because they are not observations by that child.
Final authority retains B with two admitted/one completed updates, 655,360 copied
bytes, four attempts, stoppedTrue and uncertain_workTrue. No retry is performed.

Pending and terminal adoption/transfer witnesses, a guessed C floor and the stale
original transfer witness must all refuse. A/B files and actor A stay unchanged;
no C file is created. Each family still uses four owned children and the same
native update allowance as the successful mode. All children, including the
terminated replacement, are reaped. This is an explicit local interruption, not
proof of arbitrary crash or power-loss recovery.

The added paths are `src/adapters/native_transfer_pause.py` (trusted native
delegate) and `src/adapters/native_transfer_interruption.py` (strict telemetry,
owned-peer control and existing stopped-only persistence). Focused controls are
in `tests/test_native_transfer_interruption.py`. Preserve exact original
accounting and independent witnesses when extending either path; never infer a
completed update from C telemetry or a newer disk record. See ADR-0287 and
`artifacts/runs/native-transfer-interruption-20261015/` for validation evidence.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_native_transfer_interruption.py tests/test_native_transfer_session.py tests/test_recovery_reporting.py
```

The 170 focused cases passed with zero actual model/native/process work. The one
physical interruption run passed in 7.8047 seconds with 12 owned children, all
reaped, 18 native updates and 72 energy evaluations. Twelve A/B/database/lock
files were retained; no C file was created. Exact commands and independent
source/actual review are preserved in that artifact directory.

Final inspection tightened the interruption child prediction guard from the
inherited two calls to the intended one. Seven additional pure guard/routing
controls passed with zero native work. The saved physical run performed one
prediction per child; the separate controls establish the corrected guard's
before-work enforcement. Normal replacement mode still allows two predictions.
