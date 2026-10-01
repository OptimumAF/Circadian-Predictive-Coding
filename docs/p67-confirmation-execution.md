# P6.7b2c joint confirmation execution

Date: 2026-09-30. The fixed scope remains all six families, fifty distinct
reserved sources, sixty family-seed instances and 560 cells. P6.7a settings,
roles, controls, budgets and all old scientific pins remain unchanged.
Unscored execution and independent final scoring have separate gates.

## Structure and boundary

```text
src/app/continual_confirmation_execution.py   pure request/work/resource checks
src/infra/continual_confirmation_io.py        finite JSON, source pins, exclusive files
src/infra/continual_confirmation_runtime.py   live resource and optimizer observation
scripts/run_p67_confirmation_training.py      child process, publication, readback
tests/test_continual_confirmation_execution.py
tests/test_continual_confirmation_runtime.py
tests/test_p67_confirmation_training_cli.py
```

The adapter validates the exact saved scope bytes and independently repeats
the complete twelve-bundle/twenty-usage-file inspection. A strict request
binds the complete resolved manifest, source map, adapter bytes, command,
Python/NumPy/platform/processor metadata and UTC start time before launch.
The child checks that saved request and current identities before data;
the parent and independent reader check them again.

The original selected maps plus explicit additional pins cover 78 source
files; the adapter itself is bound separately. A test traverses all static
local imports, including package files, and verifies the exact 79-file
closure without ignored artifacts. Conditional unused Torch helpers are
included in this conservative local code closure; this remains a NumPy
experiment, with no CUDA or cross-host/version portability claim.

Fixed saved scope SHA:
`622feead54f155521928341151c8496c23c02a54b355b1b3b1e0f68b76c5772f`.
Execution source-map SHA:
`b8e6ea624228c7422b61bf48fc736cd187893eedd6ce1fd49558f6b11bbd3735`.
Adapter SHA:
`e6bafe8bc6c7e2199a578b26ff991a45fbbf4d45085e760fd61d0f9417250445`.
These identities are frozen before the first reserved builder.

## Actual work and resource gates

A scoped observer wraps original baseline `train_epoch` and circadian
`_run_training_step`, the shared wake/replay optimizer seam also inherited
by parent controls. Successful calls and attempts live outside model
snapshots. Every rejected replay execution remains counted. The original
methods are restored after success or exception; no model state or setting
is added or changed by measurement.

The complete pure validator independently derives every saved family cost.
Observed totals and model-kind partitions must match those facts. The worker
stops before update 16,001, at 600 s, or when observed peak RSS exceeds
512 MiB. RSS starts before binding/training, samples every 5 ms with explicit
boundary samples, and includes held copies, validation and full result
serialization. Parent `subprocess.run` also bounds the whole child to 600 s.
The sampler retains an earlier peak after current RSS falls. Missing or
invalid telemetry fails. Brief peaks can be missed; this is observed RSS,
not an OS allocation limit. Stdout/framing and parent publication remain
outside the sampled section as prospectively declared.

All live held role, parameter, full circadian and selector state are checked
after training, independent JSON validation and serialization. Source pins
are rechecked before return/publication. Pure hashes alone cannot reconstruct
unseen tensor values or chemical moments; source-bound live capture and
complete exact repeated training remain required.

## Exclusive artifacts and failure behavior

Each new output directory gets `confirmation-train.request.json`, then a
verified `confirmation-train.result.json` and `confirmation-train.audit.json`.
Any claimed failed invocation gets `confirmation-train.failure.json`.
An exclusive `confirmation-train.claim` prevents cooperating concurrent
writers; all occupied bytes are preserved. A request collision from a writer
outside that claim is also preserved without our failure marker.

Request/result byte digests are derived from intended deterministic encoding
before writing and compared with actual stored bytes. Altered bytes cannot
receive a success audit. Failure during publication may leave an incomplete
result; the failure marker, remaining claim or missing/forged audit makes
independent readback refuse it. A failed request write never launches a child.

## Correctness evidence before reserved execution

The latest complete related gate passes **444 tests in 76.81 s**, zero
skipped, including **108 new tests**. Ruff, seven-file format and mypy
387-file checks pass. All six first development seeds and four forced
rejection families exercise counters/facts; schedule/combined rejected
replay counts are 12/72. Late last-family A parameter, B selector RNG and
role-label corruption are refused after independent validation.

Request/scope/source/configuration/type/unknown/nonfinite/duplicate fields,
occupied/colliding paths, child exit/timeout/cancel, malformed/missing/forged
results, observed work/model-kind/time/RSS, and request/result/audit write
failures are checked before any reserved builder. Three publication
regressions were reproduced and repaired before source freeze. IO success
fixtures delegate the scientific verifier to a metadata spy; those fixtures
are not a 560-cell training claim. Clean tests need no ignored artifacts.

Commands and intermediate failures/repairs are in `docs/development-log.md`.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_execution.py tests/test_continual_confirmation_runtime.py tests/test_p67_confirmation_training_cli.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_execution.py src/infra/continual_confirmation_io.py src/infra/continual_confirmation_runtime.py scripts/run_p67_confirmation_training.py tests/test_continual_confirmation_execution.py tests/test_continual_confirmation_runtime.py tests/test_p67_confirmation_training_cli.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

## Budgeted local workflow and verified execution

After c1/c2 correctness gates, use separate new directories for both complete
unscored runs. The CLI has no seed, metric, baseline or budget override.
`--scope-file` must name the exact pinned bytes. Private `--worker` requires
the saved request; there is no fixture or partial-scope production flag.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --output-dir artifacts/runs/p67-confirmation-train
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --output-dir artifacts/runs/p67-confirmation-train-repeat
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --read-only --output-dir artifacts/runs/p67-confirmation-train
.\.venv\Scripts\python.exe -m scripts.run_p67_confirmation_training --read-only --output-dir artifacts/runs/p67-confirmation-train-repeat
```

Both full processes and public independent readbacks now exit 0. Each retains
all 560 cells/sixty rows, with exactly identical 134,554,378-byte result files
at SHA `3d85c60627de63769d0f0fc0bf5ec781c77d50468673dab466d8bbe28089e547`.
Each executes 15,210 updates, including 46 rejected replay updates. Observed
worker time is 32.50/32.65 s and peak RSS 462,479,360/462,348,288 bytes under
unchanged limits. Every scope/source/request/derived-work/observed-resource
and complete artifact identity passes independently. See the complete
[training results](p67-confirmation-training-results.md) and development log.
C3/c/b2/b meet their unscored criteria. Resource and request/audit durations
remain observed metadata outside deterministic results. P6.7, the original
matrix and final/analysis criteria remain unfinished; scoring still requires
the now verified P6.11a predeclared contract and separately frozen P6.7c release.
Extend execution through separate protocols and reviewed source pins; never
revise this frozen scope after a favorable, negative or resource outcome.
