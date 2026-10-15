# Native cache transactions

This fixed local integration exercises the existing serving controller using
genuine coupled PC, matched neutral CPC and active CPC models. It demonstrates
complete state preservation through cache failures, promotion and rollback.
It does not compare learning benefit or establish durable recovery.

## Run and test

From the repository root with the installed environment:

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_cache_transactions --describe
.venv/Scripts/python.exe -m scripts.run_coupled_cache_transactions
.venv/Scripts/python.exe -m pytest -q tests/test_coupled_cache_transactions.py
.venv/Scripts/python.exe -m ruff check src/adapters/coupled_cache_transactions.py scripts/run_coupled_cache_transactions.py tests/test_coupled_cache_transactions.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/coupled_cache_transactions.py scripts/run_coupled_cache_transactions.py tests/test_coupled_cache_transactions.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/coupled_cache_transactions.py scripts/run_coupled_cache_transactions.py tests/test_coupled_cache_transactions.py
```

Use `--platform linux` for the second platform type check. Description performs
no model construction, training, serving, RSS sampling or filesystem work. The
actual demo is a separately budgeted invocation; the original development run
and its unused allowance are closed after acceptance. No arbitrary seed, quota
or threshold flags are exposed. No new dependencies or environment settings are
required; OMP/MKL/OpenBLAS are limited to one thread before NumPy import.

## Fixed behavior

Each model family begins with seed23, mean-field raw scaling, input dimension2,
width2 and two hidden layers, precision1 and gamma1.5. Two genuine warm SGD steps
precede independent actor/candidate forks. The candidate accepts a real split to
width4 and one delayed update under its original single-update budget. Its
origin, split history, raw parameters and complete CPC chemistry/policy remain
observable. CPC wake count advances2->3; actor remains at2.

Actor cache TTL is three ticks with two entries. Fixed requests demonstrate a
hit, insertion-order eviction, revisit miss and expiry exactly at the boundary.
A backward tick is refused without changing the complete bundle. Promotion
guards use separate `inner_guard` IDs and data from training; final-test role,
future arrival and training overlap are refused before native evaluation.
Role-refusal probes use these same synthetic fixtures; no final evaluation data
is loaded.

The valid guard uses real one-minus-Brier utility, native prediction latency,
canonical JSON state bytes and binary action checks. Wide bounds (utility0,
gain-1, retention1, latency60seconds, state65536bytes) are fixed engineering
controls. Why this: transaction correctness must not depend on which model wins
a tiny utility comparison. These settings provide no deployment recommendation
or scientific claim.

Preparation preserves the actor's entire bundle. Commit publishes candidate
model/version and new metadata with the original cache configuration, an empty
cache, reset cache tick and generation1. Cold then cached serving uses the new
model. Rollback restores the exact saved model/version/configuration/metadata/
cache/tick, advancing generation to2. Later hits and expiry use that restored
cache; consumed ticket and rollback handles refuse without mutation. Candidate
state, cursor and original spent budget remain unchanged by these transactions.

## Structure and limits

```text
src/adapters/coupled_cache_transactions.py   fixed native orchestration/report
scripts/run_coupled_cache_transactions.py    --describe/run and bounded JSON
tests/test_coupled_cache_transactions.py     independent behavior controls
```

The adapter composes existing app/core ports; it does not add a serving protocol,
change native learning math or manage persistent authority. Reports retain a
finite list of complete serving observations and refusal messages, candidate
states/cursors and guard evidence. JSON output is capped at2MiB; body wall time
at60seconds and boundary-observed RSS at512MiB. Boundary samples are not a hard
memory or continuous monitoring guarantee. Full recovery and indefinite workload
criteria remain open.

For extensions, add a separate prospectively bounded scenario and preserve this
fixed protocol. Reuse the existing serving controller and native builder; keep
guard data disjoint from training and final evaluation. See ADR0281 and
`artifacts/runs/coupled-cache-transactions-20261015/` for scope and saved evidence.
