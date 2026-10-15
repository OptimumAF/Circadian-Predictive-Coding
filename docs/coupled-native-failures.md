# Native failures before and after committed receipts

This finite local control exercises real raw PC, matched neutral CPC and active
CPC updates through two in-process failure positions. It verifies complete stopped
candidate state and continued actor availability using existing runtime/inbox
semantics. It adds a fixed adapter and CLI, with no change to learning rules.

## Run and check

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_native_failures --describe
.venv/Scripts/python.exe -m scripts.run_coupled_native_failures
.venv/Scripts/python.exe -m pytest -q tests/test_coupled_native_failures.py
.venv/Scripts/python.exe -m ruff check src/adapters/coupled_native_failures.py scripts/run_coupled_native_failures.py tests/test_coupled_native_failures.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/coupled_native_failures.py scripts/run_coupled_native_failures.py tests/test_coupled_native_failures.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/coupled_native_failures.py scripts/run_coupled_native_failures.py tests/test_coupled_native_failures.py
```

Use `--platform linux` for the second platform type check. Description does no
model construction, native work, serving, filesystem or RSS sampling. An actual
invocation is separately budgeted; development allowances close after acceptance.
No arbitrary seed, failure count or budget knobs are exposed. Existing installed
dependencies suffice. OMP/MKL/OpenBLAS are set to one thread before NumPy import.

## Two fixed failure positions

Each family starts at seed23, input dimension2, width2/two hidden layers, raw
mean-field scaling, gamma1.5 and precision1. Two genuine warm steps use a fixed
synthetic binary batch. The candidate accepts one split to width4, preserving
origin and full native/CPC history. Its original budget permits one update and
two consolidation attempts; only split-1 is spent. Actor A remains width2.

| Mode | Fault injection | Completed updates | Applied receipts | Owner |
| --- | --- | ---: | ---: | --- |
| uncertain | Actual native delegate updates, then wrapper raises before returning its diagnostic | 0 | 0 | Stopped |
| committed | Native update returns; observer raises at completed after original budget and receipt commit | 1 | 1 | Stopped |

Why this distinction: native work can have occurred without an acknowledged
completion. The uncertain case records physical work and retains the stopped
native state rather than pretending the original completed-update counter is1.
The committed case retains the receipt and count. This observer RuntimeError is
different from the documented late `ToyExecutionStopped` resource boundary, whose
known completed work can remain accessible. No clocks or budget data are changed
to fabricate a resource failure.

The wrapper delegates prediction, state, restore and independent forking to the
existing structural native learner. It is confined to this explicit engineering
fault scenario. The observer reads the real synchronous NativeUpdateOrigin.
Stages are `started, uncertain` or `started, completed, committed_failure`.
At every stage it serves the primed actor cache while the candidate gate is held;
the full actor bundle and cached prediction remain unchanged. Terminal reporting
does not throw or hide the primary injected error.

Full raw/CPC state advances count2->3 and remains observable as trusted local
forensics even if no applied receipt exists. For each family the final native
state is exactly equal across the two modes: the same native update ran, and the
accounting boundaries differ. PC/neutral raw parameters are matched separately;
their complete policy/chemistry metadata is not claimed equal.

## Stopped admission and scope

After either failure, four operations must refuse before native or payload work:
training retry, another consolidation, public candidate snapshot and new label
registration. Native/actor/inbox/cursor/budget state, spent IDs and revision4 stay
unchanged. The rejected consolidation callback and poisoned target copy stay at0.
Actor serving remains available after failure and after these probes.

Forensic state is read synchronously from the original private candidate solely
inside this trusted local demo. It is not a public resume, fork, restore or
promotion capability. Both stopped owners keep their original budget object and
limits; object identities in reports are process-local evidence, not authority.

```text
src/adapters/coupled_native_failures.py  fixed fault wrapper and orchestration
scripts/run_coupled_native_failures.py  describe/run and bounded JSON
tests/test_coupled_native_failures.py   independent fake/native behavior checks
```

Reports retain complete finite native/serving/inbox/cursor states, observer
events, receipts, refusals and work/resource observations. Output is capped
at2MiB including newline bytes, body time at60seconds and boundary-observed RSS
at512MiB. These are local engineering controls, not continuous hard memory bounds
or indefinite service guarantees. Repeated fixture warm-up makes no unseen-label
or scientific learning claim. No final evaluation data, promotion, automatic
recovery, native retry, quota refund or renewal occurs.

See ADR0283 and `artifacts/runs/coupled-native-failures-20261015/` for frozen
scope, results and review. Extend with a separately bounded scenario using the
existing fault/observer boundaries, preserving the distinction between physical
work, completed accounting, uncertain state and applied identity.
