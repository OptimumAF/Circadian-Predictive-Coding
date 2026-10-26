# Bounded catalog TRAIN pass count — R5.6-CLM-F.P / issue148

The existing app fitter now accepts `passes=8` through32, as a strict integer.
Eight remains the default; the complete workflow CLI remains fixed at8. A fresh
native learner must have the exact selected update allowance, twelve-row limit,
zero restores and no prior or uncertain training work. One original budget,
delayed-label inbox and logical clock cover that entire run. Nondefault reports
record `requested_passes`; default reports keep their schema.

The original limit check compared the live budget object against its own alias.
An independent copied limit value now supplements original object identity.
In-place edits and object replacement both refuse before another update. This
fix changes future validation; it does not rewrite frozen historical evidence.

## Scope and source

- Production change: `src/app/clm_workflow_training.py` only; no backend/core
  mathematics, CLI, data, encoder, optimizer or scheduler implementation changes.
- New tests: `tests/test_clm_workflow_passes.py`, deterministic literal2D heads.
- Evidence: `artifacts/runs/clm-bounded-passes-20261025/` including the before-file,
  eight SHA256 source pins, platform work/parent receipts, JUnit and logs.
- Source HEAD09ff811f with existing unrelated dirty/untracked work preserved.
  Review used the saved actual before-file because this app file is untracked.

The lead reviewed the actual source diff, tests, measurement/launch procedure and
results. This is **self-review**, not independent review. No agents were used.
Historical native/core evidence is reused only where its source and assumptions
remain unchanged. No old native run, saved review or unsaved heads were replayed.

## Executed behavior

Both platforms passed23 distinct cases with zero failures/errors/skips. Tests
reject invalid counts before native access and mismatched policy/row/restore
authority before work. Default8 and explicit8 have identical reports and full
native state, including gradients, modes, optimizer/scheduler state and RNG.
The32-pass app result equals32 ordinary native TRAIN calls under the **same
32-step policy**, including work and full native state. Caller RNG and original
loaded heads are preserved. Spent owners cannot reenter using a fresh report.

Each selected run commits exactlyN receipts at ticks1,3,...,2N-1, with12N TRAIN
queries,5N candidate rows and60N logits. After one paid update, separate tests
change limits in place, replace limits, replace the budget clock, change its
start, advance the logical clock, replace the record, alter work or alter native
policy. Each refuses before a second update and retains the preceding trusted
receipt. A controlled save failure retains that prefix and refuses reentry.

These tests compare8 and32 at the endpoints of the accepted interval; they do
not train every intermediate count. Existing episode/predictor/default full-flow
checks were not repeated: those sources are unchanged and this scope qualifies
the fitter directly. Repository-wide tests and release gates remain unrun.

## Commands, costs and failures

All fixture work used one CPU thread, offline cached runtimes, tiny literal
features and zero physical random initializers. Commands, in order:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-bounded-passes-20261025/prepare.py
.venv/Scripts/python.exe -m ruff format src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
.venv/Scripts/python.exe -m ruff check src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
.venv/Scripts/python.exe -m mypy --cache-dir artifacts/runs/clm-bounded-passes-20261025/mypy-win --explicit-package-bases --follow-imports=silent src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
```

Initial format/lint passed; mypy failed seven test-only inferred-type annotations.
No fixture ran. Corrected annotations then passed:

```powershell
.venv/Scripts/python.exe -m ruff format --check src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
.venv/Scripts/python.exe -m ruff check src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
.venv/Scripts/python.exe -m mypy --cache-dir .mypy_cache/clm-bounded-passes-20261025-win --explicit-package-bases --follow-imports=silent src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
.venv/Scripts/python.exe -m mypy --platform linux --cache-dir .mypy_cache/clm-bounded-passes-20261025-linux --explicit-package-bases --follow-imports=silent src/app/clm_workflow_training.py tests/test_clm_workflow_passes.py
.venv/Scripts/python.exe -B artifacts/runs/clm-bounded-passes-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-bounded-passes-20261025/launch.py linux-1
```

Scoped statics cost27.966095s failed attempt +23.1402777s corrected attempt =
51.1063727s of the120s allowance. Linux-target mypy ran on Windows; native Linux
Ruff/mypy are unavailable and were skipped without installing anything.

| Observed per platform | Windows | Linux |
| --- | ---: | ---: |
| Tests passed / failed / skipped |23 /0 /0 |23 /0 /0 |
| Measured body seconds |22.023548 |22.004513415 |
| Parent seconds |22.968585200 |33.252871700 |
| Sampled peak RSS bytes, after Torch import |322482176 |339484672 |
| Native TRAIN / backwards |89 /89 |89 /89 |
| TRAIN query / candidate / logit values |1068 /445 /5340 |1068 /445 /5340 |
| Head objects / forwards / projected rows |246 /178 /1513 |246 /178 /1513 |
| Scheduler constructor initial steps |17 |17 |
| Returned native captures / tensor elements |182 /926510 |182 /926510 |
| Caller RNG returned elements |1006144 |1006144 |

Each platform stayed within its original120s/1GiB/96-update/1152-query/480-candidate/
5760-logit/384-head/256-forward/2048-projected-row/32-initial-step bounds. Returned
capture+RNG elements total1932654 of24M, plus2MiB other-copy and65536 literal
reserves. These counters are not total allocations or FLOPs. Torch import RSS
is not sampled. There were zero encoder/backbone/tokenization/Transformers,
serving dispatches, replay, probes, outer evaluation or pretrained experiments.
Both logs retain11 existing scheduler warnings from native copied optimizer
ownership; this patch does not change that math. No failed fixture or rerun.

The initial mypy cache breached the8MiB evidence-directory limit:47,159,246B
observed, including a47,136,768B cache database. `storage-failure.json` preserves
that failure. Only the owned cache was moved, with resolved in-workspace paths,
to ordinary `.mypy_cache/`. Prospectively declared separate128MiB ordinary
static-cache storage; combined caches measured94,319,066B. The retained stage
returns below8MiB and logs below512KiB; this does not retroactively pass the
original storage failure or renew any experiment allowance.

## Scientific interpretation and next action

This is implemented and passing bounded training behavior. It does not establish
that32 passes help the pretrained learner. A32-step OneCycle trajectory differs
from an8-step trajectory; the comparison is not dose alone, not continuation,
and not matched total compute. The prior8-pass result remains4/12→10/12 TRAIN
correct, with legacy/password ready rows wrong; its two completable native tasks
and external permission-denial control remain scoped evidence from issue147.
No generalization, retention, context recovery or circadian advantage follows.

The concrete dependency before fresh-task adaptation is whether this same
learner can fit the fixed known TRAIN decisions. Next freeze **one32-pass
pretrained TRAIN-only qualification**, using this existing fitter and existing
read-only TRAIN diagnostics, with a fresh matching policy, original nonrenewable
budget and predeclared physical/time/copy limits. Preserve all12 row margins,
especially both wrong ready targets. No serving episodes, held-out/final access,
new seed, checkpoint selection, further horizon search or baseline tuning.
Broader reliable learning, new-task/old-task/return outcomes, R5.1/R5.3/G5/R7 and
release requirements remain open. Live handoff stays in issue148/tracker18;
historical plans and development log remain frozen under current AGENTS.md.
