# R2.1-CTX.W — Atomic class-weighted BP array port

[Issue277](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/277)
delivers the callable weighted-array prerequisite for a conventional BP training
control. Full goal and broad parents18/21/23 remain open. GitHub is the live
tracker; historical plans/log stay frozen. HEAD0aae1d6 and all inherited270–276
changes, failed/negative/SPENT captures were preserved. No install, dependency,
global setting, commit/push, old scientific read or new scientific cohort.

## Implemented and verified scope

New `src/adapters/torch_array_weighted.py` exposes one explicit array operation
on the existing native BP owner. It reuses bounded class-weight validation and
the existing `_restore_components`/`_state` detached optimizer staging. Weighted
CE is sum(target-weight*CE)/sum(target-weight); all-one dispatch uses the original
unweighted kernel for exact continuation. Diagnostic ID is separately named.
One payload assignment follows candidate validation and donor/input/RNG/mode
proofs. No base learner, native snapshot/policy format or existing game-weighted
port changed. Imports flow helper→native/readout comparator, without a reverse
native import. New module documents inputs, outputs and caller responsibilities.

The caller owns TRAIN roles/release, explicit weight derivation, original budgets
and permanent attempted-work charges. No role is invented from arrays; no final,
probe, replay, controller or selection access exists. PC/CPC and malformed inputs
are refused before candidate work. A compatible subclass can support counted
stream forks; the existing raw296 invariant owner is a different boundary.

The new tests verify exact all-one full-state continuation, independently
constructed per-row weighted CE/gradients/momentum, source/input preservation,
invalid weights/methods/arrays/mode, a lost return after SGD, and foreign changes
to policy, source parameters, gradients, optimizer state, input, three global RNGs
and grad mode. Those foreign changes are refused, never erased. Two delayed
TRAIN-only updates use the original `AttributionStream` and leave source/actor
unchanged with original observed+1 label receipts and zero scoring forwards.

Independent source/math review found one identity gap: same-valued optimizer
Parameter clones could keep serialized state equal. Before further acceptance,
the helper now validates initial optimizer-to-head binding and pins parameter-list
and Parameter identities. Two regressions cover rebinding before staging and
during a completed staged step; publication is refused while the foreign binding
remains. A bounded followup confirmed the finding resolved/no further blocker.

## Executed checks and costs

Physical cwd D:/Dev/Repositories/Circadian. TEMP/TMP D:/Dev/_Temp; Linux also
TMPDIR=/mnt/d/Dev/_Temp. Python-B, CPU1/GPU0, Torch2.14.0+cpu. Windows Python3.14.7;
Linux uses the existing interpreter
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.
Linux pytest starts with `--capture=sys`, reusing the observed D capture fix.

Prospective software allowance per platform:4pytest invocations/150s/1GiB sampled
RSS,80optimizer attempts256rows/200componentinits/9000nominal initializer slots/
300forwards/16ksnapshots30Mlogical elements. Explicit weighted-call/input-row,
SGD/backward/forward-row counts were added to the reused finite profile observer.
Stage8MiB. No model/fixture budget revision or scientific allowance.

`<stage>` = `artifacts/runs/torch-array-weighted-20261029`:

```text
<python> -B <stage>/check.py tests <stage>/<win|linux>-final-work.json [--capture=sys on Linux] tests/test_torch_array_weighted.py
```

Initial25passed/platform,0skips. Windows4.4713766s/325,464,064B sampledpeakRSS;
Linux8.796895651s/346,714,112B. Each28weighted API calls/112declared input rows,
including pre-candidate refusals.20physical SGD/backward calls/80forward rows:
16weighted candidates plus2ordinary and2manual-reference steps. Ten weighted
candidates were intentionally rejected after paid SGD; six committed.20forwards,
49componentinits882nominal initializer slots115snapshots14,423logical elements.
No PC training, task/physics dispatch, typed task inference or scientific reads.

After the binding repair, affected checks on the combined final tree:

```text
<python> -B <stage>/check.py tests <stage>/<win|linux>-binding-work.json [--capture=sys] tests/test_torch_array_weighted.py::test_should_refuse_same_valued_optimizer_parameter_rebinding tests/test_torch_array_weighted.py::test_should_match_exact_ordinary_bp_for_all_one_weights_and_continuation tests/test_torch_array_weighted.py::test_should_match_independent_per_row_weighted_ce_gradient_and_momentum tests/test_torch_array_weighted.py::test_should_run_two_arrived_train_updates_in_existing_stream_only
```

Five passed/platform: two new binding cases plus three affected successful paths,
for27distinct cases/platform total. Windows4.5430758s/326,893,568B;
Linux8.516196353s/342,638,592B. Each added8weighted calls/32declared input rows,
11SGD/backward calls44forward rows,20componentinits360nominal slots53snapshots/
2376logical elements. Before-binding corruption is refused before staging;
during-binding failure retains one completed unpublished private step.

Total two invocations/platform: Windows9.0144524s peak326,893,568B;
Linux17.313092004s peak346,714,112B. Each36weighted calls144declared input rows,
31physical SGD/backward calls124forward rows (23weighted,4ordinary,4manual),
69componentinits1242nominal initializer slots31forwards168snapshots16,799logical
elements. Twelve weighted candidates committed and11failed after paid SGD across
all fixtures, including repeated affected checks. API input rows include refused
calls and are not all optimization exposures. Legacy observer fields named
native_train_attempts count an unused older ticket wrapper and remain zero;
the weighted/SGD/forward-row counters above describe this port's actual work.
Nominal slots/snapshot counts are not exhaustive allocations/RNG instructions or
FLOPs; sampled RSS is not a hard OS peak.

Scoped quality commands:

```text
.venv/Scripts/python.exe -B -m ruff check src/adapters/torch_array_weighted.py tests/test_torch_array_weighted.py
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/torch_array_weighted.py tests/test_torch_array_weighted.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_array_weighted.py tests/test_torch_array_weighted.py
```

PASS; last command also passed `--platform=linux`. Whitespace passed with inherited
line-ending advisories. Initial NumPy RNG tuple-union and optional optimizer/
gradient fixture typing errors were fixed before numerical checks.0test failures,
skips or budget revisions. Source-review identity finding was corrected rather
than weakening its acceptance test. All logs/receipts remain in the stage.

Lead source/procedure/result self-review completed. Independent source/math review
used2readonlycalls; followup1call confirmed the repair.0imports/models/tests/raw
reads/writes. Both finite exceptions ended. This is source/math review, not
independent experimental replication or a learned-task result.

Skipped broad regression/CI/clean clone/installed wheel, physics/GPU, native
ticket task runs, BP/PC/CPC comparison or any scientific cohort. Existing native
base/game-weighted code is unchanged and its accepted evidence is reused, not
recertified wholesale. No claim of improved acquisition, retention, recovery,
class-balance benefit, circadian advantage, generalization or release readiness.
The275/276negative ready-action outcomes remain unchanged and their raw files
closed. This port alone does not answer whether weighting fixes them.

## Exact next implementation action

Wire an explicit class-balanced BP option into the existing recurring native
workflow/factory. Derive weights only when the current ordinary TRAIN batch has
arrived, using its target counts and existing bounded validation. Use this helper
for that one candidate update, preserve the original method/configuration/dose,
source/actor/RNG and original immutable execution envelope. Record objective,
class counts/weights, attempted/returned work and receipts before admitting
further updates; refuse changed/replaced preceding evidence. Keep the original
unweighted default and all-one compatibility. Neither static diagnostic truth
nor final/probe labels may drive weights. Start role/causal/metadata/failure
checks before a separately frozen small development qualification. No new
study allowance, favorable-seed selection or assumption that weighting helps.
