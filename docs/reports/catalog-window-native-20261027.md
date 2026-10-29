# Native guarded publication after a live window

[R7.7-CAT.PN / issue253](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/253)
completed its sole frozen native invocation. Source/procedure/result
**self-review**, not independent acceptance. The full repository goal remains
active; this is one development-task qualification.

The learned candidate passed the original guards and served all four ordinary
tasks successfully after publication. The original live actor's earlier legacy
failures remain in the record. This closes the private-candidate-to-serving gap
for the declared bounded case, without extending training or consuming the two
pending old-version packets.

## Fixed procedure and observations

The existing `run_case` used the saved245 chooser, original source and BP
seed17/D7/H3/K6/lr.03/momentum.9, eight clean six-row TRAIN batches, delay1,
`LiveServingPlan(9,(2,6))` and `publish_after_window=True`. Original nine-decision
envelope, wake/replay budgets, controller choices and guard thresholds stayed
fixed. No new runner, model algorithm, controller fit or production source edit.
The live controller requires its owned runtime, so this distinct publication
question fully paid its necessary six-update prefix anew; saved251 JSON was
not used to manufacture ownership. Neither251 nor248 was rerun.

| Observation | Result |
| --- | --- |
| Native learning | 6 returned SGD updates, 36 rows; full history102/612 |
| New-task guard | Original actor5/6 → candidate6/6 |
| Old-task guard | Original actor6/6 → candidate6/6 |
| Guard decision | Accepted at tick9; one prepare and one commit |
| Publication | Generation1, qualified-96 → c0, native102 |
| Post-window ordinary tasks | Invoice, parcel, legacy, password: all4 complete/successful |
| Post-window requests | 12; every request uses generation1/c0/native102 |
| Original pending work | batch-6/batch-7, observed6/7, labels arrived7/8, qualified-96; unapplied |
| Retirement | Original runtime/inbox/chooser retired; both refusal APIs reject further work |
| Replay/probe/final/new controller trials/PC-CPC/physics | Zero |

Before publication, the original actor succeeds on invoice/parcel/password and
fails legacy. All nine live episodes complete, with six successes and29requests,
all generation0/qualified-96/native96. Invoice at tick4 recovers from the planned
stale revision; legacy at1/5 still fails; the deliberately revoked parcel commit
at6 is refused. Zero applied permission/revision violations were observed.
Six native wake admissions and two serving deferrals preserve the original cap;
all eight labels arrive, with two packets pending. Private candidate readouts
also succeed on all four tasks, but the useful new observation is actual served
success after the guarded swap.

The original gate's final snapshot is exactly its prepublication snapshot with
`paused=True`: counters, deferrals and limits remain unchanged. Both original
`runtime.train_ready` and `chooser.before_work` raise their retired-owner errors;
they perform no further native work. Six preceding receipts, original source/
label versions, clocks, budget bindings, paid history and source/fit bytes remain.
The original source, caller NumPy/Torch RNG and frozen policy are unchanged.
No serving/private endpoint supplies TRAIN labels, promotion labels, replay,
probe selection or controller feedback. The independent inner guards use disjoint
IDs but repeated synthetic grammar; they are not unseen generalization.

## Costs and resource qualification

Native body14.099830594s, sampled peak process RSS340,066,304B. Parent17.8345983s,
EXIT0, no timeout; raw stdout1936B/stderr0B. Original inner120s512MiB budgets and
outer180s512MiB/parent195s limits hold. Main and native serving worker share
synchronized counters; the memory sampler starts before Torch/metadata admission.
This is sampled process RSS, not a process-tree or allocator hard bound.

| Instrumented work | Observed |
| --- | ---: |
| Model components / initializer slots / normal draws | 138 / 5382 / 276 |
| Full snapshots / logical elements | 895 / 122544 |
| Native forwards / rows | 89 / 179 |
| Backward / SGD / returned fits | 6 / 6 / 6 |
| Typed requests / sandbox dispatches | 71 / 143 |
| TRAIN-bank certification dispatches | 36 |
| Case owners / controller selections | 1 / 9 |
| New utility observations / historical consistency checks | 0 / 6 |
| Publication attempts / prepares / commits / refusal checks | 1 / 1 / 1 / 2 |
| Tensor clones / logical elements | 6705 / 85504 |
| Tensor deepcopies / logical elements | 13272 / 159264 |
| Source+fit reads / logical bytes | 138 / 64824942 |
| Code-pin reads / bytes | 66 / 1449516 |

Forward accounting reconciles6training+8causal-monitor+71typed+4guard calls;
rows36+48+71+24=179. Typed calls include59serving/phase requests and12private
candidate decisions. The29live and12post-window requests are subsets, not extra
work. The original complete TRAIN bank is prepared once despite selected replay0.
Snapshot, clone and deepcopy counts overlap; model-construction proxies are not
exact FLOPs or allocated bytes. Latency includes instrumentation/source guards:
live predictor p50=15.312082ms, p95=19.310674ms, maximum22.935962ms. Episode elapsed
also includes queue/reservation/workflow. No SLO or cross-run efficiency claim.

Saved245 costs remain distinct:48discarded fit updates240rows/48replayrows,
228prediction calls468rows and216prepared bank rows. The whole source study has
56fits288rows,304predictions624rows and288prepared bank rows; original96fits576rows
and source pretraining costs are separate. These groups overlap and are not
summed as disjoint costs. Exact fit overhead is inseparable; all-in utility is
unavailable. No baseline or chooser was refit/retuned.

All prospective caps in local `PROTOCOL.md` and `native-work.json` pass. The
actual result is584547B, SHA256
`17069ca98103a83809bc1445acea96854c9505a7076f89b73d14ecbb5456d443`.
Stage1138258B before result-review/closeout receipts, below8MiB; result below3MiB,
logs below1MiB and other metadata551775B below1MiB. Mtime-selected cache113262828B is
below160MiB; this is footprint, not growth, and excludes other inherited cache.

## Commands, gates and review

Finite local ignored evidence: `artifacts/runs/catalog-window-native-20261027/`.
Source pins, `PROTOCOL.md`, `run.py`, `check.py` and `review-go.json` froze before
the single native launch. Canonical forward-slash paths fixed portability before
models; no scientific-source preflight read was added. Original source/fit hashes
were verified inside the native prologue and guarded throughout.

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout --kill-after=2s 190s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-window-native-20261027/run.py
```

The bounded parent recorded PID122000 and polled the same process to terminal;
native PID384. No observation-timeout restart. NumPy1.26.4/Torch2.14.0+cpu matched
the existing offline producer runtime. `attempt.json` permanently spends this
allowance, including any prologue failure; it is now SPENT and must not be rerun.

Software observer checks (no project/NumPy/Torch/scientific payload imports):

```text
./.venv/Scripts/python.exe -B artifacts/runs/catalog-window-native-20261027/check.py
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-window-native-20261027/check.py
```

14stdlib tests/platform PASS,0failures/errors, Windows0.146263400s/Linux0.579144587s.
They cover permanent overruns, lost returns, original dimensions, forbidden PC,
worker/shared counters, portable pin refusal, publication/prepare/commit limits,
unknown evidence, negative tasks/known rejection and actual post-publication
versions. The first draft's wrong guard-report key was corrected from actual
source before any test/native invocation; no failed test or native attempt.

Adding `--pins` to each command passed once/platform:33code files724758B,
Windows0.086226500s/Linux0.274380966s; Linux distribution metadata matched without
Torch import. One preparation pass read the same33code files once separately.
Scientific source/fit preflight reads0. Reuse252's172qualified software cases per
platform; no unchanged model-fixture rerun.

Scoped statics, with `<files>` = this stage's `run.py check.py`:

```text
./.venv/Scripts/python.exe -m ruff format --check <files>
./.venv/Scripts/python.exe -m ruff check <files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <files>
```

All PASS first qualification; exact outputs/argv in `static.json`, measured
14.4652148s. Initial formatting was not separately timed and is not reported as0.
One saved-only stdlib read of new `actual.json` produced `readback.json` and
24PASS checks of terminal/protocol status, caps, causal/native costs, known
publication, guard roles, original pending IDs, retirement/refusals, task
completeness and actual versions. Further interpretation used that projection,
not another actual/source read. No model rebuild or rescoring. Self-review
inspected source, native procedure and actual outcomes, not merely green checks.

No protocol amendment, failed execution, retry or budget increase occurred.
Some initial broad console projections were truncated; the saved result and
readback remained complete, and later output was narrowed. An incorrect guessed
recipe filename returned a read error; inspecting the actual import resolved it,
with no model work. Skipped full suite,
clean clone/wheel/install/browser/independent review, unchanged baseline/math/
simulator cohorts, comparative/PC-CPC/sequential recovery/long-duration studies
and release gates. No new dependencies/install/download/GPU/CI/agents/Git
mutations/reset/raw publication. Inherited changes remain intact; historical
plans/log stay frozen under AGENTS and current handoff lives in issues253/18.

## Critical path and exact next implementation action

The bounded native policy now learns legacy, retains old tasks, and actually
serves those tasks after publication. This is known development grammar and
one clean short window, not demonstrated relearning after actual loss, unseen
transfer, controller/CPC advantage, long-duration robustness or release readiness.
Earlier245negative utilities,248OPEN FAILED-SPENT unknown comparison outcomes,
251private success and all other failures remain unchanged.

R2.1 diagnostics and isolated fresh-task probes already exist and have accepted
evidence; the historical plan records101checks, isolated four-arm freshC learning
curves and negative fixed-four-step endpoints. Reuse those implementations/results,
not another diagnostic/probe framework. Existing post-publication fresh learning
also exists (#66); do not automatically add another publication or capture window.

The concrete remaining R7.7 prerequisite is varied/noisy live feedback for the
frozen catalog policy. Actual `run_case` rejects `live_training_recipe` whenever
catalog execution is selected, while existing live recipes and deferred catalog
recipes already supply bounded truthful/mixed/corrupted TRAIN packets separately.
Next implementation action: compose existing mixed/corrupted `LiveTrainingRecipe`
inputs with the frozen catalog's private LIVE path, with explicit role/source/
recipe binding, original nine-opportunity envelope/charges and unchanged original
recipe delays. Inspection corrected an initial delay1 assumption: the retained
corruption recipe actually uses delay2, so its final label arrives beyond this
window. Preserve that pending/not-yet-arrived distinction; do not shorten delay
or extend the horizon to make completion easier. Keep terminal publication
refused for these combinations until its stricter prerequisite is explicitly
qualified; the existing clean publication remains working. Start with pure
causal/identity/paid-failure/default checks, then a genuinely separate prospective
usefulness question if justified. Do not renew an old owner, automatically run
another noisy comparison or retry248. This enables the actual robustness gap
through existing components; it is not a new runner or another capture increment.
Broader R2/R4/R7, recovery/science/release parents and the full goal remain open.
