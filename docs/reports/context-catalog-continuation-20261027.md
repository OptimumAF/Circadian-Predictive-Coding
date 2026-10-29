# R7.4-CAT.CTX.W — Deferred selected wake completed; retention loss observed

[Issue263](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/263),
2026-10-27. Full repository goal and parents21/23/26/33 remain open. The previous
turn made progress on the contextual catalog/output boundary. This increment
implements and executes an explicit selected-action completion contract. Lead
source/procedure/result **self-review**, not independent acceptance or release
readiness. Original262 genuine fit/application failure remains FAILED/SPENT:
its lost results/costs were not reconstructed, rerun or replaced.

## Working behavior

```text
scripts/context_catalog_continuation.py    One pending wake + exact poll/progress checks
scripts/context_catalog_control.py         Explicit opt-in, binding and capture
scripts/context_head_live.py               resume_catalog_wake=False public default
scripts/run_bp_ticket_serving.py           Existing schedule execute/observe seam
tests/test_context_catalog_continuation.py  Real conventional/refusal/worker checks
```

Why this: replay and wake are separately paid phases; choosing a fresh action
after late deferral can repeat replay or leave the selected wake unfinished.
Opt-in continuation retains one original returned replay poll/receipt and FIFO
index, with no native state chain, new scheduler, budget, clock or learner.
Existing CategoricalReplayAction validates original source/candidate/actor,
bank/versions/full native state, budgets, receipts and permanent costs.
The adapter validates exact issued/returned poll identity and value, causal
tick/FIFO target, declared effects, progress and sharing admission/deferral deltas.
Only actual successful replay plus retryable late deferral can arm it. Additional
paused/serving/resource polls perform no native work. Only the original successful
wake clears it. Missing/replaced/corrupted evidence, unknown work, changed owners,
options, counters or a post-selection clock advance refuse before further work.

Default execution adds no callbacks/capture fields. Existing frozen six-trial
method and D7 catalog remain unchanged. Explicit private catalog mode rejects
conflicting/non-Boolean options before IO. No promotion/final/probe/tuning access.

## Executed checks

48 distinct scoped cases/platform pass through unchanged evidence plus affected
final combined-tree checks:33 continuation and15 default/mode/admission cases.
Windows7invocations, Linux2; exact arguments/outcomes/costs in local stage
`artifacts/runs/context-catalog-continuation-20261027/*-checks.json` and XML.
Main command `.venv/Scripts/python.exe -B <stage>/check.py win-N <targets>`;
Linux existing WSL Ubuntu Python `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`
with same observer. No skips in selected tests; full suites/unchanged BP/PC/CPC,
robotics/game/archive/probe studies deliberately not rerun.

Failures retained:win-1 fault injection reused an already-consumed serving lease;
win-3 used a nonexistent resource callback attribute;win-4 patched an instance
instead of replay's actual detached native fork. Corrected fixtures then passed.
All work below includes these failures. Static failures (fixture import lint,
two annotations, three harness typing errors) are retained; final8file Ruff
format/check and default/Linux-target mypy pass via `statics.py 7`.
Windows seventh pytest invocation was prospectively added only for review-found
pre-execution clock refusal and positive pause/resume regression; original
numerical/time/memory limits unchanged. No retrospective scientific changes.

| Software scope | Windows | Linux |
|---|---:|---:|
| Native conventional passes / rows |73 /396|48 /258|
| Learner constructors / zero heads |253 /506|173 /346|
| Cumulative pytest seconds |15.116909900|14.614028859|
| Maximum sampled process RSS bytes |78110720|73707520|

Only literal synthetic source fixtures; zero research payload reads, random
model draws, Torch/SGD/PC/physics/probe/final/fitter/study calls. Conservative
copy visits/slot counts and all remaining totals in cost-rollup.json. They are
not allocation/FLOP counts; sampling does not prove a hard RSS ceiling.
Recorded static commands42.665413501s plus four observed formatting calls
1.242606400s =43.908019901s; touched default Python3.11 mypy/Ruff0.16.9 cache
subtrees150301776B including inherited contents, below160MiB. Global workspace
caches628799358B include untouched historical/versioned caches, not new growth.
Stage before final receipts/report659483B; final size recorded in self-review.

## Sole source77 functional smoke

Prospectively frozen SMOKE_PROTOCOL.md; GO comment6455902027 after software gates.
Command `.venv/Scripts/python.exe -B artifacts/runs/context-catalog-continuation-20261027/smoke.py`,
terminal EXIT0, wall1.0790445s; whole observed body0.805482300s,
sampled RSS50819072B. Original277216B/SHAaddf3740…be943 source reused as data
through existing decoder; no archived runtime authority or baseline retraining.

Exact original14-opportunity sequence:tick0 wait;tick1 actual replay6 completes,
harness explicitly pauses original gate, wake defers;tick2 original remaining
wake defers again with no native work, observer explicitly resumes original gate;
tick3 wake3 succeeds;tick4–13 wait. Continuation armed1/completed1/deferred2,
pendingnull. Native12/72→14/81, separate replay6/wake1 charges. Original source,
actor and inactive head unchanged; no publication. One14tick typed worker.

Fixed known TRAIN task readouts: **new1/2→2/2; old4/4→3/4**. Acquisition improved
and an old task was lost. There is no later old-task return phase here, therefore
no recovery result. This is a complete negative retention finding, not evidence
of circadian/controller advantage, generalization, matched computation or causal
superiority. The changed execution contract is explicit, not silently applied
to prior policies or the failed262 study.

Measured whole smoke:2passes9rows,8constructors16zeroheads,3matrixprediction rows,
82typed requests112sandbox steps,2teacher2trace builds,1077context1691head copy
visits184560logical coefficient slots;190source reads/52671040returned bytes/
99614910bounded attempted bytes. Includes source decode, copies, preparation,
readouts, guards and worker; no new controller fit/seed choice/probe/final work.
smoke-work.json flushed **before** JSON-safe writer; smoke.json569814B and one
metadata read; output processing0.011841700s. All within original limits.

## Reassessment and next action

R2.1 diagnostics and isolated fresh-task/return probes already exist (#78/#259/
#260); do not rebuild them or repeat their spent observations. #77 already
demonstrates four-task context acquisition/retention, with greater capacity and
explicit public context. This new six-task slice exposes within-v1 interference
that public v1/v2 head separation does not prevent. Another capture, provenance,
fourth-update or completion-counter increment is not the critical path.

Next concrete implementation action: inspect `build_joint_context_batch` and
`build_context_stream` for a bounded joint nine-row ordinary TRAIN packet
(legacy3 + invoice/password6), then add an explicit recipe through the existing
stream runner if no such route exists. Freeze a small current-only versus joint
rehearsal protocol with equal row exposure before fitting; keep earlier one-pass
retention loss, frozen metrics, original probe separation and simpler controls.
This targets simultaneous acquisition/retention, followed by a declared return
test, rather than adjusting replay doses/seeds to force favorable endpoints.
