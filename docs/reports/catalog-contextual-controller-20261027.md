# Opt-in contextual replay/wake catalog

[Issue243 / R7.4-CAT](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/243)
delivers the scoped software API. Previous turn242 and this turn are PROGRESS.
Source/procedure/result **self-review**, not independent acceptance. Full goal
and broader controller, useful-adaptation, scientific and release requirements
remain open. No native/scientific experiment was authorized or executed here.

## Working behavior and files

New pure `src/core/catalog_contextual_controller.py` defines an explicit catalog
of exactly wait, wake1 and one owner-declared replay-plus-wake1 action. The latter
uses1..128 fixed TRAIN replay rows, unchanged rate/gate and zero sleep. The
catalog binds the original immutable `AdaptationEnvelope`; it cannot grant new
execution, budget, permission or data authority. Old
`src/core/contextual_controller.py` remains byte-identical to the pre-edit copy.

Existing `src/app/counterfactual_controller.py` takes optional `catalog=` and
reuses its source/version/digest, causal return, failed-attempt, lock and freeze
path. Six disposable trials cover two contexts and all three arms exactly once.
Separate explicit request/return/estimate/fit records preserve legacy v1 dataclass
and serialized dictionary shapes, without adding None fields to old evidence.
The default remains two arms/four trials; existing command/driver contracts and
inherited81+/1- working changes are preserved.

Fit maximizes the unchanged `ControllerObjective`: ties prefer wait, then
declaration order (wake1 before replay/wake). Missing/duplicate/foreign/nonfinite
rows, inconsistent derived choices, stale source/version, changed envelope,
future/nonTRAIN observations and insufficient replay credit are handled explicitly.
Insufficient credit returns wait, without inventing a smaller untested dose.

Every completed catalog trial preserves attempted/returned native updates/rows,
replay exposure and prediction calls/rows alongside original time/RSS/violation
measures. Native updates include replay and wake; replay rows are a separately
reported subset of training exposure. Unknown/inconsistent work has no utility.
Known structurally valid partial costs stay inspectable on failure; the callback's
native owner retains its own actual permanent charges. Detailed gradients,
latent inference, initialization, allocations and FLOPs remain external driver
accounting, not inferred from these scalar fields.

One source snapshot and at most six scalar rows/returns are retained. Existing
fit pins include new work/catalog records; a bounded current prefix value+identity
pin validates prior paid successes before another callback. Corrupt or replaced
row/work/request/envelope/measure, frozen fit, clock or source refuses further
work. There is no model-state chain, retry, refit, reset or renewed allowance.

`CounterfactualController.choose` validates its current source/clock/frozen fit
and explicit catalog envelope/replay credit, returning only a selected ID. The
existing `ResourceSharedRuntime.train_action` performs actual admission. A real
scripted six-trial test then executes the frozen replay/wake selection through
that executor, observing replay then fresh update, separate original charges and
unchanged source/actor. Literal utilities are test inputs, not measured task gain.

Usage/API composition and full-state source-binding responsibilities are in
[contextual-controller.md](../contextual-controller.md#opt-in-replaywake-catalog-api).
New tests are `tests/test_catalog_contextual_controller.py`. No new runner,
dependency, native mathematics, v1 schema, source loader, CLI option or installer.

## Executed evidence

**125 distinct checks pass per platform**:61new catalog,26legacy core/owner,
38existing attribution application. Final124-case combined cohorts plus one
targeted six-trial error-message regression cover the final behavior; subsequent
formatter changes are whitespace only. Ordinary original mathematical and model
cohorts are reused where unchanged rather than repeated.

| All attempts | Instances | Pass/fail/error/skip | Body seconds | Sampled peak RSS bytes | Scripted train calls/rows |
|---|---:|---|---:|---:|---|
| Windows | 220 | 219/0/1/0 | 10.476175000 | 78,970,880 | 38/60 |
| Linux | 125 | 125/0/0/0 | 21.667472796 | 71,364,608 | 30/41 |

The one Windows error is intentional preimplementation RED: missing catalog API
import, no collected productive fixture/work. All subsequent executed checks
pass. A collect-only invocation admitted no fixtures. Tests include negative
utilities/select-wait, deterministic ties, all-six coverage, malformed/unknown
costs, refused poison metrics, reentrancy, prior/frozen evidence tampering,
source/role/clock/credit binding, default schema preservation and actual shared
replay/wake integration. The latter includes six discarded scripted updates14rows
plus selected application2updates5rows; genuine model/optimizer work is zero.

Final cohorts each measure277 trial callbacks and50 literal prediction calls/
rows. Earlier Windows callbacks were not individually metered; inspected fixtures
admit at most6callbacks per case across94earlier cases, so <=564additional and
<=841total, not an exact measured total. Earlier scripted training counts were
metered and are included above. No rerun filled the missing counter. Optional
native imports are blocked and counted0; no Torch/model/teacher/physics/probe/
final/archived payload reader or random model initialization executes.

Prospective issue limits held: <=80new/100affected distinct checks and240total
instances/platform,90sbody/platform/512MiB,1024scripted train calls4096rows.
Linux uses existing offline Python and timeout110s/CPU1 environment. Four-file
Ruffformat/lint/default+Linux-target scoped mypy pass; all-attempt static time
30.072356500s/120 and conservative touched-cache113,345,268B/160MiB (not growth).
Source/procedure review checks new pure module's inward dependencies, original
core bytes, combined app changes and actual test/cost results. It is self-review.

Exact command arrays, outputs and elapsed costs are in stage `*-checks.json`,
`*.xml` and `statics-*.json`. Final combined command:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/catalog-contextual-controller-20261027/check.py win-tree tests/test_catalog_contextual_controller.py tests/test_contextual_controller.py tests/test_attribution_contextual_window.py
```

The Linux command uses the same targets through `check.py linux-tree` in the
existing WSL environment. The targeted `*-message` receipt verifies the final
clear six-trial refusal. Static receipts contain exact four-file Ruff/mypy
commands; `review-final.json` records current reviewed source pins and costs.
These are post-check review pins, not a prospectively frozen native protocol.
One documentation patch anchor and one Windows rg wildcard read failed without
edits; simplified to the valid anchor/directory source. No test/static/model
failure beyond the intended RED and no quota renewal.

## Limits, parents and precise next action

Preserved old evidence and all negative/spent studies, including236. No new
dependencies/install/download/GPU/CI/agent/Git mutation, unrelated reset, source
archive reconstruction or sweep. Historical plans/log stay frozen under AGENTS;
GitHub owns live scope/handoff. Skipped full suite/clean clone/new wheel/install,
independent review, native/model/comparative cohorts, task/simulator/controller
utility/CPC studies and release gates. Sampling is scoped process RSS, not an
instantaneous/process-tree proof. Old wheel241 predates this optional API.

Literal utilities and scripted learner ports establish selection/admission
correctness, not task usefulness, acquisition/retention/recovery, generalization,
net efficiency, matched compute or circadian advantage. Detailed native costs
and externally observed task outcomes require a separately frozen local study.
Full goal/parents21/23/26/33 and broader science/release requirements remain open.

Critical path next is native driver integration rather than another catalog,
capture or package increment. Actual `scripts/run_bp_ticket_serving.py` accepts
only legacy `ContextualChoice/FittedController`; `_reference_schedule` constructs
wait/wake1-only sharing without a replay executor. The existing
`scripts/ticket_controller_learning.py` recipe fixes four trials. A six-trial
scientific experiment cannot run honestly through those paths yet.

Next precise implementation action: add explicit catalog trial/application
parameters to that existing `run_case/_reference_schedule` path, bind the
original complete TRAIN bank and matching envelope before work, and use the
catalog selector with the original full-source, causal inbox, budgets, shared
gate and native/task cost ports. Preserve current default/fixed-reference/v1
four-trial recipes and all saved evidence. Start with zero-native dispatch/
ownership/cost/default tests; then scope the existing controller-learning recipe
to six catalog trials and freeze one small task-outcome protocol before model
execution. No new runner, favorable baseline/seed/dose/metric changes or spent
case retry. This is the concrete dependency to useful controller evidence.
