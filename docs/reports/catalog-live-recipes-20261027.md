# Mixed and corrupted feedback during catalog live service

[R7.7-CAT.R / issue254](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/254)
is complete for scoped software, with source/procedure/result **self-review**,
not independent acceptance. Native varied-feedback usefulness remains unqualified;
no scientific allowance was spent or granted. Full repository goal remains active.

## Working behavior

The existing `run_case` now composes an explicit exact `LiveTrainingRecipe` with
its frozen catalog private LIVE path. It uses the current builder, source/fit
binding, native runtime, controller, workers, sharing gate and budgets. Original
eight-packet/nine-opportunity envelope stays fixed. No new runner, model, policy,
teacher, algorithm, CLI or dependency.

Existing mixed/noisy recipes preserve twelve rows per packet and delay2. The
window ends after tick8: seven labels have arrived; the eighth is due at9 and
remains unarrived. Existing `stream.unarrived_labels`, `arrived_unapplied` and
`pending_unapplied` distinguish that future label from arrived unused work. A
complete known service window does not claim all TRAIN work was consumed.
No forced drain, delay shortening, extended horizon, new owner or renewed budget.
Delay1 remains supported with its own unchanged chronology.

Literal-port observations through the actual builder/runtime/workers:

| Frozen choice, delay2 | No reserved contention | Reserved ticks2/6 |
| --- | --- | --- |
| wait | 0fits; 7arrived/1unarrived, 8pending | Same |
| wake1 | 7fits84rows; 1pending unarrived | 5fits60rows; 2arrived unused +1unarrived |
| replay_wake | 7replays21rows, 0wake; 8pending | 5replays15rows, 0wake; 8pending |

The fixture holds the real serving lease across each action poll. For replay_wake
the admitted replay finishes, then the wake is deferred by service. This proves
original separate replay charges/admission semantics, not an algorithm result.
All nine episodes complete and use original generation0/qualified-96/native96.
Private readout is distinct from service, and the old live comparison utility
still refuses the report. Actual native task improvement is not inferred from
literal predictions or these counts.

The existing builder marks selected row1 or retained row7 corruption as
experimental label noise, never verified teacher truth. Source, frozen record,
original recipe object/full values/underlying recipe, built TRAIN packet objects/
values/roles and schedule/delay remain bound. Workers use the existing pure
`require_pin` boundary to check immutable packet evidence without advancing
either mutable budget clock. Main-thread work still checks original budgets.
Changing service outcomes leaves causal feedback, decisions, receipts and
candidate state unchanged; final/probe sources remain unavailable.

Before IO, refuse trials, private-only recipes, deferred/overlapping stream,
reference, short or wrong-horizon windows, drain, followup, policy-change and
terminal-publication combinations, malformed/subclass recipes and coerced bool/
delay metadata. Existing clean publication remains working; this combination
does not weaken its stricter prerequisite or consume unarrived labels.

## Demonstrated receipt defect and bounded repair

A new after-success fault found that an equal-valued replacement of an earlier
wake receipt could admit more learning. The existing shared replay authority
kept detached values but no original receipt references. The scoped repair also
covers preceding replay receipts: preserve original references beside the
existing finite1024wake/32consolidation metadata prefixes, and compare identity
plus exact values using existing `_same_value`. Append only at observed success;
never keep a native-state success chain or renew any quota.

The same check runs before work and when accepting replay publication. It refuses
equal replacements, changed values and in-place bool/int coercion. The source,
actor, complete expected native state, original versions, clocks, budgets and
permanent failed charges retain their original guards. A failed or lost return
stops further work and preserves the paid prefix. Why this: metadata equality
does not establish that the original observed success still exists.

Affected paths:

```text
scripts/ticket_catalog_control.py          explicit live recipe and pure packet binding
scripts/run_bp_ticket_serving.py           narrow compatibility/fixed-cap schedule gates
src/app/categorical_replay_action.py       bounded original success-reference checks
tests/test_catalog_live_recipes.py         actual builder/worker/runtime literal cases
tests/test_categorical_replay_action.py    expanded predecessor replacement/type faults
```

## Executed validation and exact commands

Local ignored evidence: `artifacts/runs/catalog-live-recipes-20261027/`.
**242distinct cases/platform qualify:**239combined PASS, then3additional exact
metadata cases PASS. Breakdown:51live-recipe cases,128affected original catalog/
live/stream/pure timeline cases,34original terminal-publication cases,29pure
replay authority cases. No final errors/skips. Four genuine native/teacher
fixtures were explicitly deselected; the optional-native import trap remained
active. No scientific payload, real teacher-bank, Torch or simulator work.

Combined target arguments:

```text
tests/test_catalog_live_recipes.py
tests/test_catalog_live_serving.py
tests/test_ticket_catalog_control.py
tests/test_catalog_ticket_stream.py
tests/test_ticket_serving_timeline.py
tests/test_catalog_window_publication.py
tests/test_categorical_replay_action.py
-k 'not compile_only_the_original_complete_train_bank and not native_live_episode_versions_original_causal_receipts_and_priorities and not native_live_task_records_do_not_become_feedback_or_promotion_guards and not native_wake_replay_wake_example'
```

Windows: `./.venv/Scripts/python.exe -B artifacts/runs/catalog-live-recipes-20261027/check.py win-combined <targets above>`.
Linux uses the same targets after:

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-live-recipes-20261027/check.py linux-combined
```

The final followups use `win-types`/`linux-types` and
`tests/test_catalog_live_recipes.py -k should_validate_exact_live_recipe_metadata_before_io`:
3PASS/platform, no model/prediction/dispatch work. Windows1.911013500s/Linux9.879933559s.
Only those tests and formatting changed after the combined run; production tree
is unchanged. Reuse combined evidence without another full rerun.

Combined Windows16.152617000s/94,580,736B/566scripted fits3948rows;
Linux45.920620690s/79,810,560B/564fits3936rows. Each821scripted predictions6786rows,
2303pure dispatches and15literal trial callbacks. The reused meter does not
attribute the two six-row fit difference to individual tests. Preserve both
actual counts; no matched-compute/deterministic cost or scientific comparison
claim follows from this worker/fault cohort. All observed work fits the frozen
caps. This software accounting limitation does not establish native behavior.

## Failures, costs and prospective scope change

All raw `*-checks.json` and JUnit receipts remain unchanged:

- Intended RED: unsupported combination before IO; EXIT1/1.773658400s,0work.
- First implementation run completed a window; test used nonexistent
  `native_history.training_examples` instead of actual `training_rows`:
  EXIT1/1.614882900s,8literal predictions96rows/29dispatches,0fits.
- 18positive cases PASS; delay1 fixture accidentally used default reserved2/6
  yet expected8updates, observing6: EXIT1/2.808032300s,78scripted fits612rows/
  152predictions1824rows/551dispatches. Set explicit no-contention for the shared
  fault/delay fixture as intended; the contention matrix was already explicit.
- 39PASS/1FAIL exposed the genuine equal-receipt replacement gap:
  EXIT1/2.975517300s,93fits792rows/184predictions2208rows/634dispatches.
- All48new live cases PASS; extended shared controls reached64PASS/1FAIL:
  EXIT1/5.634958800s,119fits954rows/210predictions2520rows/703dispatches. The
  coercion injection wrote a nonexistent optional Python attribute, leaving
  actual evidence unchanged. Corrected to real `attempt_number=True` versus1;
  production strict comparison and fault requirement remained unchanged.
- Initial static format check found one multiline assertion to reformat;
  lint/both type modes already passed. Final all pass. A documentation patch had
  an obsolete anchor and0edits; inspecting exact text resolved it without work.

The demonstrated shared defect justified adding its source/tests to scope and
raising prospective distinct software coverage240→256 at issue comment6450770691.
All original8invocation/120s/512MiB/work/storage caps stayed fixed, failures were
not erased, and native allowance stayed0. No scientific budget renewal.

| All software attempts, including failures | Windows | Linux |
| --- | ---: | ---: |
| Invocations / body seconds | 7 / 32.870680200 | 2 / 55.800554249 |
| Peak sampled process RSS | 94580736B | 79810560B |
| Scripted fits / rows | 856 / 6306 | 564 / 3936 |
| Scripted predictions / rows | 1375 / 13434 | 821 / 6786 |
| Pure dispatches / trial callbacks | 4220 / 15 | 2303 / 15 |
| Test instances / failed / errors / skips | 368 / 5 / 0 / 0 | 242 / 0 / 0 / 0 |

All genuine Native/Torch/teacher/physics/probe/final/scientific payload counters0.
Logical fixture costs are not model-training experiments or free work.

## Static checks, self-review and remaining claims

Six file arguments: the five source/test paths above plus stage `check.py`.

```text
./.venv/Scripts/python.exe -m ruff format --check <six files>
./.venv/Scripts/python.exe -m ruff check <six files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <six files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <six files>
```

Final all PASS; exact argv/outputs/times in `static*.json`, initial+final measured
27.2011677s. Earlier formatter calls were not separately timed, not reported as0.
Mtime-selected cache113434600B is below160MiB (footprint, not growth, inherited
cache separate); stage306136B before report/review/closeout, below4MiB.

Source/procedure/result self-review examined actual changes, original admission/
receipt lifecycle and executed fault outcomes. Session-start AST is unchanged
outside `run_case`/`_reference_schedule`, catalog constructor/pin/before_work,
and replay constructor/inside/receipt comparison/observed success/commit checks,
plus two explicit imports. Builders, timeline loops, CLI, objective, task truth,
native math, frozen scientific results and unrelated changes remain untouched.
`source.diff` records that boundary. No new abstraction/runner/dependency/agents,
install/download/GPU/CI/Git mutation/reset/raw publication. Historical plans/log
remain frozen under AGENTS; current status is in issues254/18.

Skipped full suite/clean clone/wheel/install/browser/independent review, genuine
native varied feedback, unchanged baseline/math/simulator cohorts and broader
scientific/release gates. Existing245negative utilities,248OPEN FAILED-SPENT
unknown comparison,251private success and253clean published success remain
within their original frozen sources/scopes and spent allowances. This stronger
software receipt boundary is not a fresh native rerun of those results.

## Exact next action

Freeze/review one genuinely distinct small live robustness protocol using this
existing API and saved245chooser. Prefer two declared clean/retained-corruption
arms with identical original eight mixed packets/delay2/nine opportunities/
contention2,6; change only the marked row7 target. Charge both new live-owned
prefixes, all source restoration/bank/monitor/private readouts/copies/IO/service
work and permanent failures. Record actual optimization exposure and refrain
from a matched-exposure conclusion if it differs. Keep original source/seed/
optimizer/actions/metrics/thresholds, no promotion for these combinations,
final/probe/controller-fit/PC-CPC/physics0, no ranking of controllers.

This answers actual learning/old-task retention during noisy service; it is not
the spent248five-policy deferred comparison, old59/60fixed-dose study or253clean
publication repeated under a new name. Freeze explicit limits and first-failure
no-retry policy before models; a negative or unequal-cost result is valid. This
software ticket grants no native allowance. Do not add a capture/CLI/audit layer
or shorten delays/extend horizons to force success. Full R2/R4/R7, loss/relearning,
long-duration deployment, comparative science and release requirements remain open.
