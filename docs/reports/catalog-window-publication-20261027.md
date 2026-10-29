# Guarded publication after a live window

R7.7-CAT.P / [issue252](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/252).
Implemented and software-qualified with source/procedure/result **self-review**,
not independent acceptance. Native publication and the full repository goal
remain incomplete. Existing251native private learning and248failed/spent
comparison evidence are reused/preserved; neither was rerun.

## Behavior and affected boundaries

`run_case(..., catalog_policy=frozen, serving_plan=LiveServingPlan(9,(2,6)),
publish_after_window=True)` uses the original promotion guards/controller after
the complete, known, closed live window. It makes no extra TRAIN decisions or
new owners/budgets. The existing fully drained and default private paths remain.
Wrong flag types and incompatible modes fail before IO. There is no new CLI.

```text
scripts/ticket_catalog_publication.py   one bounded terminal phase; no runner/model/policy
scripts/ticket_catalog_control.py       retired-work refusal; distinct cost-format refusal
scripts/run_bp_ticket_serving.py        opt-in, integration and separately guarded readout
tests/test_catalog_window_publication.py actual controller/evaluator with literal ports
docs/adr/ADR-0641-...md                  rationale for retirement with pending labels
```

The helper freezes one bounded paid prefix using the existing value/identity
guard and full inbox/native snapshots. It validates all preceding receipts,
pending IDs/versions, original owners/resource bindings and the closed service
records. Pausing the original gate changes only its pause flag; admissions,
deferrals, limits, budgets and charges must remain exact. It calls the original
`_publish_candidate` and `ServingPromotionController.prepare/commit` on the
original independent inner guards. No live outcome or private endpoint becomes
a promotion label, final/probe source or controller selection input.

Known acceptance requires the exact original controller receipt identity plus
its preserved values, generation/previous/new version and full actor model/
configuration/metadata. Known rejection leaves the full actor unchanged. All
old experiences/labels/receipts remain; pending packets stay unapplied. The old
chooser/runtime retire under the original lease, and the inbox preserves its
historical update count. No successor is materialized. A runtime is marked
retired before the ledger retirement operation, so a failed ledger cleanup
still refuses old training even if someone resumes the gate. That case remains
incomplete, retains a known commit separately and performs no post-readout.

Four ordinary post-window episodes use the exact verified publication, with
checks before and after predictions. Old choice/training APIs refuse; corruption,
equal-valued replacement, unexpected counter/pause changes or unknown commit
stops readout. No automatic rollback/retry. The new report has separate original
training-window and final sharing snapshots. Unknown promotion count/actor
preservation is null rather than fabricated zero/true. The old private utility
and cost helpers refuse the new format, including known rejection, because
guard computation and post-window service cannot be silently omitted.

Why this:251already demonstrated native private learning, but its live actor
still failed legacy and two packets remained pending at the immutable cap.
The new phase permits guarded use without extending that cap or relabeling
those packets. [ADR-0641](../adr/ADR-0641-retire-catalog-window-before-serving-published-policy.md)
records alternatives and the intentionally terminal scope.

## Executed checks and exact commands

Local ignored receipts: `artifacts/runs/catalog-window-publication-20261027/`.
Qualified coverage **172distinct cases/platform**:34new,128affected private catalog/live/
stream/pure timeline and10pure drain controls;0final errors/skips. The combined
169cases passed before the final focused hardening below. The real teacher
bank check and two genuine native live fixtures were explicitly deselected.
No scientific payload, optional native implementation, real teacher or simulator
work executes. The new tests use the actual original evaluator/controller and
real live workers, with scripted learner/guard predictions and literal public
decisions. They demonstrate mechanics, not native learning or task benefit.

The combined command target arguments are:

```text
tests/test_catalog_window_publication.py
tests/test_catalog_live_serving.py
tests/test_ticket_catalog_control.py
tests/test_catalog_ticket_stream.py
tests/test_ticket_serving_timeline.py
tests/test_drained_ticket_publication.py::test_verified_publication_invalidates_feedback_without_rebinding_or_losing_old_evidence
tests/test_drained_ticket_publication.py::test_guard_refusal_preserves_original_feedback_and_never_retries_at_tail
tests/test_drained_ticket_publication.py::test_uncertain_or_mismatched_publication_is_preserved_and_cannot_be_retried
tests/test_drained_ticket_publication.py::test_factored_original_publication_uses_same_guards_ids_metadata_and_no_live_outcomes
-k 'not compile_only_the_original_complete_train_bank and not native_live_episode_versions_original_causal_receipts_and_priorities and not native_live_task_records_do_not_become_feedback_or_promotion_guards'
```

Windows: `./.venv/Scripts/python.exe -B artifacts/runs/catalog-window-publication-20261027/check.py win-combined2 <targets above>`.
Linux uses the same targets after:

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-window-publication-20261027/check.py linux-combined
```

Each combined run:423scripted fits2862rows/579predictions4106rows/15trialcallbacks/
1459pure workflow dispatches. Windows8.793670300s/peak90,923,008B;
Linux35.964630903s/79,548,416B. Genuine Native/Torch/teacher/physics/final/probe/
scientific payload work0. Optional-native import trap and counters are in the
small reused meter; process RSS sampling precedes pytest/fixture imports.

Self-review found that `case_work` could admit a rejected new report as old
private costs while omitting guard forwards. Added an explicit format refusal
and two accepted/rejected assertions. On the final tree, both platforms passed
those two affected selectors under names `win-cost-boundary` and
`linux-cost-boundary` (same meter/native-free):

```text
tests/test_catalog_window_publication.py::test_should_publish_after_window_without_draining_or_reusing_pending
tests/test_catalog_window_publication.py::test_should_retire_rejected_window_without_retraining_or_publishing
```

Each12scripted fits72rows/24predictions112rows/118dispatches; Windows2.502679000s/
77,094,912B, Linux7.927022653s/70,184,960B. The remaining167cases are unaffected:
the change only rejects the new format in this consumer, which the producer
does not call. A test-only explicit type annotation on corrupted receipt kwargs
changes no runtime behavior. Reused the combined169evidence accordingly; no
unnecessary full rerun or native scientific work.

Final exact-type self-review found Python equality can treat `True` and `1`, or
`102` and `102.0`, as equal. Gate/configuration/metadata comparisons now use the
existing strict structural comparison. Three malformed-type cases were added.
The first cohort exposed a genuine snapshot alias: the actor's serving snapshot
shares its frozen configuration, so corruption affected both sides of the check.
Detach the entire prepublication snapshot with the existing `deepcopy`; the
core API and fault criterion remain unchanged. Both platforms initially had
3pass/1fail (Windows3.242381700s; Linux9.365395179s), each24scripted fits144rows/
48predictions224rows/224dispatches. These failed receipts remain intact.

On the final tree, `win-detached-before` and `linux-detached-before` each PASS
5cases: valid acceptance, valid rejection, malformed metadata, malformed
configuration, and malformed pause flag. Exact selector after the same meter:

```text
tests/test_catalog_window_publication.py -k 'metadata_float or config_float or pause_type or should_publish_after_window_without_draining_or_reusing_pending or should_retire_rejected_window_without_retraining_or_publishing'
```

Each30scripted fits180rows/56predictions272rows/259dispatches; Windows3.118108800s/
77,537,280B, Linux12.303844065s/70,459,392B. Reuse the other167passing cases:
the final changes strengthen value-type comparisons and detach the prepublication
snapshot only; no valid input, default path, core policy or numerical operation
changes. Final affected checks exercise both successful outcomes and all three
new fault cases; this is scoped evidence reuse, not a fresh full-suite run.

## Failures, revisions and total fixture costs

All attempts/JUnit/exact argv remain in the stage:

- Intended RED: missing `publish_after_window` keyword, EXIT1/2.138647800s,0work.
- New fixture used the older catalog literal digest at a SHA256-only promotion
  port: EXIT1/1.833057300s;6scripted fits36rows/8predictions48rows/47dispatches.
  Fixed the fixture's promotion digest; source binding/production check unchanged.
- Fixture numerical validity returned `np.bool_`: EXIT1/1.589399200s, same work.
  Explicit bool fixed the fixture; exact production boolean validation retained.
- Actual helper bug compared identity-only `PromotionReceipt` with its deepcopy
  using `!=`: EXIT1/1.494420700s,6fits36rows/12predictions56rows/47dispatches.
  Fixed to original object identity plus existing exact structural comparison.
  Failed physical commit remains an unknown/incomplete test result, not success.
- Initial16new cases then PASS/2.446578900s. Expanded fault cohort stopped after
  23pass/1fail/6.311623900s (90fits522rows/176pred864rows/812dispatches): the
  budget fault changed an unused attribute. Corrected the injection to actual
  `session.budget`, with no production weakening; final combined checks passed.
- Formatted/out-of-order patch anchors failed with0edits; inspected exact source
  and applied ordered small hunks. Initial guessed paths/Windows wildcard errors
  led to exact-path searches, no execution/model work.

Original6-invocation engineering allowance changed prospectively to8/platform
at issue comment6449268716 after7.055525s and explained fixture/receipt failures;
120sbody/512MiB/work limits stayed fixed. Further prospective amendments8→9
(6450175063) and9→10(6450235606) cover the exact-type cohort and demonstrated
configuration alias repair. No failed attempt or scientific allowance was reset.
Native allowance remained0.

| Cumulative software work | Windows | Linux |
| --- | ---: | ---: |
| Invocations / body seconds | 10 / 33.470567600 | 4 / 65.560892800 |
| Peak process RSS | 90,923,008B | 79,548,416B |
| Scripted fits / rows | 645 / 4176 | 489 / 3258 |
| Scripted predictions / rows | 1007 / 6178 | 707 / 4714 |
| Trial callbacks / pure dispatches | 15 / 3413 | 15 / 2060 |
| Test instances / failed / errors / skips | 224 / 6 / 0 / 0 | 180 / 1 / 0 / 0 |

Counts include failures and follow-up cases;172distinct cases qualify per
platform. All finite declared software limits hold. Native251/248allowances
remain spent; no models were restored or experiments repeated.

## Static and source/procedure/result review

Final file arguments:

```text
scripts/ticket_catalog_publication.py
scripts/ticket_catalog_control.py
scripts/run_bp_ticket_serving.py
tests/test_catalog_window_publication.py
artifacts/runs/catalog-window-publication-20261027/check.py
```

```text
./.venv/Scripts/python.exe -m ruff format --check <five files>
./.venv/Scripts/python.exe -m ruff check <five files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <five files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <five files>
```

Final all PASS. Initial format/lint passed; both initial mypy modes found two
test-only mixed-kwargs diagnostics. Explicit `dict[str,Any]` documents deliberate
bad receipt injection, no ignore or production type weakening. Initial+final
measured static57.649724900s across all four receipts, including exact-type and
detached-snapshot checks; earlier formatter calls not separately timed/not0.
Exact arrays/outcomes/durations in `static*.json`.
Final storage observation:113,361,844B in cache files with mtime since task start
(selected footprint, not growth); stage249,086B before closeout receipts, within
160MiB/4MiB respectively. Other inherited caches are separate.

Self-review inspected actual combined source, original core ownership/promotion
procedures and executed fault results. Session-start AST is unchanged outside
`run_case`/`_case_measures`, catalog `before_work`/`case_work`; existing CLI,
schedule, core policy/evaluator/executor/timeline/native math are untouched.
`source.diff` and summaries retain that check. Old receipt/label/budget identity
and native clocks are preserved; rejected/unknown/failed-retirement cases do not
silently become successful reports. Inherited unrelated changes remain intact.

Skipped full suite/clean clone/wheel/install/browser/independent review,
genuine native publication, unchanged math/baseline/simulator cohorts and
scientific/release gates. No dependencies/install/download/GPU/remoteCI/agents/
Git mutations/reset/raw publication or archive reconstruction. Historical plans/
log remain frozen under current AGENTS; issue252/tracker18 carry current status.
Broad R2/R4/R7/21/23/24/26/31/33/sequential recovery/long-duration mixed/noisy/
poisoning/scientific/release requirements and the full goal remain open.

## Precise next action

Freeze/review one genuinely distinct native publication qualification using the
saved245chooser, original source/seed/8clean6row/delay1/LiveServingPlan(9,(2,6))
and `publish_after_window=True`. Keep the original guards/policy fixed. Ask
whether accepted/rejected publication is known, old training refuses,2packets
remain unapplied and actual post-publication ordinary tasks use the verified
actor; a rejection/negative result is valid. Declare all guard/model-build/
forward/copy/prediction/IO/time/RSS/storage/first-failure-no-retry limits before
models; no utility/CPC/generalization or efficiency rank.

The original controller requires a live owned candidate. Saved251state/readouts
are not authority to manufacture that runtime/ledger, so this new product
qualification may execute the bounded6update prefix in its original owner,
fully charging it anew. It is not a retrained comparative baseline, tuned run,
retry or continuation of251/248. No native allowance is granted by this software
ticket. Do not add another loader/CLI/capture/audit layer or silently extend
the horizon to obtain an accepted publication.
