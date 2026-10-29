# Explicit saved catalog comparison command

R7.5-CAT.CLI / [issue249](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/249).
Single-agent source, procedure and result **self-review**, not independent
acceptance. Full repository goal remains active.

## Delivered behavior

The existing `scripts.run_bp_ticket_serving.main` accepts `--catalog-fit` only
with `--controller-comparison --fitted-report`. It forwards `catalog_fit=True`
to the qualified comparison API. Default four-trial and policy-change calls
retain their original arguments. Invalid combinations refuse before IO;
exclusive output creation, successful/failed JSON and exit codes, exception
propagation, and the original 8MiB ceiling remain intact.

Why this: the six-trial fit already has a qualified API consumer (#247), but
ordinary CLI users could not select it. This change adds only argument parsing,
mode validation and dispatch, with no new runner, learner or selection policy.
Usage is in [contextual-controller](../contextual-controller.md#comparison-with-a-saved-catalog-fit).
An outdated introductory claim that the catalog study was unexecuted was corrected
to reference its existing results; no frozen report was rewritten.

## Executed checks

Artifacts: `artifacts/runs/catalog-comparison-cli-20261027/` (local, ignored).
The small meter is copied from the previous comparison checks; its work caps
are zero, its cumulative body cap is60s/platform and sampled RSS cap512MiB.
Optional Torch/Gym/MiniGrid/ManiSkill/Sapien imports fail immediately. Tests use
literal results at CLI dispatch, not archived fits, native learners or teachers.

The exact four target arguments used on both platforms were:

```text
tests/test_catalog_comparison_cli.py
tests/test_ticket_controller_comparison.py::test_cli_requires_explicit_fit_and_protects_output_without_changing_other_modes
tests/test_reference_controller.py::test_reference_cli_optin_preserves_original_dispatch_and_occupied_output
tests/test_ticket_controller_learning.py::test_cli_is_optin_exclusive_and_preserves_failed_result
```

Windows command: `./.venv/Scripts/python.exe -B artifacts/runs/catalog-comparison-cli-20261027/check.py win-final <the four targets above>`.
Linux command, followed by the same four targets:

```text
wsl.exe -d Ubuntu --cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/catalog-comparison-cli-20261027/check.py linux-final
```

Complete argument arrays and pytest arguments are saved in `win-final-checks.json`
and `linux-final-checks.json`, with JUnit XML. Each platform passes **28 tests**
(25 new cases, three existing pure CLI controls), zero errors/skips. Windows
final body1.797726800s/peak87,302,144B; Linux9.534353693s/peak93,130,752B.
All recorded scripted fits/rows, predictions/rows, trial callbacks, teacher
dispatches and native import attempts are zero. No genuine model, simulator,
probe, sealed task, scientific payload read, training or replay work executes.

Preserved intended RED command replaces the four targets with
`tests/test_catalog_comparison_cli.py::test_should_describe_explicit_catalog_opt_in`
and name with `win-red`. It fails on the missing help flag before implementation:
EXIT1/one failure/1.910829400s/peak72,966,144B, all work counters zero.
Windows total29 instances includes this failure; cumulative body3.708556200s.
No favorable run selection or hidden numerical work is involved.

Final checks on these three files:

```text
scripts/run_bp_ticket_serving.py
tests/test_catalog_comparison_cli.py
artifacts/runs/catalog-comparison-cli-20261027/check.py
```

```text
./.venv/Scripts/python.exe -m ruff format --check <three files>
./.venv/Scripts/python.exe -m ruff check <three files>
./.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent <three files>
./.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent <three files>
```

All four pass; exact arrays, outputs and durations in `static-final.json`
(12.116371300s total). First format check failed on line endings/line wrapping
in the copied meter; first lint and both type checks passed. Formatting only
that meter repaired it; production/tests were already formatted. Initial output
is retained separately. The prior behavioral runs remain applicable: the later
meter edit changes only whitespace. No additional fixture/model reruns.
Initial grouped static timing was not captured; its duration is unavailable,
not zero, and is not included in the measured final total. Final mtime-selected
project mypy cache113,233,920B is below the160MiB touched-cache allowance; this
is neither growth nor the total inherited cache628,187,023B.

## Review and limits

Compared the final CLI and usage with session-start copies, preserving inherited
dirty edits. Production delta is the new argument, pre-IO validation and one
conditional keyword. No core, source admission, schedule, catalog, evaluation,
model, source-pin manifest, budget or scientific record was changed. Checked
the actual output branch and final tests: the ceiling boundary passes at exactly
8MiB, rejects one byte over, and preserves the existing placeholder on failure.
Occupied output never dispatches; returned incomplete reports persist and return1.
Comparison exceptions propagate without fabricating success.

The initial metadata review script incorrectly asserted that the entire
inherited cache fit the touched-cache allowance and raised `AssertionError`
at line26 after the AST/cohort/static checks passed. Its following shell command
returned0; that does not erase the script's failure. A single metadata-only
observation corrected the accounting distinction, recorded in `review.json`
and `cache-observation.json`; no production change, cache deletion, retrospective
limit increase or test/model retry followed.

The accepted #247 API141checks/platform and actual saved-fit metadata admission
are reused; they were not repeated. The negative #245 fit and failed/spent #248
native comparison remain unchanged. No native CLI qualification or comparison
outcome follows from these software tests. Existing results establish no
circadian/controller advantage, matched compute, unseen generalization,
reliable sequential relearning or release readiness.

Skipped full suite, clean clone, wheel/install, browser, independent review,
native CLI/application/comparison, old mathematical/baseline cohorts and
scientific/release gates: this change is confined to the command boundary.
No dependencies/install/download/GPU/CI/agents/Git mutations/reset/raw upload.
Historical plans/log stay frozen under current AGENTS; live issue249/tracker18
receive the handoff. Broad #21/#23/#24/#26/#31/#33 and the full goal remain open.
An exploratory `rg` call using shell wildcard filenames failed on Windows;
inspection continued with the exact existing file, with no test/model work.

## Critical path and precise next action

The private fit, deferred stream, saved-fit API and command are implemented
within their qualified scopes; do not start another equivalent loader/CLI/audit
increment or retry #248. Useful live application remains separate. Actual
`_reference_schedule` rejects `catalog_control` with `live`, and `run_case`
refuses catalog policies with `serving_plan`; current live serving uses a fixed
reference. This is a concrete R7.7 integration dependency, not a missing runner.

Next implementation action: start pure causal/contention tests for a frozen
catalog policy on the existing `LiveServingPlan(opportunities=9)` path in
`run_bp_ticket_serving.py`, before changing those two refusals. Keep trials,
deferred comparison, publication and final evaluation separate. Reuse the
existing live lease/timeline and paid evidence guards, immutable nine-decision
envelope, original budgets/charges and private candidate; no forced drain,
budget renewal, new model/dose or serving outcomes used as labels. Prove those
boundaries before declaring a genuinely distinct small native live-serving
protocol. This can deliver concurrent service and learning; it does not close
the broader long-duration/noisy/poisoning requirement or authorize a rerun of248.
