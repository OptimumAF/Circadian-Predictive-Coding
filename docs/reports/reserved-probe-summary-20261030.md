# Reserved-task outcome summary — R2.1-TW.SUM / #326

Delivered software and saved-data evidence, 2026-10-30. Live acceptance:
[issue326](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/326).
Fullgoal/#18/#21/reliable acquisition/retention/recovery/science/release OPEN.
#323 remains FAILED/SPENT; no retry or renamed scientific case. Previous turn
#325 was PROGRESS (local delivery); its frozen wheel omits this new summary.

Physical checkout D:/Dev/Repositories/Circadian, HEAD
d9119e80926a139a2f38460564d7eab1b6d2a2b0 plus inherited dirty/untracked work.
No commit/push/install/download/global settings/deletion. Project commands and
outputs stay on D, TEMP/TMP D:/Dev/_Temp; WSL TMPDIR /mnt/d/Dev/_Temp.

## Working behavior

- `src/core/probe_task_outcomes.py`: pure signed changes, acquisition, retention,
  loss and conditional recovery. Recovery requires before=True and after=False;
  improvement on return alone is insufficient.
- `src/app/reserved_probe_summary.py`: fixed source-trained4 vs fresh2, bounded
  exact-role headers admitted across ALL phases before outcomes, complete-phase
  comparisons and detached selected work counters. Only exact ordinary zero-
  training episode booleans qualify. Foreign/final roles, duplicate/order/cohort
  changes and malformed successes/work fail explicitly. Partial observed prefixes
  remain visible; aggregates/comparisons using incomplete six-task phases are null.
- Runner appends `reserved_probe_summary` after original source/failure/cost
  bookkeeping. Original `summary`, raw episodes, complete/error/work stay intact.
  Malformed projection yields `status: invalid` without erasing original failure.
  No learner, training, promotion, probe selection or evaluation access is added.
- Costs separate source, auxiliary CE fit, private F/return and typed readouts.
  Missing stays null. Counts are not FLOPs/matched total computation: preserve
  raw timing/memory/hidden-extraction/copy/receipt/permanent-budget evidence.

Why this: the existing source A/B summary groups different tasks. Keep its
definitions/directions, put small task arithmetic in core and compose reserved
old4/fresh2 records after work. No second runner or learner. Summary completeness
is neither scientific success nor authenticated source preservation.

## Checks, exact commands and failures

Stage `artifacts/runs/reserved-probe-summary-20261030/` retains frozen PROTOCOL.md,
one-off helpers, exclusive receipts, exact static argument arrays and projection.
ZERO model work: tests/projection block NumPy/Torch/torchvision imports. No model
construction/parameters/optimizer/forward/backward/replay/simulator/final/study.

Windows (first receipt windows-work.json, final command below):

```text
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-summary-20261030/check.py artifacts/runs/reserved-probe-summary-20261030/windows-final-work.json
```

Linux final:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/reserved-probe-summary-20261030/check.py artifacts/runs/reserved-probe-summary-20261030/linux-final-work.json
```

50 distinct metadata cases/platform final PASS,0FAIL/0SKIP: nine conditional
change examples, exact bools, cohorts, loss/acquisition, missing/partial phases,
header-before-outcome refusal, roles/final gate, ordinary zero-update episodes,
malformed counts, paid failed fit costs, detached output/original error/work.
Final Windows .6073389s/49,393,664B RSS; Linux4.422094996s/37,314,560B.
Initial Windows also50PASS/.6775506s/49,278,976B. All successful receipts contain
no numerical imports;150 successful case executions,50 distinct cases.

First Linux command used capture and linux-work.json: EXIT1 during pytest fd-
capture cleanup (`tmpfile.truncate` FileNotFoundError on mounted D). Internal
receipt did not survive; outcome counts/RSS UNKNOWN, not zero/passing. Full
shell7.1944114s conservatively charged in linux-capture-failed-work.json.
Prospective repair disabled capture (-s), added exception-safe receipt handling,
preserved original model/storage/test limits. Total test time including failure
12.901395896s<120s. Successful processes meet30s/256MiB; failed RSS unverified.

Scoped statics target exactly core/app modules above, runner and newtest:

```text
.venv/Scripts/python.exe -B -m ruff check <four files>
.venv/Scripts/python.exe -B -m ruff format --check <four files>
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent <four files>
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux <four files>
```

Full literal invocations/output/timing in statics-first.json/statics-final.json.
First6.4677303s: Ruff/format PASS, both mypy platforms found2Optional annotations.
Explicit annotations fixed these. Final all4PASS/6.3599564s; initial formatting
.1257705s; total12.9534572s<90s. One-off artifact helpers self-reviewed, not scoped
statics. Exploratory nonexistent report/static reads and rg accidentally given
PowerShell -ErrorAction ended EXIT1; simplified verified-path reads followed.
One documentation patch context mismatch applied nothing and was corrected.
These tooling failures changed no evidence or budgets.

## One saved-data projection

After final software gates, self-review and independent source review:

```text
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-summary-20261030/project_once.py
```

ONE call EXIT0/.081403600s/30,662,656B RSS,5fixed inputs/8,345,054 decoded bytes.
Compressed bp72,162B,pc67,951B,neutral97,182B,active97,114B,failed32338,349B;
total372,758B. Largest decoded2,064,663B. Original10s/128MiB/256KiB compressed/
4MiB decoded perfile/20MiB total held. New stage budget2MiB. All5 original
compressed bytes unchanged. No parameter reconstruction or new hashes. Output
projection.json preserves original complete/error; projection-work.json costs.

| Saved source | Old4 successes before/F/return | Fresh2 successes before/F/return | Interpretation |
| --- | --- | --- | --- |
| Linux315 BP | 0/0/0 | 0/0/0 | No acquisition; retention/recovery inapplicable |
| Linux315 PC | 0/0/0 | 0/0/0 | Same negative outcome |
| Linux315 neutral CPC | 0/0/0 | 0/0/0 | Same negative outcome |
| Linux315 active CPC | 0/0/0 | 0/0/0 | Same negative outcome |
| Failed323 CE | unknown/unknown/unknown | unknown/unknown/unknown | Private phases not run; not negative learning evidence |

Each315 keeps source12updates72rows, private32/144, F16/48,return16/96, eight
recorded score-forward calls per training phase,36typed decisions/native forwards
per readout phase (108 total). Readouts are additional to probe scoring, not free.
Absent auxiliary records remain unknown.323 keeps paid source12/72; missing
private/aux counts unknown, original error preserved. No seed/metric/cohort/
horizon/direction/native outcome changed. No CPC advantage, equivalence,
generalization or useful adaptation claim; this is no new comparative experiment.

## Review, skips and next action

Lead self-reviewed combined core/app/runner/tests and bounded projection. Finite
independent arithmetic/role/uncertainty/results review is recorded in owningissue
and stage review.txt, distinct from lead self-review. No native rerun, archive/
model/baseline reconstruction, full suite, clean clone, fresh dependency install,
package rebuild or release checks. Reuse321/322/324/325 evidence only where
relevant source/assumptions remain unchanged;325 contains no new summary.

Independent review ACCEPTED source/actual arithmetic/roles/cohorts/unknowns/costs
against all5 original reports. Four read-only commands .7240880s total, no sessions:
exits0,0,1,1. Third helper treated a list-shaped static receipt as a dict; fourth
verified all5 projections before another static-field assumption failed. Original
3command cap was prospectively amended to ONE extra<=2s read within unchanged20s
after .4980819s spent (GitHub receipt6529996558/PROTOCOL). Fourth.2260061s.
Both review helper failures preserved. Independent final static receipt reading
remains incomplete; final4staticPASS is lead-executed/lead-verified, not a fresh
independent static run. No more tooling retries. Review exception ended.

Critical path: original bounded R2.1 diagnostics/isolated probes already exist;
native BP already uses categorical CE. Four-arm315 is negative;323's native CE
trial is spent. No horizon extension, repeated summary/audit/capture or trialretry.
The simpler MinimumChangeProbeLearner already exists in ticket_linear_probe,
while ordinary reserved-probe admission accepts only BP/PC/CPC. Next exact
implementation action: compose that existing isolated minimum-change LS fresh-
task/return owner into the ordinary runner as an explicit simpler control,
starting with role/state/budget/admission tests, preserving current defaults.
Reuse310/314, no baseline/323 rerun. Any later experiment needs its own genuinely
different question/frozen protocol; this is a product task, not a renamed CEtrial.
