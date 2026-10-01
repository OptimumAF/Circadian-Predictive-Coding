# P6.11b2: Exhaustive independent confirmation seed, interval and cost report

Completed: 2026-10-01. **P6.11b2 and P6.11b are complete.** Both actual full
publications and both independent complete readbacks passed. The original b
acceptance and all scientific settings are unchanged. Original reporting
parents remain unchecked pending their own criterion-level evidence audit.

## Module boundaries

```text
src/app/continual_confirmation_report_cost_binding.py
  Complete original cost identity and compact arm/context reference binding.
src/app/continual_confirmation_report.py
  Frozen analysis repetition, complete coverage and every cost/outcome join.
src/app/continual_confirmation_report_rendering.py
  Deterministic exhaustive Markdown with raw numbers and eligibility reasons.
src/infra/continual_confirmation_report_bindings.py
  Current source/request/input binding and unchanged complete-reader ports.
src/infra/continual_confirmation_report_artifacts.py
  Exclusive publication, failures and independently rebuilt full readback.
scripts/run_p611_confirmation_report.py
  Fixed local CLI supplying unchanged complete cost/scored readers.
```

App depends inward and owns no IO/data/model/scoring. Infrastructure owns
artifact/provenance gates; the CLI supplies actual reader ports. No dependency,
environment variable or scientific override is added. ADR-0163 records rationale.

## Frozen report protocol

The 106-source consumer extends unchanged scored/cost pins. All original
scientific settings and analysis contract `5e33ef28...6b594b1` remain fixed.
The prospective source/request record was written before fabricated report
fixtures: `artifacts/runs/p611-confirmation-report-source.json`, 40,465 bytes,
SHA `971b2a18054d6d0b15763d0af1ccea93c261ddd1a1d6320a313a0cef6ed48fe6`;
source map `67063e2b13329d8fb86654cbb4acf4c4b348db0d98e1f9e7abeca5e265058517`.
It remains a historical prospective record with correctness/publication flags
false; completion requires separate terminal evidence.

The compact original cost input is 1,608,771 bytes, SHA
`865112555b1323aeb873521053e51cc163b142ca82a6aa4a4d72aa08b01a3383`.
A clean metadata-only test fixture carries those original unscored costs.
Each actual operation uses complete current readers, not that fixture.
All 60 shared proof contexts retain exact pointers into both 112,635,395-byte
original cost artifacts at SHA `214ce7ad...d81dd`; standalone projection is
`42bbe904...27868`. Full shared context stays available in those local artifacts.

## Contents and interpretation

- All 560 raw cells, three endpoints and six derived values per cell, with
  original roles, failures and undefined-retention reasons.
- All 56 arms × six metrics and 58 contrasts × five metrics: **626 vectors**,
  each retaining all ten planned seeds, **6,260 observations** total.
- Every planned/observed count, mean, sample SD, standard error, observed range,
  marginal interval and all 116 predeclared simultaneous primary statements.
  Intervals are raw/unclipped; missing or zero-dispersion vectors retain reason
  and have no eligible interval. Retention is descriptive.
- Paired differences are computed within source seed, left minus right. Means
  prefer higher values; signed forgetting prefers lower values, with weak
  initial-A performance still visible. No winner/ranking or family pooling.
- Every per-arm raw cost field, actual initial/after-A/after-B capacities and
  original shared work/storage context. Rejected-executed work is counted.
- Both complete scored run/resource/request/artifact facts. Whole-run observed
  wall/RSS remain separate from arm costs. Repeats add no seed replications.

## Publication and readback

```powershell
.\.venv\Scripts\python.exe -m scripts.run_p611_confirmation_report --publish --output-dir artifacts/runs/p611-confirmation-report
.\.venv\Scripts\python.exe -m scripts.run_p611_confirmation_report --read-only --output-dir artifacts/runs/p611-confirmation-report
```

The fixed CLI has no seed/method/metric/alpha/cap override. Publication refuses
occupied request/result/Markdown/audit/failure/claim paths. Input byte/marker and
current source/environment/command checks surround actual complete readbacks.
The audit follows successful JSON/Markdown publication; late failure or failed
failure-marker IO cannot leave our completion audit readable. Preserve every
partial/foreign output; only unchanged owned completion audits may be revoked.

Independent readback requires all four successful files, no failure/claim, and
rebuilds the entire report from both complete original scored readers and fresh
complete cost evidence. It checks complete body, Markdown, audit and current
request/input/source bytes. The local derivative budget is 180 seconds and
does not replace any original scientific cap. A separate bounded harness owns
hard timeout verification for the actual acceptance commands.

## Actual complete publication and repetition

The prospective source freeze above was followed by the final correctness
gate: **101 new / 446 related tests passed in 37.82 seconds, zero skipped**.
Ruff passes over src/tests/scripts, all eleven new Python files pass format,
mypy passes over 439 files, and `git diff --check` passes. Fixtures prove
arithmetic and failure behavior; the actual complete reader operations below
establish the separate original-artifact/source authority.

The bounded producer ran four sequential child processes, each with a hard
180-second timeout. Every process exited 0 and all 24 data/model/train/final
guards recorded zero calls. Every operation uses two unchanged complete
scored readers with fresh complete cost evidence: four original training
readbacks and two scored readbacks per operation, sixteen and eight total.

| Operation | Child seconds | Parent seconds | Exit |
| --- | ---: | ---: | ---: |
| Canonical publication | 161.8299165 | 162.4241344 | 0 |
| Repeat publication | 161.8716003 | 162.4786186 | 0 |
| Canonical independent readback | 161.0518800 | 161.6595500 | 0 |
| Repeat independent readback | 160.6489414 | 161.2716384 | 0 |

Both JSON bodies and both Markdown files are **byte-for-byte equal**. The
complete unchanged analysis repeats at SHA
`9796965cef1b5e1b3604d6ba69f84d05d6e6cc1e1b7d70ec298c91ba90f47b5c`,
1,885,930 bytes. Request paths/times and derivative audit durations differ;
these are preserved run metadata, not additional seed replications.

| Evidence in artifacts/runs | Bytes | SHA-256 |
| --- | ---: | --- |
| p611-confirmation-report/confirmation-report.request.json | 26,104 | 57db44ded2fdc9a0011dac61ac8f0b11e55f6e831df21215ba2f98003054aff7 |
| p611-confirmation-report-repeat/confirmation-report.request.json | 26,111 | 4e5bb5956f4a8d35274054bf08ec5cf32349ca43e999e93c1fa6ec4d9fa1733e |
| Both confirmation-report.result.json files | 7,537,678 | 363ed97dba808281d13224526b4b36614ae452188cade256992b530b6ab03088 |
| Both confirmation-report.md files | 2,958,568 | c989df16853868e910bf99303470927d17c6f01ed86e5dd004d35692bf03fddc |
| p611-confirmation-report/confirmation-report.audit.json | 4,474 | 5bd1f8d641b7658ef661ded7a633e1ce5a04026b64380215e91b1832f3585486 |
| p611-confirmation-report-repeat/confirmation-report.audit.json | 4,475 | 0d6b2fd124c7029b52d86623533c6bef5ecac51cbdffae448c8a1fe777d0a31a |
| p611-confirmation-report-validation.json | 17,011 | 031f398f03115cc2967be7b4c6a571ac5d8c822910bc0bd55ce74f500495a9d8 |

Validation metadata binds every command/exit/time, artifact identity, all
106 source pins, six test/fixture identities and the bounded producer
(8,520 bytes, SHA `9de6a5098299fc5be23b6d5c570682b85d94551ceeb5437a77f49cf182bc44a2`).
All original training/scored files and complete cost evidence remain local
ignored artifacts. The new metadata fixture contains original unscored costs
only; no reserved outcome is added to clean tests.

## Complete outcome and eligibility accounting

All 560 cells and 1,680 endpoints succeeded. All 626 vectors / 6,260 planned
seed observations and all 116 primary statements remain present. Interval
statuses across vectors are estimated 535, zero observed variance 35,
descriptive only 54 and incomplete observations 2. There are 105 eligible
simultaneous primary intervals. The other eleven primary statements are
retained with observed mean/SD zero and the frozen zero-dispersion reason:

| Family | Left minus right | Primary endpoint without eligible CI |
| --- | --- | --- |
| replay | backprop_on − backprop_off | final mean accuracy |
| sleep | homeostasis_only − neutral_sham | both primary endpoints |
| schedule | backprop_periodic − backprop_no_sleep | signed A forgetting |
| schedule | backprop_adaptive − backprop_no_sleep | both primary endpoints |
| schedule | backprop_adaptive − backprop_periodic | signed A forgetting |
| schedule | pc_adaptive − pc_no_sleep | both primary endpoints |
| schedule | neutral_adaptive − neutral_no_sleep | both primary endpoints |

The two undefined retention observations are schedule seed 233 / pc_12_no_sleep
and parent seed 373 / pc_13_off. Both have `A_after_A = 0`; each retains its raw
endpoints and `zero_a_after_a` reason. No denominator or interval rule changed.

All **935 negative observations** remain. This count includes signed forgetting
and paired metric differences: its sign alone is not a generic win/loss claim.
The report retains weaker initial-A values beside forgetting, all rejected work,
inactive policies, null contrasts and raw unclipped intervals. Model-based
interval assumptions and the planned ten-seed replication unit remain explicit.
No seed, comparison, interval, baseline, metric or scientific cap was selected
or changed to favor circadian.

## Commands and scope of verification

The actual bounded producer command was:

```powershell
.\.venv\Scripts\python.exe artifacts/runs/validate-p611-confirmation-report.py
```

It deliberately refuses occupied publication paths. Preserve both completed
bundles and use the public `--read-only` command above for reuse. It rebuilds
the full report; merely hashing saved files is a weaker preservation check.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests/test_continual_confirmation_report.py tests/test_continual_confirmation_report_rendering.py tests/test_continual_confirmation_report_bindings.py tests/test_continual_confirmation_report_artifacts.py tests/test_p611_confirmation_report_cli.py tests/test_continual_confirmation_analysis.py tests/test_seed_statistics.py tests/test_continual_confirmation_scoring_validation.py tests/test_continual_confirmation_report_costs.py tests/test_continual_confirmation_report_cost_references.py tests/test_p611_confirmation_cost_inspection.py tests/test_continual_confirmation_training_references.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_report_cost_binding.py src/app/continual_confirmation_report.py src/app/continual_confirmation_report_rendering.py src/infra/continual_confirmation_report_bindings.py src/infra/continual_confirmation_report_artifacts.py scripts/run_p611_confirmation_report.py tests/test_continual_confirmation_report.py tests/test_continual_confirmation_report_rendering.py tests/test_continual_confirmation_report_bindings.py tests/test_continual_confirmation_report_artifacts.py tests/test_p611_confirmation_report_cli.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

Full repository tests, fresh-clone/actual CI, Torch/CUDA runs, new training,
new final-source access and sweeps were skipped. This consumer changes no
model, dataset, scientific runner or dependency. The 180-second measurements
are derivative validation costs; original wall/RSS/work facts and scientific
caps remain historical complete-validated observations.

Closing preservation command
`python artifacts/runs/validate-p611-confirmation-report-handoff.py` exited 0.
Its exclusive record is
`artifacts/runs/p611-confirmation-report-handoff-validation.json`, 19,447 bytes,
SHA `4ca7e899a33e5a853451988c9a30b0c4712ed47835dcb649fac369e1afdf08b8`.
It rechecks all 106 frozen sources, test/producer bytes, current requests and
all original input/report identities, exact analysis/cost/render consistency,
absent markers, unchanged HEAD and correct checked/unchecked task IDs. This
closing check establishes byte preservation; the four actual operations above
establish the separate full original scientific readback evidence.

## Remaining acceptance and exact next action

This historical b2 increment closed b2/b. The subsequent
[four-criterion current reporting audit](phase6-reporting-acceptance-audit.md)
also closes the original P6.11 parent after source/input/report evidence is
revalidated. P6.9 subsequently passes its explicit stage/task matrix
presentation and original scope audit: the current prospective two-task
contract defines three endpoints, with B-after-A unmeasured before arrival.
Do not infer that
cell or open a new final view to fill it retrospectively. P6.10 needs the
accuracy/forgetting versus compute/memory presentation and explicit accounting
of unmeasured per-arm wall/RSS/guard durations. Whole-run RSS is not per-arm
RSS. P6.12 needs explicit supported/rejected/unresolved hypothesis statements
and development-versus-confirmation interpretation. P6.7/P6.3/C9 retain their
separate original criteria.

The subsequent [P6.9a matrix consumer](p69-confirmation-matrix.md) now passes
556 related tests, two actual publications and both independent complete
readbacks, all exit 0 within 180 seconds each, with exact whole JSON/Markdown
repetition. It preserves all 560 matrix rows,
1,680 endpoints, unavailable B-after-A and original metrics/evaluation seal.
P6.9a and P6.9 are checked after complete evidence and the separate original
scope audit, without adding a retrospective future-task measurement.
Next inventory every original resource field/scope under P6.10a and implement
P6.10b's compute/memory presentation, followed by explicit H1–H4 conclusions.
P6.10/P6.12 retain their separate acceptance. Add new presentation modules
through the verified report boundary; do not edit pinned scientific sources
or start new experiments to fill missing fields retrospectively.
