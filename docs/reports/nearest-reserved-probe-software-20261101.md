# #400 / isolated fixed-input nearest probe software

The existing continuous reserved-task probe now admits the working fixed-input
nearest learner. It executes one private append through the original delayed
label/runtime path, scores reserved test and up to three old TRAIN episodes before
and after, and returns costs/metrics without an adapted learner. Source and stable
actor stay unchanged. No actual bank, reserved recording, cached model, old raw
result or native episode was read or executed in this increment.

## Delivered and checked behavior

`src/adapters/continuous_probe.py` retains the existing sequence, role/spec/context
admission, budget/clock ownership, receipts and endpoint callback contract. The new
`continuous_nearest_probe.py` leaf supplies nearest state equality/admission,
identity, exact prediction receipts and logical prototype/work accounting, with no
runner, scheduling, IO or promotion. Original nearest learner and core arithmetic
are unchanged; no dependencies or public CLI options were added.

Only an exact nonempty fixed-input TRAIN nearest source is admitted. One append
must fit within its original capacity, at most256prototypes. Other update counts,
fitted-input mode, gradient/velocity options and four old readouts refuse. Old
source/actor/prototype arrays/policies/counters/identities stay protected even
without endpoint callbacks. Only arrived `probe_train` labels update the disposable
candidate; test and old TRAIN targets score only, final unavailable.

A literal example acquires the separate fresh packet, retains all three old-task
scores, and keeps the source unchanged. The candidate's prototype payload grows
32→64logical bytes; eight one-row before/after readouts return12distance candidates/
24feature values. These are tiny deterministic correctness fixtures, not a robot
or broader adaptation result. A duplicate-input conflict retains the earliest row,
stores/pays for both labels and gives zero fresh gain; no winning threshold was
introduced.

Append attempts, native acknowledgements and runtime completions remain separate.
An ACK lost inside the delegate records completed0/attempt1/unknown native work/
remaining append slots0. A postcommit observation failure retains completed1 and
its known return. Failed attempts stay paid, no candidate release or automatic
retry. Nearest optimizer/gradient/dense neural work0; retrieval, prototype growth,
copies/validation and process costs remain distinct.

## Review finding and repair

Independent full source/procedure/test-result review found a real initial gap:
within a checkpoint, callbacks could change already returned scores or erase paid
retrieval counters before the checkpoint was sealed. The first test covered only
between-phase replacement. Review correctly withheld acceptance.

The fix guards each public score/predict/snapshot/boundary callback with the
existing deep paid-report witness. Completed checkpoint0/1, append returns and
events retain nested identity/value guards through final capture. A private set
of actual work counters publishes detached guarded views; erased/replaced views
make the result incomplete while final reporting recovers the trusted paid
attempts/returns. Known scores survive refusals; gains remain unavailable when
the score path fails qualification. No new guard framework or runtime was added.

Six new cases cover score value/equal replacement/erased counters in both
checkpoints; two more cover nested equal replacement after each complete
checkpoint. Lead self-review also added a late source-getter case: paid evidence
is rechecked after all boundary callbacks, restoring known contents and refusing
before a second retrieval. All pass. Source/array replacement, role/identity/capacity refusal,
lost ACK, postcommit failure, changed budget, endpoint replacement and unchanged
four-family neural routing are also exercised.

## Commands, outcomes and costs

From physical D:\Dev\Repositories\Circadian, TEMP/TMP/TMPDIR D:\Dev\_Temp,
PythonUTF8/no bytecode, numerical threads1:

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py initial
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py repair -k 'not existing_four_neural'
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py paid -k 'replaced_preceding_score or budget_replacement or endpoint_owner'
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py score -k 'not existing_four_neural'
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py final
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py late -k 'not existing_four_neural'
```

| Phase | Observed cases | Body seconds | Sampled peak bytes |
| --- | --- | ---: | ---: |
| initial |22PASS/1FAIL|1.257258400|73441280|
| repair |22PASS/4deselected|1.118293900|72986624|
| paid |3PASS/23deselected|.887347900|68272128|
| score |29PASS/4deselected|1.232709200|73433088|
| final combined |35PASS|1.322928600|73326592|
| final nearest-only late guard |32PASS/4deselected|1.295728700|73646080|

36distinct cases;144executions143PASS/1preservedFAIL,0skips. Aggregate test bodies
7.1142667001s/max73646080B, below original120s/512MiB/200executions/8commands.
The failed initial assertion expected completed1 after native ACK loss; corrected
to completed0 with attempted/unknown work retained. No production completion
counter was redefined. Initial type errors were retained and fixed with explicit
tuple/list annotations and a simpler indexed holder check, no ignores.

Cumulative fixture work, including all warm/failed/repeated attempts:

- Nearest121empty-owner initializations,0RNG;167append attempts/167literal rows;
  149retrieval attempts/149query rows/198distance candidates/396feature values.
- Neural8tinyD2/H2 source initializations,16normal calls/64values;8single update
  probes/8literal rows (two BP/PC/neutral/active quartets). Ten BC core forwards/
  ten rows, two BC gradient evaluations/two rows,30PC energy evaluations/30rows.
  Each PC case allows at most4inference iterations; total24iteration/24rowstep
  upper bound, not a measured iteration count. Native PC work is not mislabeled
  as the BC gradient formula.
- Validation, forks, snapshots, readout restores, Python/temporary arrays are
  additional; process RSS includes imports and fixture overhead. Counts are not
  total floating point operations or matched compute.

The original quartet's4initializations/32normal values/4updates remained spent.
After the review-required shared scorer repair, PROTOCOL and issue were amended
BEFORE final execution for one additional quartet. Total8/64/8, still8literal
rows; all other original caps unchanged. This was corrective engineering
verification of changed code, no old baseline/scientific budget renewal. Failed
receipts and costs remain. Recorded phases are SPENT; do not rerun blindly.

Final three-file gates all pass:

```powershell
.venv\Scripts\python.exe -m ruff check --no-cache src/adapters/continuous_nearest_probe.py src/adapters/continuous_probe.py tests/test_continuous_nearest_probe.py
.venv\Scripts\python.exe -m ruff format --check src/adapters/continuous_nearest_probe.py src/adapters/continuous_probe.py tests/test_continuous_nearest_probe.py
.venv\Scripts\python.exe -m mypy --no-incremental --follow-imports=skip src/adapters/continuous_nearest_probe.py src/adapters/continuous_probe.py tests/test_continuous_nearest_probe.py
```

Typing skips imported modules: scoped acceptance, not a whole-project type gate.
Fullsuite/Linux/cleanclone/wheel/browser/device/GPU/remoteCI/release and unchanged
old numerical cohorts skipped. The final nearest-only guard change ran32affected
cases; unchanged neural branches reuse the final35cohort. No production source
edits after that last test (formatter ran before); documentation only afterward. No install/download/
commit/push/delete/global settings change; unrelated dirty changes preserved.

Independent review used3readonly batches1.1stools for actual initial combined
source/procedure/tests/receipts, no imports/tests/data/models. It rejected the
initial score/counter gap. A finite text-only remedial review accepted supplied
exact corrective source excerpts and final35/statics receipt, then the narrow
late-guard excerpt and32affected receipt, with no further
tools/fresh source read/numerical replication. This scope is software acceptance,
not independent execution. Exception ENDED. Lead combined source/procedure/
results self-review also complete, not described as independent.

## Handoff

[Usage](../nearest-reserved-probe.md). Local finite stage
`artifacts/runs/nearest-reserved-probe-software-20261101/` contains the original
protocol, prospective amendment, bounded cumulative test runner, all six SPENT
phase markers/logs/JSON work/resource receipts and session handoff. Stage cap8MiB.
No historical experiment inputs, matrices or raw outputs were reopened.

#400/R2.1-MEM.PROBE software criteria are evidenced. #399's four fixed native
neural failures and original inner-owner protocol deviation remain unchanged;
criterion4 stays OPEN. Working memory known-task successes and old four-family
reserved probes are reused. No recorded nearest23/24 result, native skill,
recovery/reliability/generalization/CPC advantage or repository release follows.

One next action, #401/R2.1-MEM.PROBE.E: freeze/review ONE small actual reserved23/24 memory-probe
protocol through this existing API. Bind the accepted memory state/data and
qualified role-preserving inputs before reads, declare any necessary source
restoration/construction cost explicitly, one private append, all3old TRAIN
readouts, retrieval/prototype/copy/IO/RNG/time/RSS/storage/failure limits. Reuse
accepted baseline/native outcomes, do not repeat them. A later native endpoint
requires the existing frozen port's separately checked prerequisites and quota.
No actual experiment allowance is created here. Broader parents/goal OPEN/ACTIVE.
