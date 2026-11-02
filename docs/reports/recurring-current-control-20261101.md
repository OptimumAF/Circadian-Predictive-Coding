# R4.2-MG.JC /396 — recurring current-only control

The existing ordinary affine runner now offers a cold current-only control for
the recurring A/B/C_train/G_train/A TRAIN schedule. It repeats only the currently
arrived packet with multiplicities1,1,2,2,3,3,4,4,4,4, matching the existing joint
recipe's ten fit-matrix shapes and252 physical row exposures. Of these,90 are
arriving exposures and162 are additional CURRENT repeats. Prior-TRAIN replay
and retained bank rows are0. This is not equal compute or a learning result.

Implementation: `compose_repeated_current_training(..., recurring=True)` in
`src/adapters/minigrid_train_catalogue.py` and explicit
`run_context_linear_stream(..., recurring_catalogue=True,
recurring_current_only=True)` in `scripts/run_minigrid_linear_fit_control.py`.
Default `retain_arrived=False` must remain false for this mode. Original six-arrival
recipes, recurring joint fits, A4, references and minimum-change behavior retain
their defaults; incompatible new combinations refuse before source/owner work.
No solver, environment, runner framework, dependency or CLI flag was added.

## Checkout correction and evidence reuse

Initial targeted inspection missed `minigrid_train_catalogue` and incorrectly
proposed a new public-cue StreamRehearsal bridge. Deeper inspection found the
desired ordinary causal joint fitting already implemented and accepted in
#122–124/#127, with original G acquisition/ABC retention/A return and four known
G/A game successes. Those observations are reused, not rerun or reconstructed.
The proposed four-file bridge was unexecuted; its diff/preimages remain in the
stage. Only this chat's edits were restored byte-for-byte. The separate dynamic
live-bank proposal is deferred unnecessary architecture, not completed work.

The missing ten-arrival repeated-current baseline is the smaller useful next
comparison prerequisite. Scope was amended on396 before execution. Original
requirements, review/scientific distinctions and negative results remain:383's
joint frozen-H8 negative,390XFAILED/SPENT,391PARTIAL,392auth/browser open and395
mutual forgetting. No private probe weights/data or final labels enter TRAIN.

## Executed software checks

Physical working directory `D:\Dev\Repositories\Circadian`, with process-local
TEMP/TMP/TMPDIR=`D:\Dev\_Temp`, bytecode disabled and optional native runtime
imports blocked. Local stage:
`artifacts/runs/minigrid-public-cue-joint-20261101/`.

| Command | Outcome | Body seconds | Sampled process peak |
| --- | --- | ---: | ---: |
| `./artifacts/runs/minigrid-public-cue-joint-20261101/run-check.ps1 -Attempt initial` |42 new PASS, exit0|3.2635771|79335424B|
| same launcher `-Attempt regressions` |51 existing repetition/recurrence PASS, exit0|3.4803741|89141248B|

Host3.3966684s/3.6227345s, neither expired.93 distinct cases/93 executions,
0failed/0skipped; aggregate body6.7439512s. Final combined production tree is the
same for both cohorts apart from an explanatory composer docstring. New cases
exercise the actual affine train_batch with validated literal solver results;
predictions are scripted uniform probabilities. They execute real source/shape/
public-cue validation, ownership, delayed admission, budgets, receipt/candidate
prefix checks and accounting. No genuine LS/SVD/model/physics work executed.

Checks cover exact multiplicities, same joint shapes, current-only targets and
source spans, owned arrays,72 distinct source rows/90 arrivals/252 fit rows,
598752 input-array bytes, future/earlier-target independence, metadata-independent
public cue, role/pixel/geometry/order/count/exact-bool refusals, equal source
observation/dictionary replacement, unchanged source/actor/caller RNG, original
limits and delay-one receipts. All48 TRAIN monitors follow all ten fits, with432
rows; they cannot choose the schedule or fit extra updates. There are no probe,
final, promotion or deployed-actor updates.

The ninth lost-solver-return case preserves eight paid receipts/180committed
rows, reports36attempted rows including27extra-current repeats as potentially
unobserved failed work, performs no scores and delivers no endpoint. Corrupted
receipt/candidate/source, changed budget birth, consent and cumulative charges
stop following work. Such fault injection is evidence of refusal, not a claim
that maliciously replaced evidence/budget counts are trustworthy.

Scoped commands on the two production files and new test file:

```powershell
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/minigrid_train_catalogue.py scripts/run_minigrid_linear_fit_control.py tests/test_recurring_current_control.py
.venv/Scripts/python.exe -B -m ruff check src/adapters/minigrid_train_catalogue.py scripts/run_minigrid_linear_fit_control.py tests/test_recurring_current_control.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --cache-dir artifacts/runs/minigrid-public-cue-joint-20261101/type-cache src/adapters/minigrid_train_catalogue.py scripts/run_minigrid_linear_fit_control.py tests/test_recurring_current_control.py
```

All exit0; combined6.0558071s. Type scope is three files with silent followed
imports, not repository-wide typing. Initial type cache17187045B/stage17417440B
under64MiB/32MiB. All test bodies under60s/512MiB and aggregate120s/160cases;
statics under180s. No failed test/static attempts this increment. The unnecessary
bridge draft is retained, not counted as delivered or tested functionality.

Lead self-review inspected actual deltas, admission/default branches, fixed
data/row accounting, failure procedure and observed results. A finite read-only
independent source review accepted the complete two production deltas, complete
repetition composer, exact flag admission and selected guard/failure assertions.
Final read completed0.1440254s/5883tokens without omission; no concrete delta
defect found. The first read delivered only a context/output truncation failure,
with elapsed UNKNOWN, so original cumulative120s review-time conformance remains
unverified. The remaining read had a prospectively amended30s limit; no third
read was permitted. Original failure preserved, not zero-imputed or a science
budget renewal. Exception ENDED, no file edits/tests/models/numerical replication.
The reviewer checked scalar receipts/statics, not full execution logs/test counts,
full unchanged runner/core, all new tests or independent draft byte restoration.
This is scoped delta acceptance, not independent native/repository acceptance.

## Limits and exact next action

Input arrays hold the complete finite private plan; zero retained bank excludes
source records, plan/stream copies, snapshot wrappers, Python and RSS overhead.
Matrix rank, SVD cost, memory, copies and wall time differ from joint replay.
Scripted checks establish correctness of the integration, not learning quality,
retention, recovery, reliability, generalization, CPC advantage or release.

Full suite, clean clone, installed wheel, Linux/WSL, browser/device/robotics,
GPU/CI, native experiments and release gates were skipped. No dependency
install/download, old scientific JSON/model reconstruction, Git publication,
deletion or global setting changes. Parent18/21/23/full goal remain OPEN/ACTIVE.

Next concrete action: expose the explicit control through the EXISTING
`run_workflow(recurring_training=True)` dispatch and its training-admission check
(currently hard-coded joint), with focused scripted composition checks. Then
freeze one small native recurring current-only question in that same outcome
task using original ABC/G TRAIN, fixed seed17, ten fits252 rows, deferred monitors
and existing G/A evaluator. Compare to accepted joint only if source/data/protocol
assumptions match; preserve failures and all costs, do not rerun joint, tune doses
or select favorable seeds. This issue grants no new native allowance. No further
capture/checkpoint/audit framework is needed.
