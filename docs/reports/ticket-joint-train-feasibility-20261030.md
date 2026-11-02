# Fixed joint TRAIN capacity on the accepted BP representation

R2.1-joint-TRAIN-feasibility, issue333; parents23/21/18 remain open. Previous332
was progress: a model-free finalization repair, not retrospective native331
acceptance. This increment answers the next specific learner prerequisite,
rather than repeating a checkpoint/capture increment or native baseline.

## Observed result and limits

ONE frozen read-only analysis completed. Both prescribed affine readouts
separate all18 original TRAIN rows: old four tasks plus fresh two. Every required
target/competitor margin is strictly positive under exact rational validation.

| Comparison scope | Constraints | Solver iterations | Correct in scope | Exact minimum margin, decimal |
| --- | ---: | ---: | ---: | ---: |
| Full six actions | 90 | 31 | 18/18 | 0.01577214024729886 |
| Original version-allowed five actions | 72 | 20 | 18/18 | 0.01577214024729896 |

Full exact minimum:
`305077561208682447720869/19342813113834066795298816`.
Allowed exact minimum:
`19067347575542775147361/1208925819614629174706176`.
Both also classify18/18 globally. Solver status0 in both; maximum primal
residuals3.3306690738754696e-16/2.220446049250313e-16. Coefficients, scores,
fraction margins and warnings remain in actual.json. The compatibility warning
about threads/random_seed options being passed through to HiGHS is retained.

Meaning: the computed fixed H3 TRAIN reference features can jointly support
these task decisions using the same24-coefficient affine K6 capacity as the
original output layer. This removes a necessary-capacity concern on these
particular reference rows. It does not show that ordinary SGD will find that
readout, that the fresh hidden rows match a native forward, or that rounded
float32 coefficients execute the witness. No coefficient is deployed or used
to initialize a later experiment. Native fresh-task acquisition, old-task
retention, recovery after actual forgetting, generalization, CPC advantage and
matched-compute benefit remain unproved. Numerical zero without a certificate
would have been inconclusive; no solver/precision/metric fallback was used.

## Source, isolation and delivered helper

Reconciled321 source.json.gz checkpoint12 against318 source.json.gz: complete
saved states, including parameters/SGD buffers/counters, are exactly equal.
Original permanent charges, consent/receipts/budgets/clocks remain untouched.
Hash-pinned accepted318 old TRAIN features/logits are reused; the319 old-only
witness and its four unchanged solver fixtures are not rerun. Corrupted or
replaced input bytes are refused before the diagnosis; all pins are rechecked
afterward. Saved JSON is diagnostic data, never runtime checkpoint authority.

New `scripts/ticket_train_fit.py::joint_old_fresh_teacher` prepares the fixed
old4 then fresh2 original TRAIN traces, three phases each. The existing
`context_teacher` remains the old12 default. The new helper reuses original
role/group/schema admission, verifies public-observation feature alignment and
refuses conflicting labels on identical feature rows. Six focused checks cover
old-default equality, explicit fresh-task targets, source immutability and
role/group/shape/schema/duplicate-ID refusal. No native model is imported.

Example using existing components:

```python
from scripts.run_ticket_workflow import CONTEXT_FEATURES, training_cases
from scripts.ticket_train_fit import joint_old_fresh_teacher
from src.core.decision_features import DecisionFeatureSpec
from src.core.ticket_workflow import CATALOG

cases, batch = joint_old_fresh_teacher(
    DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG), training_cases
)
```

This creates TRAIN data only. It does not run a learner or the experiment.
Task order is v1invoice/v1password/v2invoice/v2parcel then v1parcel/v2password;
sealed/ready/staged each. Fresh tasks and comparison directions were frozen
before arithmetic, using original TRAIN definitions rather than evaluation
outcomes. Final/probe data, development outcomes and witness coefficients do
not enter source training, promotion, tuning or probe selection.

Saved-parameter NumPyfloat32 affine/tanh/raw projection uses the existing
reference formula. Old12 native parity passes: hidden maximum absolute
error7.450580596923828e-9, raw-logit error2.3283064365386963e-10, both below the
fixed1e-6 gate. Fresh6 reference rows remain non-native evidence. Reused319 LP
and exact Fraction validation operate only on the resulting binary values.

## Commands, checks, costs and preserved failures

All commands from `D:/Dev/Repositories/Circadian`, PowerShell TEMP/TMP set to
`D:/Dev/_Temp`. WSL cwd `/mnt/d/Dev/Repositories/Circadian`, TEMP/TMP/TMPDIR
`/mnt/d/Dev/_Temp`. Existing environments only; no install/download/remote CI.
Stage: `artifacts/runs/ticket-joint-train-feasibility-20261030/`.

Focused checks, each6 PASS/0fail/error/skip, no Torch/SciPy import or model/LP:

```text
.venv/Scripts/python.exe -B artifacts/runs/ticket-joint-train-feasibility-20261030/check.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ticket-joint-train-feasibility-20261030/check.py linux
```

One prospectively amended final repeat uses `windows-final`/`linux-final` in
those same commands, after explicit test class annotations repaired mypy's15
attribute errors. Original passing runs and failed mypy receipt stay intact.
Final Windows0.4259299999976065s/51,101,696B; Linux1.7154217639999998s/47,263,744B.
Across both invocations/platform:48 pure teacher dispatches/216 feature
encodings. Final checks bind actual helper/test source hashes. Six distinct
tests/platform were executed twice; this is not12 distinct tests/platform.

Scoped static commands (exact argv/output/times in statics-*.json):

```text
.venv/Scripts/python.exe -B -m ruff check scripts/ticket_train_fit.py tests/test_ticket_joint_train.py artifacts/runs/ticket-joint-train-feasibility-20261030/check.py artifacts/runs/ticket-joint-train-feasibility-20261030/run.py artifacts/runs/ticket-joint-train-feasibility-20261030/launch.py
.venv/Scripts/python.exe -B -m ruff format --check <same five files>
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian-joint333-mypy-final scripts/ticket_train_fit.py tests/test_ticket_joint_train.py
.venv/Scripts/python.exe -B -m mypy --platform linux --explicit-package-bases --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian-joint333-mypy-linux scripts/ticket_train_fit.py tests/test_ticket_joint_train.py
```

Final Ruff/format/default+Linux-target mypy all EXIT0; Linux-target mypy ran on
Windows, not a Linux mypy process. AST/source/input hash and scoped tracked
whitespace checks PASS. Recorded test-process plus static-child times total
12.34213731804929s of180s allowance; test host invocation times (including WSL
startup) total10.9491154s, versus4.612339118003809s inside test wrappers. Those
measurements differ in boundary and are not additional training computation.

Original agent-created1MiB stage ceiling FAILED during engineering:14,083,451B
stage including14,065,893B mypy cache. Keep ENGINEERING_STORAGE_SPENT.json and
all cache files. Issue comment6533557823 prospectively amended stage ceiling
to16MiB and authorized the single final six-check repeat within the original
engineering wall/RSS allowance, before numerical execution. Current two later
D:/Dev/_Temp caches total28,152,266B and are retained too; stage ceiling includes
the original cache, not those separate temporary caches. No independently
attributed temporary-library memory ceiling is claimed. Three failed patch-anchor
attempts changed no files; subsequent exact stdlib edits simplified the tooling.
No evidence was removed or relocated to conceal a failure.

One scientific command, EXIT0; never rerun after permanent markers:

```text
.venv/Scripts/python.exe -B artifacts/runs/ticket-joint-train-feasibility-20261030/launch.py
```

Actual WSL Python3.12.3/NumPy1.26.4/SciPy1.15.3:1.114414532s/85,299,200B RSS;
outer4.351452500006417s. Fixed30s/256MiB process and60s outer limits held.
Stage projected14,129,585B<16MiB at result write, counting recursive cache.
Work:2teachers/24TRAIN case exposures/24sandboxdispatches;18selected rows and
60 feature encodings derived from helper paths;1float32 hidden+raw projection
18rows,486hidden+324outputaffine products/54tanh scalars;2LP calls/36input rows,
162constraints/4050matrixcells,2float+2exact readout validation calls72rows/
864exact products. Solver internals are represented by iterations/time/RSS,
not claimed to equal native SGD work.0models/restores/nativeupdates/probeupdates/
finalexamples/promotions; no benchmark learner or deployed actor was present.

Full repository suite, clean clone, installed-wheel build, unchanged native
model tests, native fresh projection/readout, learner F/return, browser/GPU,
release checks and comparative sweeps skipped: this scope changes a TRAIN
cohort helper and runs a saved-data diagnostic only. Existing accepted baseline
evidence is reused. Original331 and323 remain FAILED/SPENT with native/final
preservation criteria unproved; this result does not repair either case.

## Review and next learner outcome

Lead self-reviewed combined source, input isolation, LP signs, fixed masks,
resource/failure receipts, exact witness semantics and unchanged source hashes.
Independent prelaunch ACCEPT:3readonly commands EXIT0/0.4369458s, no imports or
arithmetic replication; REVIEW.md records concrete scope. Final source/result/
interpretation review ACCEPT: one additional read0.1995037s, four total read-only
commands0.6364495s, no independent arithmetic replication/imports/solves. This
accepts the scoped reference result, not native fresh-task behavior. Exception
ended; lead self-review remains separately identified.

The critical path now returns to useful learner behavior: the ordinary runner
already supports BP CE-initialized isolated fresh-F then old-A return via
`--context-return --method bp --reserved-probe --probe-initialization ce`.
Its software and wheel evidence exist; replaying old-only321/331 would add no
new learner outcome. One precise next action: reconcile that command's current
source with accepted324/325 software/delivery and328 LS-control evidence, then
freeze one small native ordinary CE-initialized F/return protocol measuring
initial/F/return old+fresh task success, TRAIN loss, interference and total
paid work. Preserve fixed seeds/doses/masks and source/actor/SGD authority; do
not use these witness coefficients, change hyperparameters or retry the spent
323/331 wrappers. This would address the same underlying unanswered323 adaptation
question through the ordinary entry point. Any new attempt must explicitly
justify fresh prospective funding using the corrected software and new capacity
evidence, preserve original costs/failure and avoid treating a renamed entry
point as a changed scientific question. It is not authorized by this diagnostic's
spent numerical allowance. Full R2/R4/R5/R7/science/release/goal open.

Closeout:333 closed/completed; canonical6533727591, tracker18 6533732389,
parent21 6533732774, parent23 6533733067. Final retained stage14141331B<16MiB;
closeout.json records live receipts/next action. Full goal remains active.
