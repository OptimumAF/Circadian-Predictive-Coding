# Bounded polynomial-kernel TRAIN learner

The saved-TRAIN CLI offers a separate `--kernel-perceptron` family through the
existing delayed TRAIN stream. Why this: the original linear128-pass control
reached52/54 without a full fit; a fixed nonlinear classifier can test a different
capacity without changing the original BP, PC, neutral CPC or active CPC recipes.
A finite negative linear fit does not prove linear inseparability.

## Use

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --kernel-perceptron --describe --fit-updates 128 --fit-seconds 30
```

After freezing and reviewing a small protocol, remove `--describe` and add
`--source TRAIN_REPORT --output NEW_RESULT.json`. Existing output files refuse.
Only complete original148-frame TRAIN episodes are decoded into raw296 histories;
the unchanged source/role admission precedes learner construction. No final or
reserved probe records train, select, promote or tune this learner. This option
cannot combine with other fit families, cue-relative projection or gradients.

API: `minigrid_constructive_fit.diagnose_train_fit(episodes,
kernel_perceptron=True, updates=128, max_wall_seconds=30)`. The existing default
remains H8 BP512. Limits remain1..512 passes/(0,90]seconds; fixed checkpoints are
0/min(128,updates)/updates, deduplicated. All prescribed passes execute before
scoring; there is no best endpoint, success stop, shuffle or averaging.

## Rule and owned state

For exactfloat64 BxD input in[-1,1], B1..128/D1..296/K2..16, the fixed kernel is
`(1 + dot(x,y)/D) ** 2`. The initial dictionary is empty with zero class scores.
Each pass follows supplied order. First argmax selects the predicted class. A
mistake adds1 to the target coefficient and subtracts1 from the predicted one on
that exact numerical feature key. Matching keys reuse an entry; canceled entries
remain. There is no RNG, optimizer, eviction, regularization or tunable step.
Stable softmax produces probabilities for the existing CE/accuracy/exact-trace
scorer; CE is not the update objective.

Policy caps prototypes at1..128, default128. A new mistaken key at capacity
refuses the entire native pass atomically; existing-key mistakes still update.
The typed `KernelPerceptronCapacityExceeded` retains attempted/evaluated rows,
completed kernel pairs and locally applied uncommitted mistakes before refusal.
The refused row's prediction is paid; its unapplied mistake is excluded. The
error returns no candidate state or additional authority. Other invalid inputs
refuse before native arithmetic. Stream charges/native counters are distinct;
an already completed update with a lost return remains visible and uncertain.

Complete immutable snapshots own policy, prototype tuples, integer coefficient
tuples, pass/row/mistake and committed training-pair counters. Counters are exact
integers<=2**53-1; dictionary uniqueness, shapes, finite values, row-sum-zero
coefficients and magnitude/counter bounds are validated on copy/restore/fork.
The invariant wrapper binds a distinct snapshot family to296/K7/cap128 and the
original public-history projection. Every raw row projects before native work.
Roles, IO, budgets, scheduling and scientific conclusions stay outside core.

## Costs and storage

`work.kernel_perceptron` reports committed passes, exposed rows, row mistakes,
actual committed training kernel-pair evaluations and retained prototype count.
Prediction pair totals are explicitly **derived** from owned checkpoint dictionary
size times attempted/returned rows; they are not an external operation meter.
Typed failed-pass work is reported separately; uncertain work remains flagged.
Each pair additionally performs a D-coordinate dot product and K-class weighted
accumulation. All projection/admission/validation/copies/Python/time/RSS add costs.

At296/K7/cap128 the dictionary has310272 array-equivalent bytes at full capacity
(128*296*8 prototype bytes plus128*7*8 coefficient bytes). Integer tuples and Python
objects do not have this resident size. No unbounded success chain or full Gram
matrix is retained. The existing fit runner holds at most three complete
checkpoints plus source/candidate/actor/readers and input copies for each pass.
This capacity/objective/work differs from linear and H8 neural methods; equal
row exposure would not make compute matched.

## Structure and verification

```text
src/core/categorical_kernel_perceptron.py             rule, policy, state, scores
src/adapters/kernel_perceptron_learner.py             native ownership/atomicity
src/adapters/invariant_kernel_perceptron_learner.py   raw296 projection binding
src/adapters/minigrid_constructive_fit.py             existing stream and costs
scripts/run_minigrid_linear_fit_control.py           explicit CLI family
tests/test_categorical_kernel_perceptron.py           scalar oracle/fixed XOR
tests/test_kernel_perceptron_learner.py               ownership/capacity work
tests/test_minigrid_kernel_perceptron_fit.py           CLI/isolation/paid failure
```

Focused tests: `python -m pytest -q tests/test_categorical_kernel_perceptron.py
tests/test_kernel_perceptron_learner.py tests/test_minigrid_kernel_perceptron_fit.py`.
Lint/format: `python -m ruff check` / `python -m ruff format --check` on these
five production and three test files. Types: `python -m mypy --python-version
3.11 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL
--platform win32` on the five source files; repeat with `--platform linux`.
Exact executed commands and outcomes: [RESULTS](../artifacts/runs/kernel-perceptron-fit-20261018/RESULTS.md).

No new dependency, product environment variable or orchestration framework.
Extend the native learner/stream composition with separate explicit families;
do not mutate a frozen recipe or infer useful retention/recovery from correctness.
Real-source fitting and scientific benefit require their own frozen small case.

## Observed engineering result

203distinct scoped cases accepted,306executions (focused103/finalcombined203),
all0fail/error/skip. Fixed16-pass XOR learns and independent scalar-list arithmetic
matches. Final9pathRuff/format and5sourceWin/Linuxmypy pass; independent SOURCE/
PROTOCOL/ACTUAL GO. No new Torch execution or real-source fit.

Both synthetic CLI reports complete3passes162rows/P12-108,161mistakes/615native
trainingpairs/N4 and216derived predictionpairs. FullfitFalse,12/54correct,0/6exact,
CE1.9459101490553137→1.941091959129343. Each108saved probability rows independently
checked. This negative conflicting fixture remains preserved; no neural science
or real-source acquisition follows. [Full checks/costs/skips](../artifacts/runs/kernel-perceptron-fit-20261018/RESULTS.md).

## Real-source case

The single original 54-row / 128-pass case completed and qualified: 47/54 correct,
3/6 exact, CE 0.8894917420. Full fit remains false. All 108 probability rows and
128 receipts were independently checked; no retry or dose/kernel change.
[Results, costs and next neural implementation](reports/kernel-perceptron-original-train-fit-20261018.md).
