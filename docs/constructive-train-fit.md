# Disposable H8 TRAIN fit

## Explicit cue-relative recipe

The existing route also accepts `cue_relative=True`, or the CLI flag
`--constructive-bp --cue-relative`. This selects the separately typed public
cue-relative learner while retaining the original H8 BP seed17, learning
rate0.03, momentum0.9, unweighted CE, delayed stream and fixed scoring rules.
Omitting the flag preserves the original recipe and report.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --constructive-bp --cue-relative --describe --fit-updates 128 --fit-seconds 90
```

An execution example, requiring a prospectively budgeted local protocol:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --constructive-bp --cue-relative --fit-updates 128 --fit-seconds 90 --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output cue-relative-fit.json
```

The complete six TRAIN sources, including unused next views, are admitted under
the chosen projection before model/resource construction. Catalogue and stream
inputs remain **raw296**. The bound learner applies the mapping once for both
training and prediction. Do not supply explicit codec output: red-cue mapping is
not idempotent. Bounds name the new encoding and use its projected keys. The
checkpoint guard requires its exact new snapshot family, fixed native policy and
paid clocks. Existing source/actor checks, original clock/budget, permanent
charges and partial failure records are retained. No heldout data enters this
route. `--cue-turn-gradients` is refused with `--cue-relative` because the old
reader binds a different projection. See [encoding boundary](cue-relative-encoding.md)
and [ADR0426](adr/ADR-0426-explicit-cue-relative-constructive-recipe.md).

### Executed fixed128 result

One independently reviewed, frozen128 run completed. Cue-relative BP scored
48/54rows,0/6exact traces, CE0.39177486685667096. The preserved original invariant
BP checkpoint128 scores45/54rows,0/6exact, CE0.3929002644951615. The new run has
three additional correct rows and CE lower by0.0011253976384905662, but fails the
full-fit acceptance criterion. All six later turn errors are retained; the
earlier original512negative remains unchanged. This single TRAIN representation
control establishes no retention, recovery, generalization or circadian benefit.

128paid/native/committed updates use6912training rows and12detached classifier
calls/108rows. Zero replay/probe/teacher/final work. Source and actor remain
unchanged; work-uncertainty flags are false. Native parent wall7.2790249seconds,
actual7.0064391seconds, sampled RSS345583616bytes; one stdlib saved readback costs
0.6076772parent seconds. The separate120second allowance closes at7.8867021spent;
no retry or extra dose. The reused control's whole old512 invocation used more
total work; the comparison concerns its fixed128 checkpoint, not whole-call cost.

Admission/projection work is additional: bounds108current/next plus54bound rows,
catalogue108current/next rows, training6912 and readout108 projections (7290total
cue-relative calls). Bounds allocate256176input-array bytes, the template128304,
and the existing stream owns128copies totaling16422912input-array bytes; native
states, validation, temporary Python objects and copies add measured wall/RSS.
No final data, probe selection or deployed/benchmark learner is changed.

### Focused checks

187distinct scoped cases passed (168combined plus19scripted old-gradient-route
regressions). Three initial fixture failures remain in the evidence. Five changed
paths passed Ruff, format and Win/Linux mypy. Exact commands, XML and resources:
[RESULTS](../artifacts/runs/cue-relative-fit-20261018/RESULTS.md).

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_cue_relative_fit.py tests/test_minigrid_constructive_fit.py tests/test_minigrid_train_catalogue.py tests/test_minigrid_train_fit_bounds.py tests/test_minigrid_linear_fit_cli.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py src/adapters/minigrid_train_catalogue.py src/adapters/minigrid_probe_support.py tests/test_minigrid_cue_relative_fit.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py src/adapters/minigrid_train_catalogue.py src/adapters/minigrid_probe_support.py tests/test_minigrid_cue_relative_fit.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py src/adapters/minigrid_train_catalogue.py src/adapters/minigrid_probe_support.py tests/test_minigrid_cue_relative_fit.py
```

Use `--platform linux` for the second type-check platform. No dependencies or
environment variables were added. The only new code structure is
`tests/test_minigrid_cue_relative_fit.py`; existing core/adapter/app boundaries
remain. The subsequent [public probe integration](cue-relative-probe.md) registers
the new family in the existing history factory, snapshot binder, receipts and
F/old-A/A-return path. Its synthetic native checks establish scoped functionality;
real scientific adaptation and full acquisition remain open. No additional fit
dose was required to implement that integration.

## Original invariant recipe and preserved evidence

This diagnostic asks whether more work with the original BP learner can fit the
six original MiniGrid TRAIN traces. It uses the existing delayed training-only
stream, invariant296 representation, H8 network, seed17, unweighted CE,
learning rate0.03 and momentum0.9. It changes the work cap from the original
128-update study; it is a separately named optimization diagnostic.

## Run

Describe without reading a source or constructing resources:

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_linear_fit_control --constructive-bp --describe
```

An explicit execution example (creates a new output file):

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_linear_fit_control --constructive-bp --fit-updates 512 --fit-seconds 90 --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output constructive-fit.json
```

The fixed diagnostic budget allows1..512 updates and up to90 wall seconds, with
768MiB process RSS. Options only apply with `--constructive-bp`, which is mutually
exclusive with `--bounds-only`. Existing default linear-fit and bounds-only
commands retain their behavior. Use a prospectively reviewed local protocol
before executing a new scientific case; these arguments do not renew any spent
allowance. There is no seed, learning-rate, optimizer or baseline tuning option.

API: `src.adapters.minigrid_constructive_fit.diagnose_train_fit(episodes,
updates=512, max_wall_seconds=90)`. Inputs must be an exact tuple of the six
original ordered complete TRAIN296/K7 causal episodes with nine rows each.
The existing saved-source decoder derives these from original public148 sources.
There is no probe, final, deployed or benchmark learner argument.

## Behavior and costs

The existing catalogue composer makes one54-row input template. Wake metadata
references that template; `AttributionStream` copies arrays into its own512-wake
schedule (65691648 bytes at512, plus a128304-byte template). The original source
and serving actor stay unchanged; only the ordinary stream's candidate learns.
Each wake has delayed1 label release. Original source episodes and the budget
are checked before and after each wake, including before admitting later work.
No replay, promotion, reserved task or final scoring occurs.

Training finishes before scoring fixed detached checkpoints0/128/512. Smaller
explicit caps use sorted unique0/min(128,cap)/cap. Each detached reader must be
separate before restore; native state, score inputs and original source evidence
must remain unchanged. Readout uses the original budget session, without a new
clock or time allowance. It stores all probabilities, decisions, labels and
original unweighted CE/accuracy/exact metrics for independent saved arithmetic.
Default work is512updates/27648TRAIN rows and18forwards/162nine-row readout rows.
Copies, snapshots, validation and native compute add elapsed time and RSS.
Descriptor `score_input_array_bytes` is one54-row array equivalent per checkpoint;
score arrays are constructed again per checkpoint and detached copies add work.
It is not an allocation total or peak measurement.

The report retains applied receipts, observed budget charges, committed updates,
current native counters when readable, and returned raw scores on failure.
Unknown failed native work is flagged rather than inferred from absent receipts.
A fake native port that raises after doing work can have native progress1 but
charged/committed0: the existing stream charges successful wake returns. Such
work still costs wall time; no success or receipt is invented. Corrupted/refunded
budget evidence is refused, with the observed discrepancy retained.

`fit_witness=True` requires completed requested training and all six final traces
exact. This supplies a constructive empirical fit witness for this architecture
and data. It does not establish learned live behavior, new-task adaptation,
retention/recovery, generalization or a circadian advantage. A failed or capped
fit does not prove finite H8 incapacity. The original BP12845/54, weightedBP48/54,
linear52/54 and all PC/CPC negatives remain preserved under their own protocols.

## Executed frozen case

The sole512-update case completed with52/54 correct rows,4/6 exact traces and
CE0.14254150694397866. Its update128 probabilities reproduce the original BP
result exactly:45/54rows,0/6exact,CE0.3929002644951615. Saved arithmetic checked
all162 probability rows. The full-fit witness is **false**. Both A cue traces
still miss step2 (left/right reversed); B and C_train traces are exact. The
original old-A acquisition premise remains unmet; selecting B/C afterward would
change the frozen adaptation question.

Known work is512updates/27648training rows and18predictions/162readout rows,
no unknown native work, replay, teacher, game, reserved or final calls. Parent
wall16.7481354seconds, sampled peakRSS448200704bytes; saved-only arithmetic
0.5460929seconds. The original120second grant closed17.2942283seconds spent;
no retry or extra optimization was admitted. Greater fit under a larger work
budget is not matched-compute evidence or scientific adaptation benefit.

## Verification and extension

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_minigrid_constructive_fit.py
.venv/Scripts/python.exe -m ruff check src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py tests/test_minigrid_constructive_fit.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py tests/test_minigrid_constructive_fit.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py tests/test_minigrid_constructive_fit.py
```

Linux mypy uses the same command with `--platform linux`. Scoped evidence and
retained failed attempts are in [RESULTS](../artifacts/runs/constructive-train-fit-20261018/RESULTS.md).
The native engineering check uses synthetic sources; its metrics are not
scientific learning evidence. The real saved TRAIN case is a separate budget.

Extend by composing an existing isolated adaptation probe with an explicitly
qualified acquired checkpoint and separate work limits. Keep benchmark and
final data authority fixed. Avoid changing a baseline recipe or selecting a
favorable checkpoint in response to this diagnostic.
