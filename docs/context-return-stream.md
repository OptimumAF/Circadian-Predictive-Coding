# Fresh contextual acquisition, retention and return

The existing recurring ticket runner supports `--context-return`. It starts a
zero perceptron; it does not load the jointly trained model. The original default
and `--policy-change` recipes retain their behavior. The default is perceptron;
the [native-family extension](#native-family-context-return) also supports the
existing CPU Torch learners. The [queue rehearsal extension](context-queue-rehearsal.md) supports
explicit recent/coverage replay. Incompatible options fail before work/output reservation.

```powershell
python -m scripts.recurring_ticket_workflow_example --context-return --output context-return.json
```

Use a fresh output path. Internal limits are twelve updates,30seconds,512MiB RSS;
JSON is bounded to512KiB. No extra dependencies. Both public context and task
truth return: A=v1invoice/password; B=v2invoice/parcel; A_return=v1invoice/password.
Each phase uses four six-row TRAIN passes, labels delayed one tick. Checkpoints
0/4/8/12 read all four tasks under their original policies. Endpoint reads also
exercise stale-ticket invalidation and revoked commit permission.

## Executed result

The following original perceptron evidence remains unchanged.

Issue75/R4.1-F/R2.1-U, one frozen local case:

| Updates / exposures | v1 invoice | v1 password | v2 invoice | v2 parcel |
| --- | --- | --- | --- | --- |
|0 /0 (unchanged control)|fail|fail|fail|fail|
|4 /24 (after A)|pass|pass|fail|fail|
|8 /48 (after B)|fail|pass|pass|pass|
|12 /72 (after A returns)|pass|pass|fail|pass|

Both B tasks failed before B training, then succeeded: this is new task gain,
not inherited ability. A invoice was acquired, lost, then recovered under the
same v1 truth. Password was retained, so its recovery field is null. Returning A
loses B invoice while retaining parcel. Thus retention remains incomplete.
Endpoint stale results match the three ordinary successes; all four revoked
commits are correctly denied, including the wrong staged invoice. Permission
refusal is not task success, and stale-ticket recovery is distinct from skill
recovery. Eight ordinary plus three stale successes,24total episode records.

Twelve passes72exposures13mistakes (A8/B4/return1), no random draws, probe/final
updates, replay, native SGD/PC, controller selection or promotion. The original
seed and stream actor remain unchanged; the stream trains its owned candidate.
All twelve original delay1 receipts and four full states are retained. Paid
results, input packets, frozen predictor, source/RNG and budget origin are checked
before more work; changed/replaced paid records are refused and restored.

Nominal lower bounds:4320TRAIN affine terms,4600task affine terms,260coefficient
additions,1920bytes for four retained coefficient matrices. Physical observer:
3learners/zero initializers,1155snapshot copy/validation entries69300logical
coefficient slots,92typed reads,2teachers13packet constructions188encodes116sandbox
dispatches. Copies/validation/Python overhead are additional. Recipe8.33685s,
meter8.94138s, sampled RSS44,879,872B, boundary44,609,536B; sampling is not a hard
OS peak guarantee. Actual262946B SHA256
`581a68d8badbaed81fb6dbf85b78aab3e08798149de86448feb31936e3d7b4b0`.

The zero checkpoint is reused as a no-learning control on identical deterministic
tasks. No equal-compute, statistical, unseen-task, CPC advantage, general neural
plasticity or release claim follows. The previous joint fit and native BP negative
remain unchanged; they answer different questions. This result motivates bounded
TRAIN rehearsal against measured interference, not a longer checkpoint chain.

## Validation and next extension

105distinct selected checks pass on Windows and Linux (one intentional separate
old-policy-case skip each); three affected source/test files pass Ruff, formatting
and fresh default/Linux mypy. Software failures and consumed work are preserved.
Source/procedure/result review is self-review, not independent acceptance.
Full suite, native/physics/math, packaging, clean-clone and CI gates are unrun.

Protocol, exact commands, failed checks, ledgers, raw results and zero-model
readback: `artifacts/runs/context-return-stream-20261023/`. Live handoff:
[issue75](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/75).

Subsequent issue76 implemented contextual queues through the existing rehearsal
path and preserved D7defaults. Both fixed replay cases keep the same3/4finalsuccess;
see [the extension and limitations](context-queue-rehearsal.md). This original
no-replay result remains unchanged; explicit extra exposure is charged separately.

## Native-family context return

Optional `--record-train-fit` adds four readonly original-TRAIN diagnostics with
the native mode. For example, add it to the BP command below. It reads twelve
TRAIN rows at each checkpoint and reports CE and per-action margins, paying
four extra matrix forwards/48row reads. It changes no training or task metric.
Static teacher truth is descriptive diagnosis, not online controller feedback.
Original30s512MiB/512KiB limits remain. The [executed BP diagnosis](reports/context-train-fit-20261029.md)
finds all four ready actions incorrect even while aggregate loss decreases.
This identifies a TRAIN fit gap without proving its cause or a successful repair.

Use the existing CPU Torch environment and a new output path:

```powershell
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.recurring_ticket_workflow_example --context-return --method bp --output artifacts/native-context-return.json
```

Explicit alternatives are `pc`, `neutral_cpc` and `gated_cpc`. This performs
native model work. It uses the same D9 public vocabulary, A4/B4/A4 TRAIN stream,
12updates/72row exposures, delay1, 24fixed task readouts, seed17/H3/K6 and original
method settings. It does not use the separate-head perceptron architecture.
Native replay is refused; separate heads, private adaptation and paired serving
remain perceptron-only. No command defaults change. Requires existing optional
Torch CPU support; the NumPy-only installed wheel qualification does not qualify
this new native option.

Source, actor, full frozen native state/spec and both global RNGs remain guarded.
Inference counters advance legitimately; mutation of weights/optimizer/chemistry
or source state does not. Guards run before native updates. Failed native work
keeps its attempted/returned distinction and original receipts. Original source
12update/30second/512MiB session and 512KiB output limit remain. Checkpoint0 is
an unchanged seeded initial-state control, not a zero-weight model or a
compute-matched baseline. D9 native nominal affine terms are45perrow; backward,
latent inference, copies, initialization and validation add work.

The [one frozen four-family observation](reports/native-context-return-20261029.md)
completed. All four methods failed all ordinary tasks at all checkpoints. No
skill was acquired, so null recovery is not retention. Common initial parameters
matched; PC and neutral CPC parameters and full episode traces matched. All
negative outcomes remain. No dose/seed/configuration/metric was tuned, and no
circadian, generalization, reliability, equal-compute or release claim follows.

Focused developer checks (native fixtures incur real model costs):

```powershell
.venv/Scripts/python.exe -B -m pytest -q tests/test_native_context_return.py tests/test_context_return_stream.py
.venv/Scripts/python.exe -B -m ruff check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_context_return.py tests/test_context_return_stream.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_context_return.py tests/test_context_return_stream.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_context_return.py tests/test_context_return_stream.py
```

Use `--capture=sys` for Linux pytest when D-backed FD capture is incompatible.
These are developer examples, not authority to repeat the spent scientific cohort.
Extend the existing TRAIN diagnostic/readout ports before another adaptation
comparison; do not compensate for failure by adding CPC mechanisms or changing
baselines to manufacture a winner.

## Current-packet linear control

The conventional `--method linear_fit` option is available for
ordinary context return, with replay none and no alternate adaptation or TRAIN
fit-diagnostic flags. It fits only each current arrived packet using the existing
fixed least-squares learner. The unfitted initial readout is explicitly uniform;
the original12update72row/source30s512MiB envelope and task schedule remain.
[API, distinct costs and limits](current-linear-learner.md).
Its [one chronological case](reports/linear-context-return-20261029.md) learns A
and B, loses A invoice during B and recovers it on return, then loses both B
tasks. The explicit `--separate-context-heads` extension instead fits only the
publicly selected projected head. Its [one frozen case](reports/context-linear-heads-20261029.md)
retains all four known tasks through B and A return. It uses96coefficients/768B
versus60/480B and assumes known labels; no context-discovery, matched-capacity,
deployed or comparative benefit follows. Broader reliable joint retention stays open.

## Current-batch class-balanced BP

`--class-balanced-bp` is an explicit optional objective for ordinary native BP
context-return only. It requires `--context-return --method bp --replay none`;
separate heads, private/paired adaptation and rate/gate effects are refused.
The API is `run_example(context_return=True, method="bp", class_balanced_bp=True)`.
Existing defaults and other native families retain their original updates.

Each actual arrived TRAIN call supplies its own target counts. For each present
class the weight is `rows / (present_classes * class_count)`; absent classes
receive1.0. The existing array helper applies one normalized weighted CE update.
There are still12updates/72rows with the same seed, policy, optimizer and
original30second/512MiB source session. No diagnostic, future TRAIN, final or
probe truth can choose weights; static TRAIN fit readouts remain descriptive.
Weighted training loss has its separate objective definition. Readout CE and
task metrics retain their original definitions and comparison directions.

At most12records retain counts/weights/objective/physical attempts and returns,
plus a source/label witness captured before native work. Newly produced receipts
must match that witness; subsequent checks require their original object and
values. Changes/replacements of prior records stop further work. Failed or lost
returns retain permanent costs/unknown work; corruption preserves the original
paid prefix separately. The seeded source, actor and global RNGs remain guarded.

The [sole frozen qualification](reports/context-class-balanced-20261029.md) is
complete/SPENT: ready TRAIN accuracy0/4 and all ordinary tasks still fail.
Weighting does not repair this fixed recipe. No acquisition means no retention
or recovery claim; this neither proves imbalance is the cause nor measures CPC.
The recorded scientific case is not authorized for rerun or tuning.

Focused developer checks (real model work; budget them separately):

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_context_class_balanced.py
.venv/Scripts/python.exe -B -m ruff check scripts/ticket_workflow_native.py scripts/recurring_ticket_workflow_example.py tests/test_context_class_balanced.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/ticket_workflow_native.py scripts/recurring_ticket_workflow_example.py tests/test_context_class_balanced.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/ticket_workflow_native.py scripts/recurring_ticket_workflow_example.py tests/test_context_class_balanced.py
```
