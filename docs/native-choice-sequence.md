# Native Choice acquisition, retention and return

This exploratory A→B→A example extends the native TRAIN adapter through the
existing delayed-label ExperienceInbox. It tests two declared opaque routing
codes and two original support anchors. It preserves the original Open-Jev
hard-Choice loss, LoRA/head optimizer recipe and inherited inference temperature.
It adds no replay, circadian algorithm, actor promotion or training framework.

```text
src/core/native_choice_sequence.py     fixed corpus, roles and phase order
scripts/run_open_jev_sequence.py       inbox delivery, readouts and descriptive metrics
scripts/run_open_jev_native_update.py  shared bounded native readout capture
tests/test_native_choice_sequence.py   pure/recording-fake behavior checks
```

## Run

Use the same optional CPU Torch/Transformers/PEFT environment and pinned offline
source/base/checkpoint as the [native adapter](native-choice-bridge.md#native-train-update).
The repository's lightweight NumPy installation alone does not load this model.
Read the [prospective protocol](../artifacts/runs/native-choice-recurring-20261022/PROTOCOL.md)
before execution; it declares one graph,18updates,24native readouts,90top-level
forwards/126candidate sequences,600seconds/20GiB and no outcome-based retry.

```bash
HF_HUB_CACHE=/path/to/pinned/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false timeout 600 python -B \
  -m scripts.run_open_jev_sequence --source /path/to/pinned/source \
  --checkpoint /path/to/original/pinned/checkpoint --output /path/to/fresh-result.json
```

Keep output fresh and unset inherited `JEV_*` overrides. No new checkpoint is
saved. One initial trainable copy and four bounded weight summaries identify
phase state; the native reply's checkpoint hash remains original-load provenance.
The inherited temperature is not recalibrated after training; CPU float32 differs
from the released BF16 arithmetic. Neither parity nor calibration quality follows.

## Interpretation and extension

TRAIN and development use disjoint declared template groups and wording while
sharing the intended code concepts. They are synthetic development examples,
not a final test or evidence of empirical source independence. The zorbex marker
and one readout were seen in the preceding exploratory one-update trial; its
new-task failure/old-NLL regression remains preserved. The new study starts from
the original released checkpoint, not that updated artifact.

All18predeclared updates run despite adverse task metrics. Only correctness or
resource failure stops work. Each source is polled before its delayed label;
every exposure has a unique ID. Receipts are retained before boundary checks
can raise after a committed update. A partial report remains partial, with no
retry/renewed allowance or scoring of missing readouts as successful cases.

Accuracy is higher favorable; NLL and old-anchor drift are lower favorable.
Summary contrasts are trained minus unchanged or later minus earlier. The
unchanged control reuses baseline readouts with zero additional calls and is
not a second served actor. Infinite NLL is explicit JSON null/status; nonfinite
contrasts are unavailable. Endpoint2/2 alone does not prove learning when a
pretrained baseline was already correct. Return success is separate from
recovery after observed loss. Report all small-task criteria separately.

Extend only after inspecting the fixed acquisition/retention/return outcome.
Freeze a distinct prospective task/budget before changing data, seeds, metrics,
rank, loss or replay. Keep final data unavailable and probe updates separate.
Full R5.4 consolidation/rank/calibration and R5.5 temporal workflow semantics
remain separate requirements; this is native gradient training, not PC training.

## Isolated cold-start B control

The completed stream acquired/retained A but failed B0/2. The cold-start control
tests the same six B updates from the original released model, with no prior A
or inherited optimizer state. It reuses accepted baseline readouts only after
binding all initial LoRA/head hashes, empty AdamW, evaluation mode, zero permanent
work and private RNG. It scores six new post-B readouts through the same helper.
No baseline inference is repaid; the prior failed B result is retained.

```bash
HF_HUB_CACHE=/path/to/pinned/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false timeout 600 python -B \
  -m scripts.run_open_jev_b_control --source /path/to/pinned/source \
  --checkpoint /path/to/original/pinned/checkpoint \
  --reference artifacts/runs/native-choice-recurring-20261022/actual.json \
  --output /path/to/fresh-cold-B-result.json
```

Use its [separate frozen budget](../artifacts/runs/native-cold-b-control-20261022/PROTOCOL.md):
one graph,6updates/6predictions/24top-level forwards/36sequences,600s20GiB,
no retry. This is a development control, not a reserved probe. It tests the recipe
without prior A, not intrinsic hardness or the separate effects of weight and
AdamW history. Equal B exposure does not match total sequential computation.
Source and group admission happen before optional model loading; mismatched
initial weights/RNG/optimizer stop after the paid load without work or fallback.

```powershell
python -m pytest -q tests/test_native_cold_b_control.py
python -m ruff check scripts/run_open_jev_b_control.py tests/test_native_cold_b_control.py
python -m ruff format --check scripts/run_open_jev_b_control.py tests/test_native_cold_b_control.py
python -m mypy --follow-imports=silent scripts/run_open_jev_b_control.py tests/test_native_cold_b_control.py
```

## Declared optimizer-history control

The cold B success and sequential B failure motivate a single intervention:
A6 → clear AdamW moments/steps once → B6, using the original seed/data/loss/rank.
`OpenJevTrainingPolicy(max_optimizer_resets=1)` explicitly opts in. Default0
keeps continuous AdamW. Reset preserves weights, gradient buffers, private/caller
CPU RNG and model mode; uncertainty poisons the learner. Separate reset charges
and the original training allowances survive snapshot restore.

```bash
python -m scripts.run_open_jev_optimizer_control \
  --source /path/to/pinned/source \
  --checkpoint /path/to/pinned/package/checkpoint \
  --reference artifacts/runs/native-choice-recurring-20261022/actual.json \
  --output /path/to/fresh-optimizer-reset-result.json
```

Use the [separate frozen protocol](../artifacts/runs/native-optimizer-reset-20261022/PROTOCOL.md).
All initial/after-A trainables bind to accepted hashes before reset/B work;
baseline and after-A inference have zero new calls. Six post-B development
readouts measure B acquisition, A retention and anchor drift. Same twelve TRAIN
exposures does not imply matched total compute. Return/recovery is not tested.
No new checkpoint package is saved and no final/probe data is admitted.

```powershell
python -m pytest -q tests/test_open_jev_training.py tests/test_open_jev_optimizer_reset.py tests/test_native_cold_b_control.py tests/test_native_optimizer_reset_control.py
python -m ruff check src/adapters/open_jev_training.py scripts/run_open_jev_b_control.py scripts/run_open_jev_optimizer_control.py tests/test_open_jev_optimizer_reset.py tests/test_native_optimizer_reset_control.py
python -m ruff format --check src/adapters/open_jev_training.py scripts/run_open_jev_b_control.py scripts/run_open_jev_optimizer_control.py tests/test_open_jev_optimizer_reset.py tests/test_native_optimizer_reset_control.py
python -m mypy --follow-imports=silent src/adapters/open_jev_training.py scripts/run_open_jev_b_control.py scripts/run_open_jev_optimizer_control.py tests/test_open_jev_optimizer_reset.py tests/test_native_optimizer_reset_control.py
```

Extend native state behavior through the adapter's explicit policy and permanent
ledgers; preserve the default recipe and add behavioral ownership/refusal tests.
Any new scientific control needs a distinct prospective protocol and budget.

The [accepted optimizer-control run](../artifacts/runs/native-optimizer-reset-20261022/RESULTS.md)
completed12updates/one reset/six readouts: B2/2, A1/2, anchors2/2 with meanNLL
drift+.17685. B acquisition passes; A retention and the+.1 anchor guard fail.
Independent acceptance validates the study, not combined task success. No retry
or additional dose was run.

## Bounded TRAIN-only rehearsal

The existing control CLI can replay one old A TRAIN case after each B update:
A0/A1 repeated3, then B0/replayA0/B1/replayA1 repeated3. This is18paid TRAIN
exposures, including6replay exposures from2distinct old cases. It retains one
inbox/budget/clock and does not save model checkpoints or a success-record chain.
New opted-in A experiences grant replay permission prospectively; previous
historical consent stays unchanged. Full paid source receipts and TRAIN roles
are checked before B/replay. Development/final/probe cases cannot be retained.

```bash
python -m scripts.run_open_jev_optimizer_control \
  --source /path/to/pinned/source \
  --checkpoint /path/to/pinned/package/checkpoint \
  --reference artifacts/runs/native-choice-recurring-20261022/actual.json \
  --rehearsal --optimizer-reset 0 \
  --output /path/to/fresh-continuous-rehearsal.json
```

`--optimizer-reset 1` selects the separately declared fixed after-A reset arm.
The CLI defaults to the original no-rehearsal reset control; reset0 requires
`--rehearsal`. See the [frozen pair protocol](../artifacts/runs/native-train-rehearsal-20261022/PROTOCOL.md)
before model work. Both arms use the same18exposures; reset construction/checks
still add computation. Saved no-replay12update controls have lower optimization
cost. Six post-B development readouts score B acquisition/A retention/anchors;
baseline/A readouts are reused only after exact weight/state binding. A valid
run may fail those criteria. Recovery and generalization are not tested here.

```powershell
python -m pytest -q tests/test_native_choice_sequence.py tests/test_native_cold_b_control.py tests/test_native_optimizer_reset_control.py tests/test_native_train_rehearsal.py
python -m ruff check scripts/run_open_jev_sequence.py scripts/run_open_jev_optimizer_control.py tests/test_native_train_rehearsal.py
python -m ruff format --check scripts/run_open_jev_sequence.py scripts/run_open_jev_optimizer_control.py tests/test_native_train_rehearsal.py
python -m mypy --follow-imports=silent scripts/run_open_jev_sequence.py scripts/run_open_jev_optimizer_control.py tests/test_native_train_rehearsal.py
```

Extensions should preserve original source permissions and pay new exposures
through the same inbox. Add role/consent/receipt/failure tests before changing
replay selection; freeze a separate protocol before comparative model work.

The [accepted two-arm study](../artifacts/runs/native-train-rehearsal-20261022/RESULTS.md)
retained A2/2 and passed anchors in both arms, while B remained0/2. Rehearsal
did not solve the combined task. Pair costs36TRAIN updates/12readouts/8100tokens,
187.45s command time, peak14,013,186,048B. This preserves the negative result;
the implementation passing does not establish useful adaptation or CPC benefit.

## TRAIN-only realized-update health

`--update-health` on the existing optimizer-control CLI opts into
`OpenJevTrainingPolicy(record_update_health=True)`. The default remains off.
For an already authorized protocol, add it to the command above. This option
does not grant another model run, change a dose, select replay or alter a limit.

Every successful TRAIN delivery then has:

```python
health = report["delivery"][0]["training"]["update_health"]
head = health["groups"]["head"]
print(head["gradient_l2_before_clip"], head["parameter_step_l2"])
print(health["work"])  # cumulative conservative attempted-element charges
```

Groups are `head`, `lora_A`, `lora_B`. Counts distinguish parameters with a
missing gradient from present zero gradients. Gradient norms are pre-clip;
steps are measured after native AdamW, including moment/decay effects. These
are float64 reductions over native float32 coordinates, not extra gradients or
learning-rate-times-gradient estimates. No extra forward/backward/optimizer/
RNG call is made, and no readout/final/probe data is fetched.

The separate `learner.update_health_work` dict is detached and permanent:
attempts, attempted parameter-copy/gradient-reduction/step-reduction elements,
completed reports. Restores and resets do not renew it. CLI output keeps the
final ledger even after failure. Counts describe attempted source-element
operations; casts/subtraction/reduction scratch, allocator traffic and overhead
are additional, not complete FLOP/memory accounting. At most5mtrainable values
are copied temporarily (20MBfloat32), plus float64 scratch. The frozen base is
never cloned for health. No tensor is retained in published reports/snapshots.

A diagnostic failure poisons the learner and preserves paid work. If AdamW
already ran, it remains charged without a false native commit or app receipt.
The preceding successful report remains prior evidence. Default five native
work counters and default report shape stay unchanged.

```powershell
python -m pytest -q tests/test_open_jev_training.py tests/test_open_jev_optimizer_reset.py tests/test_native_optimizer_reset_control.py tests/test_native_train_rehearsal.py tests/test_open_jev_update_health.py
python -m ruff check src/adapters/open_jev_training.py src/adapters/open_jev_update_health.py scripts/run_open_jev_optimizer_control.py tests/test_open_jev_update_health.py
python -m ruff format --check src/adapters/open_jev_training.py src/adapters/open_jev_update_health.py scripts/run_open_jev_optimizer_control.py tests/test_open_jev_update_health.py
python -m mypy --follow-imports=silent src/adapters/open_jev_training.py src/adapters/open_jev_update_health.py scripts/run_open_jev_optimizer_control.py tests/test_open_jev_update_health.py
```

[Executed scope](../artifacts/runs/native-update-health-20261022/RESULTS.md):110
checks/platform, independent scalar oracles/state equality/cost/failure tests,
and a real tiny native-learner/inbox CLI update with fake loading metadata.
This is not a released-model run or evidence of improved task learning. Extend
diagnostics through this small helper and scalar receipt schema; keep selection
and probe policy separate, and budget any additional reads explicitly.

## Seven-update TRAIN diagnosis

`scripts.run_open_jev_train_health` is a fixed diagnostic entry point: the
original A6 updates, exact accepted after-A state binding, then first B TRAIN
case once. It composes the existing phase/exposure/inbox helpers. No prediction,
replay, reset, probe, final evaluation or outcome-selected extra update occurs.
It refuses existing output paths and preserves paid prefixes on failure.

```sh
python -m scripts.run_open_jev_train_health \
  --source /path/to/cached/source --checkpoint /path/to/cached/checkpoint \
  --reference artifacts/runs/native-choice-recurring-20261022/actual.json \
  --output /path/to/new-diagnostic.json
```

Budget and review a separate model invocation before using this command.
`update_diagnostics` contains seven scalar rows, raw group norms/counts plus
RMS normalized over **all owned coordinates** (missing gradients remain explicit).
`clipping_scale_from_reported_norm` is min(1,1/(norm+1e-6)) reconstructed in
Python float64, not the bit-exact float32 tensor coefficient. Last loader
`work_observed` event includes all TRAIN forwards/sequences/tokens and zero
prediction/Scorer calls. An optional loader `work_reporter` transports this
zero-argument observation callback; existing inference callers are unchanged.

```powershell
python -m pytest -q tests/test_native_train_health.py tests/test_open_jev_native_smoke.py
python -m ruff check scripts/run_open_jev_train_health.py scripts/run_open_jev_native_smoke.py tests/test_native_train_health.py
python -m ruff format --check scripts/run_open_jev_train_health.py scripts/run_open_jev_native_smoke.py tests/test_native_train_health.py
python -m mypy --follow-imports=silent scripts/run_open_jev_train_health.py scripts/run_open_jev_native_smoke.py tests/test_native_train_health.py
```

[Protocol and evidence](../artifacts/runs/native-train-health-20261022/PROTOCOL.md).
This describes arrived TRAIN work; fresh-task probe ownership, task success,
retention/recovery, comparative benefit and release readiness remain separate.
Extend the existing native health helper for new measurements; do not use
diagnostic summaries as an implicit training controller.

## Disposable native ownership

For a separately admitted probe, construct the source with an explicit copy
allowance. The source/fork retain the **same spent native/reset/health limits**;
forking is not a way to continue an exhausted experiment.

```python
policy = OpenJevTrainingPolicy(max_updates=12, expected_lora_rank=8, max_forks=1)
# source = OpenJevTrainingLearner(model, compile_request, instructions=..., policy=policy)
child = source.fork()
birth_work = child.work
# Train child only under a separate prospective probe budget/role admission.
```

`max_forks` defaults0 and accepts only integer0/1. Both owners inherit the paid
fork attempt, so neither can fork again. Each has its own model modules/modes,
trainable parameters/gradients, registered buffers, AdamW moments/private RNG,
policy and scalar diagnostics. Frozen Parameters with no gradients share storage
and keep the ordinary version/identity guard. The stateless native compiler is
shared. No constructor, loader, checkpoint/input-gradient setup or model forward
is rerun during copying. Foreign snapshots remain refused.

The separate detached `fork_work` ledger counts attempts, successful forks and
attempted parameter/gradient/buffer/optimizer/RNG copy source-elements. It is
permanent across restore. New child work is the delta from its inherited birth
baseline, not a second charge for parent updates. Three RNG state copies/reads
are charged on success: caller before, private clone, caller after. An uncertain
copy failure poisons the source and retains charges without a false success.

Buffers must be materialized CPU tensors,≤5m elements and≤40MiB; distinct
registered tensors sharing storage or Parameter-as-buffer aliases are refused
before allocation. Source-element counters omit Python/tokenizer/allocator
overhead; copying trusted unregistered metadata/hooks relies on upstream copy
behavior. This local port does not sandbox hostile `.data`, copy or compiler
code. Qualification on literal models does not establish an executed 2B fork.

```powershell
python -m pytest -q tests/test_open_jev_training.py tests/test_open_jev_optimizer_reset.py tests/test_open_jev_update_health.py tests/test_open_jev_owned_fork.py
python -m ruff check src/adapters/open_jev_training.py src/adapters/open_jev_owned_fork.py tests/test_open_jev_owned_fork.py
python -m ruff format --check src/adapters/open_jev_training.py src/adapters/open_jev_owned_fork.py tests/test_open_jev_owned_fork.py
python -m mypy --follow-imports=silent src/adapters/open_jev_training.py src/adapters/open_jev_owned_fork.py tests/test_open_jev_owned_fork.py
```

[Decision](adr/ADR-0523-own-disposable-native-learners.md) ·
[Protocol and evidence](../artifacts/runs/native-owned-fork-20261022/PROTOCOL.md).
Use the existing inbox's `probe_train` role admission for eventual text probes;
keep reserved readout labels out of the child training port and replay. No
reserved-text corpus/experiment or learning benefit is delivered by a copy alone.

## Reserved text-task probe

The fixed `scripts.run_open_jev_reserved_probe` command warms the original
model on accepted A6, verifies its state, then trains one disposable native fork
on new contextC (`tavrix`→billing). The source remains unchanged. The child gets
six probe_train exposures and C/oldA/anchor readouts before/after. Its work is
reported as a delta from inherited history, with copy/source-proof costs.

```sh
python -m scripts.run_open_jev_reserved_probe \
  --source /path/to/cached/source --checkpoint /path/to/cached/checkpoint \
  --reference artifacts/runs/native-choice-recurring-20261022/actual.json \
  --output /path/to/new-probe.json
```

Use the separately frozen [protocol](../artifacts/runs/native-reserved-text-probe-20261022/PROTOCOL.md)
and its reviewed prospective limits. Existing output is refused. Probe_test
targets are scored in the app after all twelve readouts and never delivered to
the training inbox. Only probe_train labels reach the child. Fixed before/after
endpoints use the existing typed native owner's candidate-batched predictor;
historical serving-Scorer metrics have different batching and are not reused.
Inherited inference temperature remains unrecalibrated.

`summary` distinguishes C endpoint success from acquisition (already-correct C
is not acquired), A retention, anchor drift and untested recovery. NLL uses raw
probabilities: zero means explicit infinite, never clipped. Negative outcomes
never choose another task, seed or extra update. Source proof covers native
snapshot tensors/optimizer/private RNG plus buffers, every module mode, caller
RNG, last_update and permanent native/reset/health work. Fork work is the sole
intentional source ledger change. Paid failures stay incomplete and are retained.

```powershell
python -m pytest -q tests/test_native_reserved_text_probe.py
python -m ruff check src/core/native_choice_probe.py scripts/run_open_jev_reserved_probe.py tests/test_native_reserved_text_probe.py
python -m ruff format --check src/core/native_choice_probe.py scripts/run_open_jev_reserved_probe.py tests/test_native_reserved_text_probe.py
python -m mypy --follow-imports=silent src/core/native_choice_probe.py scripts/run_open_jev_reserved_probe.py tests/test_native_reserved_text_probe.py
```

Core owns frozen declarations; the script composes existing typed ports/inbox/
scalar metrics. [Decision](adr/ADR-0524-probe-reserved-text-on-a-disposable-native-owner.md).
Extend with a separately frozen protocol, preserving observed-data roles and
permanent budgets. A reserved-task result is not a circadian or release claim.

## Same-task cold C control

The accepted warm C probe did not acquire C and failed anchor NLL preservation.
`--cold-control --warm-reference <accepted-warm-actual.json>` starts from the
original zero-update source and uses the same observed C corpus, six-update
recipe and candidate-batched typed readouts. Default warm mode still requires
A6/cap12; cold mode requires source0/empty AdamW/cap6. Both fork once and leave
the original source unchanged. The cold child gets no A training.

```sh
python -m scripts.run_open_jev_reserved_probe \
  --cold-control \
  --warm-reference artifacts/runs/native-reserved-text-probe-20261022/actual.json \
  --reference artifacts/runs/native-choice-recurring-20261022/actual.json \
  --source /path/to/cached/source --checkpoint /path/to/cached/checkpoint \
  --output /path/to/new-cold-c.json
```

The warm report is bounded and bound to its accepted hash; it is reused without
model execution. `warm_comparison` reports cold-minus-warm before/after accuracy
and raw NLL. C is now observed, never relabeled fresh. Cold A is untrained
transfer (`A_retained=null`), not a retention endpoint. Match task, probe dose
and prediction path; disclose unequal prior history and total computation.
This comparison cannot isolate weights from optimizer history. Recovery remains
untested. No repeated warm run, outcome-selected dose or circadian benefit claim.
Use the separately frozen [cold protocol](../artifacts/runs/native-cold-c-control-20261022/PROTOCOL.md)
and source/software/executor gates before any native job.

Focused regression command: `python -m pytest -q tests/test_native_reserved_text_probe.py tests/test_native_cold_c_control.py`.
Use the same scoped Ruff/format commands above with the new test file. For the
combined type check, use `python -m mypy --explicit-package-bases --follow-imports=silent src/core/native_choice_probe.py scripts/run_open_jev_reserved_probe.py tests/test_native_reserved_text_probe.py tests/test_native_cold_c_control.py`;
the explicit option resolves the shared test helper's namespace package. No
dependencies or configuration changes. Future mechanisms require separate
prospective protocols preserving these results and observation roles.

## Explicit native adapter projection

`OpenJevTrainingPolicy(..., max_adapter_projections=1, max_optimizer_resets=1)`
admits one explicit `owner.project_adapters(effective_rank)` operation. Both
allowances default0, remain permanent through snapshots and are inherited by
owned forks. Projection always consumes one existing reset even if moments are
empty. It is not a training step, does not renew native updates, and accepts no
labels, mixtures, evaluation results or automatic rank decisions.

Supported opt-in owners bind vanilla PEFT-shaped Linear/default modules with
unmerged active adapters, alpha/r scaling, no transposition/bias/variants and
Identity dropout. All LoRA factors must be covered; frozen base/Parameter
identities, shapes and metadata are guarded. Real PEFT compatibility and native
task results are separate qualifications from literal owner correctness.

Projection stages every layer through [bounded factor math](low-rank-composition.md),
rescales B back to the live module scale and rejects invalid float32 casts.
Allocated rank and Parameter objects remain fixed. The first k rows/columns
hold projected factors; remaining B columns are zero, remaining A rows retain
the original values. This preserves zero inactive forward contribution and
allows later B gradients without random initialization. k is an immediate
projection cap, not a lasting rank constraint or reduction in stored memory.

After staging and protected-state proof, the operation resets owned AdamW and
publishes via copy_. LoRA gradients are cleared because their coordinates are
stale; head gradients/weights, frozen base, buffers, all module modes, caller/
private RNG and spent native/fork/health work remain unchanged. A successful
next TRAIN uses fresh step1 moments. Projection is called explicitly between
ordinary runtime updates. The explicit stream option below requires its own
frozen protocol and exact accepted control references.

`adapter_projection_work` is a detached permanent ledger for attempts, input
copies, core/reserved QR/SVD/product/workspace, staged/cast/written elements,
cleared gradient buffers and RNG work. Caller RNG is read three times on
success; the existing reset's four additional copies/reads are reserved before
calling it. Library internal workspace/compute, equality traffic, metadata and
optimizer Python overhead remain additional. Workspace100m aggregate/20m per
core, factors≤5m/256modules are coordinate limits, not a full RSS/FLOP bound.

A failure after paid admission poisons the owner and retains prefix costs and
spent resets; no rollback/refund/retry is promised. Invalid metadata, aliases,
rank or unavailable authority refuse before paid staging. The bounded
`last_adapter_projection` receipt describes the last completed historical
intervention; later training/restore can change the represented operator.
Caller receipt mutation cannot alter owned history. Snapshots omit permanent
projection charges; copying the owner inherits them rather than renewing them.

```powershell
python -m pytest -q tests/test_open_jev_training.py tests/test_open_jev_optimizer_reset.py tests/test_open_jev_update_health.py tests/test_open_jev_owned_fork.py tests/test_open_jev_adapter_projection.py
python -m ruff check src/adapters/open_jev_training.py src/adapters/open_jev_projection.py src/adapters/open_jev_owned_fork.py tests/test_open_jev_adapter_projection.py
python -m ruff format --check src/adapters/open_jev_training.py src/adapters/open_jev_projection.py src/adapters/open_jev_owned_fork.py tests/test_open_jev_adapter_projection.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/open_jev_training.py src/adapters/open_jev_projection.py src/adapters/open_jev_owned_fork.py tests/test_open_jev_adapter_projection.py
```

[Protocol and evidence](../artifacts/runs/native-adapter-projection-20261022/PROTOCOL.md).
No model quality, task-retention, calibration, CPC or interference-remedy claim.
Future integration must preserve data roles, explicit allowances, failed history
and original matched native controls; use the existing stream/owner interfaces.

## Fixed projection stream

The existing optimizer-control runner accepts `projection_rank=None` by default.
An explicit rank0..8 requires reset authority and the admitted reset-only report;
rehearsal and update-health combinations are initially refused. It delivers the
original A6 through the delayed-label inbox, binds exact accepted after-A weights,
projects once (including one reset), then delivers B6 and six development readouts.
Projection receipts and permanent charges are retained before later callbacks.

```powershell
python -m scripts.run_open_jev_optimizer_control --source <cached-source> --checkpoint <cached-checkpoint> --reference artifacts/runs/native-choice-recurring-20261022/actual.json --reset-reference artifacts/runs/native-optimizer-reset-20261022/actual.json --projection-rank 4 --output <fresh-report.json>
python -m pytest -q tests/test_native_optimizer_reset_control.py tests/test_native_train_rehearsal.py tests/test_native_projection_stream.py
python -m ruff check scripts/run_open_jev_optimizer_control.py tests/test_native_projection_stream.py
python -m ruff format --check scripts/run_open_jev_optimizer_control.py tests/test_native_projection_stream.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_open_jev_optimizer_control.py tests/test_native_projection_stream.py
```

Use the pinned producer and frozen resource limits in the
[protocol](../artifacts/runs/native-projection-stream-20261022/PROTOCOL.md);
the command is not permission to search ranks or rerun spent experiments.
Endpoints use `after_adapter_projection_B`. Primary signed deltas are projection
minus accepted reset-only (positive accuracy/negative NLL favorable); original
warm and unchanged endpoints remain secondary. The original acquisition,
retention and anchor tolerances remain unchanged. Historical controls are
nonconcurrent; additional algebra means this is not matched computation.
Allocated rank stays eight, later effective rank is unmeasured, and return
recovery is not tested. No new environment variable, dependency or core/app API.

[Decision](adr/ADR-0527-compare-explicit-native-projection-with-reset.md).

The frozen rank4 native case completed12TRAIN updates: B2/2, A0/2, anchors2/2
with original NLL tolerance satisfied. B acquisition and anchors pass; retention
and the combined criterion fail. Reset-only retained A1/2. This is a negative
retention result, despite lower B/anchor NLL. See
[results, exact costs and preserved failures](../artifacts/runs/native-projection-stream-20261022/RESULTS.md).

## Fixed two-state adapter consolidation

`OpenJevTrainingPolicy(..., max_adapter_consolidations=1,
max_adapter_projections=1, max_optimizer_resets=1)` enables one reference and
one consolidation. All three authorities are explicit; consolidation defaults0.
The existing owner exposes:

```python
# After the caller's declared first TRAIN phase:
reference_receipt = owner.capture_adapter_reference()
# Continue ordinary TRAIN on that same owner, then at a declared boundary:
composition_receipt = owner.consolidate_adapters(effective_rank=8)
```

Capture owns float32 copies of only the LoRA factors, bounded to one slot with
at most5m elements/20m numeric bytes. It stores source native work and module
metadata and binds tensor identities/versions. Detached public receipts contain
no tensors. Normal mutation/replacement of the retained reference refuses before
more work; `.data` writes/hostile Python remain outside the existing trusted port.
Core feasibility is admitted before capture copies, including the two-term total
rank≤64 requirement (so allocated ranks above32 refuse). No labels, model call,
random initializer, optimizer/head copy or automatic choice of capture point.

Consolidation projects `.5 * reference_operator + .5 * current_operator` through
the existing multi-term core and stages all layers before any parameter write.
These are complete same-base LoRA operators: the current factors already contain
prior history, so they are not independent task deltas. Averaging factor matrices
would introduce cross terms and is not used. The current scalar head is kept;
this operation is not a probability ensemble or a restoration of the old head.

One existing projection allowance and one existing AdamW reset are consumed.
The shared `adapter_projection_work` ledger accounts for both terms' float64
copies, core/QR/SVD/product/workspace reserves, casting, staging and publication.
`adapter_consolidation_work` separately charges reference attempts/float32 copies/
RNG reads/completions and consolidation attempts/completions. Finite scans,
metadata/proof/equality traffic, Python and library internals remain additional.
The source is rechecked after reset and before publication. Head/base/buffers/
modes/RNG/Parameters/allocated rank stay fixed; stale LoRA gradients are cleared,
inactive current A rows remain, inactive B columns are zero and can regrow.

While pending, standalone projection/reset and fork refuse before paid work.
Restore preserves the reference and permanent limits. Success clears the private
factor slot; `last_adapter_reference` retains only detached historical metadata,
and `last_adapter_projection` stores the bounded consolidation receipt identified
by `open_jev_owned_adapter_consolidation_v1`. After success, fork inherits all
spent limits. Paid failures poison without refund/rollback, retaining bounded
prefix state; unlimited reference chains are not retained.

```powershell
python -m pytest -q tests/test_open_jev_training.py tests/test_open_jev_optimizer_reset.py tests/test_open_jev_update_health.py tests/test_open_jev_owned_fork.py tests/test_open_jev_adapter_projection.py tests/test_open_jev_adapter_consolidation.py
python -m ruff check src/adapters/open_jev_training.py src/adapters/open_jev_owned_fork.py src/adapters/open_jev_projection.py src/adapters/open_jev_consolidation.py tests/test_open_jev_adapter_consolidation.py
python -m ruff format --check src/adapters/open_jev_training.py src/adapters/open_jev_owned_fork.py src/adapters/open_jev_projection.py src/adapters/open_jev_consolidation.py tests/test_open_jev_adapter_consolidation.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/open_jev_training.py src/adapters/open_jev_owned_fork.py src/adapters/open_jev_projection.py src/adapters/open_jev_consolidation.py tests/test_open_jev_adapter_consolidation.py
```

This owner capability is qualified separately from stream integration or task
benefit. The stream option below invokes two-state consolidation under its own
frozen protocol. The earlier rank4 projection experiment remains negative and
spent. No calibration, recurrence or CPC benefit follows from operator correctness.
[Decision](adr/ADR-0528-consolidate-owned-native-adapter-operators.md) ·
[Protocol and evidence](../artifacts/runs/native-adapter-consolidation-20261022/PROTOCOL.md).

## Terminal consolidation stream

The existing optimizer-control CLI accepts default-off `--consolidation-rank`
(integer0..8). This mode requires reset authority and excludes projection,
reset-reference, rehearsal and update-health combinations. Exact initial/A6/B6
weight bindings reuse the accepted warm control without rerunning it. Capture at
A6 preserves weights, gradients, modes, RNG and native work. B6 uses continuous
warm AdamW; only then does one consolidation consume the projection/reset
allowance. No TRAIN follows it. Receipts and permanent costs survive later
callback/readout failure; prior reference corruption refuses further work.

```powershell
python -m scripts.run_open_jev_optimizer_control --source <cached-source> --checkpoint <cached-checkpoint> --reference artifacts/runs/native-choice-recurring-20261022/actual.json --consolidation-rank 8 --output <fresh-report.json>
python -m pytest -q tests/test_native_optimizer_reset_control.py tests/test_native_train_rehearsal.py tests/test_native_projection_stream.py tests/test_native_consolidation_stream.py tests/test_native_cold_b_control.py
python -m ruff check scripts/run_open_jev_optimizer_control.py scripts/run_open_jev_b_control.py tests/test_native_consolidation_stream.py
python -m ruff format --check scripts/run_open_jev_optimizer_control.py scripts/run_open_jev_b_control.py tests/test_native_consolidation_stream.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/run_open_jev_optimizer_control.py scripts/run_open_jev_b_control.py tests/test_native_consolidation_stream.py
```

The [protocol](../artifacts/runs/native-consolidation-stream-20261022/PROTOCOL.md)
freezes one rank8 native case and its resource limits; these commands do not
authorize repeating spent experiments. Qualification selected121 scalar cases
on each platform; the general pytest command above includes the full loader
module, beyond the nine loader cases selected for that cohort.

Readouts use `after_adapter_consolidation_B`. Primary contrasts are consolidation
minus accepted warm after-B: positive accuracy and negative NLL are favorable.
Original baseline, after-A, acquisition, retention and anchor criteria remain
unchanged; infinity is retained explicitly. Terminal reset preserves inference
weights, whereas the older after-A reset changes B training history. The two
complete operators share A history and retain the current scalar head. Capture,
projection and validation add compute; this is not a matched-compute comparison.
Allocated rank stays8, postcast numerical rank is unmeasured, temperature is
inherited, and return recovery is untested. No new dependency or core/app API.

[Decision](adr/ADR-0529-evaluate-terminal-native-consolidation.md).

The sole frozen rank8 case completed: A2/2, B0/2, anchors2/2 with NLL drift
+.0717553573 within the original +.1 tolerance. Combined criterion fails B
acquisition. A/B NLL worsen versus the accepted warm endpoint by +.0203181824 /
+.0874570641; anchor NLL improves -.0021573529. There is no demonstrated overall
benefit from this operation. [Executed results and costs](../artifacts/runs/native-consolidation-stream-20261022/RESULTS.md).

## Sequence checks

```powershell
python -m pytest -q tests/test_native_choice_sequence.py
python -m ruff check src/core/native_choice_sequence.py scripts/run_open_jev_sequence.py scripts/run_open_jev_native_update.py tests/test_native_choice_sequence.py
python -m ruff format --check src/core/native_choice_sequence.py scripts/run_open_jev_sequence.py scripts/run_open_jev_native_update.py tests/test_native_choice_sequence.py
python -m mypy --follow-imports=silent src/core/native_choice_sequence.py scripts/run_open_jev_sequence.py scripts/run_open_jev_native_update.py tests/test_native_choice_sequence.py
```

The stage qualification also runs the affected existing one-update scalar
readout tests. It does not repeat unchanged native loss/optimizer/state fixtures
or claim repository-wide readiness from the scoped result.
