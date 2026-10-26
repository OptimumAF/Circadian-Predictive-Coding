# Two complete episodes in one recurrent update

`src/adapters/torch_recurrent_episode_batch.py` is a separately chosen learner.
It averages two episode-mean cross-entropies and makes one atomic native SGD
step with original momentum. Each episode starts with independent zero hidden
activation. Unequal lengths are allowed: episodes receive equal weight, rather
than weighting their losses by row count or concatenating their recurrent paths.

## Interface

```python
from src.adapters.torch_recurrent_episode_batch import TorchRecurrentEpisodeBatchLearner
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy

learner = TorchRecurrentEpisodeBatchLearner(TorchRecurrentPolicy(2, 3, 3, 17))
# first and second are complete, distinct-ID TRAIN DemonstrationEpisode records.
diagnostic = learner.train_episode_pair((first, second))
owned = learner.snapshot_state()
independent = learner.fork()
probabilities = independent.predict(features)
```

Inputs must be an exact two-item tuple of complete TRAIN records with distinct
IDs, bound feature/class dimensions and1..max_episode_steps transitions each.
Final and probe roles are refused. A pair has at most32rows under the original
per-episode16step bound. No environment collection or data release occurs here.
The existing policy retains its original bounds and local seeded initialization.

The returned EpisodePairDiagnostic contains the new objective ID, averaged loss,
two input IDs/losses, and attempted/completed forward/backward rows, optimizer
steps and published updates. A staged failure raises EpisodePairUpdateFailed
with owned `.work`: a physical optimizer step can be paid without publication.
Input admission errors before staging use ValueError. Partial costs and unknown
work are never silently turned into a successful update.

Candidate tensors/optimizer are owned. Publication occurs after finite capture,
full original state comparison, input/RNG/mode checks and diagnostic construction.
Unknown in-place owner changes are refused and stay visible; they are not repaired.
No-grad callers are supported within a temporary enable_grad context;
inference_mode is refused. Prediction preserves learned state and starts at zero.

## Cold public-history control

The opt-in `initialize_public_history_control` adapter extends an original cold
148-input paired snapshot to296 inputs for `encode_public_history_episode`.
The first148 input weights are zero; the last148 and all other parameters are
copied from the raw snapshot. Starting predictions therefore agree within native
float32 roundoff. Same-seed wider random initialization would not preserve them.

```python
from src.adapters.torch_recurrent_public_history import initialize_public_history_control
from src.adapters.torch_train_episode_batch import TorchTrainEpisodeBatchLearner
from src.core.minigrid_public_history import encode_public_history_episode

history_state = initialize_public_history_control(raw_learner.snapshot_state())
episodes = tuple(encode_public_history_episode(ep) for ep in raw_train_a_episodes)
control = TorchTrainEpisodeBatchLearner(history_state.policy, episodes)
control.restore_state(history_state)
# Use control with the existing AttributionStream(training_only=True).
# Transport each pair as concatenated arrays; keep episode metadata on the owner.
```

The input must have zero updates, episodes and rows. Trained snapshots, other
widths, legacy snapshot types and malformed state are refused, with no clock
reset or optimizer migration. The caller owns the returned state independently.
The existing encoder preserves roles/IDs and uses only public first/current
observations. TRAIN array admission still refuses final/probe records. Existing
raw148 experiments, default history routing and their validators are unchanged.

With hidden8 and7 actions this control has2503 parameters, versus1319 raw:
1184 additional trainable weights, plus their gradients/momentum after training.
Extra encoded observations, memory and computation must be reported. This is a
testable acquisition prerequisite, not evidence of learned skill or CPC benefit.

The existing `minigrid_policy_evaluation.evaluate_checkpoint` now supports this
condition with explicit `method="recurrent_paired_bc", public_history=True`.
Supply original148 TRAIN-A source packets and a296 paired checkpoint at existing
0/16/32/48/64 labels, with2episodes/18rows per update. The evaluator encodes each
trace independently and starts one fresh recurrent session/first-frame encoder
per live cue. Live observations stay148 in reports; prediction features are296.
No probe/final role, row override or arbitrary layout route is added. Default
paired evaluation still requires148 inputs and its original counter validator.

[Qualification](../artifacts/runs/r21-paired-history-readout-20261017/RESULTS.md):
37new/122existing direct checks, scoped statics and one actual cold two-cue
readout completed with0SGD/32physics. Both policies timed out with no goal.
The initial broader check timed out and remains incomplete. This establishes
the readout boundary, not a useful acquisition or scientific comparison.

## Recorded public-history acquisition condition

`python -m scripts.run_minigrid_paired_bc_smoke --describe --public-history`
previews the separate condition without Torch/MiniGrid setup. Execution is
separately frozen/budgeted. It retains64paired updates/five captured stream states
but only fixed0/64 raw/live readouts. Initial-law diagnostics precede all SGD.
The primary reference is the accepted raw PAIRED0/64 endpoint evidence; its old
full-five failure stays incomplete. Default/--endpoint-first behavior is unchanged.

The [sole case](reports/paired-history-acquisition-20261017.md) completed:
historyCEgain1.608734 vsraw1.655521, both17/18actions/onegoal/jointfalse.
Added public inputs/capacity did not improve these endpoints. No new dose, seed,
metric, baseline rerun or circadian benefit is implied. Existing shared training
arm's explicit296 lane preserves original clocks/state/receipts and the148 lane.

A later [saved-only first-disagreement diagnosis](reports/recurrent-first-disagreement-20261024.md)
locates both trained blue failures at decision2: Right chosen instead of Left,
with exact teacher-forced/live probabilities on the matching causal prefix.
The history input retains the public cue, so missing input alone does not explain
that condition. No model rerun or pipeline bug repair was inferred. Earlier
negative results and the incomplete raw full case remain unchanged.

## Separate snapshot clocks

The distinct snapshot records:

| Field | Meaning |
| --- | --- |
| completed_updates |Committed optimizer steps reflected in this state |
| training_episodes |Complete episode exposures, including explicit legacy conversion |
| training_examples |Total observed training rows |

The bounds are steps<=episodes<=2steps and episodes<=rows<=max_episode_steps*episodes,
with bounded native integer counts. A cold9+9pair is1step/2episodes/18rows.
Legacy single-episode snapshots cannot represent that cold state; their strict
contracts are unchanged and reject the new type. New restore accepts only its
own exact snapshot and matching policy.

Use `TorchRecurrentEpisodeBatchLearner.from_single_episode_snapshot(snapshot)`
for explicit one-way conversion, preserving parameters, gradients and optimizer
momentum while setting prior episodes=prior native steps. There is no implicit
export to a legacy snapshot. The private episode-count bridge only reuses native
tensor/optimizer validation and staged restoration, and is never a public
checkpoint or benchmark state. Physical steps on unpublished candidates are
reported separately in failure work.

## Qualification and boundaries

[Recorded protocol, checks, costs and review](../artifacts/runs/r21-episode-batch-20261017/RESULTS.md).
44pure admission cases and12native cases passed, including an independently coded
Torch recurrence/autograd/SGD oracle, unequal episode lengths, carried momentum,
full-width cold pair, ownership, modes and paid failures. The native allowance
is spent. Synthetic physical work13SGD75rows remains separate from MiniGrid totals.
Pytest's record_property/xunit2 warning was emitted; actual properties were
present and independently parsed. It is not a warning-free run.

The explicit `recurrent_paired_bc` integration now binds the original complete
episodes to the existing delayed TRAIN array stream, raw checkpoint scorer and
TRAIN A-only live evaluator. `TorchTrainEpisodeBatchLearner` owns an even schedule
of2..128 complete TRAIN A episodes. Its18-row transport arrays must exactly match
the next two owned records; recurrent paths stay separate. Live sessions fork
the original learner and start at zero hidden activation. At checkpoint n the
fixed recipe requires n optimizer steps,2n episodes and18n rows. Other methods
retain their original contracts. Serialization codecs and benchmark/probe dispatch
are separate unfinished boundaries. Never expose the private validation bridge.

Preview the fixed acquisition case without model/environment work:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_paired_bc_smoke --describe
```

The separately budgeted case uses64paired versus128historical SGD steps, both
128episodes/1152rows. Checkpoints0/16/32/48/64 match historical0/32/64/96/128
exposures. Momentum timing differs. All training finishes before diagnostic raw
scores on six original TRAIN packets and native A-only live readouts. Initial
probabilities/decision/physics must match the saved original initial law. Only
initial/final live outcomes have historical controls; intermediate live and B/C
raw readouts are descriptive. No final data, probe, promotion or baseline retuning
enters this route. See the [protocol and evidence](../artifacts/runs/r21-paired-acquisition-20261017/SCOPE.md).
Acquisition, retention/recovery and circadian benefit remain separate claims.

The sole first native invocation completed64updates/128episodes/1152rows and
initial law parity. It retained raw0/16/32 and live0/16, then hit its original
20second diagnostic limit before live32. Endpoint outcomes remain unavailable:
acquisition is inconclusive. No retry or extension is admitted. See
[actual costs and failure](../artifacts/runs/r21-paired-acquisition-20261017/RESULTS.md).
After that case the MiniGrid total was1154updates/11334rows; synthetic units stay
separate. Scoped67boundary tests pass; this is not a repository-wide release gate.

## Measured readout comparison cost

The follow-up source optimization keeps every owner, ordered field, immutable
leaf identity, array-byte check and full native state/counter proof. It collects
the same fields and compares identity in C-level operations, with no proof cache
or skipped traversal.125focused cases include67new pure field/identity corruption
cases. On one identical profiled fake route, time changed27.3941→13.1098seconds;
schedule/signature/full-state/snapshot check counts were unchanged. This is an
instrumented fake comparison, not proof that native20second readouts now finish.
That engineering increment ran no native work.
[Commands and profile](../artifacts/runs/r21-paired-readout-cost-20261017/RESULTS.md).
The original failed experiment and its paid work remain permanent. Any subsequent
native case needs a separate frozen protocol and full new work accounting.

The [separately frozen optimized native case](../artifacts/runs/r21-paired-acquisition-optimized-20261017/RESULTS.md)
then completed another64updates/1152rows but also hit its original20second cap
during live32. All old overlapping probabilities/teacher metrics and complete
prediction/physics records match exactly. New216raw rows and89returnedpredictions/
88physics are retained; endpoint48/64 absent/fulloutcomesnull. Both cases remain
failed/SPENT, acquisition inconclusive. Current permanent MiniGrid total is
1218updates/12486rows at that point, synthetic units separate; no further case
was admitted by that review. The next increment improves
observation ordering in the existing pilot so fixed endpoints precede descriptive
middle readouts, preserving the full5requirement and distinct partial evidence.

## Commands

### Warm isolated fresh-task probes

The [actual reserved-E measurement](reports/paired-reserved-e-20261017.md) is now
complete: both fixed0/64 checkpoints lower E CE under8updates/144rows. Warm starts
and ends17/18actions, cold improves2/18→10/18; neither fits both sequences. Keep
absolute fit and signed gains separate: less headroom does not establish lost
plasticity. These teacher-forced probes are not learner live success/retention.
80newSGD1440rows plus4teacher36physics are permanently charged; no retry/nextGO.

The existing MiniGrid observer now has an explicit internal
`minigrid_learnability._run_probe(..., paired=True)` option. It admits the fixed
original16+2 reserved E episode packets at148/K7, transports8training pairs and
keeps each evaluation episode separate. Probe checkpoints0/2/4/8 row-match the
legacy16update recipe's0/4/8/16 with144training rows. The original stream source
may be cold while its candidate checkpoint is warm; both stay frozen during the
probe. No CLI switch or implicit default change is introduced. The source must
be a fixed paired stream, not a legacy snapshot relabeled by changing counters.

The returned completion guard validates the original charges and paid evidence
before a caller admits later work. Retain only the fixed required probes. Call
it at phase boundaries; do not make the supplied boundary callback recursively
invoke the same completion guard.193104 reserved data-array bytes excludes the
additional original/transport/app/model/episode/report copies counted in total
invocation time and process RSS. Scoped92checks and one native synthetic case
pass; this qualifies the observer integration, not real MiniGrid adaptation.
[Commands, costs and negative/incomplete limits](../artifacts/runs/r21-minigrid-paired-probe-20261017/RESULTS.md).

`TorchPairedProbeLearner(warm_snapshot, probe_episodes)` in
`src/adapters/torch_paired_probe.py` binds an exact paired snapshot to2..32
complete, distinct-ID `probe_train` episodes. It carries all parameters,
momentum and gradients. Its absolute counters are the warm base plus the actual
probe step/episode/row prefix. The default native `train_episode_pair` remains
TRAIN-only; the separate `train_probe_episode_pair` admits unchanged probe roles.
Both refuse final/probe_test training; no role conversion is performed.

Pass this source and its snapshot to the existing
`src.app.fresh_task_probe.run_fresh_task_probe`. Each training AttributionBatch
contains concatenated arrays for its next two original episodes; the adapter
recovers their independent recurrent paths from its owned schedule. Use
metadata-free training and evaluation packets consistently, disjoint arrays and
row IDs, `probe_train` training roles and `probe_test` scoring roles. Each
evaluation packet should represent one original episode; evaluation contexts
must be unique, with every training context represented. The app reports batch
row order for metadata-free packets and does not invent episode metadata.

The app trains a separate owned fork under a fresh budget whose update limit is
the number of pairs; app checkpoints start at local0. Source/deployed learners
remain untouched. A typed native failure adds `failed_native_work`, preserving
physical work and native publication separately from app completed updates and
budget charges. Unknown changes are refused, never repaired. Array reservation
bytes exclude schedule/model/optimizer/copy/Python payloads; measure total time
and process RSS around the invocation as well. This is an adapter component,
not a fresh-task comparative experiment or evidence of retention/recovery.

### Endpoint-first observation

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_paired_bc_smoke --endpoint-first --describe
```

The explicit option has its own protocol identity and observes0,64,16,32,48 after
all64training steps. Original initial parity still precedes every later readout;
all five complete raw/live readouts remain required for full success. Defaults
keep their original order and schema. Metrics select fixed0/64 rather than the
last observation (48); the retained original comparison selects0/128.

Only this option adds `endpoint_evidence`. It remains None until both fixed
endpoints and all source/budget/metric guards passed. `status: validated` contains
owned endpoint outcomes, separately from full-run completion. A later wall-only
stop may preserve them after final integrity readback under the original45s arm
budget; no work is admitted under the expired20s diagnostic budget. Corruption,
cleanup/readback faults or unverified state instead yield `invalid_or_unverified`
with null endpoint outcomes; original paid raw/live records remain. Full outcomes
stay null on any incomplete case. This makes no promotion or release decision.

[Protocol, checks and actual evidence](../artifacts/runs/r21-paired-endpoint-first-20261017/RESULTS.md).
No previous failed case or allowance is resumed or replaced.

The sole separately reviewed endpoint-first native case then validated0/64 but
exhausted20s during a later32readout. Full outcomes remain null/incomplete. The
separate endpoints show **joint acquisition false**, blue8/9teacher actions and
death/return−1, red9/9and goal/return.49375. Original control has identical native
endpoint behavior and greater CE gain (1.8191733 versus1.6555208). No improvement
in the prespecified acquisition metrics; no circadian advantage follows.
All costs/three spent cases retained; cumulative MiniGrid1282updates13638rows,
synthetic units separate. Do not rerun to select a favorable result. Next use the
existing isolated fresh-task probe runner with an explicit warm paired probe
adapter/role admission; the current TRAIN-A schedule cannot serve that purpose.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_recurrent_episode_batch_validation.py
# Each explicit native test invocation is new paid model work; use a frozen allowance.
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_recurrent_episode_batch_native.py
$episodeBatchPaths = @('src/adapters/torch_recurrent_episode_batch.py', 'tests/test_recurrent_episode_batch_validation.py', 'tests/test_recurrent_episode_batch_native.py')
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @episodeBatchPaths
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @episodeBatchPaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @episodeBatchPaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @episodeBatchPaths
```

No dependency, configuration or environment-variable name is added. Extend other
boundaries only with explicit snapshot/counter admission and preserved defaults.
Freeze/review each subsequent experiment; correct math is not learning benefit.

## Public-cue specialization control

`minigrid_public_cue_recurrent_learner.py` composes two existing raw148 native
recurrent learners. The initial actual public blue/red cue selects one head;
neither context metadata nor targets select it. A live session chooses once,
keeps genuine native hidden activation, and ends at the original sixteen-step
bound. Spatial inputs, actions, recurrence, CE and SGD remain unchanged.
Full snapshots own both parameter/gradient/momentum states. Aggregate counters
sum the heads; a staged failed operation leaves one permanent bounded failure
record and refuses further work or restore erasure.

Why this: saved raw and history controls first fail on a correct on-trajectory
blue observation. An explicit specialization control tests shared-head
interference without changing dose, seed or horizon. It is an architectural
hypothesis, not an identified cause or circadian mechanism.

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_paired_bc_smoke --describe --public-cue
```

The distinct opt-in recipe uses the existing delayed stream and native evaluator.
It captures all five original training points, then reads only fixed0/128 TRAIN-A
endpoints after all fitting. The original128updates/1152rows are divided equally
between two heads. Both exact teacher sequences AND both native goals remain the
acquisition criterion. The two heads total2638parameters, compared with old
raw1319/history2503; optimizer schedules and computation differ. There is no
equal-compute/capacity, isolated causal, retention or generalization claim.

Execution is paid model work, requiring a prospectively frozen local allowance;
the sole recorded case uses exclusive output and no automatic retry. Tests with
scripted native ports establish boundaries only; the separately budgeted H2
native fixture checks exact direct-learner continuation and session parity.
See [issue130](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/130)
and [frozen native protocol](../artifacts/runs/public-cue-recurrent-20261024/NATIVE_PROTOCOL.md).
No existing raw/paired/history/default recipe or failed allowance is replaced.

The sole130native attempt completed128fits/1152rows but exhausted its original
TRAIN clock before any score or game. It remains incomplete/failed-spent; final
preservation is unknown and falling pre-update CE is not task success. See
[actual evidence](../artifacts/runs/public-cue-recurrent-20261024/RESULTS.md).

For a future caller-owned workflow, the existing decoder can return original
raw148 episodes without adding affine history:

```python
from src.infra.minigrid_affine_example import example_affine_training
from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters.minigrid_public_cue_recurrent_pilot import describe_protocol

with example_affine_training() as path:
    episodes = read_training_sources(path, raw_frames=True)
protocol = describe_protocol(typed_input=True)  # metadata only
```

`run_pilot(episodes)` accepts that detached, ordered six-episode TRAIN tuple;
it checks raw148/K7/complete episodes/roles/context/public cues before native
work, then reuses the existing source schedule, stream and evaluator. This
typed API reads no historical archive and grants no old model/budget authority.
Caller-owned collection provenance is still required; valid metadata alone is
not a claim that arbitrary supplied episodes came from recorded physics.
Source reading/validation is paid work in any future whole-case protocol.

`read_training_sources(path)` retains its established296history default and
`run_pilot()` retains its original archive-backed protocol. `raw_frames` and
`typed_input` require exact booleans. No CLI default or experiment was changed
or rerun. The typed route's software evidence does not authorize retrying130.

An explicit `recurring_training=True` on the typed recurrent pilot uses the
original A32/B32/C_train32/A32 schedule. `describe_protocol(typed_input=True,
recurring_training=True)` is metadata-only. The mode fits all128episodes before
scoring owned0/32/64/96/128states; it reads6TRAIN traces at each point and plays
only Ablue/red at32 and A/B/C_trainblue/red at128. It reports teacher curves,
signed phase changes and native outcomes separately, with unplayed native64/96
unknown. Fixed-A/archive/CLI defaults remain unchanged; native execution needs
its own frozen allowance.

The sole recurring case completed: final native C2/2 and A1/2, B0/2. C teacher
sequences stay exact across return to A, but B-red's earlier exact sequence is
lost; A32was never acquired. Full source/actor/RNG preservation passes, no
retention/recovery/generalization/CPC/release claim. [Results and qualifications](reports/recurrent-context-stream-20261025.md).
