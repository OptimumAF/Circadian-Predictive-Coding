# Explicit weighted categorical PC proposal

`src/core/categorical_pc_weighted.py` provides `class_weighted_pc_update`.
It accepts CPU float32/float64 features, hard int64 targets, four parameter
tensors `(feature_hidden, hidden_bias, hidden_output, output_bias)`, an explicit
class-weight tuple, positive learning rates,1..64 relaxation steps and
`role="train"`. Batch size is1..54; input width1..4096; hidden width1..256;
classes2..16. The existing class-weight validator admits finite weights in
`[1e-6,1e6]`. No weights are selected by this function.

For prior `p=tanh(X W_fh+b_h)`, state `z` and target weights `w_i=w[y_i]`,

```
E(z,theta) = sum_i w_i * (CE(z_i W_ho+b_o,y_i) + .5 ||z_i-p_i||²) / sum_i w_i
a_i = B*w_i/sum_i w_i
z <- z - inference_learning_rate * a_i * ((z-p) + (softmax(logits)-onehot(y)) W_ho.T)
theta <- theta - learning_rate * partial_theta E(z_final,theta)
```

Why this normalization: the categorical term matches the existing weighted BP
mean, and weighting the entire row energy gives one coherent objective. Native
PC uses `B*dE/dz` during relaxation and `dE/dtheta` during parameter learning.
The batch factor therefore belongs to relaxation only. Finite relaxation is
not differentiated through; this is a partial gradient at the final state,
not the gradient of an unrolled or exactly minimized objective.

Uniform observed target weights use the original native operation order;
changing weights of absent classes has no effect. All-one float32 proposals
must match both native PC and matched neutral CPC parameter updates exactly.
Nonuniform weight changes can change the finite relaxation path and cost the
same declared number of iterations; they are not a convergence guarantee.

The returned `CategoricalPCUpdate` owns detached parameter/gradient/state tensors
and reports actual weighted `energy`, the separate original unweighted
`legacy_diagnostic` MSE statistic, and nominal latent-row iterations. The legacy
statistic is not the objective. Inputs, their gradients, RNG and grad mode are
preserved. Nonfinite inputs, linear prior, final state/logits, gradients,
candidates or diagnostics are refused.

## Use and checks

```python
from src.core.categorical_pc_weighted import class_weighted_pc_update

proposal = class_weighted_pc_update(
    features, targets, parameters,
    class_weights=(1.0, 2.0, 1.0),
    learning_rate=0.03, inference_steps=2, inference_learning_rate=0.04,
    role="train",
)
```

This is a math primitive, not a learner or schedule. It does not publish
parameters, update traffic/age/chemical state or counters, admit episodes,
access evaluation, choose a probe, or charge a runtime budget. The complete-episode
adapter below now supplies source admission and native publication. Do not copy
only these four parameters into a live CPC learner and call its state complete.

Focused tests: `.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_categorical_pc_weighted.py`.
For source and test paths run `python -m ruff check`, `python -m ruff format --check`
and `python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32`
(also `--platform linux`). The engineering run retains exact executed commands,
failures and receipts in `artifacts/runs/weighted-pc-primitive-20261018/`.

No real-data experiment, acquisition benefit, circadian advantage or release
readiness follows from these correctness checks. Historical negative results,
baseline policies, metrics and defaults are unchanged.

## Atomic complete-episode PC/neutral update

`src/adapters/torch_pc_weighted.py::train_weighted_episode` now accepts an exact
`InvariantTorchFeatureLearner` with native `pc` or `neutral_cpc` policy, a complete
1..16-transition TRAIN episode with raw296/K7 public history, and explicit weights.
Current and terminal next observations are validated; only current observations
and hard actions enter training. Reserved probe/final roles, BP, active CPC,
recurrent and unbound owners are refused. There is no automatic weight selection.

```python
from src.adapters.torch_pc_weighted import train_weighted_episode
from src.core.demonstration_class_weights import episode_class_weights

diagnostic = train_weighted_episode(
    invariant_pc_learner, complete_train_episode,
    episode_class_weights(complete_train_episode),
)
```

The adapter stages an independent complete native head, runs the accepted
weighted relaxation/partials, and reuses native `_apply_training_gradients` for
the original traffic/chemistry/importance/age/counter/legacy-history transition.
Native `train_step` uses the same extracted commit with its original unweighted
gradients. This avoids maintaining two definitions of neutral adaptive state.
The actual weighted energy gets a separate diagnostic ID; neutral's history
continues storing its original unweighted MSE statistic. The intermediate core
proposal also calculates candidate parameters; the native commit recalculates
them alongside complete state. This is bounded arithmetic, not a second update.

Complete candidate validation and unchanged donor/input/policy/gradient/RNG/mode
checks precede one payload assignment. A failed staged update publishes nothing;
foreign changes are refused without erasing them. Ordinary grad mode and
`torch.no_grad()` are supported and preserved; inference mode is refused before
candidate work. No schedule budget or source-release authority is invented.

60 final-tree focused cases pass:47 new adapter cases plus13 native/active/finite
regressions. Full all-one snapshots match ordinary native PC/neutral through S2
and S64 repeated updates; weighted fork/restore continuation is exact. Scoped
Ruff/format/mypyWin+Linux pass with independent source/actual review.
`artifacts/runs/weighted-pc-episode-20261018/RESULTS.md` records exact commands,
attempted/failed work and limitations. Focused new tests:
`.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_pc_weighted.py`.

## Existing weighted TRAIN schedule

`TorchTrainWeightedLearner` now accepts bound native `bp`, `pc` and `neutral_cpc`
policies, as well as its original recurrent policy. The feature path preflights
all complete TRAIN episodes before native construction. Each update must exactly
match the next owned episode's arrays; weights derive only from that episode's
labels. Native policy remains sealed and the separate weighted objective ID is
checked on every receipt. Active CPC is still refused.

```python
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.core.demonstration_episode import episode_arrays

learner = TorchTrainWeightedLearner(invariant_pc_policy, train_episodes)
for episode in train_episodes:
    diagnostic = learner.train_batch(*episode_arrays(episode))
```

The existing detached candidate/publication path, schedule witnesses, prefix and
row counters, input/donor/RNG/mode checks, fork, restore and deepcopy are reused.
PC/neutral dispatches to the accepted complete episode update; BP and recurrent
callbacks retain their original behavior. Public prediction is update-free and
feature policies cannot create recurrent sessions. Full S2/S64 state matches
direct weighted updates; ordinary delayed TRAIN stream receipts and stable source/
actor behavior are verified.136 focused cases pass (46newPC/50BP/40fake recurrent),
with scoped Ruff/format/mypyWin+Linux and independent review.

Focused command: `.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_weighted_pc_stream_adapter.py tests/test_weighted_feature_stream_adapter.py tests/test_torch_train_weighted_learner.py`.
Exact executed commands, resources and limitations are in
`artifacts/runs/weighted-pc-schedule-20261018/RESULTS.md`.

## Explicit online API/CLI composition

The existing `minigrid_learnability.describe_protocol` and `run_pilot` accept
`weighted_pc=True` with `public_history=True`, `invariant_history=True`, and an
explicit `qualification_method="pc"` or `"neutral_cpc"`. Omitting the flag or using
False preserves original behavior. `weighted_bp` retains its existing BP meaning.
Active CPC, fixed64 aliases, memory control, PC qualification alias, simultaneous
BP weighting and joint catalogue are refused before resource construction.

Describe the frozen protocol without constructing models or collecting data:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution `
  --scenario torch-minigrid-history-learnability --invariant-history `
  --qualification-method pc --weighted-pc --distinct-probe `
  --old-task-readouts --return-to-a --describe
```

Use `neutral_cpc` for the matched neutral recipe. Omitting `--describe` runs the
existing bounded local pilot; freeze and review a separate small scientific
protocol before interpreting real-data comparisons. This increment's native
tests use synthetic sources and forbid teacher/game collection.

The fixed benchmark sees128 delayed online nine-row TRAIN packets (1152 row
exposures), native seed17/H8/lr.03/S2/inference_lr.04. The factory owns complete
TRAIN metadata and exact arrays; weights derive only from the current episode.
Benchmark receipts name actual weighted CE-plus-hidden energy. The optional F/
old-A/A-return route instead constructs separate ordinary unweighted source
templates, proves their full initial state equals the original benchmark source,
and restores each owned checkpoint only onto disposable probe forks. Original
budgets, paid prefixes, source/actor/checkpoint/scorer guards and unweighted raw
CE/accuracy/exact-sequence metrics remain authoritative. Probe diagnostics retain
the original legacy native MSE identity and are not the weighted objective.

`artifacts/runs/weighted-pc-public-20261018/RESULTS.md` records executed gates,
failures, resource amendments and qualifications.183 distinct cases are accepted
through composite scripted/native gates, not one clean invocation. Both full
native PC/neutral cases and the BP regression pass; the initial fixture failure
and interrupted native gate remain preserved. Six-path statics, actual CLI
describe and independent review pass. This establishes no learning advantage.
The optional live-game/extra
diagnostic combinations are outside this increment's native test coverage.
The joint54-row mode still rejects these online episode-weighting flags. Its
separate catalogue option below represents several complete sources per wake;
do not reinterpret them as one episode or change the weighting law. No new
runner, active-CPC weighting or scientific advantage claim.

## Typed weighted TRAIN catalogues

`src/adapters/torch_catalogue_weighted.py` admits1..6 complete released TRAIN
raw296/K7 sources and at most54 independent rows. `TrainCatalogue` owns complete
source copies and exact half-open source spans. Terminal next views and causal
first/current histories are validated even though only current-view rows feed
the native update. The catalogue is never represented as one episode.

```python
from src.adapters.torch_catalogue_weighted import (
    prepare_train_catalogue, catalogue_class_weights, train_weighted_catalogue,
)

catalogue = prepare_train_catalogue(released_train_sources)
weights = catalogue_class_weights(catalogue)
diagnostic = train_weighted_catalogue(bound_learner, catalogue, weights)
```

The fixed catalogue rule is N/(present_classes*class_count) across all its rows,
with absent classes1. It is a separately named objective, not the original
per-episode rule. Explicit weights must be a finite positive K7 tuple in the
existing bounds. BP uses weighted CE/SGD; PC/neutral reuse the accepted weighted
energy and complete native transition. Constant observed BP weights cancel and
use ordinary reduction order. All-one updates exactly match complete native
state, including continuation and fork/restore; nonuniform parameter/energy
oracles and repeated PC-neutral parameter parity also pass.

Each call commits one update and the actual row count only after candidate,
donor, source, policy, gradients, RNG, mode and counter checks. Foreign changes
are refused without rollback. PC supports no_grad; BP requires gradients and
all methods refuse inference mode. Active CPC and reserved data roles refuse.
Validation/copy/projection/native/RSS costs remain additional; caller owns paid
release authority and budgets. This low-level adapter supplies the native boundary
used by the separate public catalogue option below. Existing online/default episode paths stay
unchanged. See ADR-0409 and the stage RESULTS for exact commands/failures/costs.

Focused command:
`.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_catalogue_weighted.py`.
70 distinct cases qualify through composite gates; this was not one clean70-case
run. The initial13 fixture failures and later single counter-fixture failure are
retained. No real-data experiment, adaptation benefit or release claim follows.

## Catalogue binding to the existing sequential stream

```python
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner

learner = TorchTrainWeightedLearner.from_catalogue(bound_policy, catalogue, updates=128)
# Pass learner to the existing TRAIN-only AttributionStream. Each admitted batch
# must contain exactly catalogue's raw296 inputs and labels in source-span order.
```

`bound_policy` must be the exact invariant BP, PC or neutral policy; the update
count is an exact integer1..128. A single owned catalogue/witness/span seal is
retained independently of wake count. The original episode/recurrent constructor
remains available. Every current wake derives aggregate weights from the owned
TRAIN catalogue only; typed source evidence is not reconstructed from arrays.
Public catalogue API/CLI composition is available through the separate option below.

The same detached candidate/publication path enforces exact float64 inputs,
objective receipts, fixed update and row-prefix limits, original source/actor/
state/gradient/RNG/mode guards, full native fork/deepcopy/restore and refusal
without overwriting foreign changes. Source evidence is checked again after
catalogue fork copying. Restore never extends the configured ceiling; execution
budgets and permanent paid charges remain the caller/stream's responsibility.
Predict remains update-free; feature policies do not create recurrent sessions.

218 distinct cases (82 catalogue/136 existing) qualify through composite gates;
initial fixture/error-message failures remain recorded. Real three-wake delayed
TRAIN streams were tested for each method using synthetic sources, not game
physics or comparative learning. See ADR-0410 and
`artifacts/runs/weighted-catalogue-stream-20261018/RESULTS.md` for exact commands,
work, failures and limits. No scientific grant or negative result was changed.

## Public joint catalogue API/CLI

Use `weighted_catalogue=True` with the existing `joint_catalogue=True`, explicit
`qualification_method="bp"`, `"pc"` or `"neutral_cpc"`, both public/invariant
history modes and distinct_probe/old_task_readouts/return_to_a. The ordinary
joint recipe remains the default. The original online weighted_bp/weighted_pc
flags keep their episode semantics and cannot be combined with this option.

Describe without constructing native learners or collecting sources:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution `
  --scenario torch-minigrid-history-learnability --invariant-history `
  --qualification-method pc --joint-catalogue --weighted-catalogue `
  --distinct-probe --old-task-readouts --return-to-a --describe
```

Omitting --describe runs the existing bounded pilot; freeze/review a separate
small scientific protocol before comparative runs. Each of128 TRAIN wakes sees
the original six sources/54 independent rows (6912 exposures). Fixed aggregate
weights use all and only those TRAIN labels. The factory validates all source
spans, row IDs, arrays and labels, then uses the existing catalogue stream owner.
No new objective mathematics, orchestrator, replay or final-data access is added.

F16/old-A/A16 use ordinary unweighted learners: each original zero-update source
must match a separate ordinary template in full native state, then checkpoints
are restored only onto disposable forks. Original native policy, budgets,
support novelty gate, CE/accuracy/exact metrics, receipt/source/actor/checkpoint/
scorer guards remain authoritative. Catalogue receipts report actual weighted
objectives; ordinary probes retain their original diagnostic IDs.

The factory currently creates another transient128-wake comparison plan (~16.4MB),
not extra native updates; time/RSS includes validation and ownership copies.
Existing optional readonly margins/live-port admission is retained, but those
combinations are not natively covered here. Public weighted PC checkpoint diagnostics are now accepted with the existing
work/report/checkpoint guards; see the public command below.247 scoped cases
pass (243 scripted plus three complete weighted native methods and ordinary joint
BP regression), with scoped statics/CLI/independent review. That engineering gate contains no real-data run;
the subsequent fixed study is recorded below. Useful acquisition/retention/
recovery or circadian advantage is not established.
See ADR-0411 and artifacts/runs/weighted-catalogue-public-20261018/RESULTS.md.

## Fixed real-data joint result

Three weighted cases complete with four original controls reused. Weighted BP
48/54 correct and3/6exact versus ordinary45/54; weighted PC/neutral16/54 versus
ordinary30/54. All full acquisition/old-A both-cue flags false; no retained or
recovered acquired skill or circadian benefit. See
[protocol, full curves, commands and costs](../artifacts/runs/weighted-joint-study-20261018/RESULTS.md).
Ordinary native PC already uses CE update directions; its MSE scalar is a legacy
diagnostic, not the differentiated objective. Frozen study wording incorrectly
described a new CE law; explicit post-run erratum preserves original artifacts
and changes no metrics or scientific protocol. All-one native continuation
evidence is reused. Historical missing probe-probability vectors remain open.

## Exact weighted update diagnostics

For an owned invariant PC/neutral snapshot and released complete TRAIN sources:

```python
from src.adapters.torch_catalogue_weighted import prepare_train_catalogue, catalogue_class_weights
from src.adapters.torch_pc_update_diagnostics import diagnose_pc_update

catalogue = prepare_train_catalogue(train_sources)
weights = catalogue_class_weights(catalogue)
report = diagnose_pc_update(snapshot, train_sources, class_weights=weights)
```

Weights are explicit, validated, and never inferred from probes/final data. None
retains original unweighted behavior/schema. The diagnostic reuses the exact
weighted core proposal, does not train/publish, and retains state/RNG/gradient/
mode guards and partial failed-work receipts. The distinct weighted definition
reports ordinary and weighted prior CE separately. Gradient/delta geometry uses
the weighted prior CE gradient and the rounded native parameter change.

Fixed-state energy after holds the original final latent z fixed, recomputes the
candidate prior and output, and performs no second relaxation. It is not a fully
relaxed optimum or a task-acquisition metric. Legacy MSE is a separate diagnostic;
ordinary PC already uses CE directions. No new CE control or baseline replacement.

Weighted mode collects no intermediate latent trace: one native proposal port
is explicitly charged with rows/nominal iterations and attempts/returns. It also
pays one projection, two prior forwards, one backward and one fixed-state energy
port (candidate prior/output projections inside). Zero native learner updates/
optimizer/published counts do not mean zero computation. Source/candidate storage
and validation are included in measured process resources.

93 focused cases include independent NumPy and exact actual-native PC/neutral
S2/S64 oracles plus original unweighted regressions; scoped statics/review pass.
[Exact commands and limits](../artifacts/runs/weighted-pc-update-diagnostics-20261018/RESULTS.md).
Public weighted checkpoint composition is now accepted with those guards; see
the command below. No useful learning or benefit claim follows.
## Public weighted checkpoint diagnostics

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution `
  --scenario torch-minigrid-history-learnability --invariant-history `
  --qualification-method pc --joint-catalogue --weighted-catalogue `
  --pc-update-diagnostics --distinct-probe --old-task-readouts --return-to-a `
  --describe
```

Use explicit pc or neutral_cpc. Omit --describe only under a separately frozen,
reviewed local protocol: the existing pilot then runs its full original recipe.
Only0/128 after all six paid TRAIN scores, derive fixed aggregate weights from
those original six TRAIN sources, call the exact numerical diagnostic on an owned
snapshot, and retain original source/actor/checkpoint/receipt/scorer/budget guards.
Ordinary F/old-A/A-return templates remain separate, unchanged and unweighted.
BP/malformed flags/incompatible routes still refuse; defaults stay unchanged.

Two reads add108 TRAIN rows of weight derivation/projection/proposal/energy work,
216 nominal latent-row iterations, four prior forwards and two backwards plus
bounded copies/validation measured in wall/RSS. Numerical proposals are paid
computation, not additional learner updates. Fixed-z energy uses the candidate
prior/output with original final z, no second relaxation. Weighted payload/list/
work owners and exact types are bound; partial failures retain costs/evidence.

284 current scoped cases qualify through composite275PASS1FAIL+10PASSfixture
repair and8PASS native-file checks (four full cases/four scripted); seven-path
statics/actual CLI/independent review accepted. Prior93 numerical checks reused.
[Exact commands and costs](../artifacts/runs/weighted-pc-public-diagnostics-20261018/RESULTS.md).
No real-data mechanistic result, useful acquired-task retention/recovery or
circadian benefit follows. Original weighted16/54 negative and missing probe
vectors remain preserved/open. See ADR-0413 for the composition rationale.

## Existing fixed64 weighted catalogue controls

The public joint weighted route also admits the existing named `pc_relax64` and
`neutral_cpc_relax64` recipes. These retain the original native PC/neutral methods
and complete fixed S64 policies; S2/default and online episode weighting stay
unchanged. Use the existing runner and the original distinct F/old-A/A-return
recipe:

```powershell
python -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method pc_relax64 --joint-catalogue --weighted-catalogue --distinct-probe --old-task-readouts --return-to-a --describe
```

Replace only `pc_relax64` with `neutral_cpc_relax64` for the matched neutral recipe.
The preview constructs no learner or environment. Actual comparative execution
requires a separately frozen/reviewed local protocol and budget; this is an
existing sensitivity endpoint, not a new relaxation-horizon search or predicted
fix. Fixed64 action-margin and PC-update diagnostics remain refused.

One full route uses128 weighted54-row benchmark updates plus32 ordinary F and32
ordinary A-return nine-row updates on disposable copies. Both ordinary zero-state
probe templates must equal the complete weighted source snapshot before any
checkpoint restore. Original source/actor/receipt/scorer/budget guards and final
not_run isolation remain. Nominal latent-row iterations are442368 benchmark and
479232 including probes, versus14976 for original S2. Validation/copies/readouts/
wall/RSS are additional costs; no equal-compute or useful-learning claim follows.

Behavioral admission/ownership tests: `tests/test_minigrid_weighted_relax64_route.py`.
Full synthetic native route tests: `tests/test_minigrid_weighted_relax64_native.py`.
Run with `python -m pytest -q` (native work must fit the local allowance). For the
five changed source/test paths, run `python -m ruff check`, `python -m ruff format
--check`, and the scoped mypy commands above with both win32/linux platforms.
Exact executed commands, retained output, costs and gate status are in
[engineering results](../artifacts/runs/weighted-catalogue-relax64-20261018/RESULTS.md)
and [ADR-0414](adr/ADR-0414-weighted-catalogue-fixed64-composition.md).
