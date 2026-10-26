# MiniGrid checkpoint learnability

## TRAIN action probability margins

Add --train-action-margins to invariant-history routes to reduce existing paid
checkpoint TRAIN predictions. No extra forward/update or probe/final data access.
For example, describe the fixed BP route without constructing a learner:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --train-action-margins --describe
```

The API option is train_action_margins=True on describe_protocol/run_pilot. It
works with the existing BP/PC/neutral/active, acquisition-only and explicit weighted
routes; existing option restrictions still apply. Defaultfalse preserves outputs.
Each TRAIN score gains action_margins: all raw p[target]-max(p[other]) row gaps,
aggregate/per-class summaries and counts. Absent classes have null mean/min/max;
zero gaps are ties, separate from original first-class argmax accuracy. No epsilon,
clipping or normalization is introduced. Margins are not calibrated uncertainty.

work.train_action_margins totals attempted/completed reductions/rows, logical
detached input bytes and reduction seconds; extra native forwards/SGD/publication0.
Input bytes exclude temporary arrays/report/Python allocations; reduction time
covers copies and core math, while enclosing guards/reporting add runtime cost.
Paid raw scores/costs survive failure; changed donor state is refused, not repaired.
No primary metric, seed, acquisition criterion, update dose or baseline changes.

Extend the pure reducer through core tests; retain TRAIN admission and the existing
guarded runner. Never select classes/checkpoints or use probe/final labels to tune
it. [Commands, all saved TRAIN results and limitations](../artifacts/runs/train-action-margins-20261018/RESULTS.md),
[decision](adr/ADR-0396-paid-train-action-margins.md).

## Bound feedforward prior CE gradients

The [single frozen canonical-PC use](../artifacts/runs/minigrid-pc-prior-diagnostic-20261017/RESULTS.md)
completed all stream/probe/gradient doses within19.49s supervised wall time and
282.25MiB sampled RSS. All exact-acquisition gates fail; post32 raw argmax is
constant action2 on TRAIN traces. CE improvement alone is insufficient for task
skill. The prior unmeasured engineering RSS remains unverified.

Preview the explicit TRAIN-only diagnostic without constructing models or games:

```powershell
python -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --pc-qualification --prior-gradient-geometry --describe
```

`--qualification-method bp|pc|neutral_cpc|gated_cpc` selects another existing bound
head. Omitting qualification reads the four bound feedforward heads while leaving
the original recurrent learner's readout unchanged. The old recurrent option is
`--cue-gradient-geometry`; the two gradient options are mutually exclusive.
A-only/count recipes are rejected because this diagnostic requires the original
six TRAIN source packets. Existing activation-health readouts can compose with it.

Execution through the existing CLI keeps its fixed A/B/C_train/A stream, seed17,
five checkpoints and separate isolated probes. Freeze a small local protocol
before a new experiment; no new comparative run is established by these checks.
Final/probe packets never enter gradient reading. No updates, replay or parameter
selection occur in the diagnostic. No new environment variables/dependencies.

Each `prior_gradient_geometry` entry has mean CE for the same-context blue/red
pair, joint and four parameter-group norms/dot/cosine, and attempted/completed
projection/forward/backward work. Gradients differentiate prediction mean CE at
the tanh prior, **not native PC latent-energy updates**. Cosine is descriptive,
not proof of causal interference or benefit. Complete learned, optimizer, gradient
and RNG state is guarded; corrupted/replaced preceding evidence is refused, and
paid returned work survives failure.

`work.prior_gradient_geometry` separately sums reader seconds, logical input and
projection bytes, validation, forward/backward rows and reductions. Reader seconds
include internal validation, copies and geometry reducers; surrounding composition
guards/copies/publication are outside that field. Whole pilot clock includes all. Bytes do not measure graphs, total allocation or process RSS.
No full efficiency, learning or release claim follows from scoped checks.

[Decision](adr/ADR-0391-bound-prior-gradient-readout.md) ·
[Executed evidence and limits](../artifacts/runs/minigrid-prior-gradients-20261017/RESULTS.md)

## Optional post-training A/F endpoint games

Add `--deferred-probe-rollouts` to the complete count-control recipe:

```powershell
python -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --acquisition-only --distinct-probe --old-task-readouts --return-to-a --memory-control --deferred-probe-rollouts --describe
```

Remove `--describe` to execute one fixed local case. This is mutually exclusive
with `--probe-rollouts`. Existing inline games and omitted-option behavior keep
their original contracts. There are no new environment variables or dependencies.

The endpoint observer captures four complete paid count states at native updates
16/32/144/160. All A128/F16/A16 learning finishes before game readouts. Two frozen
readers restore these exact states and run the existing raw-argmax A blue/red
TRAIN and F blue/red probe-test games, without teachers, targets, action repair,
replay, further learning, or final data. Storage is bounded to four captures,
256KiB each/1MiB total/620544 logical table bytes.

This option explicitly moves original training cohort/source/report validation
to phase entry and exit, including exceptions. Every game still checks its frozen
full native state, original completed training-budget witnesses, counters, paid
reports and zero-update/replay20-second budget. The original180-second global
clock covers capture, training, restore and game work. The new phase is not an
inline-equivalent refusal-timing guarantee. A closing refusal retains returned
paid work; missing admitted reports leave unknown totals and lower bounds.

The qualified application boundary now owns each registry/witness proof once,
including the original closing resource observation. Legacy no-argument adapter
callbacks retain their enclosure. Final224 focused cases/statics and independent
review pass. Actual deferred-phase checks fall49.5238%; whole-case25% target
fails8.7165% reduction. One original-budget native case completes its first four
games, then times out during A return: five games complete, one censored, ten
unrun. Full16-game20s completion remains open; no recovery or circadian advantage
claim. [Engineering evidence](../artifacts/runs/deferred-qualified-boundary-20261018/RESULTS.md),
[all native outcomes and costs](../artifacts/runs/minigrid-qualified-deferred-games-20261018/RESULTS.md).

The subsequent paid scalar-report guard repair passes346 focused cases/statics
and retains the generic opaque comparison path. Its fixed saved-receipt median
cost falls31.2177%. Sole native validation remains incomplete:62legal applied
actions,63returned predictions (one vector unpublished before refusal), five
complete/one censored/ten unrun games. Unknown costs remain; no full completion
or matched native-speed claim. [Guard results](../artifacts/runs/paid-report-guard-20261018/RESULTS.md),
[all native evidence](../artifacts/runs/minigrid-paid-guard-games-20261018/RESULTS.md).
Further guard optimization is parked; next learner work is explicit bounded
categorical PC/neutral inference with fixed controls preserved.

Reports add `deferred_probe_rollouts` and separate `work.deferred_capture` costs.
Capture/preparation seconds are nested wall-time observations, not additive to
whole-run or profile seconds; logical capture bytes exclude copies and RSS.
Known finite A/F behavior does not establish heldout transfer, reinforcement
learning or circadian benefit. Preserve failed cases and fixed protocols.
Extend subsequent recurring-room work through the existing
`minigrid_action_memory_pilot` A/B/C_train/A stream and its bound learner/evaluator
ports; preserve raw-memory defaults and final exclusion. A separate frozen
protocol must define any new task/dose, without selecting favorable endpoints.

See [ADR-0388](adr/ADR-0388-deferred-known-task-games.md) and
[execution evidence](../artifacts/runs/minigrid-deferred-probe-games-20261017/RESULTS.md).

The opt-in `torch-minigrid-learnability` scenario uses the existing training-only
stream and disposable fresh-task probe. It asks how a fixed reserved task fits
from an initial checkpoint and after an A→B→C_train→A stream, alongside known-
TRAIN acquisition, retention and return fit. Implementation correctness and
scientific interpretation remain separate.

## Run and check

From the repository root, with existing optional `requirements-minigrid.txt` and
Torch dependencies installed:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-learnability --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-learnability
```

The descriptor allocates no learner/environment. The run is one fixed local
protocol, with no tuning flags or seed search. Scoped qualification commands:

```powershell
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r21-minigrid-learnability-20261016/run_checks.py statics
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r21-minigrid-learnability-20261016/run_checks.py focused
```

## Behavior and costs

Five existing learners (BP, PC, neutral CPC, active CPC and recurrent BC) keep
seed17,148 public inputs,H8,K7 and their native objectives/optimizer settings.
Each receives the original128update stream with no replay or reward updates.
All five streams finish before any diagnostic score/probe. Owned checkpoint
states0/32/64/96/128 retain original optimizer and native clock state.

Six original TRAIN teacher traces are scored at all five checkpoints, using
owned restored forks. CE/accuracy and exact nine-action sequences are raw
teacher-forced fit, not closed-loop success. Signed initial acquisition, new-task
fit, old-task loss and return changes retain their declared directions. Initial
exact fit is required before describing forgetting; live useful skill remains
unproven even when raw fit is exact.

Separate E-blue/E-red source traces are collected explicitly as probe_train and
probe_test. Sixteen alternating uniquely identified training packets reuse the
two training traces; evaluation keeps two separate nine-step packets. At source
checkpoints0/128 each learner receives an independent16update disposable fork,
with checkpoints0/4/8/16 and the existing class/ordered decision diagnostics.
Probe labels cannot update the benchmark/deployed learner or choose its settings.
The two deterministic E trace domains repeat despite disjoint IDs/arrays; this
is not a generalization test or an isolated measure of plasticity.

The fixed whole-case limits are180body/210parent seconds,768MiB process RSS,
32MiB output,CPU1/inter-op1/GPU0. Expected teacher physics90steps/10environments
(at most160), benchmark640updates/5760rows plus probes160updates/1440rows,
230score forwards/2070rows total. RNN unroll1854. Same update/row counts do not
equal computation across native objectives or parameter counts. Each probe owns
193104 input-array bytes; record, Python, native-state and diagnostic payloads
are additional and included in process RSS/time/output.

Original budgets, clocks and permanent charges remain bound. Corrupted/replaced
source, actor, candidate, checkpoint, receipt, data or preceding diagnostic
evidence stops later work; partial charges/results remain recorded. No automatic
retry, promotion or selection. Final D is never constructed in this scenario.
The original complete MiniGrid live/final readout gate remains unfinished.

## Structure and extension

```text
src/adapters/minigrid_learnability.py    fixed diagnostic composition and report
src/adapters/minigrid_learning_pilot.py existing guarded stream/owned checkpoints
src/app/fresh_task_probe.py             existing isolated updates/metrics/budget
scripts/run_coupled_attribution.py      existing CLI scenario registration
tests/test_minigrid_learnability.py     fake learner/stream/probe boundaries
```

Extend through existing ports and a prospectively reviewed protocol, preserving
old recipes and negative results. Do not replace raw fit with live success or
infer circadian advantage from probe gain. Current commands, failures and observed
results belong in [the run evidence](../artifacts/runs/r21-minigrid-learnability-20261016/RESULTS.md).

## Observed local case

The frozen case completes in78.04parent seconds with800updates/7200training rows,
230score forwards/2070rows and90teacher physics steps. Every E probe reduces CE,
but no learner meets the initial both-cue exact TRAIN-fit premise. Post-return
E probes reach one of two exact sequences for BP/RNN and zero for PC variants.
Smaller post-stream gain is confounded by better starting fit; it does not show
lost plasticity or circadian benefit. All results, including engineering failures,
remain in the linked evidence.

Saved TRAIN observations expose a concrete next dependency: at step2 the paired
E traces have identical current148vectors with opposite turn labels. A
deterministic memoryless classifier can fit at most17/18recorded rows and one
of two complete teacher sequences. This bound does not rule out alternative
closed-loop trajectories or prove a goal-success ceiling.

The separately frozen [shared public initial/current condition](minigrid-public-history.md)
is now implemented and executed, preserving this original148 recipe/RNN, reset/
role/no-future checks and65536cap. Its larger296 input/2439parameter condition
removes the observed conflicting inputs but still fails both-cue acquisition.
Native activation health is the next diagnostic dependency. No semantic cue,
context, target, private grid or final evaluation metadata enters the encoding.


## Encoding-bound public-history route

Use --invariant-history with torch-minigrid-history-learnability to select the
four existing canonical feedforward heads plus the original raw148 recurrent BC.
Native objectives, seed17/H8/K7,128-update A/B/C/A stream and source0/128 E probes
are unchanged. Raw296 packets enter bound wrappers; encoding/policy/clock and
complete TRAIN/probe roles validate before work. Optional TRAIN-only native health is now supported as described in
[native activation health](native-activation-health.md#encoding-bound-train-health).
Gradient readouts remain unsupported and refuse before resources.

```powershell
# Preview only: no models, source reads or environments.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --describe
# Full native execution is a separate, prospectively budgeted protocol.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history
```

The existing fit helper restores genuine owned snapshots and scores six TRAIN
traces; the existing probe helper trains only disposable forks with separate
16-update/144-row budgets and existing decision diagnostics. Benchmark/source/
actor state and prior permanent charges remain protected. Completed bound probes
validate locally while the parent checks every prior arm/probe in a flat loop.
Equal-valued replacement of paid reports is refused. Returned projection array
costs are separate from native/copied payloads and total process costs.

Focused commands (the native four cases perform64 updates/576 synthetic rows):

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_minigrid_invariant_learnability.py tests/test_minigrid_invariant_learnability_cli.py -q
.venv/Scripts/python.exe -m ruff check src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_invariant_learnability.py tests/test_minigrid_invariant_learnability_cli.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_invariant_learnability.py tests/test_minigrid_invariant_learnability_cli.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_invariant_learnability.py tests/test_minigrid_invariant_learnability_cli.py
```

Repeat type checking with --platform linux; this is not Linux runtime acceptance.
Qualification36new cases ran in two batches,44old paired boundary cases pass and
30closing cases overlap; no single80-case combined run claimed. Two broad gate
and one isolated diagnostic timeouts are preserved. The fake full-route fixture
stubs already accepted TRAIN machinery and reports simulated benchmark costs;
true native fixtures verify zero-source fit/four isolated probes only. No full
five-method800native run, real-game acquisition/retention/recovery, generalization
or CPC advantage is established. Previous matched game comparison remains
INCOMPLETE/SPENT and cannot be resumed from scalar JSON or granted a retry.
[Exact commands, costs, failed attempts and next qualification](../artifacts/runs/minigrid-invariant-learnability-20261017/RESULTS.md).

Extend through existing stream/probe ports with genuine in-memory states, separate
budgets and frozen exposure/metrics. The subsequent canonical-PC qualification completed with failed acquisition;
the subsequent bound TRAIN-health measurement found no dead/saturated priors.
A fixed native BP control is the next qualification before algorithm changes.

## Single canonical-PC qualification

Add --pc-qualification to the bound route to select only the original canonical PC
head. Exact admission requires invariant history and refuses unsupported gradient
flags before resources. TRAIN-only --native-activation-health is supported. Defaults and full five-method recipes are unchanged.

```powershell
# Resource-free preview.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --pc-qualification --describe
# Generic execution entry point; requires a separately frozen prospective budget.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --pc-qualification
```

The qualification reuses128 A/B/C_train/A TRAIN updates, five genuine owned
checkpoints and six TRAIN traces per point. Two independent E16 forks from0/128
bring total native work to160updates/1440rows and46scores/414rows. Final D/live
learner games/replay/promotion are absent. Known TRAIN and inspected deterministic
E are exploratory; probes never update source, benchmark or actor. Singleton
comparisons are null; original acquisition gates and signed metrics remain.

The sole2026-10-17 frozen case is COMPLETE/SPENT:15.3574806s parent,
297144320B external actual-trainer peak,ten teacher episodes/90steps. All exact
sequence acquisition conditions failed, with trained55.6% accuracy/all forward
TRAIN predictions. E-probe CE gains .242505036 at0 and .025139223 at128 do not
establish lost plasticity because starting fit differs. No retention/recovery/
generalization/circadian advantage. The original allowance cannot be rerun.

98 combined selected checks pass with native learning forbidden; four-path lint,
format and mypy win32/linux pass. Exact argv, skipped native fixtures, scoped
saved-data validation, profiling caveats and negative metrics are in
[RESULTS](../artifacts/runs/minigrid-pc-qualification-20261017/RESULTS.md).
Encoding-bound TRAIN activation diagnostics are now implemented and qualified; see
[native health](native-activation-health.md#encoding-bound-train-health). The separate
[trained-PC health measurement](../artifacts/runs/minigrid-pc-health-measurement-20261017/RESULTS.md)
completed with no dead/saturated priors and failed acquisition. The original PC
qualification report still contains no health measurements or weights. The explicit selector below and frozen BP control are now complete, with stronger
TRAIN fit but failed acquisition; see its results.


## Explicit existing-method qualifications

Use --qualification-method bp, pc, neutral_cpc or gated_cpc with the existing
history scenario and --invariant-history to run one original native head. The
same selector is available as qualification_method on describe_protocol/run_pilot.
Values are exact native method names. --pc-qualification remains the compatible
PC alias; specifying both forms refuses even when both name PC. Omitting either
selector keeps the original five-method/default routes and keyword call shapes.

```powershell
# Resource-free BP preview.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --describe
# Native execution needs its own frozen, reviewed local allowance.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp
```

Optional --native-activation-health composes the same TRAIN-only reader; bound
--cue-gradient-geometry remains unsupported. Selected policy, method/descriptor ID,
CPC configuration, input geometry and owned snapshot clocks must agree. Native
BP/PC/neutral/active objectives and settings are unchanged. Singleton comparisons
remain null; cross-method arithmetic belongs to a separately frozen comparison.

Every singleton uses the existing128 A/B/C_train/A stream, five TRAIN readouts and
two independent E16 forks, with original source/actor/paid-proof protections.
Thus160updates/1440rows and46scores/414rows if completed natively; fake orchestration
tests are not that native execution. No final/learner games/replay or useful-skill
claim follows from tests. The original PC alias and scalar/health negative results
remain preserved. [Commands, gates and fixed BP qualification](../artifacts/runs/minigrid-bp-qualification-20261017/RESULTS.md).


### Fixed BP result and next acquisition dependency

The separately frozen BP case completed/SPENT:160updates1440rows/46scores414rows,
18.4064031s parent/377937920B externalactualRSS. BP77.8% TRAIN accuracy from64,
compared with PC55.6%, and lower CE at every trained point, but exact sequences0
and originalA32/B64/C96 acquisition conditions allfalse. All15TRAIN cells and both
probe gains remain reported; exposure matching is not matchedcompute/generalization
or a circadian benefit. Native objectives/configs/settings were not tuned.

Posthoc savedTRAIN analysis localizes remaining errors to turns and finds no
conflicting labels among54 canonical initial/current inputs. This does not prove
neural expressivity or learning speed. Next add an explicit A-only acquisition
recipe via the existing delayed stream, same128 totalupdates, unchangedBP, original
fivepoints. Preserve original A/B/C/A recipe/E16 budgets. B/C readouts in that new
recipe are untrained-context fits, not learned-task retention/return; A128 differs
from original64A/32B/32C exposure. Freeze/review that diagnostic before native work.


## Canonical BP A-only acquisition diagnostic

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --acquisition-only --describe
```

`--acquisition-only` (API `acquisition_only=True`) accepts one explicit bound
BP/PC/neutral CPC/active CPC selector and refuses health/gradient flags. The default remains the original A/B/C_train/A
recipe. The new route reuses its delayed TRAIN stream and complete-state guards:
128 alternating A episodes, five checkpoints, all six TRAIN readouts, then the
same two disposable E16 probes from original checkpoints0/128. No probe updates
enter the benchmark, serving learner or saved checkpoints. No final data is used.

The report retains all15 context fits. A is trained; B/C_train are untrained-context
fits, not independent generalization. `A_fit_gains` reports CE0-CE128 and
accuracy/exact128-0; `A_acquisition` requires both A cue sequences exact at32/128.
`probe_source_stages` and `post_A128_probe_gain_change` name the actual endpoint.
Old B/C acquisition, forgetting and return summaries are absent even after failure.

Why: both fixed continual BP and PC failed initial acquisition. Single-task fit
is a concrete prerequisite for interpreting retention/recovery. Prior raw-history
A128/live results already exist and remain unchanged; this is the corresponding
canonical-history singleton with isolated probes. A128 differs from the earlier
64A/32B/32C exposure and cannot isolate interference or establish matched compute.
No native settings, seeds, actions, metrics, E budgets or original gates change.

The frozen case and actual checks/results are recorded in
[the A-only stage](../artifacts/runs/minigrid-bp-a-only-20261017/RESULTS.md).
Native execution requires its own prospectively frozen local protocol; completed
and incomplete allowances remain spent. No automatic dose extension or sweep.


### Executed A-only result and probe limitation

The sole frozen case completes/SPENT: A128 fits17/18 actions, only the red cue
sequence exact; A_at32/A_at128 both remain false. E curves retain their observed
gains, but all four E probe_train/probe_test ordered input/action sequences equal
A under canonical projection (raw history keys do not overlap). For the bound
learner this is continued A fitting on isolated clones, not learning a novel task.
Source/actor isolation and probe accounting still hold. No retention/recovery or
plasticity claim follows. The distinct F task and explicit support report below
now deliver that prerequisite; original E/default/results stay preserved. The
optional old A readouts below also measure fit on the F-updated disposable learner
with separate monitoring costs. The fixed A-return phase below now measures
subsequent A/F fit with separate update packets and budget. Joint acquisition
remains unmet; next match this A-only curriculum across existing control heads,
with no acquired-task recovery claim, dose extension or inherited native GO.


## Distinct public probe task

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --acquisition-only --distinct-probe --describe
```

`--distinct-probe` / `distinct_probe=True` requires both history flags and keeps
the existing native method, stream, original checkpoints and separate16-update
probe budgets. Omitting it preserves the original E recipe and keyword shapes.
The option composes with BP/PC/neutral CPC/active CPC selection and the existing
bound default. New F-blue/F-red layouts move the public keys to row2, keeping the
original cue rule, grey walls, unrotated view, native actions and nine-step path.
F contexts require matching cues and probe_train/probe_test roles; final D remains
unavailable to this route. No target, role or context is added to learner features.

Before any probe update, `probe_data.support` compares all18 probe episodes with
the actual128 TRAIN packets. It reports exact raw/canonical initial-current input
support, label agreement/conflicts, and counts every episode. Each F episode must
have9/9 canonical inputs unseen in TRAIN; otherwise the route refuses and retains
benchmark charges and the completed diagnostic. This checks finite observed input
support, not generalization or successful novel-task learning. All six TRAIN
readouts remain available; untrained-context fits keep their original labels.

History/key computation, logical numeric key/label payload, elapsed time and input
arrays are reported separately in `work.probe_support`. Python objects, copies,
transient next-frame records and failed allocations add cost; sampled process RSS
includes them. If preceding evidence changes during the diagnostic callback, the
original evidence is restored and `unpublished_probe_support` retains completed
diagnostic data/costs without admitting score/probe work. Existing immutable paid
records, source/actor/state isolation and permanent budgets are reused.

[Prospective protocol and executed results](../artifacts/runs/minigrid-distinct-probe-20261017/RESULTS.md).

## Old A fit during disposable F learning

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --acquisition-only --distinct-probe --old-task-readouts --describe
```

`--old-task-readouts` / `old_task_readouts=True` requires both history modes and
the distinct F task. It composes with the existing native method selector and
default arms. At each existing F checkpoint0/4/8/16, the same disposable learner
also predicts the original A blue/red TRAIN episodes. Only F probe_train batches
update that learner. The original source/actor remain isolated; final evaluation
remains unavailable. Omitting the flag preserves previous call/report shapes.

Why: source preservation measures isolation; measuring old fit requires scoring
the learner that actually received F updates. A is17/18 at the accepted A128
source, so these are old TRAIN fit curves, not acquired-skill retention or live
task success. `old_task_readout_outcomes` records CE0−CE16 and accuracy/exact16−0;
negative values mean worse old fit. Per-cue metrics and decisions are retained.

`report.training_readouts` separately records attempted/returned forwards/rows,
detached input-array bytes, partial failures and elapsed successful intervals.
The elapsed sum excludes incomplete intervals and preflight/detachment; whole
invocation time and process RSS cover additional costs. Two A batches add8
forwards/72rows per probe,42768input bytes for bound history or21456for recurrent
raw observations. The combined original F and A array envelope still fits the
unchanged65536-entry limit. The eighteen F novelty packets/gate remain separate.
Returned monitor costs are charged before post-call guards; a failed monitor
retains the preceding completed F checkpoint and stops later work. Original F
counts keep their meaning; `work.total_score_*_completed` includes A monitoring.

[Frozen protocol, executed checks and measurements](../artifacts/runs/minigrid-probe-old-fit-20261017/RESULTS.md).

In the single fixed BP case, A128-origin old A accuracy goes17/18,10/18,10/18,
14/18 as F receives0/4/8/16updates; old A CE rises .385258473 to .751821039 and its
one exact cue sequence is lost. F accuracy goes12/18 to14/18, all exact0. Original
TRAIN/F curves remain exactly unchanged by monitoring. The opt-in A-return phase
below has now executed on that isolated learner, using separate update packets
and budget with existing A/F readouts; all earlier negative results remain.


## Fixed A return after F learning

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --acquisition-only --distinct-probe --old-task-readouts --return-to-a --describe
```

`--return-to-a` / `return_to_a=True` requires the old-task readouts route. It adds
an explicit sixteen-update A return on each same disposable F16 learner and its
existing optimizer. Original F work, checkpoint metrics and elapsed time keep
their meaning; `report.return_phase` has separate progress/status/time/diagnostics
and A/F curves at return0/4/8/16. A failure there makes the overall adapter case
incomplete while preserving the preceding paid F report and known return charges.
Omitting the option preserves old report and keyword shapes.

Two fresh TRAIN update templates copy the original A blue/red data, with distinct
IDs and storage. The fixed alternating schedule reuses those two owned templates;
keeping sixteen more full history packets would exceed the original65536-entry
limit. Combined F+monitor+return input arrays are470448B for bound history or
236016B for recurrent raw features per probe. Monitoring packets remain readonly.
All inputs detach before the initial fork; return uses no second fork or restore.
Return0 explicitly repredicts F and A and must match preceding F16/A16 exactly
before updates. Original and separate fresh return budgets/controls stay bound;
new budget16 does not reset the original F charges or extend the outer deadline.

Each return phase adds16updates144rows,8F+8A forwards144rows and42768new history
input bytes (21456raw). Existing detached F evaluation/A monitor arrays are reused;
return nested monitors report0new input bytes. Whole timing/RSS include further
copies, protected report checks, Python/native state and failed or partial work.
`return_fit_outcomes.F/A` uses CE0-16 and accuracy/exact16-0 separately. These are
fit change/maintenance measurements, not acquired-task recovery, live success,
generalization or a CPC advantage. Both origins and every checkpoint are retained.
[Protocol, executed checks and complete measurements](../artifacts/runs/minigrid-a-return-20261017/RESULTS.md).


The single frozen BP return case passes344focused checks/scopedstatics/independent
review and native192updates1728rows/94scores846rows/ten teacher90steps. Post-A128
return A CE goes .751821039/.645529026/.546007579/.499832446 at0/4/8/16; accuracy
14/18,15/18,14/18,14/18, exact0. F CE .614372850/.611402768/.618564745/1.113922382;
accuracy14/18,14/18,14/18,12/18, exact0. Lower A loss does not restore pre-F17/18
accuracy; F endpoint fit worsens. The matched A-only recipe below extends the
same A128 curriculum to PC/neutral/active CPC under unchanged per-method settings,
with separate per-method cost fields.
No seed/dose/metric tuning, new algorithm or inherited native GO.


## Matched A-only native method recipes

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method neutral_cpc --acquisition-only --distinct-probe --old-task-readouts --return-to-a --describe
```

Select `bp`, `pc`, `neutral_cpc` or `gated_cpc`; the latter is the existing active
CPC configuration. This composes the same A128 alternating-cue stream and fixed
F16/old-A-readout/A16-return use-cases, with original native objectives/settings,
seed17, checkpoints and acquisition thresholds. BP descriptors remain exactly
those of the accepted cases. Omit acquisition-only for original A/B/C_train/A;
PC alias/default multi-arm, health and gradient combinations retain exclusions.

Why: old PC-family native cases used A/B/C/A exposure, so they could not serve
as matched-curriculum controls for the BP A-only result. Availability of the
same curriculum removes that dependency. It does not equalize native objectives,
inference computation or initial trained fit. Final data is unavailable; report
all TRAIN/probe/return and per-cue points and negative results. Accepted BP saved
results may be reused when source/settings remain unchanged. A new native
comparison requires a frozen local protocol and independent source review.
[Exact checks, frozen protocol and recorded startup failure](../artifacts/runs/minigrid-matched-a-only-20261017/RESULTS.md).

The matched recipe passes116focused checks and scoped lint/format/type checks;
independent source review accepted the preserved BP/default behavior. Its single
comparative launch failed at the PID-file sharing handshake before model imports
or work admission. That original allowance remains SPENT and produced no
PC/neutral/active learning results. Its future-only bounded PID-read repair passed
a dummy-process lock test (18retries/.5780677s), not a native experiment. A fresh
allowance was subsequently frozen/reviewed and completed separately below;
the original failure remains unchanged.


### Executed matched A-only adaptation assay

The fresh repaired-handshake protocol completed: PC/neutral/active192updates each,
576total/5184rows,282scores2538rows/270teachersteps,83.9727565parent seconds and
346718208B sampled actualRSS. Accepted BP case reused; old failed grant remains
SPENT. Once saved arithmetic and independent review accept exact PC-neutral
TRAIN/receipt/probe/return numerical parity. No namespace normalization needed.

All A acquisition gates fail: BP17/18 atA128 vsPC/neutral/active10/18. Post-A128
F16 CE gains BP.905895239,PC=neutral.035322485,active.014162073; A-return CE gains
.251988593/.025593162/.025593162/.012085207. PC-family accuracy10/18 remains
weak; this is not superior retention to BP's stronger initial fit. BP returns to
14/18, below pre-F17/18; its F accuracy worsens14→12/18. Both origins/all fixed
points, metrics, cues, contrasts and costs retained. No circadian advantage or
useful acquired-task recovery established. Shared runtime/order/allocator timing
is not isolated speed/equal compute. [Full results and next known-TRAIN live
integration](../artifacts/runs/minigrid-matched-a-only-execution-20261017/RESULTS.md).


## Known TRAIN A128 live games

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --acquisition-only --distinct-probe --old-task-readouts --return-to-a --train-a-rollout --describe
```

`--train-a-rollout` / `train_a_rollout=True` requires the explicit bound A-only
singleton recipe; any of the four original native methods may be selected.
After all TRAIN scores and disposable probes/returns, this opt-in composes the
existing frozen neural evaluator with the genuine owned benchmark128 checkpoint.
It executes only known TRAIN A blue/red games, seed17/max16steps each. The first
public frame resets per episode; canonical initial/current encoding contains no
target/teacher/private metadata. Actions are the raw first argmax over seven
native classes; no mask, repair, retry or training from the resulting games.
Final data stays unavailable. Existing A/E evaluator and all omitted/false recipe
outputs remain unchanged. Post-F144/return160 clocks are not substituted for128;
scalar saved reports do not contain the native weights needed for this readout.

The new zero-training/replay budget has20s; original180s whole-case/RSS/budgets
remain binding. Frozen source/actor/checkpoint/optimizer/RNG and preceding paid
stream/probe evidence must remain unchanged before more work/publication. A
partial failed evaluator report retains returned predictions, physics and timing;
`timing_complete=False` means elapsed time covers only the last completed valid
interval. No cost is silently erased or any failed budget renewed.

`methods[0].train_a_rollout.report` contains complete episode reports, success,
return, legal/illegal native actions, every prediction and observed transitions.
`work.train_a_rollout` owns separate zero update, live forwards, environment
reset/step, projection and measured interval costs. Existing teacher-forced
`total_score_forwards_completed` excludes live games; opt-in
`total_prediction_forwards_completed` and `total_prediction_rows_completed`
include them, and `total_environment_steps_returned` adds teacher and live steps.
`learner_live_episodes` counts completed environment resets, including later failed
episodes; it is not a success count. Copies/native/Python/report allocations add
cost measured in whole elapsed time/RSS. Native configuration/settings unchanged.

Why: teacher-forced fit does not establish a completed game or a success ceiling.
This is known TRAIN control behavior, not acquired-task retention, novel learning,
held-out generalization or circadian advantage. Preserve negative results.
[Checks, local protocol and observed outcomes](../artifacts/runs/minigrid-train-a-rollout-20261017/RESULTS.md).


The single frozen BP live composition completed with158focused/scopedstatics and
independent review. OriginalTRAIN/F/oldA/return numerical evidence is exact priorBP.
Blue terminates with wrong key (−1,7steps); red succeeds (+.49375,9steps):1/2known
TRAIN games, mean−.253125,16validnativeinputs/zeroinvalid. Alljointacquisition gates
remainfalse. Extra livework0updates/2resets/16forwards16rows16steps/4.4150013s;
wholecase192updates1728rows/110predictions862rows/106physicssteps,
34.4765334parentseconds/382570496B sampledactualRSS. The native allowance remains
SPENT, no retry/futureGO; no heldout/generalization/retention/CPCbenefit claim.


## Bounded invariant exact-memory adaptation control

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --acquisition-only --distinct-probe --old-task-readouts --return-to-a --memory-control --describe
```

`memory_control=True` / `--memory-control` is a separate opt-in singleton using
the existing `InvariantActionMemoryLearner`. All shown flags are required; omit
the neural qualifier, PC alias, activation/gradient diagnostics and live flag.
Original neural selectors, omitted/false descriptors and behavior stay unchanged.
The count route needs no Torch runtime. It uses the same raw296 public initial/
current inputs, canonical projection,128 alternating A updates, six TRAIN scores
at0/32/64/96/128, fixed F16 probes from0/128, readonly A monitors and separate A16
return budgets on disposable forks. Final data stays unavailable; no live games,
replay, promotion, mask, new optimizer/objective, seed/cap/dose/metric tuning.

Known canonical keys predict cumulative action-count frequencies; unseen keys
are uniform over7actions.64keys/155136logical bytes remain the original cap,
2424bytes per stored key+seven counts. Capacity refusal/contradicting labels are
valid outcomes, never triggers to enlarge the cap or choose a favorable task.
Pure count updates are not SGD or neural parameter learning. Source/actor/
checkpoint/receipts/budgets/clocks and preceding paid proof guards remain active;
probe and return updates never enter the deployed or benchmark learner.

The report labels count-update calls/rows and typed observed table occupancy,
logical bytes and conflicting keys separately from projection, lookup, validation,
input/report/copy/time/RSS costs. Logical table size excludes Python/native
allocations. Match input/exposure; native state/initial predictions/capacity and
computation differ from BP/PC/CPC, so this is a simpler finite-assay control with
no equal-compute/equivalence/generalization or circadian-advantage claim. Table
observations are only at actual exported source/readout snapshots; app probe
end states are not reconstructed from scalar reports.

Why: canonical E aliases A, while the already inspected F cohort is canonical
TRAIN-unseen. Existing memory E-return and invariant A/E games do not answer
whether this bounded control can fit F and retain old A under the same exposure.
Neural/RL/heldout and broader useful-skill requirements remain open.
[Protocol, executed checks and outcomes](../artifacts/runs/minigrid-memory-control-20261017/RESULTS.md).

Measured case: [2026-10-17 exact-memory result](../artifacts/runs/minigrid-memory-control-20261017/RESULTS.md).
110 focused tests and scoped statics pass. The genuine once-only192-update case
fits A32/A128 and distinct F by4 updates exactly, retains old A during F and both
fits on return. Maximum36keys87264logicalB,119005184B processRSS,52.5573043s.
No live F/return games were run; B/C_train stay2/18 and generalization remains
unproven. Existing neural negative results remain unchanged.

## Same-fork target-free A/F endpoint games

`probe_rollouts=True` / `--probe-rollouts` additionally requires the complete
`--memory-control` recipe. It reads Ablue/Ared TRAIN and Fblue/Fred probe_test
from the actual disposable count learner after F16 and A16 return, at both source
origins. Each endpoint has a separate20s zero-update/replay budget within the
original whole deadline. Original A/E and neural readouts remain unchanged.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --acquisition-only --distinct-probe --old-task-readouts --return-to-a --memory-control --probe-rollouts --describe
```

The [first frozen native case](../artifacts/runs/minigrid-probe-live-20261017/RESULTS.md)
is incomplete: the first endpoint expires20s after4steps, before any complete
game. Paid A128/F16 evidence is retained and no return updates follow. Full16game
validation remains open pending measured guard-overhead work. No gameplay skill
or generalization follows from passing scripted tests or earlier teacher fit.


Qualified endpoint reads now retain paid/budget/parent checks **between** original
clock, supplied snapshot/live-clock and closing-clock groups. Even failed reads
run final checks; the original error remains sticky. A350-test combined gate and
scoped statics pass. The [bounded composition profile](../artifacts/runs/minigrid-probe-boundary-20261017/RESULTS.md)
finds no parent/cohort call reduction (1297/23089 unchanged), failing its frozen
20% BOTH criterion. This correctness change does not solve the observed native
20s timeout. No new native case, complete gameplay or scientific benefit follows.


The measured signature field walk now stores flat captured leaves and checks
all fresh seven fields per transition with C iterators. Exact transitions-tuple
ownership and full array values/types/shapes remain bound; initial mutable
containers refuse. Final366 scripted checks pass. One prospectively corrected
128A-only direct microbenchmark measures24.2293% lower median validation time
(.709652→.537708ms,5fixed blocks/100checks each implementation each block).
The first timing used the wrong context sequence and is preserved as ineligible;
the original call-count target still fails. This direct engineering cost result
does not establish full readout timing or native gameplay. Full16games remain open.


Count endpoint snapshots now use full dataclass equality only after the fresh
complete typed copier validates exact records, native scalar types, canonical
finite keys, policy/encoding/caps, counts and exposure. Snapshot/copy ports still
run at every boundary; private expected state and report/neural guards unchanged.
The [fixed fresh-read benchmark](../artifacts/runs/minigrid-count-snapshot-cost-20261017/RESULTS.md)
includes deepcopy+fullvalidation+comparison, with25.895%/28.308% lower median at
18/36keys, all prescribed blocks kept.380 selected tests/scoped statics pass.
Same scripted4one-stepgame profile completes with1297/23089 parent/cohort checks;
unitclocks and fakeports establish no native20s pass or useful gameplay. The
next optional post-training readonly phase must retain original strict mode and
state/evidence/deadline guarantees under an explicit new timing contract.


## Explicit weighted BP benchmark control

`--weighted-bp` is an opt-in on the existing history-learnability scenario and
requires `--invariant-history --qualification-method bp`. It seals the original
128 encoded complete TRAIN episodes through the existing weighted array adapter.
The descriptor names the weighted benchmark objective separately. Default methods,
seed, checkpoint dose, raw unweighted fit scores and acquisition criteria remain
unchanged. No weighted PC/CPC method is introduced.

Fresh-task probes and optional old-task return use ordinary unweighted BP on
separate templates/forks, with complete checkpoint/optimizer continuation and
original benchmark guards. They never consume the sealed benchmark schedule or
supply weight-selection evidence. Known task fit, starting loss and optimizer
history still confound interpretation; smaller probe gain alone is not proof of
lost plasticity. Final evaluation stays unavailable to this route.

```powershell
python -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --weighted-bp --distinct-probe --old-task-readouts --return-to-a --describe
```

[Weighted objective and usage](weighted-feature-bp.md),
[architecture decision](adr/ADR-0394-weighted-learnability-route.md).
A separately frozen and reviewed local comparison protocol is required before
interpreting differences between weighted and unweighted native cases. Fixed
exposure does not imply equal time, memory or computation.
