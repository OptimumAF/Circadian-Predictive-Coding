# Explicit second native CE adaptation attempt — negative result

R2.1-TW.CE.P.N / issue323. Previous333 was progress: joint old+fresh TRAIN
reference capacity, not learned adaptation. This turn executes the same
unanswered323 scientific question through the corrected ordinary implementation.
Original attempt1 remains FAILED/SPENT, with its original null preservation
flags and charges. No first-attempt file or budget was repaired, restored or
renamed. A prospective amendment/start6533787627 allocated one separate attempt
after accepted324 receipt/guard repairs,325 delivery and333 capacity evidence.

## Native behavior observed

The ordinary run completed, with root/auxiliary/private completion and original
source/actor/NumPy RNG/Torch RNG preservation all true. Negative task result:

| Fixed typed tasks | Before F | After F16 | After old TRAIN return16 |
| --- | ---: | ---: | ---: |
| Four old source-trained tasks | 4/4 | 0/4 | 0/4 |
| Two fresh source-untaught tasks | 0/2 | 0/2 | 0/2 |

Before F, old tasks solved in three applied steps: v1invoice open/route/commit,
v1password open/inspect/commit, v2invoice open/inspect/commit, v2parcel
open/defer/commit. Fresh v1parcel chose inspect and v2password chose defer;
both committed the wrong workflow. After F and after return, every task chose
open six times: first transition applied, next five refused `wrong_phase`,
episode stopped `max_steps`. All18readouts are computationally complete.
No epsilon/tie/action-mask/metric/seed or outcome selection changed.

There was no fresh-task acquisition. All four initially successful old tasks
were lost at the sampled F endpoint and none recovered after the fixed return
updates. This is observed failure to recover after observed loss, unlike328's
inapplicable recovery claim when no old success had been lost. Unsampled
intermediate task behavior and unseen generalization remain unmeasured.

Frozen F probe-test scores (accuracy for each task initially2/3, then1/3):

| F updates | v1parcel reported CE | v2password reported CE |
| --- | ---: | ---: |
| 0 | 7.780184903400713 | 0.3153515197823921 |
| 4 | 18.420680743952364 | 18.420680743952364 |
| 8 | 18.420680743952364 | 18.420680743952364 |
| 16 | 18.420680743952364 | 18.420680743952364 |

All return checkpoints0/4/8/16 have the same latter scores/accuracy. The old
TRAIN monitor batches start CE0.004600240347901782/0.000991966012625625,
accuracy1; F4/8/16 and all return checkpoints become18.420680743952364,
accuracy1/3 for both. Full curves, per-row/class/input diagnostics, native
losses, raw task logits/actions/transitions and accounting remain in native.json.

These probability-based checkpoint CEs clip target probabilities to1e-12 in
unchanged `src/core/categorical_probability_scores.py`; the plateau does not
bound the native logit-based training loss. Native F minibatch losses begin
7.780185222625732 then2801.909423828125 on the next alternating batch; remaining
F values are in the thousands. Return losses begin2587.833984375/3905.7041015625
and end2578.742919921875/3896.613525390625 on alternating old batches. No
clipping change or rescoring was made. Source input diagnostics report zero
cross-conflicting exact input keys; this does not prove optimization success.

Large fitted coefficients plus the full SGD continuation are a plausible
instability mechanism, not an established causal explanation.333's fixed
reference capacity witness does not guarantee joint hidden/output SGD
convergence. Do not tune the spent case or add circadian mechanisms to conceal
this conventional-control failure.328's LS control retains its observed
4→4→4 old and1→2→2 fresh results, including near-tie/return-loss caveats.
Different objectives/donors/work prevent causal or matched-compute comparisons;
the present result establishes neither general BP failure nor CPC advantage.

## Implementation, reuse and isolation

No production code changed this turn. New stage actual.py calls the existing
ordinary CLI main directly. No experimental stream/handoff ownership wrapper,
new learner, source loader, runner, optimizer or dependency was introduced.
The unchanged original323 observe.py supplies passive counters and zero-work
guards; original actual.py is NOT executed. New stage and permanent markers
make the separate attempt and its costs explicit.

Targeted source-reconciliation.json/diff compares current path to325's saved
source. CE helper plus three math/initialization adapters are byte-identical.
Native probe/parent changes add accepted327 LS branches and332 CE-task finish
repair; teacher adds333's opt-in joint cohort. BP CE selects the native branch
and the original old12 teacher. Actual source/isolation review checked these
branches rather than treating old wheel evidence as current wheel acceptance.
No new wheel was built;325 delivery is reused only within its original scope.

Original D9/H3/K6, seed17, lr.03, momentum.9, inference2/sourceA4B4A4=12SGD72rows.
Its full saved checkpoint12, including SGD/gradients/counters, exactly equals
accepted321's checkpoint in a saved-only comparison. Source24legacy episodes
all failed, as before. Preparing this live source again was necessary for
ordinary ownership and is charged again, not counted as new baseline benefit.

One original oldTRAIN12 fixed CE fit took34closures/408gradient rows and final
native loss0.0027961048763245344. Detached initialization copies output24values
and preserves full source SGD momentum/stale gradients/policy/native counters.
Fresh16updates/48probe_train rows on v1parcel/v2password and return16updates/
96oldTRAIN rows used the same independent private descendant. Four old TRAIN
tasks were monitored without updates during F. Probe-test labels only scored;
final examples0, no probe/final labels imported into source/replay/promotion.
Task observations were never training data. Source, actor, seed and private
probe ownership remain separated through the existing guards and budgets.

## Commands, costs, checks and failures

All project commands physical `D:/Dev/Repositories/Circadian`; PowerShell
TEMP/TMP=`D:/Dev/_Temp`. WSL cwd `/mnt/d/Dev/Repositories/Circadian`, process
TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`. Existing Python3.12.3/NumPy1.26.4/Torch2.14CPU,
threads1. Stage `artifacts/runs/ticket-ce-probe-native-attempt2-20261030/`.

One actual invocation, EXIT0, exclusive markers retained; do not repeat:

```text
.venv/Scripts/python.exe -B artifacts/runs/ticket-ce-probe-native-attempt2-20261030/launch.py
```

Launcher uses WSL `timeout --signal=KILL 60s` around existing Python `-B
artifacts/runs/ticket-ce-probe-native-20261030/observe.py
artifacts/runs/ticket-ce-probe-native-attempt2-20261030/native-work.json`.
Exact argv and exit are in shell.json; raw outputs native.json/native.log.
The one live session12299 was polled twice to terminal exit0, never restarted.

Native process17.793749319s/348,057,600B RSS; host21.09745450000628s. Native60s/
768MiB held, host75s allowance only covers WSL startup/terminal handling.
Original source30s512MiB birth4.799626927,last20.089197829,updates12/replay0;
auxiliary15s512MiB birth8.007463392,last9.29135232,updates0; F16/30s512MiB
birth9.336337887,last20.032096127,updates16; return16/30s512MiB birth9.336339017,
last20.03213236,updates16. Return clock includes F. No session was reborn or
changed, and the original source clock continued through auxiliary/private work.

Physical counts/caps in native-work.json all verified:

- 44nativeSGD+backward/216rows: source12/72,F16/48,return16/96.
- One auxiliary optimizer/step,34forward+backward closures/408rows, finalTRAIN
  read1/12 and hiddenread1/12. Auxiliary oldTRAIN reuse is paid rehearsal.
- 68components,136namedseededdraws/3060values;307nativeheadforwards/591rows,
  231typedcalls,42episodes,303sandboxdispatches, no robot physics.
- Private typed endpoint readouts18before+36afterF+36afterreturn=90; source
  legacy readouts account for the other141typed calls.
- Zero standalone fitted-affine/LS/LP/SVD/autograd.grad/randommutations/GPU/
  final/remote work. Native counters do not include auxiliary optimizer steps;
  total gradient rows in this attempt are216+408=624, not216.

The observer does not separately count deepcopy calls or tensor-clone bytes;
those overheads contribute to measured wall/RSS. Counts are not total FLOPs or
a matched-compute certificate. No additional copy instrumentation was added.

Original attempt1 contributes12SGD/72rows and5.148747762s/336,187,392B with0CE/
private work. Across both attempts:56nativeSGD/288rows +408auxiliarygradientrows
=696totalgradientrows;22.942497081reported process seconds. RSS is a per-process
peak, not a sum. Original failed preservation/review-timing gaps remain unproved.

Metadata preflight command, EXIT0/.038857295s/24,100,864B, no numerical/project
imports; verifies existing runtime metadata and timeout availability:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ticket-ce-probe-native-attempt2-20261030/preflight.py
```

Scoped commands on new actual.py/launch.py/preflight.py, final all EXIT0:

```text
.venv/Scripts/python.exe -B -m ruff check artifacts/runs/ticket-ce-probe-native-attempt2-20261030/actual.py artifacts/runs/ticket-ce-probe-native-attempt2-20261030/launch.py artifacts/runs/ticket-ce-probe-native-attempt2-20261030/preflight.py
.venv/Scripts/python.exe -B -m ruff format --check artifacts/runs/ticket-ce-probe-native-attempt2-20261030/actual.py artifacts/runs/ticket-ce-probe-native-attempt2-20261030/launch.py artifacts/runs/ticket-ce-probe-native-attempt2-20261030/preflight.py
.venv/Scripts/python.exe -B -m mypy --platform linux --explicit-package-bases --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian323-attempt2-mypy artifacts/runs/ticket-ce-probe-native-attempt2-20261030/actual.py artifacts/runs/ticket-ce-probe-native-attempt2-20261030/launch.py artifacts/runs/ticket-ce-probe-native-attempt2-20261030/preflight.py
```

Exact expanded argv/output/times in statics.json/statics-final.json. All static
invocations PASS3.0493530999519862s; with preflight below90s engineering budget.
Linux-target mypy ran on Windows. No new numerical/software fixture cohort:
accepted324/327/math tests reused where native semantics unchanged; actual
ordinary combined-tree integration was the new scoped validation. No production
edits required new behavioral tests. AST/preflight plus raw saved-result checks
are evidence inspection, not additional native model work.

One saved-only self-review failed: it incorrectly required all private
transitions `applied`. First-failure receipt retained. The corrected check
requires exact actual phase-aware semantics: beforeF all3applied; afterF/return
oneapplied+fivewrong_phase, open repeated6, max_steps. This corrects an analysis
assumption, not a protocol acceptance criterion, and does not rerun the native
case. Two initial read errors requested an obsolete absent downloaded prompt
and a nonexistent guessed losses.py; current AGENTS supersedes the old prompt,
and the loss implementation was found in categorical_probability_scores.py.
Independent prelaunch read also requested nonexistent manifest.json; no manifest
requirement or new hashing system was added. Truncated reads were followed by
bounded focused source reads. All these tooling failures remain disclosed.

Broad suite/unchanged mathematical cohorts/four-family benchmarks/clean clone/
new wheel/dependency install/browser/GPU/remoteCI/release checks/sweeps skipped
for this one ordinary conventional diagnostic. No commit/push, deletion, global
settings change, source reconstruction loader or unrelated reset. Protocol and
all prior negative/FAILED-SPENT captures remain. Stage below4MiB; cache stays D.

## Acceptance and next concrete action

Implemented ordinary software324 remains distinct from this executed native
outcome. Lead self-reviewed actual source/procedure, serialized original full
state equality, guards/roles/budgets, all18raw task transitions, curves, physical
costs and negative interpretation. Computational execution and preservation
pass; acquisition/retention/recovery fail under this fixed protocol.
Independent prelaunch ACCEPT4reads0.7929912s, no numerical imports/replication;
final actual-source/result/report ACCEPT2more reads0.2628572/0.1466943s,
6total1.2025427s. Compact final read resolved earlier output truncation and
checked all18raw traces/flags/costs. No numerical replication/model work;
review exception ended. Lead self-review is separately identified.

Critical path changes from capacity and qualification to the early update
failure. Existing native probe exposes loss but no gradient/update norm fields
(`ticket_native_probe.py`, `torch_feature_learner.py`, `fresh_task_probe.py`,
`categorical_training_diagnostic.py` inspected). Next precise implementation:
add opt-in per-TRAIN-update gradient and parameter-change diagnostics to the
existing native probe, preserving its default optimizer and recording hidden
versus output gradient/momentum/update sizes. First use bounded small fixtures,
not another full F/return run. Why: locate the observed early loss explosion
before selecting a repair; large CE weights/stale momentum remain hypotheses.
No baseline tuning, fourth-update capture chain or new orchestration framework.
Full R2/R4/R5/R7/matched comparisons/science/release/repository goal remain open.

Closeout:323 scoped investigation closed/completed NEGATIVE, canonical6533996411;
tracker18 6534008259,parent21 6534008639,parent23 6534008989 preserve fullgoalOPEN.
Final retained stage1684939B<4MiB. Guide/bothplans/log updated; no production
source change. Original attempt1 remainsFAILED/SPENT, not retrospectively passed.
