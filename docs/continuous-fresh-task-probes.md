# Isolated continuous fresh-task probes

## Accepted private probe after three-context recovery

The fixed rehearsed48/2896 BP policy retains local imitation learnability under
one original inherited-velocity SGD1x18 probe. Reserved TRAIN23/test24 gains are
+.002714882499/+.003649778364. Original TRAIN17/20 fit worsens slightly and19
improves. Source and actor stay unchanged; the private49/2914 learner is discarded.
Both fresh gains and final errors are worse than the saved corrected32 probe.
All21 prior references and signed/absolute comparisons remain; no control retraining
or probe/dose selection. Source32 physical24 failures remain negative.

Independent source/actual review and scoped statics pass. This is reserved-reset
imitation on the same PickCube task, not new physical-task success, generalization,
reliable retention or CPC benefit. The existing guarded observer has now tested physical transfer and old-context
retention: test24 fails before/after50controls, while TRAIN17/19/20 retain success
in17/19/16controls. All signed native contrasts are zero. Independent acceptance
covers the exact common probe and all21 original contrasts; no new runtime.
[Native outcomes and costs](../artifacts/runs/rehearsed-probe-native-transfer-20261021/RESULTS.md).
[Full endpoints, controls, costs, commands and review](../artifacts/runs/rehearsed-policy-fresh-probe-20261021/RESULTS.md).

## Up to three old TRAIN contexts

The existing Python probe API accepts an exact tuple of zero to three TRAIN
episodes. Why this: after acquiring TRAIN20 and recovering17/19 with rehearsal,
all three old contexts must remain visible during the same private fresh-task
probe. The caller supplies already admitted episodes; this API does not qualify
external recordings or grant teacher authority.

```python
from src.adapters.continuous_probe import run_continuous_probe

report = run_continuous_probe(
    source, fixed_reserved_plan,
    training_readouts=(original_train17, original_train19, original_train20),
)
```

All three must have role `train`, matching spec and context/identities distinct
from the reserved probe. They remain score-only old-context diagnostics; only
`probe_train` enters the private update. `probe_test` stays score-only and final
evaluation is unavailable. The source and serving actor are unchanged; the report
does not return an adapted learner. Check `complete` and retain paid partial work
on refusal. Existing zero/two-readout defaults and report names are unchanged;
the third adds `old_2` to endpoint scores and old-retention metrics.

For T third-episode rows, a complete single probe adds two prediction calls/2T
rows and T*(observation_dim+action_dim)*8 logical input bytes. These costs are in
the existing attempt/return/role/byte summaries; private model owners, snapshots,
validation, reductions, copies and process RSS are additional. Diagnostics do
not update the source or demonstrate native task success. Four readouts refuse;
three is a finite bound, not an automatically expanding context archive.

Focused verification commands (synthetic fixtures, not a scientific protocol):

```text
python -m pytest -q tests/test_continuous_probe.py tests/test_continuous_probe_three_readouts.py
python -m ruff check src/adapters/continuous_probe.py tests/test_continuous_probe_three_readouts.py
python -m ruff format --check src/adapters/continuous_probe.py tests/test_continuous_probe_three_readouts.py
python -m mypy --platform linux --follow-imports silent src/adapters/continuous_probe.py tests/test_continuous_probe_three_readouts.py
```

Software execution and independent review pass:78 checks on Windows and pinned
Linux, with measured RSS below1GiB on each. The48/2896 original-source probe has
separate evidence above; no scientific result follows from the increased readout
capacity alone.
[Commands, costs, failures and acceptance](../artifacts/runs/continuous-probe-three-readouts-20261021/RESULTS.md).

## Accepted reserved robot-trajectory study

The existing probe/stream composition now has accepted recorded PickCube simulator reset data:
TRAIN A15/B22/A15 plus fixed probe_train23/score-only probe_test24,18rows each,
executedTS12. In one frozen eight-update/phase four-family study all16disposable
one-update probes improved TRAIN/test error. Allfamilies improved fixed A/B fit;
PC-neutral common results matched, BP had the lowest final errors and active CPC
showed no overall advantage. Native task success does not follow from imitation
fit. Newphysicaltasks/generalplasticity/reliability/safety remain open.
[Exact protocol, all metrics, costs and limitations](../artifacts/runs/reserved-robot-learnability-20261020/RESULTS.md).
[Role-preserving robot input usage](reserved-robot-probes.md).

## Accepted paired-control result: local learning and retention tradeoff

The missing PC, neutral CPC and active CPC arms completed once under a separately
reviewed 33-update/69-row allowance. BP's accepted partial result was reused; its
original invocation remains **FAILED/SPENT**, not retrospectively passed. The
matched-result subtask is accepted, with an explicit forward-row estimate
qualification. Historical incomplete statuses below describe the original run.

At all three trained checkpoints, every family's one-step fresh TRAIN and test
gains change from negative to positive when only disposable velocity is zeroed.
Old A retention worsens at every such point; old B retention worsens at the first
two and improves at the return checkpoint. Active CPC's fresh gains are below PC's
at all four checkpoints. This is no uniform reset benefit or circadian advantage.

| Family at return checkpoint | Inherited Ftest gain | Zero-velocity Ftest gain | Change in old A retention loss |
| --- | ---: | ---: | ---: |
| BP, reused | -.035084270448 | .011205187610 | +.054506618282 |
| PC = neutral CPC | -.032771737370 | .010999599470 | +.055327606992 |
| Active CPC | -.032298428753 | .010559717303 | +.055074426052 |

Positive fresh gain is improvement; positive old-retention-loss change is worse.
The original benchmark, inherited probes, saved final counters and complete wake
match exactly after only declared elapsed/paired-container normalization. Current
PC/neutral arrays and common reports match before active admission. No old BP
parameter matrix archive exists or is claimed. No native/generalization or broader
plasticity conclusion follows from these tiny fixed synthetic tasks.

The new entry point `scripts/continue_continuous_probe_control.py` composes the
existing runner. `tests/test_continuous_probe_continuation.py` has 38 passing
saved-metadata cases on each platform, with scoped statics and independent review.
Unit tests run no learners; their scalar arithmetic and copies are recorded.
The single model allowance is now spent: reuse `remaining-arms.log` and
`cohort-summary.json`, rather than rerunning the entry point. The original
66 candidate-forward-row estimate was incorrect (actual 69 rows across 33 calls);
scientific update/row caps were met, full estimate conformance is unqualified.

```powershell
python -m pytest tests/test_continuous_probe_continuation.py
python -m ruff check scripts/continue_continuous_probe_control.py tests/test_continuous_probe_continuation.py
python -m ruff format --check scripts/continue_continuous_probe_control.py tests/test_continuous_probe_continuation.py
python -m mypy --follow-imports=silent scripts/continue_continuous_probe_control.py tests/test_continuous_probe_continuation.py
```

The next learner increment is a bounded TRAIN-only continuous replay route using
existing candidate consolidation and a separate exposure budget. Existing
actor/shadow wake budgets reject replay limits; current rehearsal target checks
are binary/categorical. Keep those contracts intact, add an explicit continuous
route and verify failed-charge accounting before any new comparison. Keep this
no-replay recipe and all tradeoffs as controls.
[All commands, signs, costs, qualifications and next action](../artifacts/runs/continuous-probe-continuation-20261020/RESULTS.md) ·
[Independent acceptance](../artifacts/runs/continuous-probe-continuation-20261020/REVIEW.md).

## Stream CLI inputs

The existing `continuous-mlp-learnability` command accepts explicitly reserved
fresh-task episode files alongside its unchanged qualified TRAIN teacher inputs:

```powershell
python -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --warm-teacher warm.json --new-teacher new.json --learner-family pc --phase-updates 1 --probe-train fresh-train.json --probe-test fresh-test.json --probe-updates 1 --probe-diagnostics --probe-zero-velocity-control
```

Both probe paths are required together. `--probe-updates` is fixed per scope,
1..16 (default1). Diagnostics and velocity control are independent opt-ins.
Omitting all probe flags preserves the previous command/report behavior.
Invalid combinations/counts fail before reading files. Input roles, complete
episode shape and plan binding fail before fitting or training. A later paid
failure still prints the incomplete JSON report and exits1, preserving receipts.

Each file is the exact `dataclasses.asdict(ContinuousEpisode)` JSON form:

```python
from dataclasses import asdict
import json
from pathlib import Path

# These must already be independently reserved episodes with the stated roles.
Path("fresh-train.json").write_text(json.dumps(asdict(fresh_probe_train),
                                            allow_nan=False), encoding="utf-8")
Path("fresh-test.json").write_text(json.dumps(asdict(fresh_probe_test),
                                           allow_nan=False), encoding="utf-8")
```

Exact fields: episode `{episode_id, context, role, spec, transitions}`; spec
`{observation_dim, action_units, lower_bounds, upper_bounds}`; each transition
`{step, observation, action, reward, next_observation, terminated, truncated}`.
Vector/transition containers are JSON arrays. No extra fields, duplicate keys,
nonfinite values, implicit defaults, role conversion or unit conversion. Files
are UTF-8 and at most2MiB each; each contains1..128 complete sequential rows.
Core validation enforces native scalar types, bounds, joined observations and
final termination/truncation. The plan also bounds combined feature/action
entries to65,536 and requires matching specs/fresh context/distinct identities.
TRAIN, inner_guard and final_test records are refused as probe inputs.

Probe vectors must already match the final model input:42D for ordinary teachers,
789D for full visual teachers,21D for `--public-vector-only`. The CLI projects
only its TRAIN teachers; it does not silently transform probe vectors. Full
visual probe RGB coordinates21 onward must lie in[0,1] for every observation and
next observation, including the terminal next observation. Existing mixed-loss
probe_train action admission remains in the learner runner before fitting.

The opt-in `input_probes` report records each input's actual file bytes, role and
episode ID plus the per-file byte limit. Its reservation is caller-declared;
the reader neither qualifies collection nor grants runtime release. Existing
TRAIN teacher qualification remains separate. No final/guard loading, probe
selection, fitted probe transform or benchmark/deployed update is added.

```text
src/infra/continuous_probe_json.py       bounded reserved episode JSON reader
scripts/run_native_stream_demo.py       existing CLI -> plan -> existing runner
tests/test_continuous_probe_cli.py      reader/admission/routes/real tiny CLI
```

Focused commands:

```powershell
python -m scripts.run_native_stream_demo --help
python -m pytest tests/test_continuous_probe_cli.py
python -m ruff check src/infra/continuous_probe_json.py scripts/run_native_stream_demo.py tests/test_continuous_probe_cli.py
python -m ruff format --check src/infra/continuous_probe_json.py scripts/run_native_stream_demo.py tests/test_continuous_probe_cli.py
python -m mypy --follow-imports=silent src/infra/continuous_probe_json.py scripts/run_native_stream_demo.py tests/test_continuous_probe_cli.py
```

Extend input formats in the outer reader and keep inner role/plan checks intact.
New comparative work still needs frozen roles/seeds/metrics/counts/budgets and
review; CLI availability does not renew a spent experimental allowance.
[Current protocol](../artifacts/runs/continuous-probe-cli-20261020/PROTOCOL.md).

Scoped acceptance: the same 124 cases passed on Windows and pinned Linux
(93 new, 31 existing CLI regressions; 223 other cases intentionally deselected).
The actual tiny BP CLI fixture paid 11 updates and 22 training rows per platform;
all other selected cases used parsing or stubbed runners. Scoped statics, actual
help output and independent review pass. Final static checks overlapped the tail
of a zero-model legacy test run; no whole-process serial or performance claim is
made. No new comparative/native result or renewal of the failed study follows.
[Exact commands, failures, costs, limitations and next action](../artifacts/runs/continuous-probe-cli-20261020/RESULTS.md).

## Paired disposable velocity control

The LAST optional plan field is now `zero_velocity_control=False` (exact bool).
With `True`, each existing checkpoint runs its original inherited probe first,
then a separate probe from a detached source with only velocity arrays zeroed:

```python
plan = ContinuousProbePlan(fresh_probe_train, fresh_probe_test, updates=1,
                           diagnostics=True, zero_velocity_control=True)
result = run_continuous_learnability(warm_train, new_train,
                                   learner_family="pc", phase_updates=1,
                                   probe_plan=plan)
pair = result.report["probes"][0]["report"]
# pair["inherited"] and pair["zero_velocity"] are original single-scope reports.
# pair["contrast"] uses zero_velocity minus inherited; no improvement threshold.
```

Why: negative fresh-task gain and an uphill realized direction do not identify
their cause. This intervention tests the local effect of inherited optimizer
velocity while keeping the benchmark learner and comparison recipe fixed.
It neither changes deployed optimization nor establishes lost plasticity.

Both scopes preserve checkpoint parameters, complete policy, frozen A-only
statistics, global native counters and coherent CPC wake. The control gets a
new independent fixed probe_train allowance and clock; original budgets are
never reset or transferred. Only probe_train updates; probe_test remains a
read-only score. No probe learner is returned, promoted or imported back.
Default False preserves the single report and numerical calls. Diagnostics
remain independently optional; False adds no Ftrain score/gradient or contrast.

The pair must validate the preceding scope before admitting the next. Private
expected state never reaches public restore callbacks. Original returned reports
are copied immediately and checked after later callbacks; replacement fails
without losing the observed paid report. Source/control native-owned validation
finishes after public observers, including cross-owner mutation checks. Complete
initial physical checkpoints must match exactly. An incomplete scope or later
guard failure stops the benchmark and preserves paid reports/preparation attempts.

`preparation_work` reports one extra detached source, zero-array attempts/returns,
values/numeric bytes, restore attempts/returns, and zero fit/initializer/RNG work.
Each nested report separately accounts its own owners, predictions, updates,
rows and diagnostics. Selected payload counts exclude copies, validation,
elementwise work, temporaries and process RSS; they are not peak memory estimates.
Acquisition/Ftrain contrast is larger better; old-retention-loss contrast is
smaller better. Every signed/negative outcome is retained without selection.

```text
src/adapters/continuous_probe_control.py       paired adapter leaf, no probe import
src/adapters/continuous_probe.py               dispatch to unchanged single runner
tests/test_continuous_probe_control.py         ownership/receipt/failure controls
scripts/continuous_probe_control_example.py   fixed synthetic paired example
```

Focused checks: `python -m pytest tests/test_continuous_probe_control.py
tests/test_continuous_probe_direction.py tests/test_continuous_probe.py`.
Ruff/format/mypy use these changed modules, runner, example and new test module;
the exact executed command list is recorded in the linked artifacts.
The fixed example is prospectively budgeted, not a general experiment launcher:
reuse its saved output once spent. Add future interventions only with a distinct
disposable source, fixed role/metrics/budget and unchanged default behavior.
[Protocol](../artifacts/runs/continuous-probe-control-20261020/PROTOCOL.md) ·
[Independent review](../artifacts/runs/continuous-probe-control-20261020/REVIEW.md).

Current engineering evidence:120 ordered focused cases passed on each platform,
plus one later pure tuple/list reporting regression on each platform; scoped
statics and independent acceptance pass. The sole frozen example **failed after
BP** on live tuple versus saved JSON list comparison. Saved-only repair verifies
exact original scalar evidence; PC/neutral/active were not run and the experiment
allowance is spent. BP fresh gains improve after zeroing velocity at trained
checkpoints, while old A retention worsens. This establishes no uniform benefit
or CPC result. The full four-family example requirement remains open; no automatic
rerun. [Commands, actual costs, all failures and next action](../artifacts/runs/continuous-probe-control-20261020/RESULTS.md).

The existing continuous A/B/A runner accepts optional `probe_plan`. It runs a
separate disposable probe at each fixed checkpoint: initialization, after A,
after B, and after returning A. Benchmark learning and deployed actor state stay
unchanged. Default `None` preserves the existing runner and CLI behavior.

```python
from src.adapters.continuous_probe import ContinuousProbePlan
from src.adapters.continuous_learnability_demo import run_continuous_learnability

plan = ContinuousProbePlan(fresh_probe_train, fresh_probe_test, updates=1)
result = run_continuous_learnability(warm_train, new_train,
                                   learner_family="pc", phase_updates=1,
                                   probe_plan=plan)
# result.report["probes"] contains four independent scalar probe reports.
# result.candidate is the benchmark learner; adapted probes are never returned.
```

## Role and ownership contract

The plan revalidates exact complete episodes with roles `probe_train` and
`probe_test`, distinct identities, matching context/spec, 1..128 rows each, and
at most65,536 feature/action entries. Its context/identities must differ from
benchmark TRAIN episodes. Admission occurs before fitting or model creation.
Only the original A TRAIN statistics are fitted. Mixed BP objectives also check
probe_train target endpoints before fitting. Evaluation roles never enter
`continuous_training_pairs`; scoring reads an owned probe_test view separately.
Final/guard data cannot enter this recipe, fitting, selection or promotion.

Each of four scopes has an independent, prospectively declared1..16-update
allowance; failed scopes stop the benchmark without retry. Existing
ActorShadowRuntime owns probe_train actor/candidate forks, plus one detached
scoring fork. Native training policy, momentum, parameters and complete CPC wake
start from the checkpoint; global model counters continue, local budget starts0.
The raw native array boundary consumes only explicitly released probe_train
labels. Probe_test and old TRAIN labels are read-only scores. No probe learner
is imported back, promoted or returned; no RNG/parameter initialization or refit.

Full state, version, original budget/timer/limits identities, counters and logical
clock are checked around updates and observations. Native returns are saved before
app checks, receipts survive later failures, and final retained receipts must
match validated returns with native scalar types. Coherent chemistry replacement
is refused. Unknown corrupted counts remain unknown; original authority is never
reset. Partial reports preserve attempts, returned predictions, native/failed work,
observed receipts and charges. An incomplete probe stops further benchmark work
and releases no successful aggregate metrics or candidate.

## Measurements and costs

Each scope scores at0/n: fresh probe_test and both old TRAIN tasks, using common
physical normalized action half-MSE (lower better). Fresh gain is F0-Fn (positive
improvement); old retention loss is oldn-old0 (positive deterioration). Scores are
reporting only. The inherited optimizer state is deliberately preserved.

Benchmark work remains in the existing `report.work`; each probe's `report.work`
separates native events/row weights, BP-derived arithmetic, prediction attempts/
returns/rows/actual array bytes, role-specific score exposure, reserved and old
input payloads, and parameter/momentum/wake bytes. Three extra model owners are
transient per scope. Snapshot/gate validation, copies, JSON metadata, temporary
arrays and process RSS are outside numeric/event estimates. Initialization,
admission and report construction are included in probe elapsed time. Runtime
thread pools/GPU/RSS have not been measured. No unlimited model-history chain.

```text
src/adapters/continuous_probe.py             bounded plan, existing-runtime leaf
src/adapters/continuous_learnability_demo.py optional fixed-checkpoint integration
tests/test_continuous_probe.py              role/state/budget/failure behavior
scripts/continuous_probe_example.py         fixed synthetic four-family example
```

```powershell
python -m scripts.continuous_probe_example
python -m pytest tests/test_continuous_probe.py
python -m ruff check src/adapters/continuous_probe.py src/adapters/continuous_learnability_demo.py tests/test_continuous_probe.py scripts/continuous_probe_example.py
python -m ruff format --check src/adapters/continuous_probe.py src/adapters/continuous_learnability_demo.py tests/test_continuous_probe.py scripts/continuous_probe_example.py
python -m mypy --follow-imports=silent src/adapters/continuous_probe.py src/adapters/continuous_learnability_demo.py tests/test_continuous_probe.py scripts/continuous_probe_example.py
```

The accepted example is already saved; reuse its evidence rather than repeating
its spent allowance. It performs12 benchmark plus16 probe updates/60 training rows.
At checkpoints1/2/3 every family's fresh one-step gain is negative; the cause and
wider learnability are unproven. No winner, tuning or rerun follows from this.
The existing classification probe remains unchanged. Future experiments require
fixed roles/checkpoints/metrics and independent prospective review.

[Protocol](../artifacts/runs/continuous-isolated-probe-20261020/PROTOCOL.md) ·
[Commands, all outcomes and next action](../artifacts/runs/continuous-isolated-probe-20261020/RESULTS.md) ·
[Independent review](../artifacts/runs/continuous-isolated-probe-20261020/REVIEW.md).

## Opt-in TRAIN fit and realized-step direction

Set the plan field `diagnostics=True` to add `fresh_training` scores at0/n,
`metrics.fresh_training_gain` (train0-trainn), and `update_diagnostics` per native
update. Default False preserves earlier report fields and numerical calls.
Old retention still includes only old TRAIN contexts. The held-out score remains
`fresh_acquisition_gain`; no metric or probe selection changes.

Each diagnostic uses the already released probe_train batch, PRE-update
parameters and frozen A-only transform to recompute the common physical half-MSE
BP total derivative `g`. This explicitly remains physical MSE even for a mixed
binary-gripper BP training policy. Compare it with actual native displacement
`delta=after-before`, nominal inherited momentum `-lr*mu*velocity_before`, and
reference `-lr*g`. These are scalar per-group/overall norms, dots and cosines.
The native PC/CPC inferred/gated partials are different and are not measured.
Negative `g dot delta` predicts first-order decrease, not guaranteed finite-step
improvement. A zero-norm cosine is None. Norms/cosines are scaled; dot products
use ordinary float64 products/reductions including underflow, and unrepresentable
outputs fail loudly. No reset, extra update or native inference is performed.

Paid native/app receipts and an incomplete diagnostic row are saved before extra
math. Attempt/return counters survive a diagnostic failure and the benchmark
stops without successful aggregate metrics/candidate. Work separately charges
normalization, task gradients, scalar reductions, gradient rows/payload and
estimated dense MACs. Inputs/temporaries/copies/process RSS are not claimed as
measured memory; no parameter/gradient history is retained in scalar reports.

```text
src/core/continuous_step_diagnostics.py       pure common task-gradient readout
src/adapters/continuous_probe.py             opt-in scores and paid failure rows
tests/test_continuous_probe_direction.py     independent math/state/role oracles
scripts/continuous_probe_direction_example.py fixed synthetic integration
```

```powershell
python -m pytest tests/test_continuous_probe_direction.py tests/test_continuous_probe.py
python -m ruff check src/core/continuous_step_diagnostics.py src/adapters/continuous_probe.py tests/test_continuous_probe_direction.py scripts/continuous_probe_direction_example.py
python -m ruff format --check src/core/continuous_step_diagnostics.py src/adapters/continuous_probe.py tests/test_continuous_probe_direction.py scripts/continuous_probe_direction_example.py
python -m mypy --follow-imports=silent src/core/continuous_step_diagnostics.py src/adapters/continuous_probe.py tests/test_continuous_probe_direction.py scripts/continuous_probe_direction_example.py
```

The separately frozen example uses identical fixtures/seed/updates to the saved
probe example. Run `python -m scripts.continuous_probe_direction_example` only
under its declared one-run allowance; reuse saved results after execution.
Additional math/readouts cannot establish lost plasticity, a momentum cause,
generalization or CPC benefit. Future causal experiments require separate
frozen matched protocols; preserve all signed and negative outcomes.
[Protocol](../artifacts/runs/continuous-probe-direction-20261020/PROTOCOL.md).

Accepted scoped evidence:83sameordered Windows/Linux cases,0skip; scoped statics,
independent review and one saved28update/60row example. Extra16taskgradients32rows
and32Ftrainpredictions64rows were charged. All original saved controls/wakes remain
exact. At trained checkpoints1/2/3 every family has negative Ftrain AND Ftest gain
and an uphill realized direction under the common physical task gradient. This
rules out an explanation confined to held-out scoring for this tiny example;
it establishes neither a causal mechanism nor benefit from optimizer changes.
The next R4.5 integration is exact PC/CPC admission/fullwake frozen checking in
the existing robot rollout; real simulator/constraints/generalization remain open.
[All executed commands, failures, gains/costs and next action](../artifacts/runs/continuous-probe-direction-20261020/RESULTS.md) ·
[independent review](../artifacts/runs/continuous-probe-direction-20261020/REVIEW.md).
