# R2.1-CTX.FIT — Original TRAIN fit diagnostics

[Issue276](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/276)
delivers default-off native TRAIN fit reads and a bounded BP diagnosis. Full
goal and broad parents18/21/23 remain open. Live status is in GitHub; historical
plans/log remain frozen under AGENTS. HEAD0aae1d6 and inherited270–275 dirty
files, all old negative/SPENT captures and existing functionality were preserved.
No new learning algorithm, dependency, install, global setting, commit or push.

## Working behavior and boundaries

`recurring_ticket_workflow_example.run_example(record_train_fit=True)` and
`--record-train-fit` require original native context-return/replaynone. Invalid
modes and existing output are refused before work. Defaults, separate-head/
private/paired restrictions, method settings, seed17/H3/K6, original A4/B4/A4
12update72row TRAIN and24task readouts remain unchanged. Runtime30s512MiB and
output512KiB are retained.

At0/4/8/12, the current frozen native reader projects the twelve original TRAIN
rows once. It records existing global CE/probability margins and each policy/
concept/phase's allowed winner, correctness and raw margins. It constructs no
new learner or update. Teacher/public features are aligned by the existing
factory; roles, labels and inputs are pinned. These are static known TRAIN
diagnostics, including B truth, not causally arrived online feedback or controller
input. They never affect training, promotion, selection or stopping decisions.
Final/probe/replay data remain unavailable.

Diagnostic attempts, rows, returns and unknown work are separate from typed task
forward counters. Failed reads retain local charges and stop before further
training. Active row identity/checkpoint/TRAIN role/completion/exact work types
are checked before pinning. Changed work dictionaries and equal-valued type
changes such as True for1 are refused. Completed diagnostic evidence is pinned
by value and identity through the original context guard. Source/actor/full
native state/spec/optimizer/RNG protection remains.

An initial helper reuse introduced a script cycle. Before genuine execution,
the existing teacher, CE and margin calculations moved to one small shared
`scripts/ticket_train_fit.py`; qualification and saved-diagnosis entrypoints keep
their aliases/wrapper and teacher factory patch point. AST body equivalence
(factory parameter/renaming only) and pure compatibility tests evidence unchanged
calculations. Native no longer imports the qualifier. This fixes a demonstrated
architecture dependency; it adds no runner or admission framework.

Affected files: recurring/native support, new shared helper,
`qualify_ticket_bp_fit.py`, `diagnose_ticket_training.py`, and new
`tests/test_context_train_fit.py`. New helper module documents inputs, outputs and
non-responsibilities. Native costs include four matrix attempts/48TRAIN rows and
2160extra nominal affine terms; initialization/copies/validation/gradients are
additional. They are not equal-compute or FLOP measurements.

## Focused checks, costs and failures

Physical cwd D:/Dev/Repositories/Circadian, TEMP/TMP D:/Dev/_Temp; Linux TMPDIR
also uses /mnt/d/Dev/_Temp. Python-B, CPU1/GPU0, Torch2.14.0+cpu. Linux starts
with ordinary pytest `--capture=sys` because275observed FD-capture incompatibility
on D. The existing finite observer was reused; no old experiment was rerun.
Software per-platform caps:4pytest invocations/180s/1GiB sampledRSS,
64nativeattempts384rows/128componentinits/6000nominal initializer slots/
1000forwards/1200typed/1600dispatches/20ksnapshots80Mlogical elements. Original
per-stream30s512MiB stays strict. Stage cap16MiB.

With `<stage>`=`artifacts/runs/context-train-fit-20261029`, initial command:

```text
<python> -B <stage>/check.py tests <stage>/<win|linux>-final-work.json [--capture=sys on Linux] tests/test_context_train_fit.py tests/test_native_context_return.py::test_should_forward_native_cli_without_changing_default tests/test_native_context_return.py::test_should_keep_unsupported_combinations_before_data tests/test_context_return_stream.py::test_should_refuse_unsupported_modes_before_work
```

29passed/platform. Windows23.2692732s/332,623,872B sampledpeakRSS;
Linux28.052166338s/351,428,608B. Each paid24nativeupdates144rows for BP plain/
diagnostic parity; checkpoints, applied receipts, episodes and summary were
exactly equal. CPC readonly port performed zero training and preserved its state.
Scalar tests covered CE/margins, wrong role, lost/malformed/nonfinite returns,
mode/CLI/output protection, source corruption and paid-prefix corruption.

Independent source review found that active checkpoint/role could be accepted
before the row was pinned. Repaired with active metadata and returned-work
checks, work identity/value protection and strict types. Focused affected commands:

```text
<python> -B <stage>/check.py tests <stage>/<win|linux>-guard-work.json [--capture=sys] tests/test_context_train_fit.py::test_should_refuse_active_metadata_or_work_changes_before_pinning tests/test_context_train_fit.py::test_should_complete_final_guarded_diagnostic_path_without_repeating_control
<python> -B <stage>/check.py tests <stage>/win-types-work.json tests/test_context_train_fit.py::test_should_reject_counter_type_changes_even_when_values_compare_equal
```

Windows4affected passed14.2523672s/328,790,016B, then2pure type cases passed
2.9907771s/236,347,392B. Linux added that type node to the guard command and
passed6cases17.99704559s/347,181,056B. One final BP diagnostic path per platform
paid12updates72rows; unchanged plain-control and parity evidence was reused.

After moving helpers, pure compatibility/legacy math/role command:

```text
<python> -B <stage>/check.py tests <stage>/<win|linux>-shared-work.json [--capture=sys] tests/test_context_train_fit.py::test_should_keep_existing_math_entrypoints_bound_to_shared_calculations tests/test_context_train_fit.py::test_should_preserve_original_teacher_factory_entrypoint tests/test_ticket_training_diagnosis.py::test_should_reduce_literal_logits_with_stable_ce_and_ties tests/test_context_bp_qualification.py::test_should_refuse_incompatible_context_teacher_before_native_construction
```

Seven passed/platform,0model work/120teacher dispatches. Windows3.1593829s/
239,923,200B; Linux7.922395341s/258,953,216B. Calculation body equivalence and
actual aliases/teacher factories justify reusing native parity after extraction.

Main sequence42distinct cases/platform,0skips and0test failures. A later
metadata-only followup below adds one distinct case per platform.
Windows4invocations43.6718004s, peak332,623,872B; Linux3invocations53.971607269s,
peak351,428,608B. Each:36nativeupdates216rows/36returns,120componentinits,
5400nominal initializer weight slots474nativeforwards8997snapshots943,023logical
tensor elements0latent36BPcalls424typed1335dispatches. All original caps held.
Windows four-invocation window is spent; no additional fixture allowance implied.

Six-file Ruff/format/default and Linux mypy PASS:

```text
.venv/Scripts/python.exe -B -m ruff check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py scripts/ticket_train_fit.py scripts/qualify_ticket_bp_fit.py scripts/diagnose_ticket_training.py tests/test_context_train_fit.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py scripts/ticket_train_fit.py scripts/qualify_ticket_bp_fit.py scripts/diagnose_ticket_training.py tests/test_context_train_fit.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py scripts/ticket_train_fit.py scripts/qualify_ticket_bp_fit.py scripts/diagnose_ticket_training.py tests/test_context_train_fit.py
```

Last command also passed `--platform=linux`. Whitespace passed with inherited
line-ending advisories. Initial four fixture dict annotation errors were fixed
before tests. Source-review finding and circular dependency were genuine code
defects repaired before the diagnostic case. No failed numerical capture was
edited into success; all older failures remain.

## One prospectively frozen BP diagnosis

Software/source gates preceded PROTOCOL, nine source/procedure bindings and
GO6494740136. One command:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-train-fit-20261029/execute.py
```

Existing WSL interpreter runs actual module CLI through the finite observer with
`--context-return --record-train-fit --method bp --output <stage>/case/native-bp-actual.json`.
No old275raw capture, fit state or source matrix was read. This answers a new
TRAIN-fit diagnostic question, not a retry/comparison of its four-family cohort.
No setting, dose, seed, checkpoint or metric was selected from outcomes.

Prospective allowance12newTRAINupdates72rows,32componentinits1440nominal weight
initializer slots200nativeforwards6000snapshots8Mlogical elements0latent12BPcalls
144typed300dispatches. Parent/body90s/per-process sampled1GiB, original source
30s512MiB may expire first, metadata30s/output512KiB/stage16MiB. Same child
sampler through imports/runtime/writer, tiny final receipts afterward; metadata
outside child sampler, separately reported. Not hard OS peak or all-in compute.

EXIT0/complete. Parent15.8937253s/body12.670893105s, peak336,105,472B,
output436,748B/metadata0.0037767s.12updates72rows plus4matrix calls/48diagnostic
TRAIN rows/2160extra nominal affine terms.25componentinits1125nominal initializer
slots157nativeforwards2962snapshots292,788logical elements0latent12BPcalls141typed
189dispatches. Original source clock5.318952547→15.080199145,12committed, limits
and origin never renewed. Source, actor and NumPy/Torch RNG unchanged.

| Checkpoint | Mean TRAIN CE | Allowed correct /12 | Sealed /4 | Ready /4 | Staged /4 |
| --- | --- | --- | --- | --- | --- |
|0|1.7887851|6|2|0|4|
|4|1.7617772|8|4|0|4|
|8|1.7081709|8|4|0|4|
|12|1.6464442|8|4|0|4|

All ready-phase target margins remain negative and worsen. At12: v1invoice
−0.360583, v1password−0.269331, v2invoice−0.267964, v2parcel−0.427504.
The predictor chooses open/commit instead of the task-specific ready action.
All24task records complete but all ordinary workflows fail. Correct staged
teacher rows do not show that live workflows reached or solved that stage.

This establishes ready-action failure on original TRAIN inputs, not solely a
rollout mismatch. Falling aggregate CE is not useful acquisition, retention or
recovery. It does not prove class frequency, capacity or optimization as the
cause. No CPC advantage/generalization/reliability/statistical/efficiency claim.
The [accepted earlier96update joint BP result](../context-bp-qualification.md)
already reports10/12fit and two rare-ready failures; its guide is reused without
raw reconstruction. Different schedules/exposure prevent a causal dose comparison.

COMPLETE/SPENT. Sole raw byte read consumed; same bytes parsed/hashed, all48fit
rows/24task outcomes/costs emitted to execution.json. Raw capture CLOSED: no
reread/rerun/rescore/repair/late projection/partial recovery/cap renewal. Use
only scalar summaries/receipts afterward. All nine bindings matched, stage516,155B
at first postflight, under16MiB. Outcome6494788679. All275/older negatives intact.

## Final failure-accounting followup

Lead review found missing diagnostic counters could aggregate to zero with a
false unknown-work flag. Missing/invalid totals now become null; absent flags
or attempted/returned mismatches retain unknown work. The successful diagnostic
has complete known counters, so its outcomes and costs remain valid.

Before this repair, issue6494957152 prospectively authorized exactly one additional
Windows zero-model metadata call, at most10s/1GiB. The original four-call window
remains spent. Linux used its remaining fourth slot. Source/native budgets were
not renewed. The observer asserts zero model/teacher/forward/dispatch work for
this named metadata allowance.

```text
<python> -B <stage>/check.py tests <stage>/<win|linux>-metadata-work.json [--capture=sys] tests/test_context_train_fit.py::test_should_keep_missing_diagnostic_work_unknown_in_costs
```

One passed/platform, all work counters0/Torch not imported. Windows1.1389658s/
66,498,560B sampledRSS; Linux5.168736779s/63,709,184B. Final43distinct cases per
platform. Combined44.8107662s Windows/five invocations and59.140344048s Linux/
four; numerical work and prior peaks unchanged. All final six-file static gates
passed. Two nullable-type errors were fixed before these tests. Atomic patch
anchor misses changed nothing; bounded replacements succeeded.

The reporting-only repair changes native_costs and its test after the genuine
freeze. Seven bindings remain unchanged; original source-pins.json is preserved.
`metadata-repair.json` verifies all13captured native cost fields exactly equal
under the repaired scalar calculation, using only emitted summary fields. No
model/state/raw read/new projection/scoring. An initial comparison included
unrelated generic timing/RSS fields and failed; corrected to the native fields.
The genuine capture and its known charges were not replayed or repaired.
This final accounting delta is lead-reviewed with a pure regression; the earlier
independent source/result reviews retain their stated scope.

## Review, skips and one next implementation action

Lead source/procedure/result review complete. Bounded independent source review
used2readonlycalls, found active-row admission defect; followup1call confirmed
repair. A separate2call result review accepted the limited interpretation/costs
and emphasized that class weighting remains a hypothesis, not an evidenced repair.
0models/tests/new scoring/raw reads/writes; all exceptions ended. Extraction was
lead-reviewed with static body equivalence and pure compatibility tests. These
are source/result reviews, not independent experimental replication or proof of
useful learning.

Skipped broad suite/CI/new installed wheel/clean clone/other runtimes/GPU/simulator,
extra CPC/cohort/dose/seed studies/final evaluation.275baseline and earlier R2.1/
probe/math evidence reused under unchanged assumptions. No full release claim.

Next conventional learner hypothesis: a fixed class-balanced BP objective,
derived only from each currently arrived TRAIN batch. Existing validation/count
formula lives in `demonstration_class_weights.py`, but weighted feature adapters
require raw296 invariant episodes; generic `TorchFeatureLearner` has no weighted
array update. Never weaken the raw296 boundary to admit D9.

ONE next implementation action: add an optional atomic class-weighted BP array
update to the existing generic Torch learner port, reusing weight validation and
its detached optimizer staging. Require exact all-one equivalence to ordinary BP,
full source/optimizer/RNG/counter/failure preservation and a separately named
objective/cost record. Caller derives weights only from the current arrived TRAIN
batch; no future/final/probe truth or update multiplication. Start focused math/
state/role tests before any separately budgeted development qualification. Keep
all defaults and this failed native baseline frozen; do not assume weighting
fixes ready actions or launch another comparison/CPC mechanism without evidence.
