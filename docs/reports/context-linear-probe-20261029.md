# R2.1-CTX.LS.P — contextual LS through the existing isolated probe

[Issue284](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/284).
Previous goal turn PROGRESS:283delivered known-context candidate retention. This
increment delivers and qualifies the existing probe API for LS; **no new scientific
case ran**. Full goal and parents18/21/23/33/reliable adaptation/science/release
remain open. All prior negative/SPENT studies and closed raw boundaries remain.

## Working behavior and boundaries

Modified `scripts/probe_ticket_policy_learning.py`: explicit
`run_arm(..., separate_context_heads=True, linear_fit=True)` accepts the current
in-memory owned ContextLinearFitLearner state, reuses generic FreshTaskProbe,
old-TRAIN return, original two budgets and endpoint observer. It now uses the
accepted contextual LS frozen reader for initial and both endpoint six-task
readouts. It has no new CLI flag, raw loader, source reconstruction or training
runtime; native/perceptron defaults retain their original call paths.

New `scripts/ticket_linear_probe.py` binds source/data/spec/RNG and paid evidence.
New `tests/test_context_linear_probe.py` verifies the actual generic and ticket
API, roles, isolation, work accounting, refusals and corruption failures. Existing
core math, generic probe, return and endpoint runtimes remain unchanged.
[Usage and owned in-memory inputs](../context-linear-probe.md).

Actual fixture execution:2source TRAIN fits12rows (correctness setup, not carried
historical control evidence), then32private updates144rows:16×3probe_train and
16×6oldTRAINreturn. Source and actor stay unchanged; only the disposable learner
learns. Original0/4/8/16metrics/decision reports/old-TRAIN monitors and distinct
LS diagnostic preserved. All18initial/afterprobe/afterreturn typed episodes
complete without training. Each original16update30s512MiB session is born once;
return lifetime includes probe time. No source/benchmark/deployed update, final,
replay, promotion, neural/Torch model or random model initialization.

The source's complete policy/state identities+values, supplied snapshot,
cohort batch/array identities+values, fixed spec and NumPy/Python RNG are guarded.
Readout/episode original list+row identities/values and prior-history headers are
checked before and after callbacks, including exceptions. Corruption restores
original paid records then refuses more work; foreign source/input/RNG changes
are never rolled back. Lost reads preserve paid frozen-reader counters before
post-call failure. Failed LS work retains the original TrainingUpdateFailed
receipt even with0budget commits; return admission stays closed.

## Verification, failures and review

28distinct cases qualified/platform through scoped reuse+affected checks, **not
one final28case cohort**. Generic full-probe compatibility and wrong-role cases
were first executed unchanged. The18affected API/refusal/failure/default cases
then passed both platforms after episode/header pinning. Final guard-only
pre/try/finally/post repair passed9pure callback cases on the combined tree, both
platforms,0model work. The previous integration observations are reused for the
unchanged math/data/loop; the final9verify the changed guard order. No claim that
the whole18case numerical cohort reran after that final guard-only repair.
The final test edit was a type annotation only, with statics repeated.

Independent source review found an actual unsealed prior-episode gap. Fixed by
pinning each returned episode and its list, plus source-history headers. Follow-up
confirmed that repair but found the external callback could corrupt records after
the precheck. Fixed with postcheck in finally. A final tiny source-only review
accepted that actual repair and its pure tests; no remaining blocker in inspected
scope. Calls3+1+1across finite documented exceptions, each ended, no imports/tests/
models/raw/artifact reads/writes. This is independent source acceptance, not
independent execution, experimental replication or scientific acceptance. Lead
combined source/procedure/results self-review complete.

One Linux initial cohort failed the new lost-fit test's expected error string:
unchanged existing completion validator refused0of16session charges as
`original paid session limits/clocks differ`, while the test expected
`incomplete`. The actual first failed private solve, attempted3rows,0commits and
closed return were preserved. Corrected the message expectation without weakening
the refusal. Initial static annotations/unused type inference were repaired;
one patch missed its formatted anchor and changed nothing. All failures retained.

Original software route cap300was revised prospectively to384before the three
necessary review-finding regression cases: initial API used88routes, and bounded
postfit corruption cases need initial task reads plus their first paid update.
Original300limit and earlier receipts remain visible in issue6498995017; every
other numerical/time/storage cap stayed unchanged. No scientific allowance or
failed experiment was renewed. Windows4pytest allowance is fully consumed; no
additional Windows numerical work in this issue.

## Exact commands and costs

Physical cwd `D:/Dev/Repositories/Circadian`; Windows TEMP/TMP=`D:/Dev/_Temp`;
Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`; Python-B. Stage
`artifacts/runs/context-linear-probe-20261029/`,9506B before documentation.
All argv/status/costs retained in `win-1/2/3/4-work.json`, `linux-1/2/3-work.json`.

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-probe-20261029/check.py artifacts/runs/context-linear-probe-20261029/win-3-work.json tests/test_context_linear_probe.py -k "not use_existing_probe_return_and_keep_source_actor_rng_inputs"
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-probe-20261029/check.py artifacts/runs/context-linear-probe-20261029/linux-2-work.json --capture=sys tests/test_context_linear_probe.py -k "not use_existing_probe_return_and_keep_source_actor_rng_inputs"
```

Those18affected cases passed; one unchanged generic full32update case was
deselected/reused. Last invocation/platform selected
`-k "recheck_paid_evidence or unchanged_callback"`:9passed/0failed/0skipped;
unrelated cases deselected, not skipped. Win1and2selected the full file at their
then-current5/10case contents; Linux1reached the last failed regex fixture.

| Cumulative software, including failures | Windows | Linux |
|---|---:|---:|
| pytest invocations |4|3|
| observed body seconds |19.3731527|35.579890335|
| sampled peak RSS bytes |72,544,256|75,108,352|
| LS fit attempts / physical solves |148 /148|117 /117|
| source+private exposure rows |684|543|
| full-class probability calls / rows |144 /648|116 /522|
| typed affine calls / rows |180 /180|198 /198|
| outer+selected typed delegate calls |368|412|
| unfitted reads |4|8|
| context route calls / rows |332 /868|323 /749|
| TRAIN teacher traces / all sandbox dispatches |24 /252|16 /246|
| served endpoint episodes |64|74|

All original4invocation180s512MiB/160fit720row/1200fullclass3000row/
2400typeddelegate/64teacher192teacherstep/120served1000dispatch caps and amended
384route1600row bound held. Costs include fixtures and deliberate paid failures,
not just successful commits. Logical API quantities are not all-in FLOPs/copies/
guards/SVD/softmax work.0Torch/neural/GPU/physics/final/replay/randommodeldraws/
old scientific raw reads. No downloads/install/delete/global changes/commit/push.

Final three-file commands PASS: `python -B -m ruff check`,
`ruff format --check`, `mypy --explicit-package-bases --follow-imports=silent`
(also `--platform linux`) for both script files and the new test. Whitespace
check passes with inherited CRLF advisories. Broader unchanged math/native/old
raw-backed probe tests, scientific cohorts, clean clone/current wheel install,
GPU/physics/CI/release checks unrun outside scope.279wheel predates these changes.

## Scientific status and precise next action

Software compatibility and private task-read execution are verified. Tests do
not establish initial/new-task gain, useful retention/recovery, generalization,
plasticity cause, CPC/circadian benefit, comparative efficiency or release
readiness. Known legacy templates share stages/labels with source tasks; initial
ability must be measured in any genuine case rather than assumed absent.
No control retraining or old parameter reconstruction was performed.

Next substantive action: freeze one small further-learning/old-task-return
observation through this existing in-memory API, explicitly budgeting any new
source preparation separately from private probe/return work and measuring the
initial six tasks before fitting. Keep the283raw closed, use current source
ownership, preserve all negative outcomes, and do not add another checkpoint/
capture framework. No scientific budget is granted by this software closeout.
Live status/handoff:284/tracker18; historical plans/log remain preserved perAGENTS.
