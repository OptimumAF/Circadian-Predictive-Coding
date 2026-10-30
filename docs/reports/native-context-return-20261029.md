# R4.8-CTX.N — Native context-return stream and negative observation

[Issue275](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/275)
delivers BP/PC/neutral CPC/active CPC through the existing D9 public-context
A→B→A workflow. Full repository goal and broad parents18/21/23 remain open.
GitHub holds live status; historical plans/log remain frozen under AGENTS.
HEAD0aae1d6 and inherited270–274 changes/captures were preserved. No new
algorithm, dependency, runner, install, global setting, commit/push or old
scientific raw read occurred.274's future-work NumPy wording was incorrect:
the actual factory uses existing CPU Torch adapters behind array ports.

## Implemented and executed behavior

`recurring_ticket_workflow_example` accepts native methods with context-return
and replaynone, forwards the exact method through the original command, and
passes the actual vocabulary width9 to the existing factory. Old width7/default
and policy-change behavior remain. Separate heads/private adaptation/paired
serving stay perceptron-only. Original seed17/H3/K6 and method configuration,
teachers, roles, causal labels, dose and runtime/output limits are unchanged.

`ticket_workflow_native.require_readout` checks the original feature spec and
complete copied native state, then the existing readonly predictor guard.
Legitimate inference counters advance; weights, optimizer/chemistry and source
state cannot. The original context source/actor/RNG/paid-evidence guard runs
before native fitting and through readouts, now including Torch global RNG.
Nominal native affine costs use actual D9 width:45terms/row versus old D7's39.
They exclude backward/latent/tanh/copy/validation work; no equal-compute claim.

Independent source review found native results incorrectly inherited the
perceptron's zero-initial-state description. Both metadata and summary now say
seeded initial state for native methods, while old perceptron wording remains.
Scalar regression covers this. Readonly source/result followup confirmed the
finding resolved and no further blocker; no independent experimental replication.

## Focused software validation

Physical cwd D:/Dev/Repositories/Circadian; TEMP/TMP D:/Dev/_Temp, Linux TMPDIR
too, Python-B, CPU1/GPU0. Existing native profile observer was reused, with
prospective per-platform caps:4pytest invocations/180s/1GiB sampledRSS,
128nativeattempts768rows/200componentinits/2000forwards/2000dispatches; detailed
copy/init/latent counters in receipts. Original per-stream30s512MiB remains
strict. Software fixtures are separate from the genuine study below.

Stage shorthand: `artifacts/runs/native-context-return-20261029`.

```text
<python> -B <stage>/check.py tests <stage>/win-final-work.json tests/test_native_context_return.py tests/test_context_return_stream.py tests/test_native_recurring_tickets.py::test_actual_native_sequence_has_exact_original_task_schedule_and_paid_work[bp] tests/test_native_recurring_tickets.py::test_cli_passes_explicit_native_method_without_changing_old_default tests/test_native_recurring_tickets.py::test_unsupported_native_configuration_refused_before_work
<python> -B <stage>/check.py tests <stage>/win-metadata-work.json tests/test_native_context_return.py::test_should_describe_seeded_initial_control_without_constructing_a_model
```

Windows45passed initially,59.9948073s/335,556,608B sampledpeakRSS. After the
description-only repair, one affected scalar case passed2.7849625s/240,521,216B,
zero models/updates/forwards.46distinct cases, total62.7797698s/two invocations;
unchanged native math/state cases reused rather than repeated.

Linux initially FAILED during pytest FD-capture collection on D, truncate
FileNotFoundError:10.89181861s/65,708,032B, all work counters0/Torch not imported.
Original log/receipt retained. Simplified to ordinary pytest `--capture=sys`;
TEMP/TMP/TMPDIR stay on D, without a policy bypass. Same full command as Windows,
with `linux-repair-work.json` and `--capture=sys` before the test paths:
46passed,64.428045296s/361,259,008B, total75.319863906s/two invocations.0skips.
Both use Torch2.14.0+cpu and the existing environment; Linux interpreter
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.

Each platform's model fixtures paid65nativeattempts354rows/64returns (intended
lost fifth call included),24perceptronattempts144rows,152componentinits,
6690nominal initial weight slots,705nativeforwards,13,092snapshots/30,175,394
logical tensor elements,72PC latent iterations/28BPcalls,869typed requests,
1186sandbox dispatches. Tests cover four actual native methods, exact contexts/
roles/receipts, full snapshots, cost width, bounded JSON, source and frozen-reader
corruption, final-role refusal before model construction, lost-update charges,
CLI forwarding/output protection, unsupported combinations and old BP/default
and perceptron context controls. No historical scientific read/final/probe work.

Four-file Ruff/format/default and Linux mypy PASS:

```text
.venv/Scripts/python.exe -B -m ruff check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_context_return.py tests/test_context_return_stream.py
.venv/Scripts/python.exe -B -m ruff format --check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_context_return.py tests/test_context_return_stream.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_context_return.py tests/test_context_return_stream.py
```

Last command also passed `--platform=linux`. Whitespace check passed with inherited
LF/CRLF advisories. Initial fixture lambda/private-field typing errors were fixed
before tests. One prospective producer preparation assertion failed before writing
the executor/zero models; replaced by a small ordinary subprocess procedure.
One atomic documentation patch missed an anchor/no changes, then split edits
succeeded. Orientation path/rg option mistakes were corrected with direct reads.
No original budget was renewed or failed evidence rewritten.

## Sole frozen four-family observation

After final software gates/source review, PROTOCOL and seven source/procedure
bindings froze before GO6494105180. One command:

```text
.venv/Scripts/python.exe -B artifacts/runs/native-context-return-20261029/execute.py
```

It launched4sequential WSL cases through the existing finite profile meter and
actual module CLI: `--context-return --method bp|pc|neutral_cpc|gated_cpc
--output <stage>/cases/native-<method>-actual.json`. Each retains original
12update/30s512MiB session, A4/B4/A4 six-row TRAIN packets, delayed labels+1,
checkpoints0/4/8/12, all four original task truths and24declared episodes.
Final/probe/replay/promotion unavailable. Source/actor/global RNG protected.

Prospective cohort allowance:48updates288rows TOTAL,128componentinits,
5760nominal weight initializer slots,800forwards,12k snapshots/60Mlogical
elements,72latent iterations/12BPcalls,576typed/1000dispatches. Parent150s,
metered bodies cumulative120s/per-process sampled1GiB, original source30s512MiB
may expire first; output512KiB/method/stage16MiB/metadata30s. All limits held.
No old baseline raw read or retraining. Common parameter geometry/seed/initial
values match across all methods; complete optimizer/chemistry differs by algorithm.
Same update/row count does not mean matched objective or total computation.

EXIT0, complete all four cases and96episodes. Parent60.4275098s;
metadata0.0149298s after all cases. Each body sampler spans imports/model/CLI
writer; metadata is outside child samplers and reported separately. No hard
OS peak or all-in efficiency ranking is implied.

| Method | Metered body seconds | Sampled peak RSS bytes | Output bytes | Ordinary success at0/4/8/12 |
| --- | --- | --- | --- | --- |
| BP | 12.208387798 | 335,216,640 | 418,093 | 0/4,0/4,0/4,0/4 |
| PC | 9.522618670 | 262,955,008 | 411,541 | 0/4,0/4,0/4,0/4 |
| Neutral CPC | 16.053566836 | 266,354,688 | 479,979 | 0/4,0/4,0/4,0/4 |
| Active CPC | 17.036253121 | 266,080,256 | 479,595 | 0/4,0/4,0/4,0/4 |

48attempted/returned native updates288rows,100componentinits4500nominal weight
initializer slots576forwards,11,512snapshots/30,035,222logical tensor elements,
72latent iterations12BPcalls564typed660dispatches. Copies/native validation/
RNG operations/Python/gradients additional; nominal slots are not exhaustive RNG
draws or FLOPs. All four initial parameter tensors matched exactly; PC/neutral
parameters at all checkpoints and complete episode traces matched exactly.

All ordinary tasks failed at every checkpoint; all stale/revoked scenarios also
had successFalse. This does not establish that revoked commits were reached or
blocked. All task records completed, but no skill was acquired, so zero signed
changes and null recovery establish neither retention nor recovery. Active CPC
showed no benefit. The already-working separate-head perceptron is a different
architecture/control; its accepted evidence is reused without a capacity or
compute match claim. This D9 study is distinct from the older D7 negative study.
Neither is overwritten or recast as success. No tuning, dose increase, seed
selection, uncertainty/statistical significance, held-out generalization,
reliability, efficiency or full release claim follows.

COMPLETE/SPENT. Sole raw byte read per method consumed after all cases; the same
bytes parsed and hashed. Raw files CLOSED, no reread/rerun/rescore/repair/late
diagnosis/restoration. Use emitted execution.json summaries/costs and scalar
case receipts only. Source/procedure pins unchanged; stage1,873,211B before doc
closeout, under16MiB. Outcome receipt6494140383. Failed software attempts and all
older negative/SPENT captures remain preserved.

## Review, skips and exact next action

Lead source/procedure/result self-review complete. One bounded independent
source reviewer used2readonlycalls, identified seeded-control wording, then a
separate2call followup verified repair and negative interpretation from emitted
summaries/receipts.0tests/models/imports/new scoring/rawactual reads/writes.
Both exceptions ended. This is independent source/result review, not independent
experimental replication or proof of useful adaptation.

Skipped broad suite/CI/installed-current wheel/clean clone/new runtime versions,
GPU/simulators/unseen-task tests/probe or final access/new hyperparameter cohorts.
274wheel predates275native option; its NumPy-only installed qualification remains
valid for its earlier scope, not this Torch option. No export/publication.

Next substantial outcome is diagnosis of the actual native learning-to-control
gap, not another family comparison or dose change. Existing
`diagnose_ticket_training._summarize_logits` and
`qualify_ticket_bp_fit` already provide TRAIN logit/allowed-action margin tools;
the recurring runner does not expose these at its own phase checkpoints.

ONE next implementation action: add a default-off original-TRAIN fit diagnostic
to the existing recurring native context-return readout, reusing those helpers.
At fixed checkpoints, report complete TRAIN CE, phase-specific allowed-action
correctness/margins and all additional forwards/copies under the original guard/
budget, without updating source/actor or reading spent captures. Begin scalar/
role/cost/failure/isolation tests before any native diagnostic experiment. Keep
native settings, task metrics and failed cohort frozen. This distinguishes
underfit TRAIN steps from rollout alignment/coverage problems before considering
any development repair or new comparative protocol. Broader scientific/reliable
adaptation/release requirements remain open.
