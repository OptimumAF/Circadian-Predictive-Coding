# Direct LS acquisition, retention and return — #328

R2.1-TW.LS.DIRECT.N, 2026-10-30. [Owning issue](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/328).
Previous turn327 was PROGRESS: direct12/72 handoff software accepted. This turn
executed ONE genuine ordinary local observation, with no production code change.
Fullgoal/#18/#21/reliable adaptation/science/release remain OPEN.323 remains
permanently FAILED/SPENT; this different LS/earlier-donor case is not its retry.

## Observed behavior

| Typed task group | Before fresh training | After F16 | After old TRAIN return16 |
| --- | --- | --- | --- |
| Four source-trained tasks | 4/4 | 4/4 | 4/4 |
| Two source-untaught tasks | 1/2 | 2/2 | 2/2 |

Only v2password was newly acquired. v1parcel was already correct before F.
The four old tasks and both fresh tasks succeeded after return. All18readouts
are complete three-step sandbox episodes. Before F, v2password performed
open→defer→commit and stopped `wrong_workflow`; after F/return it performed
open→inspect→commit and succeeded. Raw traces and all failures remain saved.

The initial failure is numerically fragile: defer raw score
0.36363636363636354 versus inspect0.36363636363636337, an approximately1.7e-16
gap. Exact existing selection rules were preserved; no epsilon/tie/seed change.
This is an observed failure→success under the frozen producer, not robust initial
failure across arithmetic environments. After F inspect=1 versus others0;
after return inspect=.7355371900826446 versus defer=.15112160566706007.

No loss was observed in the saved typed readouts, so **recovery is inapplicable,
not demonstrated**. Unsampled intermediate policy behavior remains unobserved.
These are repeated known templates with explicit public context, not unseen-task
generalization. Disposable probe updates were never promoted into source/actor.
This establishes scoped conventional acquisition/retention behavior, not reliable
broader adaptation, CPC benefit, comparative superiority or release readiness.

## Full fixed curves and adverse result

Cross-entropy shown rounded; full original scores/decisions and diagnostics are
in native.json/raw.json.gz. No checkpoint omitted or selected as a new horizon.
Old v1/v2 curves coincide to displayed precision. Old accuracies are1 at every
point; fresh parcel accuracy1 throughout, password2/3 at F0 then1.

| Phase | Updates | Fresh parcel CE | Fresh password CE | Old v1 CE | Old v2 CE |
| --- | --- | --- | --- | --- | --- |
| F | 0 | 1.226349 | 1.226349 | 1.043592 | 1.043592 |
| F | 4 | 1.043592 | 1.043592 | 1.032291 | 1.032291 |
| F | 8 | 1.043592 | 1.043592 | 1.032291 | 1.032291 |
| F | 16 | 1.043592 | 1.043592 | 1.032291 | 1.032291 |
| Return | 0 | 1.043592 | 1.043592 | 1.032291 | 1.032291 |
| Return | 4 | 1.114393 | 1.114393 | 1.043592 | 1.043592 |
| Return | 8 | 1.114393 | 1.114393 | 1.043592 | 1.043592 |
| Return | 16 | 1.114393 | 1.114393 | 1.043592 | 1.043592 |

Return training worsened fresh confidence/CE from1.04359177819 to1.11439266347,
and old CE from1.03229090277 back to1.04359177819, despite retaining all task/class
successes. Preserve this adverse result. LS trains one-hot squared residual,
not CE; unchanged argmax success does not establish unchanged probabilities.
No inference that early class accuracy proves a typed rollout at checkpoint4.

## Frozen procedure and actual execution

[Predeclared protocol](../ordinary-linear-probe-native.md). Physical checkout
D:/Dev/Repositories/Circadian, HEADd9119e80926a139a2f38460564d7eab1b6d2a2b0 plus
inherited dirty/untracked work preserved. Project outputs/TEMP/TMP/TMPDIR on D.
No commit/push/install/download/deletion/global setting changes. The old C:
execution prompt is superseded by AGENTS/AGENT_PROMPT; no old archive rebuilt.

Exact sole command, output redirected to stage/native.log by PowerShell:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ordinary-linear-probe-native-20261030/producer.py
```

The producer calls actual ordinary entry.main with frozen --method linear_fit
--context-return --separate-context-heads --reserved-probe --probe-update
minimum_change --train-update minimum_change --output <stage/native.json>.
Actual Python3.12.3/NumPy1.26.4. NumPy comes from the existing linked SDK, not a
new install. No scripted fits/episodes, new controls, parameter interception or
restored expired authority. Exclusive native.attempted consumed before numerical/
project imports; ONE invocation, EXIT0, no automatic retry.

Producer5.152574424s/65,773,568B sampled RSS; full shell8.1817451s, within60s512MiB.
Ordinary app3.936321165s includes private3.168840954s: do not add overlapping
times. Source original30s limit/start3.600962426/last7.529426977 held12updates;
F born4.372475229/last7.494483505 and return born4.372476961/last7.494513927 each
held16updates and30s512MiB. Return clock includes F, no renewal. Source/actor/
NumPy/Python RNG unchanged flags all TRUE; consent/receipts/sharing remain with
original application guards. Final/remote/physics/Torch/neural/SGD/gradient/
latent/modelrandom/optimizer work ZERO, replay API counters0. Explicit return
reuses old TRAIN templates for96paid row exposures; it is not free or96independent
new observations. All phases repeat fixed templates. Original native and retained23/144 controls
were not retrained; historical310/314/315/326 stay scoped descriptive references.

## Work and memory evidence

All named attempts equal returns in work.json:

- Source12fits72rows; private F16/48 +return16/96:44actual NumPy LS solves216rows.
 2unfitted cold core fits12rows,42minimum core updates204rows;88minimum owner
 ports432rows count context+generic layers, not88learner updates;0cold owner ports.
- 128conservatively scheduled training design products624rows. Minimum-change has
 THREE products: outer residual, nested solver residual, final updated residual.
 Count _solve_design1 plus _solve_minimum_change2; not exact BLAS FLOPs.
- 32probability kernel calls144rows, including F/return evaluation and old TRAIN
 monitors.116typed affine calls116rows exactly reconcile62source+54private.
 Source98typed calls include36unfitted reads without a matrix product; private
18typed calls/phase.152typed requests/304context+generic reader invocations.
- 42workflow episodes,200dispatcher calls,4teacher constructions.5context/44generic
 owner initializations and restores,3context forks/0generic forks, no model RNG.
- Post-run input diagnostics3bounds calls120rows1080coordinates9600array bytes.
 F reserved arrays4320B, old monitors960B, return960B, return-monitor reuse0newB.
 Model coefficients768B/two public-context heads; retained source checkpoint
 coefficients1920B. These are selected logical/array records, not exact heap,
 copies, total memory or matched computation. Outer RSS/time includes additional
 Python, guard, serialization and NumPy costs. Source sampler peak55,603,200B
 is narrower than outer65,773,568B; do not attribute their difference to a model.

Actual probability kernel and typed aliases were observed before work, eliminating
327's missed-alias ambiguity for this case. Source preparation is included, not
a free reused baseline. Sampling is not process-tree/exact-allocation proof.

## Prerequisites, failures and retained artifacts

Reused327's unchanged scoped software tests; no numerical fixtures repeated.
Three --preflight calls used stdlib AST/metadata only,0project/numeric imports,
total observed shell7.0391149s<90s, each<30s/max24,227,840B<128MiB. Receipts
preflight.json/preflight-final.json/preflight-checked.json; final .027646620s
internal/24,215,552B. Repeated only for local typing/receipt changes, not modelwork.

Producer-only static commands, Windows .venv/Scripts/python.exe -B:

```text
-m ruff check artifacts/runs/ordinary-linear-probe-native-20261030/producer.py
-m ruff format --check artifacts/runs/ordinary-linear-probe-native-20261030/producer.py
-m mypy --explicit-package-bases --follow-imports=silent <same producer>
-m mypy --explicit-package-bases --follow-imports=silent --platform linux <same producer>
```

Final statics-checked all4PASS/4.9021877s, literal args/exits/outputs retained.
Total statics+format21.4091515s<90. First both-platform8mypy method/stream annotations
failed; later default-platform resourceAPI/name narrowing failed. Simplified to
dynamic stdlib resource import; failed receipts preserved, no ignores/cap renewal.
One guessed oldPROTOCOL read failed, and initial result projection expanded
diagnostics/truncated output and used wrong action field (null). Simplified to
scores-only/raw selected_id reads; a source rg no-match ended EXIT1. These are
read/typing failures, not native failures; sole actual native case succeeded.

Stage artifacts/runs/ordinary-linear-probe-native-20261030/ initially14files/
1,559,474B<8MiB: native.json1,503,802B, raw.json.gz22,208B (same parsed full report),
work.json/native.log/preflights/static receipts/producer/review/native.attempted.
All curves,18typed episodes, trusted3phase copies, uncertainty and charges retained.

## Review and critical path

Lead self-reviewed combined source, actual traces/curves/costs and negative return
confidence. Independent finite protocol/result review is recorded in stage
review.txt/owning issue, distinct from self-review. Full suite/new model fixtures/
native controls/package build/clean clone/fresh dependencies/release checks skipped;
this is an actual scoped task observation, not repository-wide readiness.

Independent protocol + ACTUAL + report ACCEPTED:4readonly commands/.7044599s,
allEXIT0/no live sessions. Raw gzip equals native JSON, all attempts/returns,
source/actor/RNG flags, original limits/clocks/charges,18typed records/preserved
copies, full eight-point CE curves and adverse confidence verified. Three-product
accounting and actual kernel/alias/fork hooks independently inspected. Third-read
diagnostic display truncation resolved with compact curves in fourth read;
raw evidence unchanged. No project/numeric imports/models/tests/writes/network/
children; exception ended. No independent whole-repository/release qualification.

This case supplies local conventional acquisition/retention, not recovery or
robust initial-failure evidence. Do not force forgetting, extend horizons, select
seeds or automatically add another capture/provenance increment. The full roadmap
still includes R4.7's shared inspector: existing saved-ticket/context projections
accept older report families, not this ordinary reserved-probe report/summary.
Next concrete implementation: extend the existing shared inspector's report
dispatch/projection to display these recorded private phases, full CE curves and
separate costs with saved-event pointers, unknowns/failures and inactive controls
visible. Use existing source-cell/renderer plumbing, no new runner or learner work.
Show probe-versus-deployed distinction and the adverse return CE/near-tie caveat.
Broader learning/generalization/native-family claims and release stay OPEN.
