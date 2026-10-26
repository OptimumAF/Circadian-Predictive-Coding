# Workflow32 source: isolated fresh-task fit and old-TRAIN damage

[Issue151](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/151),
R2.1-CLM-W.X. This is one prospectively bounded native source32→probe6 case.
It asks whether the disposable warm-weight probe can fit the fixed reserved
markers and what happens to its known old-task decisions. The benchmark/source
learner stays unchanged during probe work. Context-return relearning and deployed
adaptation are outside this case; no circadian advantage or release claim.

## Frozen procedure and metrics

Use the existing public-observation workflow preparation, catalog fitter and
read-only TRAIN diagnostics, then the qualified guarded workflow-probe adapter.
No production code, CLI, update mathematics, optimizer or metric definition
changes. The source has32catalog CE updates/384query160candidate1920logits,
original twelve TRAIN rows/six unique queries/five actions and32-step OneCycle.
Its runtime must complete before Fencoding or probe admission. Do not select a
seed/checkpoint from accuracy; a failed source stops the case.

The initial raw encoding remains its original eleven-sequence batch. Encode
`tavrix`, `vemlot`, `tavrix please`, `vemlot please` plus the existing instructions
in a separate four-sequence batch. Use the original raw action rows for
defer/inspect/route, with targets inspect/route. These are reserved from the main
TRAIN corpus, not claimed novel to pretraining. The probe owns fresh AdamW/
six-step OneCycle, separate weights/budget/clock/training-only consent. Its six
delayed updates consume12marker query/18candidate rows/36logits. Source and probe
authority are original and nonrenewable; no source update during the probe.

Before/after probe groups are F TRAIN2x3, suffix2x3 and original old TRAIN12x5:
six reads/32query/22candidate rows. Source diagnostics add two TRAIN reads/24query/
10candidate rows. No replay, final data, promotion, installation, source/model
export, serving rollout or return-A update. All32+6updates finish before cached
scoring. Existing `_reserved_probe_metrics`/`_metrics` produce per-case accuracy,
raw NLL/status, Brier and five-bin ECE; `summarize_training_fit` retains old-task
phase/row target-minus-maxOTHER probability margins. No new clamp/tolerance.
All cases and raw before/after distributions are retained, not selected by gain.

Accuracy/margins increase toward better fit; NLL/Brier/ECE decrease. Two markers
are a tiny split. Twelve old TRAIN rows contain six unique inputs and repeated
sealed/staged rows; keep their phase-specific errors visible. Do not pool
confidence across the three- and five-candidate groups or infer generalization,
calibration, deployment retention, recovery or CPC benefit from these endpoints.

## Software prerequisites and source review

Three literal cases passed per Windows/Linux, zero failures/errors/skips:
fullsource32/probe6/deferredmetrics and source/loaded-head/callerTorchRNG
preservation; incomplete source blocks further encoding/probe; incomplete probe
blocks scoring. No fake-model accuracy assertion. Existing #15037-case isolation/
default/mutation evidence is reused, not rerun. The accepted native update/math
implementations are unchanged.

Per platform38Tiny updates396query178candidate1956logits,108heads94forwards679
projected rows,3constructor initial steps;2FakeBackbones3forwards26sequences/
tokenizations/IDs/padded slots;24local TRAIN teacher dispatches.160returned native
captures814660elements+1749376callerRNG,4MiBothercopyreserve/literal65536. All
original software bounds held, zero physical random/Transformers/assets/
pretrained/physics/final/replay work. Existing optimizer-copy scheduler warnings
remain in logs. No failed fixture/static check or cap revision.

| Software observation | Windows | Linux |
| --- | ---: | ---: |
| Body seconds |6.761434200 |20.632544605 |
| Parent seconds |7.303368500 |26.535772200 |
| Sampled peak RSS bytes, after Torch import |328843264 |344862720 |

The inherited scope label incorrectly said TRAIN-only. It was corrected after
Windows qualification; a saved before-file/AST comparison proves this was the
only numerical/control-source difference, so no model rerun was spent on a label.
Linux ran the final label. Scoped Ruff/format/default and Linux-target mypy and
final AST checks pass, total43.0826773s of120s before native work. Saved review is
Windows-only and skipped for Linux-target typing; native Linux statics/full
unchanged suites/clean-clone/install/GPU/CI/release/independent result checks are
unrun. Lead source/procedure/result review is **self-review, not independent
acceptance**. The actual sources and flat qualification map are saved locally.

## Native resources and commands

Original cached Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218,
original75,557,149B CLM head, unchanged local asset/digest/key validation, offline
WSL Python3.12/Torch2.14cpu/Transformers5.10.2, CPU2. Native4096→512 geometry and
2048/cap2047 token recipe unchanged. One Qwen/two encoder forwards/fifteen
sequences; physical batch≤11/width≤256 before backbone body, no extra truncation.
Caps112Headobjects100forwards720projectedrows (expected100/92/662),38TRAIN/
backward/AdamW/scheduler updates396query178candidate1956logits,2constructor
initial steps, physical random0/meta initializer visits16B. Native captures≤240/
18Btensor elements/caller RNG8M plus6Bother-copyreserve≤26Belements. Copies are
not all allocations or FLOPs. Twelve local TRAIN teacher transitions only.

The one native allowance is720s inner including imports/load and750s parent
including fresh RAM preflight,22GiB sampled RSS. Fresh available Windows RAM
36,116,496,384B/WSL32,609,841,152B; no other WSLPython process. Stage8MiB/log512KiB/
actual2MiB/ledger128KiB. Ordinary caches95,621,578B stay outside the stage under
128MiB. Source/protocol/prerequisite receipts were bound before invocation.
No retries, cap renewal, resumption, refit, reseeding, dose or metric selection.
Only OS/resource meters were inspected while live.

Recorded commands:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-fresh-native-20261025/prepare.py
.venv/Scripts/python.exe -m ruff format artifacts/runs/clm-workflow-fresh-native-20261025 tests/test_clm_workflow_fresh_case.py
.venv/Scripts/python.exe -m ruff check artifacts/runs/clm-workflow-fresh-native-20261025 tests/test_clm_workflow_fresh_case.py
.venv/Scripts/python.exe -m mypy --cache-dir .mypy_cache/clm-workflow-fresh-native-20261025-win --explicit-package-bases --follow-imports=silent artifacts/runs/clm-workflow-fresh-native-20261025/fresh_case.py artifacts/runs/clm-workflow-fresh-native-20261025/execute.py artifacts/runs/clm-workflow-fresh-native-20261025/review_saved.py tests/test_clm_workflow_fresh_case.py
.venv/Scripts/python.exe -m mypy --platform linux --cache-dir .mypy_cache/clm-workflow-fresh-native-20261025-linux --explicit-package-bases --follow-imports=silent artifacts/runs/clm-workflow-fresh-native-20261025/fresh_case.py artifacts/runs/clm-workflow-fresh-native-20261025/execute.py tests/test_clm_workflow_fresh_case.py
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-fresh-native-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-fresh-native-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-fresh-native-20261025/execute.py
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-fresh-native-20261025/review_saved.py
```

Both commands exited0. The sole native attempt and saved review are now **spent**;
do not retry, refit, reconstruct unsaved weights or reparse the original case
inputs. The detached `saved-review.json` retains the already-computed scores for
reporting. All older native/saved-review allowances remain spent; source149weights
were not exported or reconstructed. This case pays for its own fresh source as
a prerequisite of the different F/old-task-damage question.

## Observed learning and interference

The fresh source fit improves known TRAIN4/12→12/12. Both original source TRAIN
readouts validate preservation. The probe begins from that successful source,
learns only one reserved marker and loses two known old decisions. The source
remains unchanged during the probe; the probe is neither installed nor exported.

| Fixed group | Correct before→after | Mean NLL before→after | Mean Brier before→after | Five-bin ECE before→after |
| --- | ---: | ---: | ---: | ---: |
| F TRAIN, two markers, three candidates |0/2→1/2 |5.726722→1.582033 |1.833795→0.916550 |0.937604→0.473932 |
| F suffix, two queries, three candidates |0/2→1/2 |5.205237→1.714701 |1.740707→0.935767 |0.914301→0.479796 |
| Old TRAIN, twelve weighted rows, five candidates |12/12→10/12 |0.015935→1.254415 |0.001203→0.267742 |0.015610→0.097615 |

All candidate recalls are1; all NLLs finite, with zero zero-target-probability
cases. Raw probabilities, calibration bins and cases remain in the detached
review. Rounded display values do not replace those observations.

| F input | Target | Selected before→after | Target probability before→after |
| --- | --- | --- | ---: |
| tavrix |inspect |defer→route |0.049298078→0.042655889 |
| vemlot |route |defer→route |0.000215280→0.990568101 |
| tavrix please |inspect |defer→route |0.102846123→0.032658000 |
| vemlot please |route |defer→route |0.000292820→0.992294192 |

Thus improved mean F losses coexist with a lower target probability on both
tavrix inputs. The six-update dose does not acquire both markers. The suffix
result repeats the same one-marker pattern; this tiny template split establishes
neither robust generalization nor calibration.

Every old-row endpoint follows. Margins are target probability minus maximum
other probability, as already computed by the fixed diagnostic.

| Concept | Phase/target | Selected before→after | Target probability before→after | Margin before→after |
| --- | --- | --- | ---: | ---: |
| invoice |sealed/open |open→open |0.999924660→0.999087691 |+0.999888369→+0.998179424 |
| invoice |ready/inspect |inspect→inspect |0.978690207→0.884540200 |+0.957765440→+0.813859507 |
| invoice |staged/commit |commit→commit |0.988241732→0.982500970 |+0.977035208→+0.969337734 |
| legacy |sealed/open |open→open |0.999924660→0.999087691 |+0.999888369→+0.998179424 |
| legacy |ready/defer |defer→route |0.983941972→0.000653029 |+0.972219324→−0.680409431 |
| legacy |staged/commit |commit→commit |0.988241732→0.982500970 |+0.977035208→+0.969337734 |
| parcel |sealed/open |open→open |0.999924660→0.999087691 |+0.999888369→+0.998179424 |
| parcel |ready/defer |defer→route |0.926756084→0.000838234 |+0.855346695→−0.654993759 |
| parcel |staged/commit |commit→commit |0.988241732→0.982500970 |+0.977035208→+0.969337734 |
| password |sealed/open |open→open |0.999924660→0.999087691 |+0.999888369→+0.998179424 |
| password |ready/inspect |inspect→inspect |0.970623314→0.645378053 |+0.943378942→+0.460621506 |
| password |staged/commit |commit→commit |0.988241732→0.982500970 |+0.977035208→+0.969337734 |

Sealed/staged remain4/4 each; ready falls4/4→2/4 and its mean margin falls
+0.932177600→−0.015230544. Repeated sealed/staged inputs account for eight of the
twelve rows, so10/12 understates the loss among distinct ready decisions. These
are old TRAIN predictions in a disposable learner, not a benchmark retention
score or deployed forgetting measurement. No return-A updates or learned
closed-loop workflow after probe are executed. Source equality is isolation
evidence, not proof that an adapting deployed model retains its task.

## Observed costs, integrity and limits

Native inner time356.070164764s, outer362.567103000s; sampled peak
16,721,719,296B. Original32source receipts apply at ticks1..63 and separate6probe
receipts at1..11. Both streams complete, with no unreturned work unknown. Exactly
38backward/AdamW/scheduler attempts,396query178candidate1956logit exposures and
2constructor scheduler initial steps. There is no replay exposure. Native
100Headobjects/92headforwards/662projectedrows match the preregistered counts.
One Qwen/two encoder forwards/fifteen tokenizations visit148token IDs and176padded
slots, below original bounds. Twelve local TRAIN teacher dispatches; zero learned
workflow episodes, physics or game effects. Physical random initializer elements0;
meta initializer visits15,173,957,632.

There are156returned native captures/10,993,421,492tensor elements plus1,628,032
caller-RNG elements. Adding the original6Bother-copyreserve gives
16,995,049,524elements, below26B. This finite accounting is not total allocations,
resident bytes or measured FLOPs. Original source budgets, consent, counters,
clocks, owners and loaded-head/caller-Torch-RNG preservation hold; source work
stays32commits/two diagnostics throughout the probe. All36flat source/protocol/
qualification pins hold. Actual JSON SHA256:
`3a3343bdb8b78be81ec0e0d3f2d025007755d7c2bd2f860f6a4eb716758af62d`.

The sole postterminal stdlib review passes structure/cost checks in0.015339s
body/0.2052901sparent, peak22,065,152B, reading161,113B current inputs under its
20s/256MiB/3MiB allowance. Zero prohibited imports, additional scoring, model work,
effects or input errors. Native and review have no execution failure. Negative
learning endpoints above remain failures of full acquisition/retention, not
infrastructure errors. No outcomes were inspected while the native process lived.
The report was prepared from the detached review after its one-time input read.
Final local stage28files/404,545B, native log1,751B, actual154,536B and ledger820B
remain below the original storage caps. Documentation-only closeout changes the
report, README introduction and probe usage result paragraph; executable source
has not changed since the passing qualifications and native source binding.
`git diff --check -- README.md docs/clm-reserved-probes.md` passes (only the existing
README line-ending conversion notice). No unchanged model tests or native run
were repeated for those documentation edits.

Final source/procedure/result review by the lead accepts the scoped research
question as answered with partial acquisition and interference. This is
**self-review only**. Existing BP/PC/neutral/active controls and all earlier failed,
negative or unknown cases remain unchanged; this case compares no model families
and establishes no CPC advantage, efficiency improvement or repository readiness.

## Critical path and next action

The R2.1 probe and workflow connection now exist and have a native observation;
another provenance/capture increment is not the next dependency. The old-task
route errors and one-marker acquisition justify testing whether parameter scope
matters. They do not identify which parameter changes caused either result.
`CLMCatalogProbe` and the temporal runner already support `state_only`, but the
workflow32 entry deliberately refuses that combination.

**Next implementation action:** add an explicit opt-in `state_only` argument to
the existing guarded workflow32 probe entry, preserving the all-parameter default,
original six-update dose, fixed data/metrics and original12x5TRAIN readout. Qualify
fixed action-head/logit-scale and source/caller preservation with tiny focused
tests before freezing any distinct native comparison protocol. Do not rerun this
spent case or add F steps/seeds; any later comparison must count fresh source
preparation and all work and retain this negative all-parameter result. This is a
bounded conventional control, not a claim of remedy or a new orchestration layer.
Reliable new-task acquisition, joint retention, context-return recovery, broader
R2/R5.1/R5.3/G5/R7 and release requirements remain open. Historical plans/log are
frozen under AGENTS.md; live handoff is issues151/18.
