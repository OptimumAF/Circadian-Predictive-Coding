# R4.8-CLM.LW — direct fit to typed local workflow

2026-10-25, [issue #158](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/158).
Implemented and executed scoped software behavior; single-agent source/procedure/
results **self-review**, not independent acceptance. Full goal and broad
R2/R4/R5.1/R5.3/G5/R7/release requirements remain open.

## Delivered behavior

The existing CLI has a default-off `--linear-fit-control` route. It calls the
original twelve-row TRAIN builder once, performs one detached categorical fit
and after-TRAIN readout, then runs the original three local development scenarios
through the existing typed dispatcher. There are no fabricated eight-step SGD
receipts. Unchanged/contrastive-fine-tuned controls remain explicitly unobserved.
The existing public-observation data flag stays explicit; incompatible
contrastive-only flags, including false-like non-booleans in APIs, fail before
loading. Default `run_loaded`/`run_case` code is AST-identical after removing only
the new opt-in argument/conditional statements; empirical before/current default
reports match on both platforms except observed timing.

New `CLMLinearDecisionPredictor` accepts a TRAIN fit, frozen encoder and exact
fixed raw-action declaration/catalog IDs/descriptions/order. Probe fits cannot
serve. Each returned request has exactly one query encoding and linear readout,
without contrastive head or optimizer work. Native insertion-order ties, margin
confidence and retained-input-token observation are explicit; output tokens stay
unknown. Exact producer-observed encoder deltas are checked rather than broadly
ignoring counter changes. Uncertain paid failures stop that owner without retry.
Only one last scalar readout and bounded frozen state are retained.

Loaded heads/modes/gradient ownership/storage/scale, frozen backbone, caller RNG,
original features and paid records remain guarded. Callback changes to prior fit,
episode owners/values, counters, request clock or save failures refuse more work
and preserve trusted prefixes. Final save also guards summary/pending status and
preserves their trusted values on corruption. This is trusted local Python, not
hostile-process containment or a dataset provenance certificate.

Files: new `src/adapters/clm_linear_decision.py`, `scripts/clm_linear_workflow.py`,
minimal opt-in in `scripts/run_clm_ticket_workflow.py`, focused
`tests/test_clm_linear_workflow.py`, README and [usage guide](../clm-linear-workflow.md).
No dependencies, new loader, learning rule, stream framework or effect rule.
Three #157 numeric/control sources remain byte-identical; accepted math is reused.
The helper composes existing components and imports adapters at the script boundary.

## Executed results and retained failure

Stage `artifacts/runs/clm-linear-workflow-20261025/` contains prospective PROTOCOL,
pre-edit script, old/repaired meters, exact command receipts, XML/stdout/work,
initial and final rollups/current source pins, and marked literal CLI JSON.

| Run | Observed cases | Body seconds | Parent seconds | Peak RSS bytes |
| --- | --- | ---: | ---: | ---: |
| Windows1 | 35 pass | 60.169693600 | 60.730122800 | 329695232 |
| Linux1 | 30 pass, 1 fail; tail unexecuted | 51.536234067 | 56.419221300 | 339308544 |
| Windows admission | 10 pass | 4.254592400 | 4.654073400 | 244486144 |
| Linux repair | 15 pass | 15.594004312 | 21.096191800 | 347611136 |
| Windows CLI | 1 pass | 3.816427400 | 4.177083800 | 243937280 |
| Linux CLI | 1 pass | 9.191830789 | 13.930582700 | 270598144 |
| Windows final save | 3 pass | 4.597967400 | 4.958044800 | 244486144 |
| Linux final save | 3 pass | 9.411442695 | 13.618128500 | 271097856 |

All but Linux1 EXIT0; zero errors/skips/timeouts. Final **42 distinct qualified
cases/platform** combine unchanged first-cohort evidence and targeted final
admission/CLI/publication checks. This is not one final42-case full rerun. The
Linux repair executes the failed default comparison and four previously unrun
legacy checks. No repeated full cohort after measured head/RNG/body costs would
exceed the cumulative budget. Source/change review identifies the affected guards;
initial cohort had no per-function hash manifest. Final pins bind current sources.

The actual public `main`/`run_case` path executes with only the loader replaced by
a deterministic six-dimensional literal backbone and heads. It requests two
threads; the fixture records that request and forces one to honor its engineering
allowance. Real wall/RSS sampling and exclusive JSON output run. A second invocation
refuses the occupied file before loading. Final marked JSON:

- `windows-final-save-literal-cli.json` —87775B.
- `linux-final-save-literal-cli.json` —85185B.

Each captures one12-row fit, twelve correct original TRAIN decisions, ordinary
invoice success, stale parcel recovery and external invoice commit denial. Three
episodes complete, raw success2/3, one recovery/one denial. There are11 returned
typed requests,12 encoder forwards/22 sequences including initial11,11 runtime
dispatches and9 local applied effects. Teacher construction separately reports12
demonstration transitions. No real external actions. Denial is model-external
permission enforcement; stale-ticket recovery is not context-return relearning.
These are software fixtures, not released-model task outcomes or a two-thread
native/hardware qualification.

Linux1 failed before any legacy TRAIN/backward/optimizer when the meter read
arbitrary frame locals during Torch's lazy `torch._dynamo`/`torch._numpy` import.
Python3.12 inlined-comprehension variables became None, producing retained warnings
and a TypeError. Narrowing the profiler to relevant call/return events/paths and
specific method-entry locals fixed it. No product code/algorithm/data changed to
repair that failure, no warm model work/install or hidden retry. Initial paid
LAPACK/encoding/RNG work remains charged. Expected existing scheduler-wrapper
warnings from legacy tests are retained separately.

The original Windows meter also matched only `adamw.py`, missing Torch2.14's
inherited `Adam.step` implementation in `adam.py`. Its raw `optimizer_steps=0`
is preserved and **not zero work**. Conservatively charge all24 successful legacy
native TRAIN calls as24 optimizer attempts based on their source/completed
receipts; do not claim a corrected Windows runtime observation. Linux's repaired
meter observes24. Original/repaired harness copies remain, no model rerun/refund.

## Cumulative engineering costs

| Per-platform total including failures/repeats | Windows | Linux |
| --- | ---: | ---: |
| LAPACK fits / rows | 30 / 360 | 30 / 360 |
| Linear readouts / rows | 98 / 241 | 98 / 241 |
| Profiled linear/design cells | 17809 | 17809 |
| Oracle-cell reservation | 262144 | 262144 |
| Literal backbone / head objects | 35 / 142 | 36 / 150 |
| Fake encoder forwards / sequences | 228 / 1098 | 228 / 1098 |
| Token IDs / padded slots | 1098 / 1098 | 1098 / 1098 |
| Legacy Tiny TRAIN / backward / charged optimizer / scheduler | 24 each | 24 each |
| Legacy TRAIN query / candidate / logit exposures | 288 / 120 / 1440 | 288 / 120 / 1440 |
| Head forwards / projected rows / scheduler initial steps | 264 / 1056 / 3 | 264 / 1056 / 3 |
| Returned native capture + RNG elements | 13813253 | 13818309 |
| Body / parent seconds | 72.838680800 / 74.519324800 | 85.733511863 / 105.064124300 |

The direct route itself has zero native head/optimizer steps; the24 Tiny updates
are selected legacy/default-equivalence engineering work and are not free tests.
All literal parameter initializers are deterministic; metered random initializers0.
No real pretrained encoder/backbone/assets, Transformers, robotics/game physics,
final data, replay, remote CI or old scientific raw/detached-review reads.

Original limits held without amendment:96fit/1536rows,256readout/8192rows,
512fakeforwards/2048sequences/8192IDs/16384slots,128backbones/1024heads,
512headforwards/4096projectedrows,64Tiny updates/768query384candidate3840logits,
128initial steps,16M returned tensor/RNG elements+8MiB other-copy reserve,
2M linear/design/oracle cells,120sbody150sparent/1GiB sampledRSS/CPU1 sequential/
96cases per run. Logical counts/reserves are not total allocations, comparisons,
IO or FLOPs. RSS starts after imports. Profiler overhead changed after repair;
these timings do not compare algorithm efficiency.

Initial mypy reported seven missing-container/index typing errors, corrected by
annotations. statics2..6 and final four-file Ruff/format/default+Linux-target mypy
pass. Six attempts130.548063500s; conservative formatter-inclusive141.819838200s
plus metadata and bounded source/JSON/whitespace checks within180s. Actual Linux Ruff/mypy
unavailable and not installed; Linux-target typing runs on Windows. Ordinary
task-touched whole cache files109741764B<160MiB, global inherited+task596776488B
separately **not** within that reservation. Final stage before final rollup487008B/
stdout19170B below8MiB/512KiB. Earlier rollup/CLI snapshots remain frozen.

## Reproduction commands and limits of acceptance

From repository root:

```powershell
.venv/Scripts/python.exe -m ruff format src/adapters/clm_linear_decision.py scripts/clm_linear_workflow.py scripts/run_clm_ticket_workflow.py tests/test_clm_linear_workflow.py
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/statics.py statics-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/statics.py statics-3
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/statics.py statics-4
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/launch.py windows-cli tests/test_clm_linear_workflow.py::test_should_route_public_cli_explicit_flag_without_overwriting_output
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/launch.py linux-cli tests/test_clm_linear_workflow.py::test_should_route_public_cli_explicit_flag_without_overwriting_output
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/statics.py statics-5
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/statics.py statics-6
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-workflow-20261025/rollup.py '2026-10-25T16:58:35Z'
git diff --check -- scripts/run_clm_ticket_workflow.py scripts/clm_linear_workflow.py src/adapters/clm_linear_decision.py tests/test_clm_linear_workflow.py README.md docs/clm-linear-workflow.md
```

Exact targeted `windows-admission`, `linux-repair`, `windows-final-save` and
`linux-final-save` node arguments/expanded cached WSL commands are retained in
their `*-parent.json` command arrays; `focused.py` lists the initial cohort.
Additional measured formatting of edited files and bundled pre-statics4 formatter
is included above. These frozen names refuse reuse; a fresh name is not renewed
experiment authority. Python/Torch/NumPy environment and executable paths are
retained in receipts; Windows Python3.14.7/NumPy2.4.6 and Linux Python3.12.3/NumPy1.26.4,
Torch2.14.0+cpu both, no
cross-platform bitwise claim.

No full unchanged suite, native pretrained run, native task/retention/recovery
study, clean clone/install/new wheel, browser, GPU, remote CI, independent review
or release qualification. No installs/downloads/commit/push/deletion/unrelated
reset. Historical plans/log frozen under AGENTS; live issue/tracker owns handoff.
Local evidence is not a claim that these files are committed or published.

All earlier scientific negatives/permanent charges stay unchanged: #151 partial
F acquisition and old-task interference, #153 OPEN FAILED-SPENT preflight before
model load/no retry or lower gate, #154 manual UI policy gate, and older failures/
unknowns. #157's original dense precision failure remains; no numerical/tolerance
changes here. #155 wheel predates these changes. No CPC advantage, matched compute,
generalization or full repository readiness.

**One precise next action:** freeze and execute one small native direct-fit
development question through this existing producer: original twelve public TRAIN
rows/full diagnostics, complete ordinary/stale/denied task outcomes and all
feature-production/fit/readout/dispatch/time/RSS/storage costs. Prefer informative
known TRAIN gaps (legacy/password) over repeating only previously solved concepts;
freeze that case binding before observation and label it development, not unseen
evaluation. First check prepared assets/capacity without consuming a native
attempt; retain one bounded fitted-control summary before process exit if needed
for interpretation. No spent #153 retry, old-weight reconstruction, seed selection,
baseline tuning, final access or another checkpoint/audit increment. Native
usefulness, fresh-task retention/context-return and broad parents remain open.
