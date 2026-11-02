#397 — ordinary recurring current-only workflow and native control

The existing workflow now exposes #396's repeated-current baseline through
`run_workflow(recurring_training=True, recurring_current_only=True)` and
`--example-affine-recurring --recurring-current-only`. The option is cold-only,
ordinary TRAIN-only, and refuses incompatible modes before data/model work.
Before gameplay it requires the exact control definition/protocol, paid10/252/
head36 endpoint, verified original receipts and90arrival+162CURRENTrepeat rows;
a same-shaped joint-replay report is refused. Defaults remain unchanged.

Production: `scripts/minigrid_affine_workflow.py` and
`scripts/run_minigrid_linear_fit_control.py`. No new solver, representation,
scheduler, loader, dependency or framework. Existing ABC source, original G
collector, delayed stream, physical solver and G/A evaluator are reused.

## Actual result: acquisition scores, forgetting and return

One frozen source-workflow API case completed. All entries below are correct
actions out of18 on the original TRAIN source pair, scored only after all fits.
They are not native episode success measurements at each checkpoint.

| Updates | A | B | C_train | G_train |
| ---: | ---: | ---: | ---: | ---: |
|0|2|2|2|2|
|2|18|13|11|15|
|4|13|18|15|12|
|6|11|14|18|12|
|8|15|13|13|18|
|10|18|13|11|15|

Each current task fits its teacher actions. G reaches18/18, then falls to15/18
after A returns; B/C also fail retention. A's action score actually loses and
regains correctness. There was no native evaluation at update8 or earlier;
new native G acquisition or native goal recovery after observed failure is
unmeasured, even though TRAIN action acquisition/forgetting is observed.

| Final native episode | Goal | Steps | Return | End |
| --- | --- | ---: | ---: | --- |
|G blue|false|16|0|truncated|
|G red|false|16|0|truncated|
|A blue|true|9|0.49375|terminated|
|A red|true|9|0.49375|terminated|

All four episodes completed and closed with unchanged readout state; all roles
TRAIN, all seed17/max16/K7 firstargmax. Both original G teachers also completed
and closed in9steps each. Six environment owners/reset/local Gym seed calls,
68native steps (18teacher+50game), no extra cleanup required. Source, actor,
NumPy/Python caller RNG and original authorities remained unchanged.

The accepted historical joint recipe #124/#127 retained all four TRAIN tasks
at18/18 and solved all four final G/A games. Its ten fit matrices have the same
physical row counts as this control. The result is consistent with old-data reuse
mattering beyond extra row exposure in this recipe. It is a historical reference,
not a freshly byte-paired or concurrently replicated comparison. Do not infer
statistical reliability, generalization, matched computation or CPC advantage.

The confidence tradeoff remains: current-only final A CE1.165422180486, versus
accepted joint1.189879. Current-only final CE B1.401678037849,
C1.525572125637,G1.273079259439; after G it was1.165422180486 on G. Every per-source
metric and action is preserved in the actual report; no favorable endpoint,
seed, weight, dose or metric was selected.

## Counts, memory and timing

Exactly10analytic fits/10LAPACK calls,252design rows, no SGD/backward/optimizer or
random weight initialization. Matrix rows9,9,18,18,27,27,36,36,36,36.90arrivals
include18return-A reused exposures, from72distinct source rows;162extra CURRENT
repeats and0prior-packetreplay/0retainedbank. Zero bank is not zero data reuse or
zero memory:598752materialized input bytes plus original source, stream copies,
six checkpoint models and validation/metadata overhead. Final two heads each
rank9 despite36fit rows, with4158coefficients/33264logical float64 bytes total.

48TRAIN metric calls432rows;98port predictions482rows, including50native actions;
90fitted-core predictions410rows (initial8uniform source reads need no fitted
core).2966snapshot-copy calls,3395fit validations,2064enumerated public projection
calls. Wrapper scopes are in the producer, not FLOPs/all allocations, and cannot
be compared directly with old observer implementations. Seven LEARNER constructor
entries are distinct from six ENVIRONMENT owners.

Host3.6725296s; observer through actual capture3.5012352s/80797696B sampled process
peak; workflow2.4156055s/79630336B; training/report phase0.2534994s. Original
TRAIN10update/90s/512MiB limits unchanged; readout0update/0replay/150s/512MiB,
original started_at equals observed_started_at. No workstop, expiry or cap breach.
Host time includes final metadata; process RSS excludes parent/process-tree.
Do not compare the3.50seconds to old joint128.98seconds as efficiency evidence:
the observer/environment/overhead accounting is not a matched-compute trial.
Local Gym draws are unmeasured; six seeded owners and caller RNG preservation
are separately verified. No Torch or model-training RNG was used.

Actual719536B SHA256
`1f7ec05b7636282a4322f5cdd39a0e62516aacca43c1eda7d6988d6248ed3cec`.
Producer serialization, sole lead byte read and sole peer byte read exhaust the
three prepared capture/read passes. Use LEAD-SCALARS.json afterward; do not read
the raw report again or restore its models.13direct source pins matched before
and after the case; this is not whole-repo/dependency/authenticity verification.

## Executed checks and failures

PhysicalD checkout and TEMP/TMP/TMPDIR=`D:\Dev\_Temp`; existing Windows venv,
Python3.14.7/NumPy2.4.6/MiniGrid3.1.0/Gym1.2.3, threads2. Numerical work occurred
only in the single separately frozen native case below. Scripted checks used
the actual delayed stream/train_batch with literal numerical/collection/games.

From the physical repository root, use the recorded launcher prefix
`./artifacts/runs/recurring-current-workflow-20261101/run-check.ps1` with these
exact arguments (do not rerun a spent attempt name):

| Arguments | Outcome | Body seconds | Process peak B |
| --- | --- | ---: | ---: |
|`-Attempt initial`|29newPASS|4.3297024|76316672|
|`-Attempt regressions`|63legacyPASS|2.2955981|82325504|
|`-Attempt typed-forward -Selection execute_actual_delayed`|1affectedPASS|1.3653518|76357632|
|`-Attempt meter-literals`|3actual-helper literalsPASS|0.5970710|47255552|
|`-Attempt meter-rss -Selection latch_actual_workstop`|1affectedPASS|0.5578919|48037888|

95distinctpassing/97executions,0testFAIL/0SKIP, aggregate9.1456152s, all host exits0/
noexpiry/optional runtimeimportsblocked[]. All within frozen140executions/120s/
60s-per-body/512MiB. New mode/default/role/intent/control-evidence/source/receipt/
lost-return/budget/clock/corruption and CLI boundaries covered. #396unchanged
math checks accepted/reused, not repeated.

Scoped commands: `python -B -m ruff format --check` and `ruff check` on both
production scripts, both new tests and native_observe.py; `mypy --follow-imports=silent
--cache-dir <software-stage>/type-cache` on the two scripts/two test files.
Final all0/.7662692s (earlier final three-file batch.8764127s). Initial one
boundary-kwargs type error and later two AST-list typing errors are preserved
and fixed by explicit annotations, not ignores/behavior weakening. Initial
timed static batch3.3595981s failed typing. Supplemental format/type commands
were not separately timed as a single total. Native producer is lint/AST/literal/
source-reviewed; it is not claimed fully typed or independently numerically proved.

Native command, exactly once after gates/PRE GO:

```powershell
./artifacts/runs/recurring-current-native-20261101/run-native.ps1
```

Host PID81696 exit0/child0/statuscomplete/noexpiry. No live handle remains.
Original budgets/failures are SPENT; no retry. Final source/report self-review
is separate from independent review. CLI parsing/dispatch was scripted; the
native case used the workflow API, not installed console/wheel qualification.

## Review and storage correction

Finite independent exception: software source delta ACCEPT, first native PRE
CHANGES_REQUIRED because stage/log limits were unenforced, then compact host
storage correction PRE GO, then actual POST ACCEPT. Original host/finding and
initial truncated read retained. Host now checks stage/log totals every200ms and
after exit, reserves4KiB for its receipt, and permanently fails/spends on excess.
Polling overshoot is retained failure, never deleted or described as in-budget.
Three AST-extracted actual PowerShell boundary predicates PASS/.0426164s, parse0,
nativeprocess0. No real cap-exhaustion/kill/cleanup episode was run; literal tests
and source review are distinct from the successful normal native closure.

Reviewer made5attempts/4complete batches, reported cumulative observed shell
time0.7976s. First output-truncated read preserved; prospective bounded revisions
were recorded before follow-up reads. No reviewer edits/tests/models/replication.
Full changed workflow/CLI deltas, producer/protocol/host/storage delta and one
actual scalar projection were read; full unchanged code/test suite and software
test-count recount were not independently qualified. Nested readout/workflow
budgets were parsed but not independently printed; detailed clock equality is
lead-derived and enforced in the original runtime. Exception ENDED/no carryover.

Host-recorded stage748400B/log1411B; later scalar/review metadata remains under
the8MiB stage cap. Software stage/type cache remain under64MiB/32MiB. Native
actual4MiB/log256KiB caps and all numerical/time/RSS caps held. Full suite,
Linux/WSL, clean clone, wheel, browser/device/robot/GPU/CI/release skipped. No
install/download/Git publication/deletion/global setting changes. Preserve all
383/390X/391/392/395negatives/partials and unrelated work.

## Critical path beyond this increment

This closes397's scoped implementation/one-control investigation, not the full
repository goal. Reconciliation found that successful PickCube joint acquisition
and reserved continuous imitation/native probes already exist; reuse them.
PushCube remains genuinely unresolved: the accepted aggregated198 correction
study fails all four physical endpoints, and the existing19503-pair diagnostic
already rules out exact raw/native32 target conflicts. Do not duplicate either.

Next398: inspect the existing continuous readout LS primitive, frozen-rollout
policy admission and qualified198data/weight extraction. Freeze a minimal bounded
raw-physical continuous affine control and meaningful cases before edits or any
data/model access. Keep the SAME original198TRAIN rows/six-group weights and
all earlier failures; no new corrections/dose/seed/weight search, hidden/final
selection, model reconstruction or silent widening of old128row contracts.
No software/source-read/fit/native allowance exists for398 yet. Full parents
18/21/23/33, general useful adaptation/science/reliability/release remain OPEN.
