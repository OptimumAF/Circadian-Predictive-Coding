# Fixed continuous MLP learning sequence

## Latest real cached lighting evidence — 2026-10-20

The existing CLI completed one jointly frozen BP/PC/neutral CPC/active CPC
A8/B8/A8 cohort on qualified cached TRAIN17 A/B29-row RGB789 releases. Same seed,
architecture, A-only transform, physical metrics and update exposure; native
computation differs and is charged separately. All four completed24updates696rows.
PC/neutral common reports match exactly under the prospective exclusions. Active
CPC has higher final error on both contexts; this is a negative comparative result.
All six signed metrics, all axes/endpoints/conditioning/native costs and resources
are retained. It is appearance adaptation on the same physical trajectory, not a
new physical task or held-out/native skill result. Reuse saved evidence; these
four experimental allocations are spent.

The isolated continuous probe and fit/direction APIs described in the older
next-work sections below have since been implemented and accepted; see
[continuous probes](continuous-fresh-task-probes.md). The current functional gap
is a runnable opt-in composition of the returned detached stream candidate with
[the existing frozen robot API](frozen-bc-robot-rollout.md), with fake-boundary
refusal/isolation tests before any separately reviewed native study. No new
checkpoint, provenance or orchestration layer is required.

[Joint protocol](../artifacts/runs/matched-continuous-lighting-20261020/PROTOCOL.md) ·
[Executed results, exact commands, failures and next action](../artifacts/runs/matched-continuous-lighting-20261020/RESULTS.md) ·
[Independent review](../artifacts/runs/matched-continuous-lighting-20261020/REVIEW.md).

## Matched continuous learner families — 2026-10-20

The same runner and JSON wrapper accept LAST optional `learner_family="bp"`.
Choices are `bp`, `pc`, `neutral-cpc`, `active-cpc`. Default and explicit BP preserve
legacy report/call shapes. PC/CPC use the existing continuous PC learner with the
same default base H64/seed17/LR.01/momentum.9 and single A-only transform. Precision1,
inference cap4/rate.05 and existing stopping defaults are fixed; neutral uses real
chemistry with unit gates, active uses the existing default chemistry. No seed or
optimizer search is performed. Mixed binary-gripper likelihood is refused for
PC/CPC before fitting; legacy BP keeps it. CLI `--learner-family` is allowed only
with `--scenario continuous-mlp-learnability`, before teacher reads.

```python
result = run_continuous_learnability(warm_train, new_train,
                                    phase_updates=1, learner_family="neutral-cpc")
assert result.report["complete"]
# candidate is a detached complete NumpyContinuousPCLearner in PC/CPC modes.
```

Run the fixed synthetic integration with
`python -m scripts.continuous_learnability_example`. It executes four phase1 A/B/A
runs (12updates/28rows total), not the real-data teacher command below. The example
checks exact PC/neutral common state/reports and preserves every signed metric
without a gain or winner threshold. Use any larger/real-data/comparative run only
under a separately frozen reviewed protocol; prior spent native cases stay spent.

For PC/CPC, endpoint conditioning describes the same read-only task-head BP total
gradients/saturation as BP, explicitly labeled
`common_task_head_bp_total_gradients_read_only`. It is not PC inferred energy or
its free-state gradient. Common physical action half-MSE/signed metrics retain
their original definitions; training diagnostics keep their native energy ID.
All inputs remain released complete TRAIN episodes. No final/probe labels enter
this learning recipe, fitting, selection or promotion. No serving publication.

The native runtime port returns only TrainingDiagnostic. A bounded recording
facade therefore preserves each actual PC `train_batch_with_work` scalar return
in `native_updates` before app postcommit/resource checks. It retains no model
history, input, gradient or latent chain. The sink capacity equals the original
run update allowance and is checked before work. Complete state comparisons
revalidate exact snapshots plus policy/counters/parameters/momentum/wake/config;
one expected candidate protects prior successes and final owner observations.
A changed but internally coherent chemistry state is refused. Partial reports
retain native returns, failed work and app receipts separately; incomplete work
releases no learner, aggregate success work or metrics.

Cost accounting for PC/CPC separates actual native counters/row-weighted event
exposures, returned task-conditioning work, readout normalization, derived residual/
actor forwards, fitted/diagnostic/training/repeated rows and numeric owner payloads.
`native_event_row_weights` are exposures, not arithmetic. No BP training dense-MAC
formula is assigned to PC/CPC. Four extra CPC transition counters exclude repeated
validation gate recomputation; state comparisons also validate/copy snapshots.
Metadata, temporaries, elementwise math, copies and process memory remain outside
numeric payload/event summaries. Wake adds16Hnumericbytes/owner; the bounded scalar
sink/report and temporary full expected snapshot have additional storage overhead.
Wall time includes runner operations but neither runtime pools nor peak RSS/GPU
use are measured. Historical BP cost formulas remain scoped estimates, not a new
measurement of the stronger state checks.

```text
src/adapters/continuous_learnability_demo.py   existing recipe, family selection
src/adapters/continuous_learnability_modes.py bounded native facade/common readouts
scripts/run_native_stream_demo.py            existing CLI, explicit family option
scripts/continuous_learnability_example.py   fixed four-family synthetic sequence
tests/test_continuous_learnability_modes.py  independent integration/failure tests
```

```powershell
python -m pytest tests/test_continuous_learnability_modes.py tests/test_continuous_learnability_demo.py tests/test_continuous_learnability_result.py tests/test_continuous_learnability_binary.py tests/test_continuous_phase_updates.py tests/test_continuous_a_only_control.py -k 'not direct_reference'
python -m ruff check src/adapters/continuous_learnability_demo.py src/adapters/continuous_learnability_modes.py scripts/run_native_stream_demo.py scripts/continuous_learnability_example.py tests/test_continuous_learnability_modes.py
python -m ruff format --check src/adapters/continuous_learnability_demo.py src/adapters/continuous_learnability_modes.py scripts/run_native_stream_demo.py scripts/continuous_learnability_example.py tests/test_continuous_learnability_modes.py
python -m mypy --follow-imports=silent src/adapters/continuous_learnability_demo.py src/adapters/continuous_learnability_modes.py scripts/run_native_stream_demo.py scripts/continuous_learnability_example.py tests/test_continuous_learnability_modes.py
```

The next learner outcome is an optional isolated continuous fresh-task probe using
the existing stream's probe_train runtime and common physical residuals. Existing
fresh_task_probe validation/scoring is classification-specific (B,1class labels/
probability metrics); continuous multi-axis actions must not be passed as classes.
Reuse its isolation/budget rules and the existing stream, preserving a separate
probe learner/allowance and probe_test-only score labels. A small matched protocol
must freeze checkpoints/roles/metrics/costs before new comparative execution.
R2.1's accepted classifier diagnostics/probes remain accepted; continuous probing,
robot/generalization/benefit/release requirements stay open.

[Protocol](../artifacts/runs/continuous-learnability-modes-20261020/PROTOCOL.md) ·
[Executed results and limits](../artifacts/runs/continuous-learnability-modes-20261020/RESULTS.md) ·
[Independent review](../artifacts/runs/continuous-learnability-modes-20261020/REVIEW.md).

The existing stream CLI has an explicit `continuous-mlp-learnability` mode. It
trains a private continuous BC candidate on A, B, then returning A, reporting
acquisition, retention, recovery and costs. Source and stable actor stay exact.
This is offline TRAIN imitation; the report does not establish autonomous robot
skill, new-task generalization, domain drift or a circadian advantage.

## Run and boundaries

```powershell
.venv/Scripts/python.exe -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --visual-observation --warm-teacher artifacts/runs/pick-cube-rgb-features-20261019/teacher-visual.json --new-teacher artifacts/runs/visual-context-return-20261019/teacher-b.json
```

This command performs training; run it only under a declared local experiment
budget. The accepted case is stored, so reuse its evidence instead of rerunning
it to confirm documentation. Omitting `--visual-observation` keeps the existing
state teacher reader. Seeds are bound to TRAIN17/19; `--seed` is refused. Existing
CLI defaults, nearest mode, matched BP/PC/neutral CPC/active CPC and teacher readers
are unchanged. No held-out or guard record is converted to TRAIN.

```text
scripts/run_native_stream_demo.py               Existing CLI, new explicit mode
src/adapters/continuous_learnability_demo.py     Fixed recipe and scalar reports
src/adapters/numpy_continuous_bc.py              Existing owned MLP/diagnostics
src/app/actor_shadow.py                         Existing arrival/budget/isolation
tests/test_continuous_learnability_demo.py       Independent behavior/reference tests
```

Python API: `run_continuous_learnability_demo(warm, new,
expected_standardization=None)`. Both episodes must be complete, exact TRAIN
records with distinct identities and matching specifications. Admission uses the
existing continuous stream helper before fitting or model creation. One A-only
fit is frozen throughout. Optional expected statistics gate that single fit
before learner construction, allowing the accepted case to require exact reuse
of its earlier A-only transform without fitting twice. No B refit or tuning.

## Fixed behavior

Default MLP seed17, hidden64, learning rate0.01, momentum0.9 and MSE. The sequence
is eight complete-batch A updates, eight B updates, then eight A-return updates.
All 24 updates use one original runtime budget and 24 explicitly identified
packets, with label delay one logical tick. Each pending packet must leave the
candidate unchanged; each successful receipt/state/charge is validated before
the next packet. This is repeated exposure to saved TRAIN labels, not 24 newly
collected episodes. No replay during the absent context or serving publication.

Endpoints 0/8/16/24 each report both contexts in A/B order. Existing conditioning
and residual APIs read a temporary owned restored learner; the actual stable
actor predicts both contexts. Full policy, parameters, velocity, counters and
source/actor/candidate isolation are checked. The final empty poll must preserve
full state and charges. Returned readouts are recorded before later assertions;
the bounded cumulative receipt list survives even a postcommit return failure.

Admission errors raise; execution failures return `complete=False` with partial
events/checkpoints/charges and error. The CLI prints that evidence and exits1.
Successful work totals/metrics are absent on an incomplete run. A failed call
may have consumed additional diagnostic/copy work beyond returned reports.

## Metrics and costs

Primary endpoint metric is normalized physical action half-MSE, lower better.
All active-axis/stage residuals, declared objective loss, saturation, gradients,
actor predictions and every returned training loss are retained. Signed fields:

| Field | Definition | Positive means |
| --- | --- | --- |
| warm_acquisition_gain | A0 minus A8 | A fit improves |
| new_acquisition_gain | B8 minus B16 | B acquisition improves |
| warm_retention_loss | A16 minus A8 | A retention worsens |
| warm_return_recovery | A16 minus A24 | A return improves |
| warm_return_deficit | A24 minus A8 | A remains worse than its first endpoint |
| new_retention_loss | B24 minus B16 | B retention worsens on return |

For accepted A29/B30: 24 SGD updates/704 training-row exposures, 59 unique source
rows and 645 repeats. Diagnostics: eight conditioning calls/236 rows, eight
residual calls/236 rows; stable actor: eight predictions/236 rows. Total dense
work is 131,586,816 MACs, 56 forwards/1,648 rows and 32 backwards/940 rows.
Normalization is 48 calls/1,412 rows, with initial A fit29 separate. Generated
gradient payload is 1,626,240 values/13,009,920 bytes across all 32 gradients.

Four named owners each have 813,120 parameter/velocity numeric bytes plus12,624
statistics bytes. The bounded inbox has 4,466,176 feature/target numeric bytes.
Snapshots, Python, copies, validation, reductions, elementwise math and process
memory are separate; these numeric sums are not whole-process memory peaks.
Input/hidden/output diagnostic counts describe tensor sizes, not total repeated
activation work. Compact actual-case report is capped prospectively at256KiB;
no model/activation/gradient archives or unlimited success chain is retained.

## Checks and safe extension

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_learnability_demo.py tests/test_continuous_stream_demo.py tests/test_visual_continuous_stream.py
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_learnability_demo.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py tests/test_continuous_learnability_demo.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/continuous_learnability_demo.py scripts/run_native_stream_demo.py
```

The accepted gate runs the direct-reference test exactly once globally; repeat
or platform checks exclude it with `-k 'not direct_reference'`. One module fixture
shares the full run; corrupt-receipt/postcommit and empty-poll-mutation cases
separately test permanent charges and incomplete evidence. Tests use tiny
synthetic TRAIN data and no simulator or stored real teacher records.

Keep the mode fixed and reproducible. Add a separately named, prospectively
reviewed protocol before changing any model, optimizer, transform, update count
or comparison. Preserve all negative results; keep guard/probe/final data out
of fitting, tuning and selection. Use the existing runtime for future learners.
[Frozen protocol, exact checks, actual results and limitations](../artifacts/runs/visual-mlp-learnability-20261019/RESULTS.md).

## Continuous probe followup — 2026-10-20

The optional fixed-checkpoint continuous probe dependency above is now implemented.
Use an owned ContinuousProbePlan through the existing runner/JSON API; the CLI
retains its existing teacher readers and defaults. Four disposable probe_train
scopes score fresh probe_test and old TRAIN tasks, with separate work and complete
benchmark state/budget isolation. All later one-step fresh gains in the fixed
synthetic example were negative; no benefit or cause is inferred.
[API, checks and evidence](continuous-fresh-task-probes.md).
