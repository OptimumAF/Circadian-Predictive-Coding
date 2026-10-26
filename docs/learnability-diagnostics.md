# Learnability and reserved fresh-task probes

## Categorical complete-episode extension

The existing disposable probe now supports explicit K>=3 CE/accuracy and reserved
complete episode roles, while retaining the binary rules below.197 scoped cases
including actual recurrent source/actor/checkpoint/RNG isolation pass. See
[categorical probe usage and boundaries](categorical-fresh-task-probes.md).
This is component correctness; a frozen checkpoint-based scientific case remains.

R2.1 now has feature-health diagnostics and an isolated fresh-task learning
probe, composed with the existing delayed-label A → B → A stream.

```powershell
.venv/Scripts/python.exe -m scripts.run_learnability_pilot --describe
.venv/Scripts/python.exe -m scripts.run_learnability_pilot --run
```

The fixed local protocol compares existing matched BP, coupled PC, neutral CPC,
and active CPC. It preserves their initialized tensors, objectives and update
settings. One seed, 23, is declared in advance. The budget is 136 updates total,
60 seconds, one BLAS CPU thread, 256 MiB process RSS and 32 MiB output. There
are no downloads, replay, sleep, promotion or adjustable experiment knobs.

The [retained pilot report](reports/learnability-fresh-task-20261016.md) shows
intermediate C improvement in every probe, followed by negative fixed-four-step
endpoints in15/16 cases. It supports no claim of plasticity loss or CPC advantage.

## Fixed sixteen-update follow-up

The separately named `sixteen-update` scenario extends probe exposure without
changing the benchmark, learner settings, seeds, metrics or original results:

```powershell
.venv/Scripts/python.exe -m scripts.run_learnability_pilot --describe --scenario sixteen-update
.venv/Scripts/python.exe -m scripts.run_learnability_pilot --run --scenario sixteen-update
```

Its probe curve is fixed at 0/1/2/4/8/16 updates, with the original four training
batches as an exact prefix and twelve additional consecutive batches from the
same generator. Probe evaluation and the label-free reference are unchanged.
All four arms and all sixteen probes remain in the output, including failures.
Initial levels and every scoring point are retained; neither an early best point
nor a method-specific update count becomes a new policy.

This is an exploratory follow-up motivated by the original short curves. It is
the same synthetic data family and seed, not independent confirmation or proof
of task generalization. The original fixed-four-update negatives remain valid.
The new budget is 72 benchmark plus 256 isolated probe updates (328 total),
2624 training-row exposures, 96 probe scoring forwards and 32 benchmark scoring
forwards. Sixteen diagnostic passes use 32 SVDs. The existing 60-second, single
CPU-thread, 256 MiB RSS and 32 MiB output limits apply. No replay, promotion,
sleep, download or environment interaction is added.

Without `--scenario`, or with `--scenario four-update`, the original protocol
remains selected. There are no seed, rate or arbitrary horizon controls.

The [retained sixteen-update report](reports/learnability-sixteen-probe-20261016.md)
records the sole 328-update run. BP improved at all four sixteen-update endpoints;
the three PC-family arms improved only initially and had nine negative trained
endpoints. Every curve improved at eight, then nine rebounded by sixteen. All
original benchmark results and four-point probe prefixes matched exactly. This
supports no CPC advantage or causal plasticity-loss claim. Full curves, cost
counts, the initial test-controller failure and independent review are retained.

## What the observations mean

The benchmark retains its original 18 delayed-label updates and checkpoints at
0, 6, 12 and 18. A/B final outcomes are scored only after all training and
checkpoint observers finish. The report separates:

- Learning B: initial B BCE minus B BCE after its phase.
- Retaining A: A BCE after B minus A BCE after the first A phase.
- Recovering A: A BCE after B minus A BCE after the returning A phase.
- Learning reserved C: the entire 0/1/2/4-update probe curve, its initial BCE,
  and initial minus final BCE.

Each checkpoint probe restores a disposable fork of a separate untrained
template. Probe labels and updates never reach the benchmark learner or its
scorer. C uses separately generated data and the rule `x0*x1 >= 0`. This is a
fixed nonlinear composition of A/B inputs, not proof of general fresh-task
learning. Probe evaluation labels are used only by fixed metrics; neither probe
scores nor final benchmark scores select an update count or policy.

All four method rows and sixteen planned probe rows remain present on failure.
Failed probes preserve attempts, commits, completed score checkpoints and their
reason. Incomplete arms have no retention/recovery comparison. There is no retry
or seed replacement. The original R1.7 study and its negative/inconclusive CPC
findings remain unchanged. This pilot establishes no circadian advantage.

## Feature health and costs

On a fixed label-free reference, each tanh layer reports mean absolute activity
per unit and its min/mean/max. Dead units have mean absolute activity ≤ 0.001;
active is the complement. Saturation is the fraction of entries with absolute
activation ≥ 0.99, so it can overlap activity.

Feature effective rank uses the centered activation matrix. For singular values
above `eps*max(shape)*max(1,max(s))`, normalize the values themselves to weights
`p` and report `exp(-sum(p*log(p)))`. Numerical zero has rank zero. These are
singular values, not squared variances. Constant active features can have zero
centered rank; the absolute cutoff is not invariant at tiny scales. Rank is a
descriptive diagnostic and does not establish a cause of impaired learning.

The report measures complete diagnostic and probe call time, including probe
preflight/copies. The remaining time is labeled benchmark and orchestration,
not training time. Inner probe timing begins after data/state detachment and
reports unavailable/partial timing honestly on failure. Retained snapshot and
reserved-data array bytes exclude Python objects and temporary allocations;
process RSS is measured separately. Probe update/example/forward counts are
separate from benchmark counts. Native BP and PC computation differs, so equal
updates do not imply equal compute.

**Full R2.1 remains open.** The coupled models do not store native per-neuron
importance or ages. Those fields are explicitly unavailable; chemistry and wake
counts are not substituted. Longer learning horizons and broader task/seed
generalization also remain unverified.

## Native importance and age observations

The legacy CPC learner already records per-unit importance, age and stable
lineage. Read these fields without training or taking a full model snapshot:

```python
from src.adapters.native_unit_diagnostics import read_legacy_unit_diagnostics

observation = read_legacy_unit_diagnostics(legacy_cpc_model)
```

The existing delayed-label demo has an opt-in mode:

```powershell
.venv/Scripts/python.exe -m scripts.run_native_stream_demo --seed 23 --diagnose-units
```

This adds initial/final `native_unit_diagnostics` and the combined measured
`native_unit_diagnostic_seconds` to its legacy circadian method row. Default
output and other scenario routes are unchanged. The flag requires the `stream`
scenario. This engineering example uses the original legacy BP/CPC learners;
it is separate from the matched four-arm pilot and establishes no comparison
between their learning objectives or computational costs.

Importance is the stored EMA of reward-scaled mean absolute outgoing gradient,
before plasticity scaling. Historical replay can update it; wake-only replay
preserves it. Split children inherit parent importance, and pending-prune decay
attenuates it. It is a model-specific history, not causal feature importance.
Age is the stored native float count of successful epoch-state updates since
birth/reset. Replay that skips epoch-state updates does not increment it. A
split child starts at zero; its parent retains its age. Age is neither elapsed
time nor a substitute for the global wake count.

The reader returns frozen Python records with per-unit IDs/parent IDs and
min/mean/max summaries. Removed parents may still be named by surviving children.
Only the last adaptive hidden layer is supported. The current weight matrix
width is used because initial geometry can be stale after split/prune.
Exact legacy model binding, aligned int64 IDs and finite nonnegative float64
histories are required, with at most 4096 units. Means scale by the maximum to
avoid overflowing valid large values. The adapter checks weight metadata,
not full weight values or complete model validity/provenance. It calls no
model method and retains no model/array references. Keep the model quiescent
during a read; this adds no concurrent snapshot guarantee.

Coupled models still lack corresponding native importance/age fields. Their
pilot entries remain explicitly unavailable. This addition does not alter or
rerun the retained fresh-task experiment.

Verification: 107 focused cases passed; three affected preservation cases were
then requalified after strengthening array-flag checks. Six-path Ruff/format and
Windows/Linux mypy passed. The independently reviewed native CLI run completed
eight updates and two reads: ages 0→4, output 12598 B, measured read/conversion
time 1.62 ms. Full state/RNG remained unchanged around reads. These are scoped
engineering results, with failures and counter limitations retained in
`artifacts/runs/r21-native-units-20261016/RESULTS.md`.

## Structure and extension

```text
src/core/learnability_diagnostics.py   bounded activation and native-unit summaries
src/app/fresh_task_probe.py           disposable learner, reserved roles, budget
src/adapters/learnability_pilot.py     fixed data and existing-stream composition
src/adapters/native_unit_diagnostics.py read-only legacy CPC field binding
scripts/run_learnability_pilot.py     describe/run JSON boundary
tests/test_learnability_diagnostics.py
tests/test_fresh_task_probe.py
tests/test_learnability_pilot.py
tests/test_native_unit_diagnostics.py
tests/test_native_unit_demo.py
```

Core does not know about learners, labels or IO. The app accepts owned learner
ports and `probe_train`/`probe_test` batches, refusing benchmark `final_test`.
The adapter owns native construction and fixed data. No stream runner or learner
algorithm is replaced. See [ADR-0307](adr/ADR-0307-isolate-reserved-fresh-task-probes.md).

Run focused tests with:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_learnability_diagnostics.py tests/test_fresh_task_probe.py tests/test_learnability_pilot.py tests/test_attribution_checkpoint_observer.py
.venv/Scripts/python.exe -m pytest -q tests/test_native_unit_diagnostics.py tests/test_native_unit_demo.py
.venv/Scripts/python.exe -m pytest -q tests/test_learnability_sixteen.py tests/test_learnability_scenario_cli.py tests/test_learnability_pilot.py tests/test_attribution_checkpoint_observer.py
```

Scoped Ruff and Windows/Linux mypy argv and actual outcomes are preserved in
`artifacts/runs/r21-learnability-20261016/`. Freeze a separate protocol before
changing thresholds, exposure, metrics, tasks or seeds; retain all original
outcomes. Add native importance/age support through explicit observations rather
than inferring histories from unrelated fields.

## Opt-in committed histories for coupled PC/CPC

The ordinary coupled models retain their original fields and format-1 snapshots.
Use `TrackedCoupledPredictiveCodingNetwork` or
`TrackedCoupledCircadianPredictiveCodingNetwork` to start native last-hidden
histories at construction. The existing `CoupledLearner` owns and forks either
tracked type, preserving its full history and fixed decay policy.

```python
import numpy as np
from src.core.tracked_coupled_predictive_coding import TrackedCoupledPredictiveCodingNetwork
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.native_unit_diagnostics import read_coupled_unit_diagnostics

native = TrackedCoupledPredictiveCodingNetwork(2, (8, 8), seed=23, history_decay=0.95)
learner = CoupledLearner(native, learning_rate=0.03, inference_steps=4,
                         inference_learning_rate=0.2)
learner.train_batch(np.array([[0.2, -0.4]]), np.array([[1.0]]))
summary = read_coupled_unit_diagnostics(learner.snapshot_state())
assert summary.successful_updates == 1
assert native.read_unit_history().updates == 0  # adapter owns its copy
```

The statistic is an EMA of the absolute **batch-aggregate outgoing coupled-energy
gradient before CPC plasticity gating**. It is not mean absolute per-example
sensitivity: opposite contributions can cancel. It has no reward scaling and is
not causal importance, chemistry, or a validated renewal criterion. PC and neutral
CPC histories match when their native trajectories match. Active CPC records its
own raw gradient along its own trajectory.

Age counts successful native updates since tracked construction, including each
replay call. It counts neither rows nor wall time. Predictions and failed native
calls do not advance it. Parameters, CPC chemistry/gates/count, and prospective
history are validated before publication. Fixed IDs are0..width−1; there is no
replacement, birth log, retrospective history, or unbounded success chain.
The last hidden width is1..4096; each history retains24×width array bytes
(IDs, EMA and ages), excluding metadata, native parameters, snapshots/copies and
validation temporaries. Counts stop at2**53−1 to keep float64 ages exact.

Tracked snapshots use explicit format2 and own their history arrays; full in-memory
restore/fork preserves history, native state and decay binding. Parameter-only
restore preserves history as declared. Complete format1 state cannot initialize
tracked history retrospectively. Existing exact format1 disk codecs and managed
capture schemas **refuse tracked states/models**. No persistence migration, durable
resume or authority extension is provided. The reader checks declared history
geometry and its CPC clock; it does not validate every native parameter, provenance
or recovery authority.

The existing isolated learnability observer keeps unavailable importance/age in
its original untracked protocols. Its explicit tracked mode now validates and
reports native histories before feature/SVD work, using independent tracked
probe templates and copied complete checkpoints. Native histories are counted in
snapshot bytes and the same diagnostic timer. Corrupt tracked state is refused;
no history is inferred for ordinary BP or untracked models.

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --scenario tracked-learnability
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --run --scenario tracked-learnability
```

This fixed scenario already includes rotating replay and sixteen-step probes;
omit single-case flags. All original single/paired descriptors and results remain
available. The accepted [tracked observer verification](reports/tracked-learnability-20261016.md)
retains all original benchmark reports,96probe points and negative endpoints,
while exposing12history records. Original bounded R2.1 implementation criteria
are accepted; renewal, causal utility, persistence and scientific benefit remain
separate unfinished work. Runtime project-completion flags do not certify plan
acceptance; independent implementation evidence and canonical plans do.

The accepted native integration in
`artifacts/runs/r21-coupled-unit-history-20261016/` runs the existing delayed stream
and rehearsal helper:2wake+2replay calls produce ages0/2/4 at wake checkpoints0/1/2.
Ordinary/tracked complete reports and PC/neutral histories match; source learners
remain unchanged. Six streams cost24native calls,36training-row exposures,
36score forwards and96energy evaluations, with zero probes/SVD. This is engineering
correctness evidence, not a comparative learning result or circadian benefit.

Commands, failures, scope and timing are in that stage's RESULTS.md. Add future
renewal only after defining maturity/utility eligibility, replacement bounds and
complete affected-state reset semantics; these histories alone do not implement it.
