# Capacity at training boundaries

The existing attribution runner accepts an optional `StreamCapacity` helper.
It measures the committed candidate at initialization, after successful wake
calls and after its one optional scheduled split. No final data reach the helper.

```powershell
python -m scripts.run_stream_rehearsal_pilot --describe --scenario capacity
python -m scripts.run_stream_rehearsal_pilot --run --scenario capacity
python -m pytest -q tests/test_capacity_cost.py tests/test_stream_capacity.py tests/test_capacity_pilot.py
```

The fixed six-arm engineering example uses PC, neutral CPC and active CPC with
four two-row delayed batches. Each method has a growing width2→4 candidate and
a fixed width4 reference initialized from the same isometric split. It has
24 native updates, 48 row exposures, 36 final-score forwards and six split-kernel
calls (three fixed-reference initializations and three scheduled transitions).
Limits are CPU1, GPU0, 60 seconds, 256MiB RSS and 2MiB output. Declare a new local
execution envelope before another run. No new dependencies or environment
variables are required; the CLI sets BLAS threads before importing NumPy.

## Modules and boundaries

```text
core/capacity_cost.py        # immutable units and pure dense parameter count
adapters/numpy_capacity.py   # supported snapshot array validation/counting
app/stream_capacity.py       # original-budget runtime observations and split
adapters/capacity_pilot.py   # frozen native/data binding and engineering report
```

Core owns `CapacityUsage(parameters, parameter_bytes, auxiliary_bytes)` and
`count_dense_parameters(input_dim, hidden_widths)`. All supported parameters are
float64, so parameter bytes equal eight times the actual dense weight/bias count.
The adapter reads ordinary/tracked/renewable coupled and matched BP snapshots,
raw scaling state, and structural/topology envelopes. Chemistry, gates, unit
IDs, importance and age arrays count as auxiliary state. Invalid or unsupported
types refuse explicitly. Measurement does not construct/restore native models,
materialize effective weights, run inference or train a learner.

The app takes a measurement callable rather than importing an adapter:

```python
helper = StreamCapacity(
    learner, original_budget, total_updates=4,
    measure_state=measure_numpy_capacity,
    max_parameters=17, max_state_array_bytes=200,
    split_at_update=2, max_width=4,
)
result = run_attribution_stream(
    learner, training, heldout, checkpoint_updates=(0, 2, 4),
    label_delay=2, budget=original_budget, capacity=helper,
)
```

The helper must be born with fresh counters and bind to its original source,
budget, clocks, measurement callable, ceilings and schedule. It is single use.
The existing runtime owns native training, candidate state and publication;
the helper stores bounded scalar samples and receipts. Default execution with
`capacity=None` keeps its original report schema and continuation behavior.
Observation-only capacity can accompany [bounded renewal](renewal-capacity.md),
with separate pre/post-event samples and permanent charges. Scheduled split plus
renewal, capacity/rehearsal composition and capacity continuation remain unsupported.

## Means, peaks and costs

Means weight the model actually used for each **successful native wake call**.
They do not weight snapshots, rows, elapsed time or post-split samples. With two
width2 wakes followed by two width4 wakes, the parameter mean is
`(9 + 9 + 17 + 17) / 4 = 13`, with a committed peak of 17. A fixed width4 model
has mean and peak 17. This remains true with unequal batch sizes; row exposures
are a separate count. Initial/post-split samples affect maxima and future wakes.

Parameter-byte means are 104 versus 136. CPC's two native width-sized auxiliary
arrays add means 48 versus 64 bytes, giving committed native-array means 152
versus 200 and peak 200. These are stored field payloads for the candidate.
They exclude staging/checkpoint/scorer/actor/source copies, latent/gradient
arrays, metadata, Python overhead and process memory. Physical RSS is measured
separately. Duplicated units count as stored parameters; counts do not measure
functional rank or effective representational capacity.

Reports keep completed calls/row exposure, requested/attempted/committed split
stages, per-boundary usage, peaks, wake-weighted means and measurement seconds.
Measurement time includes the helper's snapshot/copy/validation and failed
observations. Pilot checkpoint-reporting time is separate; full wall time also
includes all other work. Counts and timings do not establish matched FLOPs or
efficiency. Unknown failed native attempts are not inferred to cost zero from a
completed-wake counter.

## Growth admission and failures

The one scheduled split uses the existing structural learner and split
consolidator. Dense target parameters, CPC auxiliary bytes and maximum width are
checked before invoking the expanding transform. The actual committed result is
measured again. Source/actor learning is isolated from candidate growth; no
promotion, final-score selection, fresh budget or retry is introduced.

Known completed wake work remains recorded if a measurement or split fails.
A runtime receipt determines whether publication happened. If the final budget
check fails after publication, the report retains that receipt and records the
available post-split usage without renewing admission or continuing training.
Failure of that read is recorded too; the primary failure is preserved.

## Scientific scope and extension

This example verifies accounting and guarded scheduled growth. The fixed
reference has replicated units; the native split maps precisions and changes
width-dependent raw/effective update factors. Equal final stored width and
initial function do not establish equal compute, effective capacity or a fair
full mechanism comparison. All factors and final scores are reported without
using them to tune schedules or budgets.

The original **R2.6 comparison criteria are now independently accepted** in the
[full capacity comparison](capacity-comparison.md), which composes the existing
helpers and [structural matched BP port](structural-matched-bp.md). Its 23 frozen
cohorts report actual costs and outcomes. All four guarded prunes rejected and
renewal selected zero units at all nine events; these results establish no
scientific benefit or efficiency advantage. The smaller example above retains
its original scope and limitations. Preserve defaults, negative results and
final-data isolation when extending the helpers.
