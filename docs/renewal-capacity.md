# Capacity observations around bounded renewal

The existing attribution runner accepts `renewal=StreamUnitRenewal(...)` together
with `capacity=StreamCapacity(...)` when the capacity helper has no scheduled
split. Both helpers must bind to the same original source, wake budget and total
update count. `measure_numpy_capacity` already handles complete renewable native
snapshots, including fixed-size unit history and CPC chemistry arrays.

```python
from src.adapters.numpy_capacity import measure_numpy_capacity
from src.app.stream_capacity import StreamCapacity
from src.app.stream_unit_renewal import StreamUnitRenewal

# learner is an existing CoupledLearner owning a renewable PC/CPC model.
# budget is the original fresh ToyBudgetSession used by run_attribution_stream.
renewal = StreamUnitRenewal(
    learner, budget, every_updates=2, total_updates=4, max_replacements=1,
)
capacity = StreamCapacity(
    learner, budget, total_updates=4, measure_state=measure_numpy_capacity,
    max_parameters=13, max_state_array_bytes=224,
)
# Pass both helpers to the existing run_attribution_stream call.
```

The example ceilings fit D2/H1/W3: thirteen float64 parameters (104 bytes),
72 bytes of renewable history and up to 48 bytes of CPC chemistry/gates. Choose
ceilings prospectively for the actual declared model; observed results never
increase an existing allowance. No dependency or environment variable is added.

## Ordering and cost

At each completed wake the runner first records `post_wake` capacity using the
prior measured model for its wake-weighted means. Renewal then selects from the
current candidate using its original policy and permanently charges replacement
units before preparation. Each reached scheduled renewal boundary gets one
`post_consolidation` observation, including zero selection and exhausted
allowance. Those observations affect current usage/peaks, not wake counts, row
counts or mean weights. Fixed-size renewal must leave capacity equal; a different
usage stops the observer rather than admitting unexplained growth.

The capacity report retains native parameter/auxiliary bytes and measurement
seconds. The renewal report retains charged/attempted/returned/committed units,
per-event `work_seconds` and their total `event_seconds`. Event time includes
selection, preparation, reset and publication, including failures and scheduled
allowance checks. It is separate from measurement time and native wake counts;
neither timing is a causal overhead or comparative efficiency estimate.

## Failure and authority

Original learner/helper identities, schedule, limits, clock and birth budget are
checked before new work. Permanent charges must match recorded event charges;
unobserved counter resets or allowance changes are refused. Measurement receives
only a detached state and cannot legitimately change any of these bindings.

A renewal failure stops the stream. Available post-event state is read without
another budget-admission callback; that read does not train, reset a unit or
renew a budget. It still checks original bindings and actual completed receipts.
The original renewal error and any issued publication receipt remain primary.
If this final observation also fails, `capacity_observation_error` records that
secondary error on the already identified event. Known wake work, returned reset
work, measured peaks and all permanent charges remain available in helper reports.

Only scalar usage samples and bounded event slots are retained: at most one
initial sample, one per wake and one post-consolidation sample per wake. Source,
actor and final-scoring learners remain isolated from candidate reset. Final
evaluation is unavailable to selection, measurement and training.

## Limits and extension

Replay with these helper pairs, simultaneous scheduled split plus renewal, and
continuation of renewal/capacity helpers remain unsupported. Ordinary renewable
snapshots and raw structural BP/PC snapshots are separate native formats and
objectives; equal stored counts do not make them interchangeable or equal work.

The engineering qualification verifies identical measured/unmeasured trajectories
for PC, neutral CPC and active CPC. It is not fresh-task acquisition, useful
adaptation, a full R2.6 comparison or a circadian advantage. Existing negative
renewal and matched-baseline experiments remain unchanged.

The original R2.6 comparison criteria are now independently accepted in the
[full capacity comparison](capacity-comparison.md), which composes these helpers
with the existing runner. Its 23 frozen cohorts report actual parameters, bytes,
work and outcomes, including separate pruning-validation costs. All four prunes
rejected and all nine renewal events selected zero units. Those negative results
establish no scientific benefit; the qualification above retains its own scope.

## Checks

```powershell
python -m pytest -q tests/test_renewal_capacity.py tests/test_stream_unit_renewal.py tests/test_stream_capacity.py tests/test_capacity_pilot.py
python -m ruff check src/app/stream_capacity.py src/app/stream_unit_renewal.py src/app/coupled_attribution.py tests/test_renewal_capacity.py
python -m ruff format --check src/app/stream_capacity.py src/app/stream_unit_renewal.py src/app/coupled_attribution.py tests/test_renewal_capacity.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/app/stream_capacity.py src/app/stream_unit_renewal.py src/app/coupled_attribution.py tests/test_renewal_capacity.py
```

Repeat type checking with `--platform linux` for that target. Scoped checks do
not establish repository-wide or release readiness.

Executed evidence: [results and exact commands](../artifacts/runs/r26-renewal-capacity-20261016/RESULTS.md)
and [independent review](../artifacts/runs/r26-renewal-capacity-20261016/REVIEW.md).
Qualification includes 121 cases/363 passed phases (120 + one closing fake-clock
control); the original three native pairs were not repeated.
