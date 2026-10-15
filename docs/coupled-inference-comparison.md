# Allocated-time coupled inference comparison

## Opt-in granularity-aware protocol

```powershell
python -m scripts.run_coupled_inference_comparison --granularity-aware --describe
python -m scripts.run_coupled_inference_comparison --granularity-aware > comparison-v2.json
```

The new flag selects the separately frozen
`coupled_allocated_time_inference_granularity_v2` protocol:30ms allocated training
per case, inclusive30..33ms observed window, and an inclusive3ms observed complete
iteration ceiling. It preserves the same six cases, paired initial tensors,
data, native policies, rates, metrics and failure vectors. The original default
below and its incomplete result remain unchanged. A run is a local experiment,
requiring its own prospective budget; --describe performs no training.

Why30ms: using only timing fields from all six retained v1 cases, the largest
`max(call_seconds) + window_seconds - sum(call_seconds)` envelope was
2.32390001474414ms. Attributing every noncall cost to the largest call is
conservative for those retained intervals. Round upward to3ms and allocate ten
times that ceiling. This justifies the new granularity without using final
scores or changing the10% overshoot rule; it cannot guarantee future scheduling.

`run_timed_inference_case(..., max_iteration_seconds=...)` validates a finite
positive native ceiling no greater than10% of the allocated window before copies,
clocks or callbacks. Opt-in intervals partition the timed procedure: the first
starts at window_start, later native intervals start at loop boundaries, and the
last extends through the enclosing terminal window-end read. Native calls,
copies, resource checks, records, cap checks and bookkeeping stay inside the
window. Common qualification/scoring follows that same terminal marker and is
included in total elapsed. The ceiling is observed qualification, not preemption.

An exceeded interval fails the case with no initial/final scores, retaining the
source, committed records and known measured maximum. Unknown failed-call timing
is not reported as zero. Opt-in results add `max_iteration_seconds` and
`observed_max_iteration_seconds`; default None retains v1 behavior. Summaries
accept explicit `window_seconds` and `max_iteration_seconds` and reject complete
rows with missing/mismatched limits or unknown/excessive observed maxima. Their
defaults still require the original10ms window.

The new protocol caps80updates per case/480total and preserves60s execution,
256MiB sampled process RSS, CPU1 and32MiB output. Engineering calls are separately
reserved within the original512call ceiling. No calibration, warmup, retry or
outcome-dependent adjustment is part of it. The prospective contract and actual
receipts live in artifacts/runs/coupled-inference-granularity-20261015.

The [v2 report](reports/coupled-inference-granularity-20261015.md) retains the
incomplete outcome: fixed seed59 exceeded3ms, its scores are null, and no stopping
savings are demonstrated. The boundary passes115focused tests; that engineering
result does not qualify the failed scientific case or complete parent R1.5.

## Original protocol

Run from the repository root:

```powershell
python -m scripts.run_coupled_inference_comparison --describe
python -m scripts.run_coupled_inference_comparison > comparison.json
```

The first command performs no training. The second runs six fixed PC cases once:
seeds 23/41/59, fixed six-step inference versus the unchanged default guarded
policy, with exact shared initial tensors and identical available training data.
It uses one CPU thread, at most 480 updates, 60 seconds, sampled process RSS
256 MiB and output 32 MiB. It exits 2 when any case/global limit is incomplete.
An invocation is an experiment; consult its prospective budget before running.

Each case has the same **10 ms allocated wall-time training window**. The observed
window must be between 10 and 11 ms inclusive. Native updates cannot be preempted.
All in-loop resource/deadline checks, records and bookkeeping are timed alongside
the entire native call, including stopping arithmetic and parameter validation.
Initial setup/model/data copies and deferred scoring are outside that window,
and count in total elapsed time. Fresh training-buffer copies on each call are
inside both call and window timing. Whole call times, window/overshoot and last iteration are
retained separately. Diagnostic counts do not enumerate every scalar operation.

Why this: counting just accepted latent steps omits policy and orchestration
costs. Equal allocated time permits different update/row-exposure counts. It is
neither equal CPU time nor exactly equal observed wall time; the explicit 10%
qualification and very short windows limit interpretation.

The application validates roles, unique IDs and disjoint buffers, and copies
native PC state/policy/data before external callbacks. Only training labels reach
inference. After all training in that case, detached initial/final states make
target-free heldout predictions. Common clipped BCE and accuracy use final labels
only in metrics. Scores never affect subsequent frozen cases. Any failure,
including a late scoring/resource/clock failure, clears both score dictionaries.

Every planned case is retained. Overrun, no commit, update-cap exhaustion before
the deadline, or numerical/resource/clock errors produce an unscored failed case;
there is no retry or extended window. Known completed records remain available;
unknown failed-call work is not fabricated as zero. Complete-vector summaries
retain null failed seeds and report adaptive minus fixed differences. Nominal
marginal 95% t intervals use df2/critical4.303; they mix initialization and runtime
variability on fixed data, with no simultaneous or general speed/learning claim.

## Structure and extension

`app/timed_coupled_inference.py` owns one copied case and isolation;
`app/coupled_inference_summary.py` owns pure complete-vector summaries;
`adapters/coupled_inference_comparison.py` owns fixed data/composition/resources;
`scripts/run_coupled_inference_comparison.py` owns CLI/JSON/thread configuration.
There are no new dependencies or environment settings beyond the CLI's local
thread limits. Existing native learners/default inference are unchanged.

To compare another window or policy, add a separate prospectively frozen protocol
and budget; preserve this protocol and its complete failure vectors. Do not choose
thresholds/windows/seeds after looking at final scores. Focused checks:

```powershell
python -m pytest tests/test_timed_coupled_inference.py tests/test_coupled_inference_comparison.py -q -o addopts= -p no:cacheprovider
```

Use scoped Ruff, format and mypy commands recorded in the development log.
The [first report](reports/coupled-inference-comparison-20261015.md) preserves a
failed timing row and no demonstrated stopping benefit; no retry was performed.

For the new boundary, also run
`python -m pytest -q tests/test_timed_inference_granularity.py tests/test_inference_granularity_protocol.py`.
See ADR0271 for the timing-only protocol decision. Extend timing experiments only
with a separately frozen protocol; preserve old failures and all final-data roles.
