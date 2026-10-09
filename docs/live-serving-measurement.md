# Matched actual live serving latency

This engineering measurement observes the real `ResourceSharedRuntime.predict`
path while an independent candidate executes its original native `train_batch`.
It measures serving availability and timing; it does not score scientific utility,
compare native losses, select seeds or establish a general performance advantage.

```text
src/core/serving_latency.py             pure timing/nearest-rank/overlap populations
src/app/live_serving_measurement.py     native facade, finite requests and worker joins
scripts/measure_shared_serving_latency.py reserved two-native fixture/JSON boundary
tests/test_live_serving_measurement.py  pure metrics and deterministic fake concurrency
```

## Fixed local protocol

The prospective artifact `artifacts/runs/r35c-live-serving-20261007/PROTOCOL.json`
is authoritative. Both methods use seed23, data seed9012,input16,hidden64, learning
rate0.03. CPC retains its native two inference steps and rate0.2; backprop retains
its native ordinary-gradient update. No sleep/growth occurs. Training batch8192
and serving batch64 are identical; each of three arrived training pairs has the
same synthetic inputs/targets. Data/actor/source/state bindings are retained.

One declared run in fixed backprop/circadian order: four serving warmups,96 idle
requests,then three blocks of32 actual serving requests alongside exactly one
candidate native update per block. Total ceilings:6 native wakes,392 predictions,
zero sleeps. BLAS thread variables are all1 before NumPy import. Per-method
measurement/budget wall limit20 seconds,worker join/start limit5 seconds,observed
process-RSS ceiling512MiB,sampling interval5ms. In-flight native calls cannot be
preempted by the cooperative gate; an outer55-second process timeout sits within
the separate60-second engineering child ceiling. There is no automatic retry.

## Population and timing definitions

Serving timestamps immediately surround the real shared prediction call, covering
gate admission,input/output copies,actor read lock and original native prediction.
Metadata reads and statistics are outside the interval. Native timestamps surround
the original adapter `train_batch`; readiness notification is outside, and no
event/wait/join synchronization extends native intervals. Instrumentation is a
trusted facade around independent forks; production native equations and policy
are unchanged. These are elapsed dispatch intervals, not exclusive CPU time.

Retain every warmup/request/native interval,poll,ID/status and sharing/resource
observation. Report nearest-rank p50/p95/max with counts for:

- all96 idle requests;
- all96 requests in the shared training phase;
- requests with nonzero timestamp overlap with a completed native interval;
- requests fully contained in a completed native interval.

Overlap depends on phase/block/timestamps only,never on latency or whether an
outcome looks favorable. Requests outside native intervals remain in the phase
population and raw records. Empty populations are unavailable,not zero latency.
Acceptance requires at least20 fully-contained requests per method,all three
native updates,complete quotas/IDs/stable actor/source binding and resource gates.
Insufficient coverage/slowdown/no advantage are retained outcomes. The protocol
does not rerun/tune a method to obtain coverage or improve a comparison.

## Reproducibility and failure boundaries

Complete fake/core/regression/type/static/AST checks and exact whole-source binding
precede native launch. The fixture creates an exclusive `experiment.reserved`
before work; an existing reservation refuses a repeat. Raw per-method files are
published before proceeding to the next declared method. Failure preserves those
files,partial timings,exception and live-worker status. A daemon worker is never
silently restarted: callers must establish its actual termination before any new
work. This harness neither certifies crash recovery nor grants science authority.

The existing cooperative resource gate is exercised without changing it. Record
actual admitted/deferred work and quiescence. A resource probe that reports true
is a declared fixed availability control; real process-RSS is separately observed
through the original budget/sampler. Observed RSS does not promise hard allocation
limits. Fixed method order,one machine,one small run and instrumentation limit
what conclusions these timings support; no result selection or broad speed claim.

## Runnable fake example

```python
from src.core.serving_latency import NativeCallTiming, ServingRequestTiming, summarize_latencies, select_native_overlap
requests = (
    ServingRequestTiming("shared",0,0,10,20,"actor-0",True),
    ServingRequestTiming("shared",0,1,20,40,"actor-0",True),
)
calls = (NativeCallTiming(0,5,25,True),)
all_requests = summarize_latencies(requests)
contained = select_native_overlap(requests,calls,fully_contained=True)
assert all_requests.count == 2 and all_requests.p95_ns == 20
assert contained == (requests[0],)
assert summarize_latencies(()).p50_ns is None
```

## Commands and next extension

```powershell
python -m pytest -q tests/test_live_serving_measurement.py
python -m ruff check src tests scripts
python -m ruff format --check src/core/serving_latency.py src/app/live_serving_measurement.py scripts/measure_shared_serving_latency.py tests/test_live_serving_measurement.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

The new native fixture is run once through the stage runner after correctness and
source binding. Its CLI is `python -m scripts.measure_shared_serving_latency
--protocol <frozen-protocol> --output-dir <new-owned-directory>`; establish a new
explicit work/storage allowance before any later run. Preserve all original raw
records when adding another workload/platform. Durable checkpoint restart remains
R3.5b2; original failure/long-run validation remains R3.7 after bounded engineering
measurement,with scientific guards and human deferrals still in force.
