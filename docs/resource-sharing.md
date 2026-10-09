# Cooperative serving priority

R3.5a adds an opt-in shared boundary around the existing actor/candidate runtime.
It admits bounded serving requests and defers new candidate updates under serving
contention, manual pause, another active update, a lifetime work quota or an attached
resource probe. Native equations, roles, clocks, diagnostics and final-data rules
remain the existing contracts. Complete checkpoint/restore is R3.5b; actual idle
and training serving p50/p95 measurements are R3.5c. Parent R3.5 stays open.

## Structure and boundaries

```text
src/core/resource_sharing.py       limits and detached admission/poll/snapshot records
src/app/resource_sharing.py        serving/work gate and actual actor/runtime wrapper
src/app/experience_inbox.py        optional bounded drain and per-update admission context
src/app/actor_shadow.py            forwards compatible optional update admission
tests/test_resource_sharing.py     deterministic priority/defer/native parity controls
docs/adr/ADR-0203-admit-work-at-native-update-boundaries.md
```

Core has no model, resource sampler, clock, IO or scheduler. The app composes native
update, actor serving and trusted resource callbacks; no background worker or new
dependency is introduced. Inner modules import no infra or adapters. An outer
caller controls event arrival, threads, monitoring and when to poll training.

## Supported shared path

Use `ResourceSharedRuntime(runtime, gate)` for every participating `predict`, cached
`serve`, and `train_ready` call. Direct calls to the original runtime/actor continue
to implement their original APIs and bypass this optional shared admission policy.
Cached `serve` requires the R3.4b promotable actor and preserves its atomic complete
generation frame. Uncached `predict` supports both actor types.

`SharingLimits` declares:

- `max_serving_requests`: positive capacity for active **and queued** wrapper calls.
  Admit before copying features or waiting for the actor read gate. Reject overload
  explicitly; outer code may defer the request.
- `max_admitted_updates`: nonnegative lifetime quota for admitted update attempts,
  including failures after admission. Zero disables new work while serving remains
  available. Empty polls and denied requests consume no admission.
- `max_updates_per_poll`: positive upper bound on native updates per polling call.
  The next poll continues the remaining ready pairs in arrival/identity order.

Serving admission takes no candidate gate and never waits for active training.
Training checks manual pause, serving count, active training, and lifetime quota
under a short lock. Only if those permit work, call `resource_available()` **outside**
the lock, require an exact bool, and recheck contention/pause/quota afterward. A
serving arrival or pause during the probe wins that second check. False defers work;
probe failure/malformed output propagates before work. Attach a trusted resource
monitor appropriate to the outer application's declared limits. Existing native
budget wall/RSS pre/post checks still apply independently.

Charge the attempt and mark training active before native payload copy/update.
Release the active flag on every exception/cancellation; never refund quota after
an admitted failure. Default inbox `drain()` retains its previous complete ready
drain behavior. Optional `max_updates=` bounds each call, while
`before_each_update=` supplies an exact-bool context around each eligible native
update. A denial stops the current poll before its next payload copy/native call.
Future labels remain pending; duplicate/applied IDs and diagnostics are retained.
Original native failure poisoning and late budget-stop completed receipts remain.

## Pause, observations and limits

`gate.pause()` defers new work, and `gate.resume()` permits admission checks again.
An active native update finishes its complete boundary; this is cooperative pause,
with no hard preemption or rollback of work already admitted. A serving request
arriving just after training admission may overlap that call. Native GIL/CPU/memory
contention can still increase serving latency; the subsequent matched live timing
task must measure it. The admission gate alone establishes no latency improvement.

`SharingSnapshot` detaches configuration/counters/pause/active flags and finite
per-reason deferral counts. `TrainingPoll` contains committed update receipts and
the actual refusal reason, if one occurred. `deferred_reason=None` does not assert
that all future/pending work is drained, especially when a poll limit is reached.
Snapshots/partial polls provide observations, **not** a durable checkpoint or
authority to reset cumulative budgets. Native attempt limits, outer event bounds
and bounded requests constrain work/records; arbitrary payload bytes and resource
callbacks are not hard allocation or OS priority guarantees.

All participating runtimes must use their owned candidate/budget and this shared
gate consistently. The candidate owner still refuses nested/concurrent operations
as `busy`; the caller can defer at that boundary. Role permissions, monotone logical
arrival ticks and matched native policies are preserved. No final/outer label is
released, scored or trained through a sharing admission.

## Local example

This fixed fixture demonstrates admission/cursors only, without a research study.

```python
from src.app.actor_shadow import ActorShadowRuntime
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits

class Learner:
    def __init__(self, state): self.state = list(state)
    def fork(self): return Learner(self.state)
    def snapshot_state(self): return self.state[:]
    def restore_state(self, state): self.state = list(state)
    def predict(self, features): return self.state[:]
    def train_batch(self, features, targets):
        self.state = list(targets)
        return TrainingDiagnostic("fixture_native_v1", float(sum(self.state)))

clock = LogicalClock()
runtime = ActorShadowRuntime(Learner([2,2]), actor_version="actor-0",
    candidate_version="candidate-0", clock=clock,
    budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda:0.0))
gate = ServingPriorityGate(SharingLimits(2,2,1), resource_available=lambda:True)
shared = ResourceSharedRuntime(runtime, gate)
for sample, targets in [("s1",[8,8]),("s2",[9,9])]:
    runtime.record_experience(Experience(sample,"e1",1,"actor-0",["x"],"train",
        ExperiencePermissions(training=True)))
    runtime.record_label(LabelArrival("label-"+sample,sample,"e1",3,"actor-0",targets))
clock.advance_to(3)
gate.pause()
assert shared.train_ready().deferred_reason == "paused"
assert shared.predict(["x"]).prediction == [2,2]
gate.resume()
first = shared.train_ready()
assert [u.sample_id for u in first.updates] == ["s1"]
with gate.serving():
    assert shared.train_ready().deferred_reason == "serving_active"
second = shared.train_ready()
assert [u.sample_id for u in second.updates] == ["s2"]
assert runtime.candidate_snapshot().state == [9,9]
assert shared.predict(["x"]).prediction == [2,2]
assert gate.snapshot().admitted_updates == 2
```

## Verification and next extension

```powershell
python -m pytest -q tests/test_resource_sharing.py tests/test_experience_inbox.py tests/test_actor_shadow.py -k "not apply_real_native_cpc_consolidation"
python -m ruff check src tests scripts
python -m ruff format --check src/core/resource_sharing.py src/app/resource_sharing.py src/app/experience_inbox.py src/app/actor_shadow.py tests/test_resource_sharing.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

R3.5b must persist/restore the supported complete native/candidate, inbox source/
label/applied/duplicate/arrival/stopped cursors, consolidation history, cumulative
budget and sharing admission/pause state. Specify supported clock/resource adapters
and reject stale/busy/corrupt/unsupported snapshots without exposing partial native
restore or replaying consumed work. Preserve time/work quotas rather than renewing
them on resume. Keep actor generation and pending promotion authority separate.

Only after those correctness gates pass, R3.5c measures actual wrapper actor calls
while idle and while candidate work runs, with fixed identical workloads, policies,
seeds, measurement/count/sample paths and finite budgets for both native methods.
Record raw samples/counts/p50/p95 and deferred work; negative slowdown/no advantage
is valid. Existing guard timing cannot substitute for these live measurements.


## Complete inbox capture prerequisite

R3.5b1 adds [validated full inbox cursor capture](inbox-cursors.md). It preserves pending/future/label-first and consumed/applied histories with owned payloads and observed work/time/stopped state. These records remain observations. R3.5b must still bind them to native/candidate/actor generation, cumulative budget/time/resource/sharing ownership, prepare complete restore off serving and retire the old owner without duplicate work or quota renewal. The parent and live timing gates remain open.


### Complete candidate checkpoint ownership

Supported same-process full candidate handoff is implemented through app/candidate_checkpoint.py; see docs/candidate-checkpoints.md and ADR-0205. Retain original cumulative budget/clocks/RSS sampler/resource gate and stable actor, restore independent native state plus full inbox/consolidation histories, retire old owner and invalidate its promotion authority. Identity tokens and preparation work are bounded. R3.5b acceptance remains pending current full validation; durable process recovery and actual live latency remain unfinished.

R3.5b current supported owned checkpoint acceptance:371 tests,647-file Windows/Linux types,full scoped/static/source/resource/guide gates pass. Evidence:artifacts/runs/r35b-owned-handoff-20261007/. Actual live R3.5c latency and durable R3.5b2 recovery remain unfinished.


### Actual live serving measurement

The generic app native observer/shared-request harness and pure core timing/
nearest-rank/overlap records are documented in docs/live-serving-measurement.md
and ADR-0206. The reserved script boundary executes the finite declared matched
two-native protocol after full correctness/source binding. Retain all raw requests,
native windows and incomplete worker status; no automatic repeat/outcome filtering.
Current measurement acceptance is pending; no scientific advantage is claimed.

R3.5c/original R3.5 current acceptance:396 cases,full651-file platform types/static/source/resource gates and one reserved actual native serving run pass;96/96 shared requests per method fully native-contained. Negative circadian p95 slowdown retained. Evidence:artifacts/runs/r35c-live-serving-20261007/measurement-summary.md. Next R3.6 privacy/replay lifecycle;durable R3.5b2 and broader guards remain open.
