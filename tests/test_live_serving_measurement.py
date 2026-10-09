"""Pure timing populations and deterministic actual serving/native concurrency."""

from dataclasses import replace
from threading import Event, Lock

import pytest

from src.app.live_serving_measurement import (
    MeasurementIncomplete,
    MeasurementLimits,
    NativeTimingObserver,
    ObservedLearner,
    measure_shared_serving,
)
from src.core.serving_latency import (
    NativeCallTiming,
    ServingRequestTiming,
    summarize_latencies,
    select_native_overlap,
)
from src.core.learner_ports import TrainingDiagnostic


def request(start=10, end=20, index=0):
    return ServingRequestTiming("shared", 0, index, start, end, "actor-0", True)


@pytest.mark.parametrize("count,p50,p95", [(1, 1, 1), (2, 1, 2), (20, 10, 19), (21, 11, 20)])
def test_should_use_declared_nearest_rank_quantiles_without_discarding_tail(count, p50, p95):
    rows = tuple(request(0, i, i) for i in range(1, count + 1))
    summary = summarize_latencies(rows)
    assert (summary.count, summary.p50_ns, summary.p95_ns, summary.max_ns) == (
        count,
        p50,
        p95,
        count,
    )


def test_should_report_unavailable_empty_population_instead_of_zero_latency():
    summary = summarize_latencies(())
    assert summary.count == 0 and summary.p50_ns is None and summary.p95_ns is None


@pytest.mark.parametrize(
    "change",
    [
        {"start_ns": True},
        {"end_ns": -1},
        {"end_ns": 9},
        {"phase": "unknown"},
        {"actor_version": " "},
        {"completed": 1},
        {"block": None},
    ],
)
def test_should_refuse_invalid_request_metadata(change):
    with pytest.raises(ValueError):
        replace(request(), **change)


def test_should_classify_actual_overlap_and_containment_only_from_full_timestamps():
    calls = (NativeCallTiming(0, 15, 25, True),)
    rows = (request(10, 15, 0), request(10, 20, 1), request(16, 24, 2), request(25, 30, 3))
    assert select_native_overlap(rows, calls) == (rows[1], rows[2])
    assert select_native_overlap(rows, calls, fully_contained=True) == (rows[2],)
    assert len(rows) == 4


def test_should_refuse_summarizing_failed_requests_and_exclude_failed_native_intervals():
    with pytest.raises(ValueError):
        summarize_latencies((replace(request(), completed=False),))
    assert select_native_overlap((request(),), (NativeCallTiming(0, 0, 30, False),)) == ()


class Clock:
    def __init__(self, release=None, counter=None):
        self.tick = 0
        self.lock = Lock()
        self.release, self.counter = release, counter

    def __call__(self):
        with self.lock:
            self.tick += 1
            tick = self.tick
            if self.counter is not None and self.counter[0] == 5:
                self.release.set()
            return tick


class Learner:
    def __init__(self, release=None, counter=None, state=0, fail=False):
        self.release, self.counter, self.state, self.fail = release, counter, state, fail

    def fork(self):
        return Learner(self.release, self.counter, self.state, self.fail)

    def train_batch(self, features, targets):
        if self.release is not None:
            assert self.release.wait(1)
        if self.fail:
            raise RuntimeError("native failed")
        self.state += 1
        return TrainingDiagnostic("fixture", float(self.state))

    def predict(self, features):
        if self.counter is not None:
            self.counter[0] += 1
        return self.state

    def snapshot_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state


def fixture(source, clock=None):
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
    from src.core.resource_sharing import SharingLimits

    observer = NativeTimingObserver(clock=clock or Clock())
    runtime = ActorShadowRuntime(
        ObservedLearner(source, observer),
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=LogicalClock(3),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
    )
    runtime.record_experience(
        Experience("s1", "e1", 1, "actor-0", 0, "train", ExperiencePermissions(training=True))
    )
    runtime.record_label(LabelArrival("label", "s1", "e1", 3, "actor-0", 0))
    gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
    return ResourceSharedRuntime(runtime, gate), observer, runtime


def test_should_measure_actual_actor_requests_while_candidate_native_body_is_running():
    release = Event()
    source = Learner(release, [0])
    shared, observer, runtime = fixture(source, Clock(release, source.counter))
    result = measure_shared_serving(shared, 0, observer, MeasurementLimits(2, 1, 3, 1, 5))
    assert len(result.requests) == 5 and len(result.native_calls) == 1
    assert (
        len(select_native_overlap(result.requests[2:], result.native_calls, fully_contained=True))
        == 3
    )
    assert result.polls[0].updates[0].sample_id == "s1"
    assert runtime._budget.updates_completed == 1 and source.state == 0
    assert shared._sharing.snapshot().admitted_updates == 1


def test_should_preserve_failed_native_interval_and_partial_requests():
    shared, observer, _ = fixture(Learner(fail=True))
    with pytest.raises(MeasurementIncomplete) as error:
        measure_shared_serving(shared, 0, observer, MeasurementLimits(2, 1, 3, 1, 5))
    assert not error.value.live_worker
    assert len(error.value.native_calls) == 1 and not error.value.native_calls[0].completed
    assert error.value.requests[:2][0].phase == "idle"


def test_should_report_live_owned_worker_on_timeout_without_silently_retrying():
    release = Event()
    shared, observer, _ = fixture(Learner(release))
    try:
        with pytest.raises(MeasurementIncomplete) as error:
            measure_shared_serving(shared, 0, observer, MeasurementLimits(1, 1, 1, 0.02, 5))
        assert error.value.live_worker and error.value.active_native is not None
    finally:
        release.set()


@pytest.mark.parametrize(
    "change",
    [
        {"idle_requests": 0},
        {"training_blocks": True},
        {"requests_per_block": -1},
        {"worker_timeout_seconds": 0},
        {"max_wall_seconds": float("inf")},
    ],
)
def test_should_refuse_unbounded_or_invalid_measurement_limits(change):
    with pytest.raises(ValueError):
        replace(MeasurementLimits(2, 1, 3, 1, 5), **change)


def test_should_refuse_observer_reuse_without_resetting_captured_native_work():
    shared, observer, _ = fixture(Learner())
    measure_shared_serving(shared, 0, observer, MeasurementLimits(2, 1, 3, 1, 5))
    with pytest.raises(ValueError, match="fresh"):
        measure_shared_serving(shared, 0, observer, MeasurementLimits(2, 1, 3, 1, 5))


def test_should_refuse_aliasing_native_fork_before_exposing_actor_or_candidate():
    class AliasedLearner(Learner):
        def fork(self):
            return self

    with pytest.raises(ValueError, match="distinct"):
        ObservedLearner(AliasedLearner(), NativeTimingObserver()).fork()


def test_should_stop_cli_before_next_method_when_owned_worker_is_still_live(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import measure_shared_serving_latency as cli

    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "methods": ["backprop", "circadian"],
                "method_order": ["backprop", "circadian"],
                "reruns_allowed": 0,
            }
        ),
        encoding="utf8",
    )
    (tmp_path / "source-binding.json").write_text("{}", encoding="utf8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["measurement", "--protocol", str(protocol), "--output-dir", str(tmp_path / "run")],
    )
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(cli, "verify_binding", lambda path, protocol: None)
    calls = []

    def incomplete(method, protocol):
        calls.append(method)
        return {
            "method": method,
            "PASS": False,
            "status": "incomplete",
            "live_owned_worker": True,
            "active_native": (0, 1),
            "native_calls": [],
            "requests": [],
            "warmups": [],
        }

    monkeypatch.setattr(cli, "measure_method", incomplete)
    assert cli.main() == 1 and calls == ["backprop"]
    saved = json.loads((tmp_path / "run" / "result.json").read_bytes())
    assert saved["methods"] == ["backprop"] and saved["native_wake_attempts"] == 1
