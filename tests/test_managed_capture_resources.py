"""NEW original-sampler graphs, bounded contention, no training/native restore."""

from pathlib import Path
from threading import Event, Thread, Lock
from unittest.mock import patch
import json

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as module
from src.adapters.numpy_composite_capture import (
    capture_numpy_managed_composite,
    copy_numpy_composite,
)
from src.adapters.numpy_learners import BackpropLearner, make_managed_data_lifecycle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import (
    ToyBudgetSession,
    ToyExecutionBudget,
    ToyExecutionProgress,
    ToyExecutionStopped,
    ToyProcessRssUnavailable,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.data_lifecycle import (
    LifecycleDeclaration,
    DataProvenance,
    DataConsent,
    LifecycleLimits,
)
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import LogicalClock, Experience, ExperiencePermissions, LabelArrival
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits
from src.shared.process_memory import ProcessRssSampler
from test_managed_composite_capture import LIMITS, values


class Reading:
    def __init__(self, value):
        self.value, self.calls = value, 0
        self.callback = None

    def __call__(self):
        self.calls += 1
        if self.callback is not None:
            self.callback()
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


@pytest.fixture(scope="module")
def ledger(request):
    counts = dict(
        graphs=0,
        readers=0,
        contenders=0,
        copies=0,
        updates=0,
        predictions=0,
        snapshots=0,
        restores=0,
        sleeps=0,
        structural=0,
        workers=0,
        live_threads=0,
    )
    yield counts
    assert counts["graphs"] <= 32 and counts["live_threads"] == 0
    p = Path(request.config.option.basetemp) / "resource-native-work.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(counts, indent=2), encoding="utf8")


@pytest.fixture
def graph(monkeypatch, ledger):
    for name in ("train_batch", "predict", "snapshot_state", "restore_state"):
        monkeypatch.setattr(
            BackpropLearner, name, lambda *a, **k: pytest.fail("native method work")
        )
    ledger["graphs"] += 1
    wall, rss = Reading(8.0), Reading(100)
    sampler = ProcessRssSampler(read_rss_bytes=rss)
    sampler.sample()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_wall_seconds=50, max_process_rss_bytes=1000),
        clock=wall,
        progress=ToyExecutionProgress(),
    )
    budget.attach_memory(sampler)
    learner = BackpropLearner(BackpropMLP(3, 2, 23), learning_rate=0.01)
    runtime = ActorShadowRuntime(
        learner,
        actor_version="actor",
        candidate_version="candidate",
        clock=LogicalClock(0),
        budget=budget,
        max_experiences=4,
    )
    shared = ResourceSharedRuntime(
        runtime, ServingPriorityGate(SharingLimits(2, 0, 1), resource_available=lambda: True)
    )
    owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(4, 4))
    life = make_managed_data_lifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096,
            20,
            PayloadOwnershipLimits(8, 12),
            owned_payload_copies=PayloadCopyLimits(4096),
            max_retention_seconds=1000.0,
        ),
    )
    for name in ("one", "two"):
        owner.declare(
            LifecycleDeclaration(
                ("episode", name),
                DataProvenance("local", "person", True, False),
                DataConsent(True, True),
                "replay",
            )
        )
    owner.record_experience(
        Experience(
            "one",
            "episode",
            0,
            "actor",
            np.ones((1, 3)),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(
        LabelArrival("label", "one", "episode", 0, "actor", np.ones((1, 1)), "train")
    )
    shared._sharing.pause()
    yield owner, life, budget, sampler, wall, rss
    assert budget.updates_completed == 0 and sampler._thread is None
    assert rss.calls <= 128 and wall.calls <= 128 and life._copy_budget._charged <= 4096
    ledger["readers"] += rss.calls


def test_should_return_final_original_observations_without_copying_arrays_again(
    graph, monkeypatch, ledger
):
    owner, life, budget, sampler, wall, rss = graph
    original_count = sampler.sample_count
    original_start = sampler.start_bytes
    seen = []

    def copying(state, limits, memo):
        result = copy_numpy_composite(state, limits, memo)
        array = values(values(result.holders[1][1])["_inbox"])["_experiences"][
            ("episode", "one")
        ].features
        seen.append((id(memo), array))
        rss.value, wall.value = 200, 9.0
        ledger["copies"] += 1
        return result

    monkeypatch.setattr(module, "copy_numpy_composite", copying)
    result = capture_numpy_managed_composite(owner, limits=LIMITS)
    assert len(seen) == 2 and seen[0][0] == seen[1][0] and seen[0][1] is seen[1][1]
    assert sampler.start_bytes == original_start == 100
    assert sampler.sample_count == original_count + 2 and sampler.peak_bytes == 200
    observed = values(result.state.budget)
    progress = values(observed["progress"])
    rss_state = values(observed["process_rss_sampler"])
    assert observed["process_rss_segment"] is progress["process_rss_segment"]
    assert observed["process_rss_segment"].sample_count == sampler.sample_count
    assert observed["process_rss_segment"].peak_bytes == rss_state["peak_bytes"] == 200
    assert observed["last_clock"] == budget.last_clock == 9.0 and observed["started_at"] == 8.0
    assert result.state.records.lifecycle.lifecycle.last_seconds == 9.0
    assert life._copy_budget._charged == result.state.records.lifecycle.copy.charged_bytes == 64


def test_should_refuse_contended_sampler_before_reader_or_copy(graph, ledger):
    owner, life, _, sampler, _, rss = graph
    ready, release = Event(), Event()

    def hold():
        with sampler._lock:
            ready.set()
            assert release.wait(1.0)

    worker = Thread(target=hold)
    ledger["contenders"] += 1
    ledger["live_threads"] += 1
    worker.start()
    try:
        assert ready.wait(1.0)
        before = rss.calls
        with patch.object(module, "deepcopy", side_effect=AssertionError("premature copy")):
            with pytest.raises(PayloadOwnershipBusy, match="busy"):
                capture_numpy_managed_composite(owner, limits=LIMITS)
        assert rss.calls == before and life._copy_budget._charged == 32
    finally:
        release.set()
        worker.join(1.0)
        assert not worker.is_alive()
        ledger["live_threads"] -= 1


@pytest.mark.parametrize("invalid", [None, True, -1, 1.0, "100", OSError("reader failed")])
def test_should_refuse_invalid_reader_and_keep_original_failure(graph, invalid):
    owner, life, _, sampler, _, rss = graph
    rss.value = invalid
    before = sampler.sample_count
    with patch.object(module, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises((ValueError, OSError, ToyProcessRssUnavailable)):
            capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == 32 and sampler.sample_count == before
    assert sampler._stop.is_set() and sampler._error is not None
    if isinstance(invalid, Exception):
        assert sampler._error is invalid


@pytest.mark.parametrize("reason", ["rss", "wall", "retention", "stop", "reader_error"])
def test_should_refuse_after_copy_and_preserve_consumed_original_state(graph, monkeypatch, reason):
    owner, life, budget, sampler, wall, rss = graph
    calls = 0

    def copying(state, limits, memo):
        nonlocal calls
        calls += 1
        result = copy_numpy_composite(state, limits, memo)
        if reason == "rss":
            rss.value = 1001
        elif reason == "wall":
            wall.value = 58.0
        elif reason == "retention":
            wall.value = 1008.0
        elif reason == "stop":
            sampler._stop.set()
        elif reason == "reader_error":
            rss.value = OSError("after copy")
        return result

    monkeypatch.setattr(module, "copy_numpy_composite", copying)
    with pytest.raises((ValueError, OSError, ToyExecutionStopped)) as caught:
        capture_numpy_managed_composite(owner, limits=LIMITS)
    assert calls == 1 and life._copy_budget._charged == 64 and budget.started_at == 8.0
    assert budget.updates_completed == 0
    if reason == "rss":
        assert isinstance(caught.value, ToyExecutionStopped)
        assert caught.value.stop.reason == "max_process_rss_bytes"
        assert budget.process_rss_segment.peak_bytes == sampler.peak_bytes == 1001
        assert budget.progress.process_rss_segment is budget.process_rss_segment
    if reason == "wall":
        assert isinstance(caught.value, ToyExecutionStopped)
        assert caught.value.stop.reason == "max_wall_seconds"


@pytest.mark.parametrize("terminal", ["stop", "error", "unavailable_baseline"])
def test_should_refuse_original_terminal_state_before_read(graph, terminal):
    owner, life, _, sampler, _, rss = graph
    if terminal == "stop":
        sampler._stop.set()
    elif terminal == "error":
        sampler._error = OSError("old error")
    else:
        sampler.start_bytes = None
    before = rss.calls
    with pytest.raises((ValueError, RuntimeError)):
        capture_numpy_managed_composite(owner, limits=LIMITS)
    assert rss.calls == before and life._copy_budget._charged == 32


def test_should_expire_original_sampler_capability_and_refuse_other_threads(ledger):
    sampler = ProcessRssSampler(read_rss_bytes=lambda: 100)
    sampler.sample()
    errors = []
    with sampler._lease_observation() as observe:

        def attempt():
            try:
                observe()
            except ValueError as error:
                errors.append(str(error))

        worker = Thread(target=attempt)
        ledger["contenders"] += 1
        worker.start()
        worker.join(1.0)
        assert not worker.is_alive() and len(errors) == 1
        segment = observe()
        assert segment is not None and segment.start_bytes == 100
    with pytest.raises(ValueError, match="outside"):
        observe()
    assert sampler.sample_count == 2


@pytest.mark.parametrize("field", ["reader", "interval", "gate", "stop", "thread"])
def test_should_refuse_changed_original_sampler_source_before_read(field):
    rss = Reading(100)
    sampler = ProcessRssSampler(read_rss_bytes=rss)
    sampler.sample()
    with sampler._lease_observation() as observe:
        if field == "reader":
            sampler.read_rss_bytes = lambda: 100
        elif field == "interval":
            sampler.interval_seconds = 2.0
        elif field == "gate":
            sampler._lock = Lock()
        elif field == "stop":
            sampler._stop = Event()
        else:
            sampler._thread = Thread()
        before = rss.calls
        with pytest.raises(ValueError, match="changed"):
            observe()
        assert rss.calls == before


@pytest.mark.parametrize("reason", ["rss", "wall", "retention"])
def test_should_refuse_original_cap_before_copy(graph, reason):
    owner, life, budget, _, wall, rss = graph
    if reason == "rss":
        rss.value = 1001
    elif reason == "wall":
        wall.value = 58.0
    else:
        wall.value = 1008.0
    with patch.object(module, "deepcopy", side_effect=AssertionError("premature copy")):
        with pytest.raises((ValueError, ToyExecutionStopped)):
            capture_numpy_managed_composite(owner, limits=LIMITS)
    assert life._copy_budget._charged == 32 and budget.started_at == 8.0


@pytest.mark.parametrize("change", ["reenter", "reader", "baseline", "stop"])
def test_should_refuse_reader_callback_reentry_or_original_state_change(change):
    rss = Reading(100)
    sampler = ProcessRssSampler(read_rss_bytes=rss)
    sampler.sample()
    with sampler._lease_observation() as observe:

        def callback():
            if change == "reenter":
                observe()
            elif change == "reader":
                sampler.read_rss_bytes = lambda: 100
            elif change == "baseline":
                sampler.start_bytes = 1
            else:
                sampler._stop.set()

        rss.callback = callback
        with pytest.raises((ValueError, RuntimeError)):
            observe()
        assert sampler._stop.is_set() and sampler._error is not None
