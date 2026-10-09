"""Tiny direct replay storage parity; execute only after the declared native gate."""

import pickle
import json
import sys
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
)
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.replay_write_origin import ReplayWriteLimits, observe_replay_writes


@pytest.fixture(scope="module", autouse=True)
def native_work_ledger(tmp_path_factory):
    work = dict(
        constructors=0,
        stores=0,
        predictions=0,
        array_copies=0,
        training=0,
        restores=0,
        snapshots=0,
        sleeps=0,
    )
    path = tmp_path_factory.mktemp("native-work") / "native-work.json"
    previous = sys.getprofile()

    def profile(frame, event, function):
        if (
            event == "c_call"
            and frame.f_code.co_name == "_store_replay_snapshot"
            and getattr(function, "__name__", None) == "copy"
            and isinstance(getattr(function, "__self__", None), np.ndarray)
        ):
            work["array_copies"] += 1
        if previous is not None:
            previous(frame, event, function)

    def counted(name, original):
        def call(*args, **kwargs):
            work[name] += 1
            return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for name, method in [
            ("constructors", "__init__"),
            ("stores", "_store_replay_snapshot"),
            ("predictions", "predict_proba"),
        ]:
            original = getattr(CircadianPredictiveCodingNetwork, method)
            stack.enter_context(
                patch.object(CircadianPredictiveCodingNetwork, method, counted(name, original))
            )
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
            path.write_text(json.dumps(work, indent=2), encoding="utf8")
            assert work["constructors"] <= 12 and work["stores"] <= 28
            assert work["predictions"] <= 27 and work["array_copies"] <= 88


def model(policy, max_bytes=64):
    network = CircadianPredictiveCodingNetwork(
        3,
        2,
        circadian_config=CircadianConfig(replay_memory_size=2, sleep_mode="disabled"),
        seed=23,
        min_hidden_dim=2,
    )
    if policy != "batch":
        network.configure_replay_retention(
            ReplayRetentionBudget(2, max_bytes),
            policy=ReplayRetentionPolicy(policy, 23 if policy == "seeded_reservoir" else None),
        )
    return network


@pytest.mark.parametrize("policy", ["batch", "content_hash", "recent_fifo", "seeded_reservoir"])
def test_should_preserve_native_storage_and_bind_actual_copies_without_content_matching(policy):
    observed, original = model(policy), model(policy)
    fields = tuple(vars(observed))
    batches = [
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0], [1.0], [0.0]]),
        ),
        (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0.0], [1.0]])),
        (np.array([[1.0, 1.0, 1.0]]), np.array([[1.0]])),
    ]
    copied: dict[int, Any] = {}
    reports = []
    for call, (features, targets) in enumerate(batches):
        local = []

        def observer(stage, read):
            value = read()
            local.append((stage, value))
            assert (
                value.model is observed and value.features is features and value.targets is targets
            )
            if stage == "copied":
                copied[id(value.snapshot)] = value
                snapshot = value.snapshot
                start, count = value.row_start, value.row_count
                assert np.array_equal(snapshot.input_batch, features[start : start + count])
                assert np.array_equal(snapshot.target_batch, targets[start : start + count])
                assert not np.shares_memory(snapshot.input_batch, features)
                assert not np.shares_memory(snapshot.target_batch, targets)

        with observe_replay_writes(
            observed, features, targets, observer, ReplayWriteLimits(3, 2, 16)
        ):
            observed._store_replay_snapshot(features, targets)
        original._store_replay_snapshot(features, targets)
        assert local[0][0] == "begin" and local[-1][0] == "retained"
        assert all(
            actual is expected
            for actual, expected in zip(local[-1][1].retained, observed._replay_memory)
        )
        for snapshot in observed._replay_memory:
            assert copied[id(snapshot)].snapshot is snapshot
        assert tuple(vars(observed)) == fields
        assert pickle.dumps(observed.__dict__, protocol=5) == pickle.dumps(
            original.__dict__, protocol=5
        )
        reports.append(local)
    assert all(r[1].row_count <= 3 for rows in reports for r in rows)
    assert observed._epoch_count == original._epoch_count == 0


def test_should_refuse_native_observer_fault_before_copy_without_changing_original_state():
    network = model("batch")
    features, targets = np.ones((1, 3)), np.zeros((1, 1))
    before = pickle.dumps(network.__dict__, protocol=5)

    def observer(stage, read):
        read()
        if stage == "before_copy":
            raise RuntimeError("native observer refusal")

    with pytest.raises(RuntimeError, match="native observer refusal"):
        with observe_replay_writes(
            network, features, targets, observer, ReplayWriteLimits(1, 2, 4)
        ):
            network._store_replay_snapshot(features, targets)
    assert pickle.dumps(network.__dict__, protocol=5) == before


def test_should_report_oversized_row_refusal_without_claiming_a_retained_copy():
    network = model("content_hash", max_bytes=16)
    features, targets = np.ones((1, 3)), np.zeros((1, 1))
    reports = []
    with observe_replay_writes(
        network,
        features,
        targets,
        lambda stage, read: reports.append((stage, read())),
        ReplayWriteLimits(1, 2, 4),
    ):
        network._store_replay_snapshot(features, targets)
    assert [stage for stage, _ in reports] == ["begin", "before_copy", "retained"]
    assert reports[-1][1].retained == () and not network._replay_memory
    assert network._epoch_count == 0


def test_should_preserve_already_retained_native_payload_after_late_observer_fault():
    network = model("batch")
    features, targets = np.ones((1, 3)), np.zeros((1, 1))
    copied = []

    def observer(stage, read):
        value = read()
        if stage == "copied":
            copied.append(value.snapshot)
        if stage == "retained":
            assert value.retained[0] is copied[0]
            raise RuntimeError("late observer fault")

    with pytest.raises(RuntimeError, match="late observer fault"):
        with observe_replay_writes(
            network, features, targets, observer, ReplayWriteLimits(1, 2, 4)
        ):
            network._store_replay_snapshot(features, targets)
    assert network._replay_memory[0] is copied[0] and network._epoch_count == 0


def test_should_report_disabled_replay_without_prediction_or_payload_copy(monkeypatch):
    network = CircadianPredictiveCodingNetwork(
        3,
        2,
        seed=23,
        min_hidden_dim=2,
        circadian_config=CircadianConfig(replay_memory_size=0, sleep_mode="disabled"),
    )
    features, targets = np.ones((1, 3)), np.zeros((1, 1))
    monkeypatch.setattr(
        type(network), "predict_proba", lambda *_: pytest.fail("prediction entered")
    )
    stages = []
    with observe_replay_writes(
        network,
        features,
        targets,
        lambda stage, read: stages.append(stage),
        ReplayWriteLimits(1, 2, 4),
    ):
        network._store_replay_snapshot(features, targets)
    assert stages == ["begin", "retained"] and not network._replay_memory
