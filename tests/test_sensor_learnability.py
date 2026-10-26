"""Fixed sensor probe ownership and partial work with fake native arithmetic."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import sensor_learnability as subject
from src.app.coupled_attribution import AttributionBatch
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState
from src.core.learnability_diagnostics import ActivationHealth
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.shared.process_memory import ProcessRssSampler


def snapshot(*, coupled=False):
    parameters = CoupledParameters(
        (np.zeros((16, 8)),), (np.zeros((1, 8)),), np.zeros((8, 1)), np.zeros((1, 1))
    )
    if coupled:
        wake = CoupledWakeState(0.995, 0.02, 0.7, 0.2, np.full(8, 0.4), np.full(8, 0.8), 9)
        return CoupledSnapshot(1, parameters, (1.0,), wake)
    return MatchedBackpropSnapshot(1, parameters)


class FakeLearner:
    def __init__(self, state=None, *, trace=None, hooks=None):
        self.state = snapshot() if state is None else deepcopy(state)
        self.trace = [] if trace is None else trace
        self.hooks = {} if hooks is None else hooks
        self._learning_rate = 0.1

    def snapshot_state(self):
        return deepcopy(self.state)

    def fork(self):
        self.trace.append(("fork", self._learning_rate))
        child = FakeLearner(self.state, trace=self.trace, hooks=self.hooks)
        child._learning_rate = self._learning_rate
        return child

    def restore_state(self, state):
        self.trace.append(("restore", pickle.dumps(state)))
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        self.trace.append(("train", features.copy(), targets.copy(), self._learning_rate))
        hook = self.hooks.get("train")
        if hook:
            hook(self)
        self.state.parameters.output_bias[:] += 0.01
        return TrainingDiagnostic("fake_native", 0.5)

    def predict(self, features):
        self.trace.append(("predict", features.copy(), pickle.dumps(self.state)))
        return np.full((len(features), 1), 0.5 + float(self.state.parameters.output_bias[0, 0]))


@pytest.fixture
def setup_observer(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(subject, "monotonic", lambda: now[0])
    memory = ProcessRssSampler(read_rss_bytes=lambda: 1000)
    memory.sample()
    health_trace: list[tuple[Any, ...]] = []

    def feedforward(parameters, features):
        health_trace.append(("forward", features.copy()))
        return (np.full((32, 8), 0.2),)

    def health(activations):
        assert len(activations) == 1 and activations[0].shape == (32, 8)
        health_trace.append(("svd",))
        return (ActivationHealth(0, 32, 8, 1.0, 0.0, 0.0, 0.2, 0.2, 0.2, (0.2,) * 8, 1.0),)

    monkeypatch.setattr(subject, "feedforward_latents", feedforward)
    monkeypatch.setattr(subject, "diagnose_activations", health)
    source = FakeLearner()
    boundaries = []
    observer = subject.SensorLearnability(
        source, deadline=40.0, memory=memory, check_boundary=lambda: boundaries.append(now[0])
    )
    return observer, source, now, memory, health_trace, boundaries


def arrived():
    return AttributionBatch("train", "A", ("sensor-3-0",), np.ones((1, 16)), np.ones((1, 1)))


def test_should_freeze_disjoint_roles_data_target_rule_and_costs():
    training, evaluation, reference = subject.build_reserved_data()
    assert len(training) == 4 and len(evaluation) == 1
    assert all(
        batch.role == "probe_train" and batch.features.shape == (8, 16) for batch in training
    )
    assert evaluation[0].role == "probe_test" and evaluation[0].features.shape == (32, 16)
    assert reference.shape == (32, 16)
    arrays = [reference]
    ids: list[str] = []
    for batch in (*training, *evaluation):
        arrays.extend((batch.features, batch.targets))
        ids.extend(batch.row_ids)
        np.testing.assert_array_equal(
            batch.targets, batch.features[:, [0]] * batch.features[:, [1]] >= 0
        )
    assert len(ids) == len(set(ids)) == 64
    assert all(not identifier.startswith("sensor-3-") for identifier in ids)
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    assert (
        sum(a.nbytes for a in arrays)
        == subject.describe_learnability()["reserved_data_array_bytes"]
        == 12800
    )
    second = subject.build_reserved_data()
    np.testing.assert_array_equal(second[0][0].features, training[0].features)
    training[0].features[:] = 99
    assert not np.any(second[0][0].features == 99)


def test_should_execute_four_fixed_curves_on_owned_forks_and_detach_report(setup_observer):
    observer, source, _, _, health, boundaries = setup_observer
    state = snapshot(coupled=True)
    state_before, source_before = pickle.dumps(state), pickle.dumps(source.snapshot_state())
    for point in subject.CHECKPOINTS:
        observer.observe(point, state, None if point == 0 else arrived())
    report = observer.report()
    assert report["status"] == "complete" and report["charged_probe_updates"] == 16
    assert [row["updates"] for row in report["observations"]] == [0, 4, 8, 12]
    assert [entry[0] for entry in health] == ["forward", "svd"] * 4
    assert boundaries
    for row in report["observations"]:
        assert row["status"] == "complete" and row["probe_source_checkpoint_and_data_unchanged"]
        assert [p["updates"] for p in row["probe"]["checkpoints"]] == [0, 1, 2, 4]
        assert row["probe"]["completed_updates"] == row["charged_probe_updates"] == 4
        assert row["probe"]["completed_training_examples"] == 32
        assert row["probe"]["score_forward_calls_completed"] == 4
        assert row["diagnostic_activation_bytes"] == 2048
        assert row["checkpoint_snapshot_array_bytes"] == 1288
        assert row["importance"] is row["neuron_age"] is None
        curve = row["probe"]["checkpoints"]
        assert row["fresh_gain"] == curve[0]["scores"]["C"]["bce"] - curve[-1]["scores"]["C"]["bce"]
    assert (
        pickle.dumps(state) == state_before
        and pickle.dumps(source.snapshot_state()) == source_before
    )
    restores = [entry for entry in source.trace if entry[0] == "restore"]
    assert len(restores) == 4 and all(entry[1] == state_before for entry in restores)
    assert all(
        entry[3] == 0.1 and entry[1].shape == (8, 16)
        for entry in source.trace
        if entry[0] == "train"
    )
    assert report["costs"]["completed_updates"] == 16
    assert report["costs"]["health_svd_calls_completed"] == 4
    json.dumps(report, allow_nan=False)
    report["observations"][0]["probe"]["checkpoints"].clear()
    assert len(observer.report()["observations"][0]["probe"]["checkpoints"]) == 4


def test_should_allocate_all_distinct_sessions_at_setup_without_checkpoint_renewal(setup_observer):
    observer, _, now, memory, _, _ = setup_observer
    sessions = observer._sessions
    assert len(sessions) == len({id(session) for session in sessions}) == 4
    assert all(
        session.started_at == 10 and session.budget.max_training_updates == 4
        for session in sessions
    )
    assert all(session.process_rss_sampler is memory for session in sessions)
    now[0] = 39
    observer.observe(0, snapshot(), None)
    assert observer._sessions is sessions
    assert observer.report()["probe_session_origins"] == [10.0] * 4
    now[0] = 40
    with pytest.raises(RuntimeError, match="original.*deadline"):
        observer.observe(4, snapshot(), arrived())
    report = observer.report()
    assert report["status"] == "failed" and report["charged_probe_updates"] == 4
    assert [row["status"] for row in report["observations"]] == [
        "complete",
        "failed",
        "not_run",
        "not_run",
    ]


@pytest.mark.parametrize("point", [True, 1, 4, 12, -1, 0.0])
def test_should_refuse_out_of_order_or_non_integer_checkpoints_before_work(point, setup_observer):
    observer, source, _, _, health, _ = setup_observer
    with pytest.raises(ValueError, match="ordered once-only"):
        observer.observe(point, snapshot(), None)
    assert not source.trace and not health


def test_should_refuse_duplicate_checkpoint_without_more_native_work(setup_observer):
    observer, source, _, _, health, _ = setup_observer
    observer.observe(0, snapshot(), None)
    sizes = len(source.trace), len(health)
    with pytest.raises(ValueError, match="once-only"):
        observer.observe(0, snapshot(), None)
    assert (len(source.trace), len(health)) == sizes


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_accept_only_arrived_training_metadata(role, setup_observer):
    observer, source, _, _, health, _ = setup_observer
    observer.observe(0, snapshot(), None)
    sizes = len(source.trace), len(health)
    with pytest.raises(ValueError, match="batch role must be train"):
        observer.observe(4, snapshot(), replace(arrived(), role=role))
    assert (len(source.trace), len(health)) == sizes


def test_should_refuse_initial_arrival_before_native_work(setup_observer):
    observer, source, _, _, health, _ = setup_observer
    with pytest.raises(ValueError, match="initial.*no arrived"):
        observer.observe(0, snapshot(), arrived())
    assert not source.trace and not health


@pytest.mark.parametrize(
    "mutation", ["checkpoint", "template", "policy", "reference", "training", "evaluation"]
)
def test_should_stop_before_probe_when_health_changes_owned_inputs(
    mutation, setup_observer, monkeypatch
):
    observer, source, _, _, _, _ = setup_observer
    state = snapshot()

    def corrupt(parameters, reference):
        if mutation == "checkpoint":
            parameters.output_bias[:] = 5
        elif mutation == "template":
            source.state.parameters.output_bias[:] = 5
        elif mutation == "policy":
            source._learning_rate = 0.9
        elif mutation == "reference":
            reference[:] = 5
        elif mutation == "training":
            observer._training[0].targets[:] = 0.7
        else:
            observer._evaluation[0].features[:] = 5
        return (np.full((32, 8), 0.2),)

    monkeypatch.setattr(subject, "feedforward_latents", corrupt)
    with pytest.raises(RuntimeError, match="changed"):
        observer.observe(0, state, None)
    report = observer.report()
    assert not source.trace and report["status"] == "failed"
    row = report["observations"][0]
    assert row["health_forward_calls_attempted"] == row["health_forward_calls_completed"] == 1
    assert row["health_svd_calls_attempted"] == 0 and row["probe"] is None
    with pytest.raises(RuntimeError, match="stopped"):
        observer.observe(4, snapshot(), arrived())


def test_should_preserve_partial_probe_failure_and_refuse_later_work(setup_observer):
    observer, source, _, _, _, _ = setup_observer
    attempts = []

    def fail_third(child):
        attempts.append(child)
        if len(attempts) == 3:
            raise RuntimeError("injected third native failure")

    source.hooks["train"] = fail_third
    with pytest.raises(RuntimeError, match="third native failure"):
        observer.observe(0, snapshot(), None)
    row = observer.report()["observations"][0]
    assert row["status"] == "failed" and row["charged_probe_updates"] == 2
    assert row["probe"]["attempted_updates"] == 3
    assert row["probe"]["completed_updates"] == 2
    assert [point["updates"] for point in row["probe"]["checkpoints"]] == [0, 1, 2]
    assert [r["status"] for r in observer.report()["observations"][1:]] == ["not_run"] * 3
    with pytest.raises(RuntimeError, match="stopped"):
        observer.observe(4, snapshot(), arrived())
    assert len(attempts) == 3


def test_should_charge_committed_native_work_before_failed_guard(setup_observer):
    observer, source, _, _, _, _ = setup_observer
    source.hooks["train"] = lambda child: source.state.parameters.output_bias.fill(5)
    with pytest.raises(RuntimeError, match="template"):
        observer.observe(0, snapshot(), None)
    row = observer.report()["observations"][0]
    assert row["charged_probe_updates"] == row["probe"]["completed_updates"] == 1
    assert row["probe"]["attempted_updates"] == 1
    assert row["probe"]["score_forward_calls_completed"] == 1


def test_should_preserve_failed_svd_attempt_without_starting_probe(setup_observer, monkeypatch):
    observer, source, _, _, _, _ = setup_observer

    def fail(activations):
        raise RuntimeError("injected SVD failure")

    monkeypatch.setattr(subject, "diagnose_activations", fail)
    with pytest.raises(RuntimeError, match="SVD failure"):
        observer.observe(0, snapshot(), None)
    row = observer.report()["observations"][0]
    assert row["health_svd_calls_attempted"] == 1 and row["health_svd_calls_completed"] == 0
    assert row["health_forward_calls_completed"] == 1 and not source.trace


def test_should_measure_preflight_and_probe_invocation_time(setup_observer, monkeypatch):
    observer, _, now, _, _, _ = setup_observer
    original = subject.run_fresh_task_probe

    def measured(*args, **kwargs):
        now[0] += 3
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "run_fresh_task_probe", measured)
    observer.observe(0, snapshot(), None)
    row = observer.report()["observations"][0]
    assert row["invocation_seconds"] == row["probe_invocation_seconds"] == 3
