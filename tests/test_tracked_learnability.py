"""Tracked checkpoint measurements remain separate from disposable probe work.

These small native examples exercise the existing observer and reserved probe;
they test ownership and measured state, not comparative learning performance.
"""

from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.learnability_pilot as pilot
from src.adapters.coupled_learners import CoupledLearner
from src.app.coupled_attribution import AttributionBatch
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.matched_backprop import MatchedBackpropNetwork
from src.core.tracked_coupled_predictive_coding import (
    TrackedCoupledCircadianPredictiveCodingNetwork,
    TrackedCoupledPredictiveCodingNetwork,
    TrackedCoupledSnapshot,
)


def native(kind):
    if kind == "pc":
        return TrackedCoupledPredictiveCodingNetwork(2, (8, 8), 23)
    return TrackedCoupledCircadianPredictiveCodingNetwork(
        2,
        (8, 8),
        23,
        config=CoupledCircadianConfig.matched_pc_control() if kind == "neutral" else None,
    )


def learner(kind):
    return CoupledLearner(
        native(kind), learning_rate=0.03, inference_steps=2, inference_learning_rate=0.02
    )


def wake_batch():
    return np.array([[0.2, -0.4], [-0.5, 0.3]]), np.array([[1.0], [0.0]])


def history(snapshot):
    assert type(snapshot) is TrackedCoupledSnapshot
    assert snapshot.unit_history is not None
    return snapshot.unit_history


def array_values(snapshot):
    parameters = snapshot.parameters
    result = [
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    ]
    if snapshot.wake_state is not None:
        result.extend((snapshot.wake_state.chemical_state, snapshot.wake_state.plasticity_state))
    stored = history(snapshot)
    result.extend((stored.unit_ids, stored.importance_ema, stored.update_age))
    return result


def observer(template, *, training=None, evaluation=None, reference=None):
    if training is None:
        training, evaluation, reference = pilot.build_reserved_data(extended=True)
    slots = pilot.empty_learnability_observations()
    callback = pilot._observer(
        template,
        training,
        evaluation,
        reference,
        slots,
        60.0,
        probe_checkpoints=pilot.EXTENDED_PROBE_CHECKPOINTS,
    )
    return callback, slots


def forbidden(*args, **kwargs):
    pytest.fail("refused checkpoint reached feature, SVD, or probe work")


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_measure_restored_checkpoint_history_without_counting_disposable_probe_updates(
    monkeypatch, kind
):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    source = learner(kind)
    features, targets = wake_batch()
    for _ in range(2):
        source.train_batch(features, targets)
    checkpoint = source.snapshot_state()
    # A restore supplies existing history; neither observing nor probe setup may
    # reset it, even when the independent template is newly constructed.
    benchmark = learner(kind)
    benchmark.restore_state(checkpoint)
    checkpoint = benchmark.snapshot_state()
    template = learner(kind)
    assert history(template.snapshot_state()).updates == 0
    training, evaluation, reference = pilot.build_reserved_data(extended=True)
    input_before = pickle.dumps(
        (benchmark, source, template, checkpoint, training, evaluation, reference)
    )
    stored = history(checkpoint)
    importance = stored.importance_ema.copy()
    forks: list[CoupledLearner] = []
    restored: list[int] = []
    trained: list[int] = []
    original_fork = CoupledLearner.fork
    original_restore = CoupledLearner.restore_state
    original_train = CoupledLearner.train_batch

    def fork(self):
        child = original_fork(self)
        assert self is template
        assert child is not self
        assert history(child.snapshot_state()).updates == 0
        forks.append(child)
        return child

    def restore(self, state):
        assert state is not checkpoint
        for left, right in zip(array_values(state), array_values(checkpoint), strict=True):
            np.testing.assert_array_equal(left, right)
            assert not np.shares_memory(left, right)
        original_restore(self, state)
        restored.append(history(self.snapshot_state()).updates)

    def train(self, x, y):
        assert self is forks[0]
        index = len(trained)
        np.testing.assert_array_equal(x, training[index].features)
        np.testing.assert_array_equal(y, training[index].targets)
        assert not np.shares_memory(x, training[index].features)
        assert not np.shares_memory(y, training[index].targets)
        result = original_train(self, x, y)
        trained.append(history(self.snapshot_state()).updates)
        # Native inputs are borrowed only during this call. Deliberately damage
        # those copies to prove later batches and the caller's reserve stay owned.
        x.fill(900)
        y.fill(901)
        return result

    monkeypatch.setattr(CoupledLearner, "fork", fork)
    monkeypatch.setattr(CoupledLearner, "restore_state", restore)
    monkeypatch.setattr(CoupledLearner, "train_batch", train)
    observe, slots = observer(
        template, training=training, evaluation=evaluation, reference=reference
    )
    observe(0, checkpoint, None)
    assert len(forks) == 1 and restored == [2] and trained == list(range(3, 19))
    assert history(forks[0].snapshot_state()).updates == 18
    assert (
        pickle.dumps((benchmark, source, template, checkpoint, training, evaluation, reference))
        == input_before
    )
    row = slots[0]
    assert row["status"] == "complete" and row["reason"] is None
    assert row["probe_source_and_checkpoint_unchanged"] is True
    assert all(slot["status"] == "not_run" for slot in slots[1:])
    assert row["importance"]["count"] == 8
    assert row["importance"]["minimum"] == float(importance.min())
    assert row["importance"]["maximum"] == float(importance.max())
    assert row["importance"]["mean"] == pytest.approx(float(importance.mean()), rel=2e-15)
    assert row["neuron_age"] == {"count": 8, "minimum": 2.0, "mean": 2.0, "maximum": 2.0}
    assert row["unavailable_reason"] is None
    measured = row["native_unit_history"]
    assert measured["successful_updates"] == 2 and measured["array_bytes"] == 192
    assert measured["family"] == "tracked_coupled" and measured["layer"] == "last_hidden"
    assert measured["history"]["importance"] == row["importance"]
    assert measured["history"]["age"] == row["neuron_age"]
    assert [unit["neuron_id"] for unit in measured["history"]["units"]] == list(range(8))
    assert all(
        unit["parent_id"] is None and unit["age_updates"] == 2
        for unit in measured["history"]["units"]
    )
    assert "batch-aggregate" in measured["importance_definition"]
    assert "including replay" in measured["age_definition"]
    arrays = array_values(checkpoint)
    independent_bytes = sum(
        value.nbytes
        for index, value in enumerate(arrays)
        if all(value is not prior for prior in arrays[:index])
    )
    assert (
        independent_bytes
        == row["checkpoint_snapshot_array_bytes"]
        == (1032 if kind == "pc" else 1160)
    )
    assert row["diagnostic_activation_bytes"] == 32 * 16 * 8
    assert [health["units"] for health in row["feature_health"]] == [8, 8]
    probe = row["probe"]
    assert probe["status"] == "complete"
    assert probe["attempted_updates"] == probe["completed_updates"] == 16
    assert probe["attempted_training_examples"] == probe["completed_training_examples"] == 128
    assert probe["score_forward_calls_attempted"] == probe["score_forward_calls_completed"] == 6
    assert [point["updates"] for point in probe["checkpoints"]] == [0, 1, 2, 4, 8, 16]
    assert probe["reserved_data_bytes"] == 3840
    # Reports contain scalar data, with no mutable history-array ownership link.
    saved_report = deepcopy(measured)
    stored.importance_ema.fill(999)
    assert measured == saved_report
    measured["history"]["units"][0]["importance_ema"] = 888
    assert not np.any(history(benchmark.snapshot_state()).importance_ema == 888)


@pytest.mark.parametrize("fault", ["history", "format", "ids", "importance", "age", "wake_clock"])
def test_should_refuse_corrupt_tracked_history_before_feature_or_probe_work(monkeypatch, fault):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    state = native("active").snapshot_state()
    stored = history(state)
    if fault == "history":
        object.__setattr__(state, "unit_history", None)
    elif fault == "format":
        object.__setattr__(state, "format_version", 1)
    elif fault == "ids":
        stored.unit_ids[0] = 99
    elif fault == "importance":
        stored.importance_ema[0] = float("nan")
    elif fault == "age":
        stored.update_age[0] = 0.5
    else:
        assert state.wake_state is not None
        object.__setattr__(state.wake_state, "wake_updates", False)
    before = pickle.dumps(state)
    template = learner("active")
    template_before = pickle.dumps(template)
    monkeypatch.setattr(pilot, "feedforward_latents", forbidden)
    monkeypatch.setattr(pilot, "diagnose_activations", forbidden)
    monkeypatch.setattr(pilot, "_measure_probe", forbidden)
    observe, slots = observer(template)
    with pytest.raises(ValueError):
        observe(0, state, None)
    assert pickle.dumps(state) == before and pickle.dumps(template) == template_before
    assert slots[0]["status"] == "failed" and slots[0]["probe"] is None
    assert slots[0]["diagnostic_seconds"] == 0.0
    assert "feature_health" not in slots[0]
    assert all(slot["status"] == "not_run" for slot in slots[1:])


@pytest.mark.parametrize("phase", ["feedforward", "health"])
def test_should_keep_checkpoint_template_and_benchmark_unchanged_on_feature_failure(
    monkeypatch, phase
):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    benchmark, template = learner("pc"), learner("pc")
    state = benchmark.snapshot_state()
    before = pickle.dumps((benchmark, template, state))
    problem = RuntimeError("feature measurement failed")
    read = pilot.read_coupled_unit_diagnostics
    reads = []

    def reader(snapshot):
        result = read(snapshot)
        reads.append(result.successful_updates)
        return result

    def broken(*args, **kwargs):
        assert reads == [0]
        raise problem

    monkeypatch.setattr(pilot, "read_coupled_unit_diagnostics", reader)
    monkeypatch.setattr(
        pilot, "feedforward_latents" if phase == "feedforward" else "diagnose_activations", broken
    )
    monkeypatch.setattr(pilot, "_measure_probe", forbidden)
    observe, slots = observer(template)
    with pytest.raises(RuntimeError) as caught:
        observe(0, state, None)
    assert caught.value is problem
    assert pickle.dumps((benchmark, template, state)) == before
    assert reads == [0]
    assert slots[0]["status"] == "failed" and slots[0]["probe"] is None
    assert slots[0]["diagnostic_seconds"] == 0.0


def test_should_retain_failed_probe_work_without_mutating_native_checkpoint_or_template(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    benchmark, template = learner("active"), learner("active")
    features, targets = wake_batch()
    benchmark.train_batch(features, targets)
    state = benchmark.snapshot_state()
    before = pickle.dumps((benchmark, template, state))
    original_fork, original_train = CoupledLearner.fork, CoupledLearner.train_batch
    children = []
    problem = RuntimeError("failure after disposable native commit")

    def fork(self):
        child = original_fork(self)
        children.append(child)
        return child

    def train(self, x, y):
        original_train(self, x, y)
        raise problem

    monkeypatch.setattr(CoupledLearner, "fork", fork)
    monkeypatch.setattr(CoupledLearner, "train_batch", train)
    observe, slots = observer(template)
    with pytest.raises(RuntimeError, match="fresh probe incomplete"):
        observe(0, state, None)
    assert pickle.dumps((benchmark, template, state)) == before
    assert len(children) == 1 and history(children[0].snapshot_state()).updates == 2
    row = slots[0]
    assert row["status"] == "failed"
    assert row["native_unit_history"]["successful_updates"] == 1
    assert row["neuron_age"]["mean"] == 1.0
    probe = row["probe"]
    assert probe["status"] == "failed" and str(problem) in probe["reason"]
    assert probe["attempted_updates"] == 1 and probe["completed_updates"] == 0
    assert probe["score_forward_calls_completed"] == 1
    assert [point["updates"] for point in probe["checkpoints"]] == [0]
    assert all(slot["status"] == "not_run" for slot in slots[1:])


@pytest.mark.parametrize("kind", ["pc", "cpc", "bp"])
def test_should_preserve_the_exact_untracked_feature_schema_and_unavailable_values(
    monkeypatch, kind
):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23)
    state: Any
    if kind == "pc":
        state = initial.snapshot_state()
    elif kind == "cpc":
        state = CoupledCircadianPredictiveCodingNetwork(2, (8, 8), 23).snapshot_state()
    else:
        state = MatchedBackpropNetwork(initial.snapshot_parameters()).snapshot_state()
    before = pickle.dumps(state)
    reference = np.array([[0.2, -0.4], [-0.5, 0.3], [0.1, 0.7]])
    monkeypatch.setattr(pilot, "read_coupled_unit_diagnostics", forbidden)
    row: dict[str, Any] = {}
    pilot._measure_features(state, reference, row)
    assert set(row) == {
        "feature_health",
        "diagnostic_activation_bytes",
        "checkpoint_snapshot_array_bytes",
        "importance",
        "neuron_age",
        "unavailable_reason",
        "diagnostic_seconds",
    }
    assert row["importance"] is row["neuron_age"] is None
    assert row["unavailable_reason"] == "No native per-neuron importance/age in coupled models."
    assert row["checkpoint_snapshot_array_bytes"] == (968 if kind == "cpc" else 840)
    assert row["diagnostic_activation_bytes"] == 3 * 16 * 8
    assert row["diagnostic_seconds"] == 0.0 and len(row["feature_health"]) == 2
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("role", ["final_test", "probe_test"])
def test_should_refuse_heldout_metadata_before_reading_tracked_history(monkeypatch, role):
    monkeypatch.setattr(pilot, "_measure_features", forbidden)
    monkeypatch.setattr(pilot, "_measure_probe", forbidden)
    metadata = AttributionBatch(role, "C", ("heldout",), np.zeros((1, 2)), np.zeros((1, 1)))
    slots = pilot.empty_learnability_observations()
    before = deepcopy(slots)
    observe = pilot._observer(object(), (), (), None, slots, 60.0)
    with pytest.raises(ValueError, match="arrived training"):
        observe(0, object(), metadata)
    assert slots == before
