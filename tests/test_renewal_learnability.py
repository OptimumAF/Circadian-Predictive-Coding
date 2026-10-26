"""Renewed native ages are observations, independent of fresh-probe exposure.

Data-only readout controls complement small real PC/CPC observer/probe examples.
No endpoint value is used as an eligibility or comparative success criterion.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.learnability_pilot as pilot
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.native_unit_diagnostics import read_renewable_coupled_unit_diagnostics
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledWakeState
from src.core.coupled_unit_history import initial_unit_history
from src.core.coupled_unit_renewal import RenewableUnitHistory, RenewalPolicy
from src.core.matched_backprop import MatchedBackpropSnapshot
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledCircadianPredictiveCodingNetwork,
    RenewableCoupledPredictiveCodingNetwork,
    RenewableCoupledSnapshot,
)
from src.core.tracked_coupled_predictive_coding import TrackedCoupledSnapshot


def declared_policy():
    return RenewalPolicy(2, 1e308, 1, 29)


def learner(kind):
    if kind == "pc":
        native: Any = RenewableCoupledPredictiveCodingNetwork(
            2, (8, 8), 23, renewal_policy=declared_policy()
        )
    else:
        native = RenewableCoupledCircadianPredictiveCodingNetwork(
            2,
            (8, 8),
            23,
            config=CoupledCircadianConfig.matched_pc_control() if kind == "neutral" else None,
            renewal_policy=declared_policy(),
        )
    return CoupledLearner(
        native, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.02
    )


def stored_history(state):
    assert type(state) is RenewableCoupledSnapshot
    assert type(state.unit_history) is RenewableUnitHistory
    return state.unit_history


def renewed_checkpoint(benchmark):
    features, targets = np.array([[0.2, -0.4], [-0.5, 0.3]]), np.array([[1.0], [0.0]])
    for _ in range(2):
        benchmark.train_batch(features, targets)
    result = benchmark.renew_units(1)
    assert len(result.indices) == 1
    benchmark.train_batch(features, targets)
    checkpoint = benchmark.snapshot_state()
    ages = stored_history(checkpoint).update_age
    assert ages[result.indices[0]] == 1 and np.count_nonzero(ages == 3) == 7
    return checkpoint, result


def observer(template):
    training, evaluation, reference = pilot.build_reserved_data(extended=True)
    slots = pilot.empty_learnability_observations()
    return (
        pilot._observer(
            template,
            training,
            evaluation,
            reference,
            slots,
            60.0,
            probe_checkpoints=pilot.EXTENDED_PROBE_CHECKPOINTS,
        ),
        slots,
        (training, evaluation, reference),
    )


def declared_snapshot():
    parameters = CoupledParameters(
        (np.zeros((2, 3)),), (np.zeros((1, 3)),), np.zeros((3, 1)), np.zeros((1, 1))
    )
    return RenewableCoupledSnapshot(
        3,
        parameters,
        (1.0,),
        None,
        RenewableUnitHistory(
            2,
            0.95,
            np.array([8, 2, 5], dtype=np.int64),
            np.array([0.0, 0.25, 0.5]),
            np.array([0.0, 2.0, 5.0]),
            5,
            10,
        ),
        declared_policy(),
    )


def forbidden(*args, **kwargs):
    pytest.fail("refused renewable history reached feature or probe work")


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_restore_heterogeneous_history_into_probe_without_renewing_or_mutating_checkpoint(
    monkeypatch, kind
):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    benchmark, template = learner(kind), learner(kind)
    checkpoint, replacement = renewed_checkpoint(benchmark)
    stored = stored_history(checkpoint)
    ids, ages, importance = (
        stored.unit_ids.copy(),
        stored.update_age.copy(),
        stored.importance_ema.copy(),
    )
    assert stored.updates == 3 and stored.next_unit_id == 9
    assert replacement.new_unit_ids == (8,)
    observe, slots, reserve = observer(template)
    before = pickle.dumps((benchmark, template, checkpoint, reserve))
    original_fork, original_restore = CoupledLearner.fork, CoupledLearner.restore_state
    children: list[CoupledLearner] = []
    restored: list[tuple[int, int]] = []

    def fork(self):
        assert self is template
        child = original_fork(self)
        assert stored_history(child.snapshot_state()).updates == 0
        children.append(child)
        return child

    def restore(self, state):
        assert state is not checkpoint
        supplied = stored_history(state)
        np.testing.assert_array_equal(supplied.unit_ids, ids)
        np.testing.assert_array_equal(supplied.update_age, ages)
        for name in ("unit_ids", "importance_ema", "update_age"):
            assert not np.shares_memory(getattr(supplied, name), getattr(stored, name))
        original_restore(self, state)
        restored.append((supplied.updates, supplied.next_unit_id))

    monkeypatch.setattr(CoupledLearner, "fork", fork)
    monkeypatch.setattr(CoupledLearner, "restore_state", restore)
    # Disposable probe training advances native histories but has no benchmark
    # renewal allowance/helper. A hidden renewal call must therefore fail here.
    monkeypatch.setattr(CoupledLearner, "renew_units", forbidden)
    monkeypatch.setattr(RenewableCoupledPredictiveCodingNetwork, "renew_units", forbidden)
    monkeypatch.setattr(RenewableCoupledCircadianPredictiveCodingNetwork, "renew_units", forbidden)
    observe(0, checkpoint, None)
    assert restored == [(3, 9)] and len(children) == 1
    probe_history = stored_history(children[0].snapshot_state())
    assert probe_history.updates == 19 and probe_history.next_unit_id == 9
    np.testing.assert_array_equal(probe_history.unit_ids, ids)
    np.testing.assert_array_equal(probe_history.update_age, ages + 16)
    assert pickle.dumps((benchmark, template, checkpoint, reserve)) == before
    row = slots[0]
    assert row["status"] == "complete" and row["reason"] is None
    assert row["probe_source_and_checkpoint_unchanged"] is True
    assert row["unavailable_reason"] is None
    assert row["importance"]["count"] == 8
    assert row["importance"]["minimum"] == float(importance.min())
    assert row["importance"]["maximum"] == float(importance.max())
    assert row["importance"]["mean"] == pytest.approx(float(importance.mean()), rel=2e-15)
    assert row["neuron_age"] == {"count": 8, "minimum": 1.0, "mean": 2.75, "maximum": 3.0}
    measured = row["native_unit_history"]
    assert measured["family"] == "renewable_coupled" and measured["layer"] == "last_hidden"
    assert measured["successful_updates"] == 3 and measured["array_bytes"] == 192
    assert [unit["neuron_id"] for unit in measured["history"]["units"]] == ids.tolist()
    assert [unit["age_updates"] for unit in measured["history"]["units"]] == ages.tolist()
    assert all(unit["parent_id"] is None for unit in measured["history"]["units"])
    assert row["checkpoint_snapshot_array_bytes"] == (1032 if kind == "pc" else 1160)
    assert row["diagnostic_activation_bytes"] == 4096
    probe = row["probe"]
    assert (
        probe["status"] == "complete"
        and probe["completed_updates"] == probe["attempted_updates"] == 16
    )
    assert probe["completed_training_examples"] == 128
    assert [point["updates"] for point in probe["checkpoints"]] == [0, 1, 2, 4, 8, 16]
    assert all(slot["status"] == "not_run" for slot in slots[1:])
    saved = deepcopy(measured)
    stored.importance_ema.fill(999)
    assert measured == saved
    measured["history"]["units"][0]["age_updates"] = 999
    assert not np.any(stored_history(benchmark.snapshot_state()).update_age == 999)


def test_should_report_ids_and_native_heterogeneous_ages_as_owned_immutable_scalar_data():
    state = declared_snapshot()
    before = pickle.dumps(state)
    arrays = [
        *state.parameters.hidden_weights,
        *state.parameters.hidden_biases,
        state.parameters.output_weight,
        state.parameters.output_bias,
        stored_history(state).unit_ids,
        stored_history(state).importance_ema,
        stored_history(state).update_age,
    ]
    identities = [(id(array), array.flags.writeable, array.tobytes()) for array in arrays]
    result = read_renewable_coupled_unit_diagnostics(state)
    assert pickle.dumps(state) == before
    assert [(id(array), array.flags.writeable, array.tobytes()) for array in arrays] == identities
    assert result.family == "renewable_coupled" and result.layer == "last_hidden"
    assert result.successful_updates == 5 and result.array_bytes == 72
    assert [unit.neuron_id for unit in result.history.units] == [8, 2, 5]
    assert [unit.age_updates for unit in result.history.units] == [0, 2, 5]
    assert all(unit.parent_id is None for unit in result.history.units)
    assert asdict(result.history.importance) == {
        "count": 3,
        "minimum": 0.0,
        "mean": 0.25,
        "maximum": 0.5,
    }
    assert result.history.age.mean == pytest.approx(7 / 3)
    assert "batch-aggregate" in result.importance_definition
    assert "replacement" in result.age_definition.lower()
    with pytest.raises(FrozenInstanceError):
        setattr(result.history.units[0], "age_updates", 9)
    stored_history(state).unit_ids[:] = [9, 7, 6]
    stored_history(state).update_age.fill(0)
    assert [unit.neuron_id for unit in result.history.units] == [8, 2, 5]
    assert [unit.age_updates for unit in result.history.units] == [0, 2, 5]


@pytest.mark.parametrize(
    "fault",
    [
        "snapshot_format",
        "missing_history",
        "history_format",
        "ids",
        "next_id",
        "age",
        "newborn_importance",
        "missing_policy",
        "policy",
        "outgoing_dtype",
        "outgoing_width",
        "incoming_dtype",
        "incoming_width",
        "bias_width",
    ],
)
def test_should_refuse_invalid_renewable_state_before_feedforward_svd_or_probe(monkeypatch, fault):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    state = declared_snapshot()
    stored = stored_history(state)
    if fault == "snapshot_format":
        object.__setattr__(state, "format_version", 2)
    elif fault == "missing_history":
        object.__setattr__(state, "unit_history", None)
    elif fault == "history_format":
        object.__setattr__(stored, "format_version", 1)
    elif fault == "ids":
        stored.unit_ids[:] = [8, 8, 5]
    elif fault == "next_id":
        object.__setattr__(stored, "next_unit_id", True)
    elif fault == "age":
        stored.update_age[1] = 2.5
    elif fault == "newborn_importance":
        stored.importance_ema[0] = 1
    elif fault == "missing_policy":
        object.__setattr__(state, "renewal_policy", None)
    elif fault == "policy":
        assert state.renewal_policy is not None
        object.__setattr__(state.renewal_policy, "min_age_updates", True)
    else:
        p = state.parameters
        changes: dict[str, Any] = {
            "outgoing_dtype": {"output_weight": np.zeros((3, 1), dtype=np.int64)},
            "outgoing_width": {"output_weight": np.zeros((2, 1))},
            "incoming_dtype": {"hidden_weights": (np.zeros((2, 3), dtype=np.float32),)},
            "incoming_width": {"hidden_weights": (np.zeros((2, 2)),)},
            "bias_width": {"hidden_biases": (np.zeros((1, 2)),)},
        }
        object.__setattr__(state, "parameters", replace(p, **changes[fault]))
    before = pickle.dumps(state)
    monkeypatch.setattr(pilot, "feedforward_latents", forbidden)
    monkeypatch.setattr(pilot, "diagnose_activations", forbidden)
    monkeypatch.setattr(pilot, "_measure_probe", forbidden)
    row: dict[str, Any] = {}
    with pytest.raises(ValueError):
        read_renewable_coupled_unit_diagnostics(state)
    with pytest.raises(ValueError):
        pilot._measure_features(state, np.zeros((2, 2)), row)
    assert row == {"diagnostic_seconds": 0.0}
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("clock", [False, 4])
def test_should_refuse_cpc_clock_disagreement_before_feature_work(monkeypatch, clock):
    state = declared_snapshot()
    wake = CoupledWakeState(0.995, 0.02, 0.7, 0.2, np.zeros(3), np.ones(3), clock)
    object.__setattr__(state, "wake_state", wake)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "feedforward_latents", forbidden)
    monkeypatch.setattr(pilot, "diagnose_activations", forbidden)
    row: dict[str, Any] = {}
    with pytest.raises(ValueError):
        pilot._measure_features(state, np.zeros((2, 2)), row)
    assert row == {"diagnostic_seconds": 0.0}


def test_should_preserve_renewed_checkpoint_and_template_after_partial_disposable_probe_failure(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    benchmark, template = learner("active"), learner("active")
    state, _ = renewed_checkpoint(benchmark)
    before = pickle.dumps((benchmark, template, state))
    original_fork, original_train = CoupledLearner.fork, CoupledLearner.train_batch
    children: list[CoupledLearner] = []

    def fork(self):
        child = original_fork(self)
        children.append(child)
        return child

    def train(self, x, y):
        original_train(self, x, y)
        raise RuntimeError("disposable return failed")

    monkeypatch.setattr(CoupledLearner, "fork", fork)
    monkeypatch.setattr(CoupledLearner, "train_batch", train)
    observe, slots, _ = observer(template)
    with pytest.raises(RuntimeError, match="fresh probe incomplete"):
        observe(0, state, None)
    assert pickle.dumps((benchmark, template, state)) == before
    assert len(children) == 1
    assert stored_history(children[0].snapshot_state()).updates == 4
    row = slots[0]
    assert row["status"] == "failed" and row["native_unit_history"]["successful_updates"] == 3
    assert row["neuron_age"]["minimum"] == 1
    assert row["probe"]["attempted_updates"] == 1 and row["probe"]["completed_updates"] == 0
    assert [point["updates"] for point in row["probe"]["checkpoints"]] == [0]
    assert all(slot["status"] == "not_run" for slot in slots[1:])


@pytest.mark.parametrize("kind", ["tracked", "untracked", "bp"])
def test_should_preserve_original_tracked_and_unavailable_default_dispatch(monkeypatch, kind):
    supplied = declared_snapshot()
    state: Any
    if kind == "tracked":
        state = TrackedCoupledSnapshot(
            2, supplied.parameters, (1.0,), None, initial_unit_history(3)
        )
    elif kind == "untracked":
        state = CoupledSnapshot(1, supplied.parameters, (1.0,))
    else:
        state = MatchedBackpropSnapshot(1, supplied.parameters)
    monkeypatch.setattr(pilot, "read_renewable_coupled_unit_diagnostics", forbidden)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        pilot, "feedforward_latents", lambda parameters, reference: (np.zeros((2, 3)),)
    )
    monkeypatch.setattr(pilot, "diagnose_activations", lambda values: ())
    before = pickle.dumps(state)
    row: dict[str, Any] = {}
    pilot._measure_features(state, np.zeros((2, 2)), row)
    assert pickle.dumps(state) == before
    keys = {
        "feature_health",
        "diagnostic_activation_bytes",
        "checkpoint_snapshot_array_bytes",
        "importance",
        "neuron_age",
        "unavailable_reason",
        "diagnostic_seconds",
    }
    if kind == "tracked":
        assert set(row) == keys | {"native_unit_history"}
        assert row["native_unit_history"]["family"] == "tracked_coupled"
        assert row["native_unit_history"]["successful_updates"] == 0
        assert row["neuron_age"]["maximum"] == 0 and row["unavailable_reason"] is None
    else:
        assert set(row) == keys
        assert row["importance"] is row["neuron_age"] is None
        assert row["unavailable_reason"] == "No native per-neuron importance/age in coupled models."
