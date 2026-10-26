"""Optional stream-factory policy binding, without native training or inference."""

from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

import src.adapters.coupled_attribution_pilot as pilot
from src.adapters.coupled_learners import CoupledLearner
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.tracked_coupled_predictive_coding import (
    TrackedCoupledCircadianPredictiveCodingNetwork,
    TrackedCoupledPredictiveCodingNetwork,
)


@pytest.fixture(autouse=True)
def forbid_native_updates(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("factory tests must not perform native SGD or inference")

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "train_epoch", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "train_epoch", forbidden)


@pytest.fixture
def initial():
    return CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("tracked", [False, True])
def test_should_keep_owned_bounded_policy_and_original_tensors_through_fork_restore(
    method, tracked, initial
):
    supplied = CoupledInferencePolicy(2, 1.0, None)
    reference = asdict(initial)
    learner = pilot.build_learner(
        method, initial, track_unit_history=tracked, inference_policy=supplied
    )
    assert isinstance(learner, CoupledLearner)
    expected_kind = (
        TrackedCoupledPredictiveCodingNetwork if tracked else CoupledPredictiveCodingNetwork
    )
    if method != "pc":
        expected_kind = (
            TrackedCoupledCircadianPredictiveCodingNetwork
            if tracked
            else CoupledCircadianPredictiveCodingNetwork
        )
        expected_config = (
            CoupledCircadianConfig.matched_pc_control()
            if method == "neutral_cpc"
            else CoupledCircadianConfig()
        )
        assert isinstance(learner._model, CoupledCircadianPredictiveCodingNetwork)
        assert learner._model.config == expected_config
    assert type(learner._model) is expected_kind
    policy = learner.checkpoint_policy()
    assert (policy.learning_rate, policy.inference_steps, policy.inference_learning_rate) == (
        0.03,
        4,
        0.2,
    )
    assert policy.inference_policy == supplied
    assert learner._inference_policy is not supplied
    assert policy.inference_policy is not learner._inference_policy
    assert learner.snapshot_state().precisions == (1.0, 1.0)
    np.testing.assert_equal(asdict(learner.snapshot_state().parameters), reference)

    fork = learner.fork()
    assert isinstance(fork, CoupledLearner)
    snapshot = learner.snapshot_state()
    learner.restore_state(snapshot)
    fork.restore_state(snapshot)
    assert fork._inference_policy is not learner._inference_policy
    assert fork.checkpoint_policy() == learner.checkpoint_policy() == policy

    object.__setattr__(supplied, "gradient_tolerance", 99.0)
    object.__setattr__(policy.inference_policy, "min_steps", 4)
    initial.output_weight.fill(999)
    snapshot.parameters.output_weight.fill(-999)
    for owned in (learner, fork):
        assert owned.checkpoint_policy().inference_policy == CoupledInferencePolicy(2, 1.0, None)
        np.testing.assert_equal(asdict(owned.snapshot_state().parameters), reference)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("tracked", [False, True])
def test_should_preserve_old_constructor_call_shape_when_policy_is_none(
    monkeypatch, method, tracked, initial
):
    calls = []
    marker = object()

    def capture(model, **kwargs):
        calls.append(kwargs)
        return marker

    monkeypatch.setattr(pilot, "CoupledLearner", capture)
    assert pilot.build_learner(method, initial, track_unit_history=tracked) is marker
    assert (
        pilot.build_learner(method, initial, track_unit_history=tracked, inference_policy=None)
        is marker
    )
    assert (
        calls == [{"learning_rate": 0.03, "inference_steps": 4, "inference_learning_rate": 0.2}] * 2
    )


def test_should_preserve_backprop_constructor_call_shape_for_none(monkeypatch, initial):
    model = object()
    learner = object()
    calls = []
    monkeypatch.setattr(pilot, "MatchedBackpropNetwork", lambda value: model)

    def capture(native, **kwargs):
        calls.append((native, kwargs))
        return learner

    monkeypatch.setattr(pilot, "MatchedBackpropLearner", capture)
    assert pilot.build_learner("bp", initial) is learner
    assert pilot.build_learner("bp", initial, inference_policy=None) is learner
    assert calls == [(model, {"learning_rate": 0.03})] * 2


def corrupted_policy(kind):
    policy = CoupledInferencePolicy(2, 1.0, None)
    mutations = {
        "bool_min": ("min_steps", True),
        "zero_min": ("min_steps", 0),
        "over_cap": ("min_steps", 5),
        "bool_gradient": ("gradient_tolerance", True),
        "negative_gradient": ("gradient_tolerance", -1.0),
        "infinite_gradient": ("gradient_tolerance", float("inf")),
        "nan_relative": ("relative_energy_tolerance", float("nan")),
        "extra_field": ("unrecognized", 1),
    }
    if kind == "missing_field":
        object.__delattr__(policy, "min_steps")
    else:
        field, value = mutations[kind]
        object.__setattr__(policy, field, value)
    return policy


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc", "bp"])
@pytest.mark.parametrize(
    "kind",
    [
        "wrong_type",
        "subclass",
        "bool_min",
        "zero_min",
        "over_cap",
        "bool_gradient",
        "negative_gradient",
        "infinite_gradient",
        "nan_relative",
        "extra_field",
        "missing_field",
    ],
)
def test_should_reject_invalid_policy_before_any_native_construction(monkeypatch, method, kind):
    policy: Any

    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached native construction")

    for name in (
        "MatchedBackpropNetwork",
        "CoupledPredictiveCodingNetwork",
        "CoupledCircadianPredictiveCodingNetwork",
        "TrackedCoupledPredictiveCodingNetwork",
        "TrackedCoupledCircadianPredictiveCodingNetwork",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    if kind == "wrong_type":
        policy = {"min_steps": 2}
    elif kind == "subclass":

        class OtherPolicy(CoupledInferencePolicy):
            pass

        # The native policy constructor itself refuses subclasses.
        policy = object.__new__(OtherPolicy)
    else:
        policy = corrupted_policy(kind)
    with pytest.raises(ValueError):
        pilot.build_learner(method, cast(Any, object()), inference_policy=policy)


def test_should_reject_valid_bounded_policy_for_backprop_before_native_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported policy reached backprop constructor")

    monkeypatch.setattr(pilot, "MatchedBackpropNetwork", forbidden)
    with pytest.raises(ValueError, match="backpropagation"):
        pilot.build_learner(
            "bp", cast(Any, object()), inference_policy=CoupledInferencePolicy(2, 1.0, None)
        )
