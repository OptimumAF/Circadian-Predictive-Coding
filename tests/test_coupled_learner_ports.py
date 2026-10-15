"""Real native coupled learners at the owned adapter boundary.

Compare native and adapter trajectories, including CPC wake state, across restore
and continuation. Refusal tests assert adapter publication is atomic. These are
tiny deterministic integration controls, not model or scientific comparisons.
"""

from dataclasses import replace
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledWakeState,
)
from src.core.learner_ports import TrainingDiagnostic

Array = NDArray[np.float64]
WIDTHS = ((2,), (2, 3), (2, 3, 2, 1))
RATE, STEPS, INFERENCE_RATE = 0.025, 3, 0.07


def batch() -> tuple[Array, Array]:
    return np.array(
        [
            [-0.8, 0.2, 0.4],
            [0.3, -0.6, 0.1],
            [0.7, 0.4, -0.2],
            [-0.1, 0.8, -0.5],
        ]
    ), np.array([[0.0], [1.0], [0.0], [1.0]])


def native_model(kind: str, widths: tuple[int, ...] = (2, 3)) -> CoupledPredictiveCodingNetwork:
    precisions = (0.7, 1.4, 0.6, 2.1)[: len(widths)]
    if kind == "pc":
        return CoupledPredictiveCodingNetwork(3, widths, seed=41, precisions=precisions)
    assert kind == "circadian"
    return CoupledCircadianPredictiveCodingNetwork(3, widths, seed=41, precisions=precisions)


def adapter(model: CoupledPredictiveCodingNetwork) -> CoupledLearner:
    return CoupledLearner(
        model, learning_rate=RATE, inference_steps=STEPS, inference_learning_rate=INFERENCE_RATE
    )


def parameter_arrays(parameters: CoupledParameters) -> tuple[Array, ...]:
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def assert_complete_snapshot(actual: CoupledSnapshot, expected: CoupledSnapshot) -> None:
    assert type(actual) is CoupledSnapshot
    assert (
        type(actual.format_version) is int and actual.format_version == expected.format_version == 1
    )
    assert type(actual.precisions) is tuple and actual.precisions == expected.precisions
    for left, right in zip(
        parameter_arrays(actual.parameters), parameter_arrays(expected.parameters), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    if expected.wake_state is None:
        assert actual.wake_state is None
        return
    wake, reference = actual.wake_state, expected.wake_state
    assert type(wake) is CoupledWakeState
    assert wake is not None
    for name in (
        "chemical_decay",
        "chemical_buildup_rate",
        "plasticity_sensitivity",
        "min_plasticity",
    ):
        assert getattr(wake, name) == getattr(reference, name)
    assert type(wake.wake_updates) is int and wake.wake_updates == reference.wake_updates
    np.testing.assert_array_equal(wake.chemical_state, reference.chemical_state)
    np.testing.assert_array_equal(wake.plasticity_state, reference.plasticity_state)


def assert_native_step(
    learner: CoupledLearner,
    direct: CoupledPredictiveCodingNetwork,
    features: Array,
    targets: Array,
) -> None:
    before_features, before_targets = features.copy(), targets.copy()
    expected = direct.train_epoch(features, targets, RATE, STEPS, INFERENCE_RATE)
    diagnostic = learner.train_batch(features, targets)
    assert type(diagnostic) is TrainingDiagnostic
    assert diagnostic.definition == expected.energy_definition == "numpy_coupled_binary_energy_v1"
    assert diagnostic.value == expected.energy
    assert_complete_snapshot(learner.snapshot_state(), direct.snapshot_state())
    np.testing.assert_array_equal(learner.predict(features), direct.predict_proba(features))
    np.testing.assert_array_equal(features, before_features)
    np.testing.assert_array_equal(targets, before_targets)


@pytest.mark.parametrize("kind", ("pc", "circadian"))
@pytest.mark.parametrize("widths", WIDTHS, ids=("one-layer", "two-layers", "four-layers"))
def test_adapter_matches_native_wakes_restore_and_continuation(
    kind: str,
    widths: tuple[int, ...],
) -> None:
    source, direct = native_model(kind, widths), native_model(kind, widths)
    learner = adapter(source)
    assert_complete_snapshot(learner.snapshot_state(), direct.snapshot_state())
    features, targets = batch()
    for rows in ([0, 1, 2], [3, 1]):
        assert_native_step(learner, direct, features[rows], targets[rows])
    saved = learner.snapshot_state()
    if kind == "circadian":
        assert saved.wake_state is not None and saved.wake_state.wake_updates == 2
        assert np.any(saved.wake_state.chemical_state > 0)
    learner.train_batch(features[[2]], targets[[2]])
    # Widen only the consumer's return type so the actual None result is checked.
    restore: Callable[[CoupledSnapshot], object] = learner.restore_state
    assert restore(saved) is None
    direct.restore_state(saved)
    assert_complete_snapshot(learner.snapshot_state(), direct.snapshot_state())
    # The snapshot contains state, not a replacement adapter learning policy.
    assert_native_step(learner, direct, features, targets)


@pytest.mark.parametrize("kind", ("pc", "circadian"))
def test_constructor_fork_snapshots_predictions_and_source_have_independent_ownership(
    kind: str,
) -> None:
    features, targets = batch()
    source = native_model(kind)
    source.train_epoch(features, targets, RATE, STEPS, INFERENCE_RATE)
    saved = source.snapshot_state()
    learner = adapter(source)
    fork = learner.fork()
    assert type(fork) is CoupledLearner and fork is not learner
    assert_complete_snapshot(learner.snapshot_state(), saved)
    assert_complete_snapshot(fork.snapshot_state(), saved)
    source.train_epoch(features[:2], targets[:2], RATE, STEPS, INFERENCE_RATE)
    assert_complete_snapshot(learner.snapshot_state(), saved)
    assert_complete_snapshot(fork.snapshot_state(), saved)
    returned = learner.snapshot_state()
    for array in parameter_arrays(returned.parameters):
        array.fill(99)
    if returned.wake_state is not None:
        returned.wake_state.chemical_state.fill(99)
        returned.wake_state.plasticity_state.fill(0)
    assert_complete_snapshot(learner.snapshot_state(), saved)
    prediction = learner.predict(features)
    expected_prediction = prediction.copy()
    prediction.fill(99)
    np.testing.assert_array_equal(learner.predict(features), expected_prediction)
    direct = native_model(kind)
    direct.restore_state(saved)
    assert_native_step(learner, direct, features[:2], targets[:2])
    assert_complete_snapshot(fork.snapshot_state(), saved)
    fork_diagnostic = fork.train_batch(features[:2], targets[:2])
    assert fork_diagnostic.definition == "numpy_coupled_binary_energy_v1"
    assert_complete_snapshot(fork.snapshot_state(), learner.snapshot_state())


COMMON_BAD_SNAPSHOTS = (
    "format",
    "bool-format",
    "foreign-header",
    "precision",
    "nan-parameter",
    "topology",
    "family",
)


def malformed_snapshot(case: str, snapshot: CoupledSnapshot, kind: str) -> object:
    if case == "format":
        return replace(snapshot, format_version=2)
    if case == "bool-format":
        return replace(snapshot, format_version=True)
    if case == "foreign-header":
        return {"format_version": 1, "parameters": snapshot.parameters}
    if case == "precision":
        return replace(snapshot, precisions=(0.7, 1.2))
    if case == "nan-parameter":
        snapshot.parameters.hidden_weights[0][0, 0] = np.nan
        return snapshot
    if case == "topology":
        return native_model(kind, (3, 2)).snapshot_state()
    assert case == "family"
    return native_model("circadian" if kind == "pc" else "pc").snapshot_state()


def assert_restore_refused_without_publication(learner: CoupledLearner, bad: object) -> None:
    before = learner.snapshot_state()
    features, _ = batch()
    prediction = learner.predict(features)
    with pytest.raises(ValueError):
        learner.restore_state(cast(CoupledSnapshot, bad))
    assert_complete_snapshot(learner.snapshot_state(), before)
    np.testing.assert_array_equal(learner.predict(features), prediction)


@pytest.mark.parametrize("kind", ("pc", "circadian"))
@pytest.mark.parametrize("case", COMMON_BAD_SNAPSHOTS)
def test_adapter_refuses_incompatible_snapshot_without_publishing_any_state(
    kind: str, case: str
) -> None:
    learner = adapter(native_model(kind))
    features, targets = batch()
    learner.train_batch(features, targets)
    bad = malformed_snapshot(case, learner.snapshot_state(), kind)
    assert_restore_refused_without_publication(learner, bad)
    # Refusal also preserves the fixed adapter policy and ability to continue.
    direct = native_model(kind)
    direct.restore_state(learner.snapshot_state())
    assert_native_step(learner, direct, features[:2], targets[:2])


@pytest.mark.parametrize("case", ("negative-count", "bool-count", "zero-count", "gate", "config"))
def test_adapter_refuses_malformed_cpc_wake_state_atomically(case: str) -> None:
    learner = adapter(native_model("circadian"))
    features, targets = batch()
    learner.train_batch(features, targets)
    snapshot = learner.snapshot_state()
    wake = snapshot.wake_state
    assert wake is not None
    if case == "negative-count":
        wake = replace(wake, wake_updates=-1)
    elif case == "bool-count":
        wake = replace(wake, wake_updates=True)
    elif case == "zero-count":
        wake = replace(wake, wake_updates=0)
    elif case == "gate":
        wake = replace(wake, plasticity_state=np.zeros_like(wake.plasticity_state))
    else:
        wake = replace(wake, plasticity_sensitivity=wake.plasticity_sensitivity + 0.1)
    assert_restore_refused_without_publication(learner, replace(snapshot, wake_state=wake))


BAD_POLICY: tuple[tuple[str, object], ...] = (
    ("learning_rate", 0),
    ("learning_rate", -0.1),
    ("learning_rate", True),
    ("learning_rate", float("nan")),
    ("learning_rate", float("inf")),
    ("learning_rate", "0.1"),
    ("learning_rate", 10**1000),
    ("learning_rate", np.float64(0.1)),
    ("inference_learning_rate", 0),
    ("inference_learning_rate", False),
    ("inference_learning_rate", float("nan")),
    ("inference_learning_rate", float("inf")),
    ("inference_steps", 0),
    ("inference_steps", -1),
    ("inference_steps", True),
    ("inference_steps", 1.5),
    ("inference_steps", 10001),
)


@pytest.mark.parametrize("name,value", BAD_POLICY)
def test_adapter_refuses_invalid_fixed_training_policy_at_construction(
    name: str, value: object
) -> None:
    declared: dict[str, object] = {
        "learning_rate": RATE,
        "inference_steps": STEPS,
        "inference_learning_rate": INFERENCE_RATE,
    }
    declared[name] = value
    source = native_model("pc")
    before = source.snapshot_state()
    with pytest.raises(ValueError):
        CoupledLearner(
            source,
            learning_rate=cast(float, declared["learning_rate"]),
            inference_steps=cast(int, declared["inference_steps"]),
            inference_learning_rate=cast(float, declared["inference_learning_rate"]),
        )
    assert_complete_snapshot(source.snapshot_state(), before)


@pytest.mark.parametrize("kind", ("pc-subclass", "cpc-subclass", "unrelated"))
def test_adapter_requires_an_exact_supported_native_model(kind: str) -> None:
    class PcSubclass(CoupledPredictiveCodingNetwork):
        pass

    class CpcSubclass(CoupledCircadianPredictiveCodingNetwork):
        pass

    model: object = (
        PcSubclass(3, (2,), seed=41)
        if kind == "pc-subclass"
        else CpcSubclass(3, (2,), seed=41)
        if kind == "cpc-subclass"
        else object()
    )
    with pytest.raises(ValueError):
        adapter(cast(CoupledPredictiveCodingNetwork, model))
