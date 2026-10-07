"""Exercise pinned release/predict adapters using fabricated fields only."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from src.core.confirmation_final_roles import (
    EndpointFailure,
    EndpointResult,
    FinalRole,
    final_content_digest,
)
from src.infra import continual_confirmation_final as adapter
from src.infra.continual_roles import PhaseDecisionRoles, RoleAvailability


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("final adapter opened original outer/train/guard fields")


@dataclass
class _FabricatedSource:
    inputs: np.ndarray
    targets: np.ndarray
    reads_input: int = 0
    reads_target: int = 0

    @property
    def train_input(self) -> np.ndarray:
        return _forbid()

    @property
    def train_target(self) -> np.ndarray:
        return _forbid()

    @property
    def test_input(self) -> np.ndarray:
        self.reads_input += 1
        return self.inputs

    @property
    def test_target(self) -> np.ndarray:
        self.reads_target += 1
        return self.targets


def _original(phase: str) -> tuple[Any, _FabricatedSource]:
    source = _FabricatedSource(
        np.arange(80, dtype=np.float64).reshape(40, 2),
        (np.arange(40) % 2).astype(np.float64).reshape(40, 1),
    )
    ids = {"final_test": tuple(f"phase_{phase}/seed_11/final/{i}" for i in range(40))}
    blocked: Any = object()
    roles = PhaseDecisionRoles(
        phase,
        11,
        blocked,
        blocked,
        blocked,
        ids,
        {name: "a" * 64 for name in ("train", "inner_guard", "outer_selection")},
        {"final_test": RoleAvailability("global_freeze", "global_freeze")},
        40,
        _source=source,
    )
    held: Any = type("FixtureHeld", (), {"roles_a": roles, "roles_b": roles})()
    return held, source


def _role() -> FinalRole:
    item, _ = _original("a")
    return adapter.release_confirmation_final(item, "a")


@pytest.mark.parametrize("phase", ["a", "b"])
def test_should_release_fabricated_fields_once_and_keep_original_sealed(phase: str) -> None:
    item, source = _original(phase)
    original = item.roles_a
    role = adapter.release_confirmation_final(item, phase)
    assert (source.reads_input, source.reads_target) == (1, 1)
    assert original.final_released is False and original.final_test is None
    assert original._source is source
    assert set(original.split_hashes) == {"train", "inner_guard", "outer_selection"}
    assert role.phase == phase and role.seed == 11
    assert role.input is source.inputs and role.target is source.targets
    assert final_content_digest(role) == role.sha256


@pytest.mark.parametrize("phase", ["c", None, []])
def test_should_refuse_wrong_phase_before_source_reads(phase: Any) -> None:
    item, source = _original("a")
    with pytest.raises(ValueError, match="phase a/b"):
        adapter.release_confirmation_final(item, phase)
    assert (source.reads_input, source.reads_target) == (0, 0)


@pytest.mark.parametrize(
    "change",
    [
        "phase",
        "seed",
        "train",
        "ids",
        "policy",
        "hash",
        "extra_hash",
        "retained_source",
        "seal",
        "same_object",
    ],
)
def test_should_refuse_changed_released_metadata_without_opening_other_roles(
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item, source = _original("a")
    original_release = adapter.release_final_test

    def broken(original: PhaseDecisionRoles) -> PhaseDecisionRoles:
        released = original_release(original)
        updates: Any = {
            "phase": {"phase": "b"},
            "seed": {"seed": 12},
            "train": {"train": object()},
            "ids": {"sample_ids": dict(released.sample_ids)},
            "policy": {"release_policy": dict(released.release_policy)},
            "hash": {"split_hashes": {**released.split_hashes, "train": "0" * 64}},
            "extra_hash": {"split_hashes": {**released.split_hashes, "unknown": "0" * 64}},
            "retained_source": {"_source": source},
            "seal": {"final_released": False},
            "same_object": {},
        }[change]
        return original if change == "same_object" else replace(released, **updates)

    monkeypatch.setattr(adapter, "release_final_test", broken)
    with pytest.raises(ValueError, match="changed original role"):
        adapter.release_confirmation_final(item, "a")


def test_should_preserve_original_threshold_with_exact_correct_count() -> None:
    role = _role()
    probabilities = np.tile(np.array([0.499, 0.5, 0.501, 0.5]), 10).reshape(40, 1)
    seen = []

    class Prediction:
        def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
            seen.append(inputs)
            return probabilities

    model: Any = Prediction()
    result = adapter.evaluate_confirmation_final(model, role)
    assert result == EndpointResult(30)
    assert len(seen) == 1 and seen[0] is role.input


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_should_retain_nonfinite_probability_as_explicit_numerical_failure(value: float) -> None:
    probabilities = np.full((40, 1), 0.5)
    probabilities[-1, 0] = value
    model: Any = type("Prediction", (), {"predict_proba": lambda self, inputs: probabilities})()
    assert adapter.evaluate_confirmation_final(model, _role()) == EndpointResult(
        None, EndpointFailure("nonfinite_predictions")
    )


def test_should_retain_floating_point_error_without_swallowing_other_exceptions() -> None:
    class Prediction:
        def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
            raise FloatingPointError("fabricated numerical prediction failure")

    model: Any = Prediction()
    assert adapter.evaluate_confirmation_final(model, _role()) == EndpointResult(
        None, EndpointFailure("numerical_prediction_error", "FloatingPointError")
    )


@pytest.mark.parametrize("error", [ValueError("bad shape"), RuntimeError("unexpected")])
def test_should_propagate_undeclared_prediction_exception(error: Exception) -> None:
    class Prediction:
        def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
            raise error

    model: Any = Prediction()
    with pytest.raises(type(error), match=str(error)):
        adapter.evaluate_confirmation_final(model, _role())


@pytest.mark.parametrize(
    "prediction",
    [
        np.zeros(40),
        np.zeros((40, 2)),
        np.zeros((40, 1), dtype=np.float32),
        np.full((40, 1), -0.1),
        np.full((40, 1), 1.1),
        None,
    ],
)
def test_should_refuse_invalid_prediction_contract_instead_of_substituting_accuracy(
    prediction: Any,
) -> None:
    model: Any = type("Prediction", (), {"predict_proba": lambda self, inputs: prediction})()
    with pytest.raises(ValueError, match="prediction"):
        adapter.evaluate_confirmation_final(model, _role())


def test_should_refuse_changed_final_content_before_model_prediction() -> None:
    role = _role()
    role.input[-1, 0] += 1
    model: Any = type("Prediction", (), {"predict_proba": _forbid})()
    with pytest.raises(ValueError, match="digest"):
        adapter.evaluate_confirmation_final(model, role)
