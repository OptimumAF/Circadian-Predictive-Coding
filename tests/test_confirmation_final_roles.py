"""Pure fabricated final-role/count contracts; no dataset or model construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.core.confirmation_final_roles import (
    EndpointFailure,
    EndpointResult,
    FinalRole,
    capture_final_role,
    final_content_digest,
    validate_endpoint_result,
)
from src.infra.continual_roles import _hash_role
from src.infra.datasets import LabeledData


def _role(phase: str = "a") -> FinalRole:
    count, seed = 40, 11
    inputs = np.arange(count * 2, dtype=np.float64).reshape(count, 2) / 100
    targets = (np.arange(count) % 2).astype(np.float64).reshape(count, 1)
    ids = tuple(f"phase_{phase}/seed_{seed}/final/{index}" for index in range(count))
    data = LabeledData(inputs, targets)
    return FinalRole(
        phase, seed, ids, inputs, targets, _hash_role(phase, seed, "final_test", ids, data)
    )


@pytest.mark.parametrize("phase", ["a", "b"])
def test_should_match_original_role_hash_for_complete_fabricated_final_bytes(phase: str) -> None:
    role = _role(phase)
    facts = capture_final_role(role, 40)
    assert facts.sha256 == role.sha256
    assert final_content_digest(role) == role.sha256
    assert facts.phase == phase and facts.seed == 11 and facts.count == 40
    assert facts.sample_ids == role.sample_ids
    swapped = replace(role, sample_ids=role.sample_ids[::-1])
    assert final_content_digest(swapped) != role.sha256


@pytest.mark.parametrize(
    "change",
    [
        "seed_bool",
        "phase",
        "id_list",
        "duplicate",
        "id_type",
        "id_ascii",
        "id_nul",
        "count",
        "input_dtype",
        "target_dtype",
        "input_shape",
        "target_shape",
        "nonfinite",
        "soft_labels",
        "hash",
        "hash_type",
    ],
)
def test_should_refuse_ambiguous_or_changed_role_identity_and_values(change: str) -> None:
    role = _role()
    updates: Any = {}
    if change == "seed_bool":
        updates = {"seed": True}
    elif change == "phase":
        updates = {"phase": "c"}
    elif change == "id_list":
        updates = {"sample_ids": list(role.sample_ids)}
    elif change == "duplicate":
        updates = {"sample_ids": (role.sample_ids[0], *role.sample_ids[:-1])}
    elif change in {"id_type", "id_ascii", "id_nul"}:
        bad: Any = {"id_type": 1, "id_ascii": "é", "id_nul": "a\0b"}[change]
        updates = {"sample_ids": (bad, *role.sample_ids[1:])}
    elif change == "count":
        updates = {"sample_ids": role.sample_ids[:-1]}
    elif change == "input_dtype":
        updates = {"input": role.input.astype(np.float32)}
    elif change == "target_dtype":
        updates = {"target": role.target.astype(np.int64)}
    elif change == "input_shape":
        updates = {"input": role.input[:, :1]}
    elif change == "target_shape":
        updates = {"target": role.target.reshape(-1)}
    elif change == "nonfinite":
        inputs = role.input.copy()
        inputs[-1, 0] = np.nan
        updates = {"input": inputs}
    elif change == "soft_labels":
        targets = role.target.copy()
        targets[-1, 0] = 0.2
        updates = {"target": targets}
    elif change == "hash":
        updates = {"sha256": "0" * 64}
    else:
        updates = {"sha256": None}
    with pytest.raises(ValueError, match="final role"):
        capture_final_role(replace(role, **updates), 40)


def test_should_detect_changed_content_even_when_shape_and_ids_stay_fixed() -> None:
    role = _role()
    role.input[-1, 1] += 0.1
    with pytest.raises(ValueError, match="digest"):
        capture_final_role(role, 40)


@pytest.mark.parametrize("count", [0, -1, True, 40.0])
def test_should_refuse_invalid_declared_final_count(count: Any) -> None:
    with pytest.raises(ValueError, match="final role"):
        capture_final_role(_role(), count)


@pytest.mark.parametrize("correct", [0, 1, 21, 40])
def test_should_preserve_exact_discrete_count_accuracy(correct: int) -> None:
    assert validate_endpoint_result(EndpointResult(correct), 40) == correct / 40


@pytest.mark.parametrize("correct", [-1, 41, True, 21.0, np.int64(21), None])
def test_should_refuse_invalid_or_ambiguous_count_without_failure(correct: Any) -> None:
    with pytest.raises(ValueError, match="endpoint result"):
        validate_endpoint_result(EndpointResult(correct), 40)


@pytest.mark.parametrize(
    "failure",
    [
        EndpointFailure("nonfinite_predictions"),
        EndpointFailure("numerical_prediction_error", "FloatingPointError"),
    ],
)
def test_should_retain_declared_numerical_failure_as_null(failure: EndpointFailure) -> None:
    assert validate_endpoint_result(EndpointResult(None, failure), 40) is None


@pytest.mark.parametrize(
    "failure",
    [
        EndpointFailure("unknown"),
        EndpointFailure("nonfinite_predictions", "TypeError"),
        EndpointFailure("numerical_prediction_error"),
        EndpointFailure("numerical_prediction_error", "ValueError"),
    ],
)
def test_should_refuse_undeclared_failure_or_exception_policy(failure: EndpointFailure) -> None:
    with pytest.raises(ValueError, match="endpoint result"):
        validate_endpoint_result(EndpointResult(None, failure), 40)


def test_should_refuse_success_that_also_declares_failure() -> None:
    with pytest.raises(ValueError, match="endpoint result"):
        validate_endpoint_result(EndpointResult(20, EndpointFailure("nonfinite_predictions")), 40)
