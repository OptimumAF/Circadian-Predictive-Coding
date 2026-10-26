"""Exact saved-prefix parity without native training or environment work."""

from copy import deepcopy
from typing import Any

import pytest

from src.adapters import minigrid_warm_start as warm


@pytest.fixture(scope="module")
def accepted() -> dict[str, Any]:
    # Reuse existing admission once. No native models/probe/final data.
    return warm._admit_reference(warm._load_reference())


@pytest.mark.parametrize("index", range(5))
def test_should_accept_exact_saved_readout_with_equivalent_json_containers(
    accepted: dict[str, Any], index: int
) -> None:
    original = accepted["evaluations"][index]
    current = deepcopy(original)
    for score in current["scores"]:
        score["probabilities"] = tuple(tuple(row) for row in score["probabilities"])
    warm._parity(original, current)


@pytest.mark.parametrize("index", range(5))
def test_should_refuse_tiny_probability_drift_at_every_warm_checkpoint(
    accepted: dict[str, Any], index: int
) -> None:
    original = accepted["evaluations"][index]
    current = deepcopy(original)
    # Preserve normalized mass; acceptance has no float tolerance or repair.
    current["scores"][0]["probabilities"][0][0] += 1e-8
    current["scores"][0]["probabilities"][0][1] -= 1e-8
    with pytest.raises(ValueError, match="raw TRAIN"):
        warm._parity(original, current)


@pytest.mark.parametrize("target", ["action", "next_observation", "reward", "truncated"])
def test_should_refuse_changed_native_prefix_law(accepted: dict[str, Any], target: str) -> None:
    original = accepted["evaluations"][-1]
    current = deepcopy(original)
    transition = current["episodes"][0]["environment"]["transitions"][0]
    if target == "next_observation":
        transition[target][0] = 0.125
    elif target == "truncated":
        transition[target] = True
    else:
        transition[target] += 1
    with pytest.raises(ValueError, match="physics"):
        warm._parity(original, current)


@pytest.mark.parametrize("target", ["action", "probabilities", "success", "return"])
def test_should_refuse_changed_native_prediction_or_outcome(
    accepted: dict[str, Any], target: str
) -> None:
    original = accepted["evaluations"][-1]
    current = deepcopy(original)
    episode = current["episodes"][0]
    if target == "action":
        episode["predictions"][0]["action"] = 6
    elif target == "probabilities":
        episode["predictions"][0]["probabilities"][0] += 1e-8
    elif target == "success":
        episode["success"] = not episode["success"]
    else:
        episode["return"] += 1
    with pytest.raises(ValueError, match="decision/outcome"):
        warm._parity(original, current)
