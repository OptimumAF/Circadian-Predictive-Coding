"""Circadian constructors reject nonfinite numeric controls by field name."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import pytest

from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
)
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead


def _construct(backend: str, config: Any) -> None:
    if backend == "numpy":
        CircadianPredictiveCodingNetwork(2, 2, seed=73, circadian_config=config, min_hidden_dim=2)
    else:
        torch = pytest.importorskip("torch")
        CircadianPredictiveCodingHead(
            2, 2, 3, torch.device("cpu"), seed=73, config=config, min_hidden_dim=2
        )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_all_numeric_circadian_config_fields_reject_nonfinite_values(
    backend: str, invalid: float
) -> None:
    config: Any = CircadianConfig() if backend == "numpy" else CircadianHeadConfig()
    checked = 0
    for field in fields(config):
        if not isinstance(field.default, (int, float)):
            continue
        checked += 1
        with pytest.raises(ValueError, match=f"{field.name}.*finite"):
            _construct(backend, replace(config, **{field.name: invalid}))
    assert checked > 40


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_finite_default_and_named_control_configs_still_construct(backend: str) -> None:
    if backend == "numpy":
        configs: tuple[Any, ...] = (CircadianConfig(), CircadianConfig.matched_pc_control())
    else:
        configs = (CircadianHeadConfig(), CircadianHeadConfig.matched_pc_control())
    for config in configs:
        _construct(backend, config)
