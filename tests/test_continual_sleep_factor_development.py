"""Scored sleep factors wait for a complete matching train-only gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_sleep_factor_development import (
    CONTRASTS,
    run_sleep_factor_development,
)
from src.app.continual_sleep_factor_preflight import (
    ARMS,
    fixed_sleep_factor_manifest,
    run_sleep_factor_preflight,
)
from src.core.continual_metrics import TwoTaskAccuracy
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("sleep-factor development opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("sleep-factor development opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("sleep-factor development opened outer input before global gate")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("sleep-factor development opened outer label before global gate")


def _sealed_generator(**kwargs: object) -> _SealedSource:
    source = generate_two_cluster_dataset_with_transform(**kwargs)  # type: ignore[arg-type]
    return _SealedSource(source.train_input, source.train_target)


@lru_cache(maxsize=1)
def _local_reference() -> dict[str, Any]:
    # Why this: this portable gate test must not overwrite historical evidence.
    result = run_sleep_factor_preflight(fixed_sleep_factor_manifest())
    return json.loads(json.dumps(asdict(result), allow_nan=False))


def _reference() -> dict[str, Any]:
    return deepcopy(_local_reference())


def test_should_reject_changed_reference_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()
    reference["protocol_id"] = "changed"

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("source opened before reference validation")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    with pytest.raises(ValueError, match="sealed c3 manifest"):
        run_sleep_factor_development(fixed_sleep_factor_manifest(), reference)


def test_should_fail_global_train_fact_mismatch_before_any_outer_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = deepcopy(_reference())
    reference["seed_results"][-1]["arms"][-1]["final_parameter_sha256"] = "0" * 64
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    original_a = arrived._build_phase_a_roles
    original_b = arrived._build_phase_b_roles
    monkeypatch.setattr(
        arrived,
        "_build_phase_a_roles",
        lambda *args: replace(original_a(*args), outer_selection=_BlockedOuter()),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        arrived,
        "_build_phase_b_roles",
        lambda *args: replace(original_b(*args), outer_selection=_BlockedOuter()),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="train facts differ"):
        run_sleep_factor_development(fixed_sleep_factor_manifest(), reference)


def test_should_score_every_prespecified_cell_after_global_gate_with_final_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    result = run_sleep_factor_development(fixed_sleep_factor_manifest(), _reference())
    assert result.outer_selection_scored is True
    assert result.final_released is False
    assert len(result.scored_seeds) == 3
    assert sum(len(seed.arms) for seed in result.scored_seeds) == 27
    for seed in result.scored_seeds:
        assert tuple(arm.name for arm in seed.arms) == ARMS
        assert tuple((row.left, row.right) for row in seed.contrasts) == CONTRASTS
        arms = {arm.name: arm for arm in seed.arms}
        assert arms["pc_8"].a_after_a == arms["neutral_sham"].a_after_a
        assert arms["pc_8"].a_after_b == arms["neutral_sham"].a_after_b
        assert arms["pc_8"].b_after_b == arms["neutral_sham"].b_after_b
        for arm in seed.arms:
            metrics = TwoTaskAccuracy(arm.a_after_a, arm.a_after_b, arm.b_after_b)
            assert arm.final_mean_task_accuracy == metrics.final_mean_task_accuracy
            assert arm.signed_forgetting_a == metrics.signed_forgetting_a
            assert arm.retention_ratio_a == metrics.retention_ratio_a
