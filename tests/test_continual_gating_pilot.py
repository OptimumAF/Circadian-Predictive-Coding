"""Development-only mechanism pilot has matched PC work and sealed finals."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_gating_pilot import (
    ARMS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    fixed_gating_pilot_manifest,
    run_gating_pilot,
    validate_gating_pilot_manifest,
)
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _DevelopmentSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("pilot opened final-test inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("pilot opened final-test labels")


def _sealed_generator(**kwargs: object) -> _DevelopmentSource:
    source = generate_two_cluster_dataset_with_transform(**kwargs)  # type: ignore[arg-type]
    return _DevelopmentSource(source.train_input, source.train_target)


def test_should_preflight_fixed_seeds_and_work_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixed_gating_pilot_manifest()
    assert manifest.seeds == DEVELOPMENT_SEEDS
    assert manifest.confirmation_seeds == CONFIRMATION_SEEDS
    assert manifest.arms == ARMS
    assert validate_gating_pilot_manifest(manifest) == 216

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("source opened before manifest preflight")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    with pytest.raises(ValueError, match="frozen manifest"):
        run_gating_pilot(replace(manifest, seeds=(41, 43, 59, 61)))


def test_should_match_neutral_pc_and_never_open_final_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)

    first = run_gating_pilot(fixed_gating_pilot_manifest())
    second = run_gating_pilot(fixed_gating_pilot_manifest())

    assert first == second
    assert first.final_released is False
    assert tuple(row.seed for row in first.seed_results) == DEVELOPMENT_SEEDS
    for seed_row in first.seed_results:
        assert set(seed_row.source_role_hashes) == {
            "a_train",
            "a_inner_guard",
            "a_outer_selection",
            "b_train",
            "b_inner_guard",
            "b_outer_selection",
        }
        assert seed_row.source_role_counts["a_train"] == 72
        assert seed_row.source_role_counts["b_train"] == 36
        assert tuple(row.method for row in seed_row.methods) == ARMS
        ordinary, neutral, gating = seed_row.methods
        assert len({row.initial_parameter_sha256 for row in seed_row.methods}) == 1
        assert ordinary.final_parameter_sha256 == neutral.final_parameter_sha256
        assert ordinary.development == neutral.development
        assert gating.minimum_plasticity is not None
        assert 0.2 <= gating.minimum_plasticity < 1.0
        for row in seed_row.methods:
            assert row.wake_updates == 24
            assert row.train_presentations == 1296
            assert row.latent_iterations == 48
            assert row.example_inference_iterations == 2592
            assert row.sleep_attempts == row.replay_updates == 0
            assert row.hidden_width_start == row.hidden_width_end == 8
            assert row.parameter_count_start == row.parameter_count_end == 33
            metrics = row.development
            assert metrics.signed_forgetting == pytest.approx(metrics.a_after_a - metrics.a_after_b)
            assert metrics.final_mean_task_accuracy == pytest.approx(
                (metrics.a_after_b + metrics.b_after_b) / 2.0
            )
