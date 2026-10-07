"""The replay factor preserves source seals, shared supply, and PC parity."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_replay_factor_pilot import (
    ARMS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    ON_ARMS,
    fixed_replay_pilot_manifest,
    run_replay_pilot,
    validate_replay_pilot_manifest,
)
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedDevelopmentSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("replay pilot opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("replay pilot opened final labels")


def _sealed_generator(**kwargs: object) -> _SealedDevelopmentSource:
    source = generate_two_cluster_dataset_with_transform(**kwargs)  # type: ignore[arg-type]
    return _SealedDevelopmentSource(source.train_input, source.train_target)


def test_should_preflight_frozen_replay_factor_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixed_replay_pilot_manifest()
    assert manifest.seeds == DEVELOPMENT_SEEDS
    assert manifest.confirmation_seeds == CONFIRMATION_SEEDS
    assert manifest.arms == ARMS
    assert validate_replay_pilot_manifest(manifest) == 684

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay source opened before preflight")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    with pytest.raises(ValueError, match="frozen manifest"):
        run_replay_pilot(replace(manifest, seeds=(41, 43, 59, 61)))
    with pytest.raises(ValueError, match="frozen manifest"):
        run_replay_pilot(replace(manifest, planned_width=16))


def test_should_match_replay_ids_and_leave_final_values_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    first = run_replay_pilot(fixed_replay_pilot_manifest())
    second = run_replay_pilot(fixed_replay_pilot_manifest())

    assert first == second
    assert first.final_released is False
    assert tuple(row.seed for row in first.seed_results) == DEVELOPMENT_SEEDS
    for seed_row in first.seed_results:
        assert seed_row.source_role_counts == {
            "a_train": 72,
            "a_inner_guard": 24,
            "a_outer_selection": 24,
            "b_train": 36,
            "b_inner_guard": 12,
            "b_outer_selection": 12,
        }
        assert len(seed_row.source_role_hashes) == 6
        assert [(row.phase, row.epoch) for row in seed_row.boundaries] == [
            ("a", 4),
            ("a", 8),
            ("a", 12),
            ("b", 4),
            ("b", 8),
            ("b", 12),
        ]
        for boundary in seed_row.boundaries:
            assert len(boundary.selected_ids) == 2
            assert boundary.retained_examples == 8
            assert boundary.retained_bytes == 192
            assert tuple(boundary.retained_order_ids[-2:]) == boundary.selected_ids
            assert set(boundary.applied_ids_by_method) == set(ARMS)
            assert all(
                boundary.applied_ids_by_method[name]
                == (boundary.selected_ids if name in ON_ARMS else ())
                for name in ARMS
            )
        methods = {row.method: row for row in seed_row.methods}
        assert tuple(methods) == ARMS
        assert len({methods[name].initial_parameter_sha256 for name in ARMS[:6]}) == 1
        assert (
            methods["pc_off"].final_parameter_sha256
            == methods["circadian_off"].final_parameter_sha256
        )
        assert (
            methods["pc_on"].final_parameter_sha256
            == methods["circadian_on"].final_parameter_sha256
        )
        for name, row in methods.items():
            assert row.wake_updates == 24
            assert row.wake_presentations == 1296
            assert row.replay_updates == (12 if name in ON_ARMS else 0)
            assert row.replay_presentations == row.replay_updates
            assert row.sleep_attempts == (6 if name.startswith("circadian") else 0)
            width, parameters = (12, 49) if "width12" in name else (8, 33)
            assert row.hidden_width_start == row.hidden_width_end == width
            assert row.parameter_count_start == row.parameter_count_end == parameters
            metrics = row.development
            assert metrics.final_mean_task_accuracy == pytest.approx(
                (metrics.a_after_b + metrics.b_after_b) / 2
            )
            assert metrics.signed_forgetting_a == pytest.approx(
                metrics.a_after_a - metrics.a_after_b
            )
