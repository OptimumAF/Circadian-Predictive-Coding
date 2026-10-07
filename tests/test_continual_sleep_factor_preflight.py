"""No-replay sleep factors keep roles sealed and guard proposals auditable."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_sleep_factor_preflight import (
    ARMS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    SLEEP_ARMS,
    fixed_sleep_factor_manifest,
    run_sleep_factor_preflight,
    validate_sleep_factor_manifest,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("sleep-factor preflight opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("sleep-factor preflight opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("sleep-factor preflight scored outer inputs")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("sleep-factor preflight scored outer labels")


def _sealed_generator(**kwargs: object) -> _SealedSource:
    source = generate_two_cluster_dataset_with_transform(**kwargs)  # type: ignore[arg-type]
    return _SealedSource(source.train_input, source.train_target)


def test_should_preflight_frozen_sleep_factor_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixed_sleep_factor_manifest()
    assert manifest.seeds == DEVELOPMENT_SEEDS
    assert manifest.confirmation_seeds == CONFIRMATION_SEEDS
    assert manifest.arms == ARMS
    assert validate_sleep_factor_manifest(manifest) == 648

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("sleep-factor source opened before manifest preflight")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    with pytest.raises(ValueError, match="frozen manifest"):
        run_sleep_factor_preflight(replace(manifest, seeds=(67, 71, 73, 79)))
    with pytest.raises(ValueError, match="frozen manifest"):
        run_sleep_factor_preflight(replace(manifest, planned_width=16))


def test_should_run_train_only_factors_with_outer_and_final_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    first = run_sleep_factor_preflight(fixed_sleep_factor_manifest())
    second = run_sleep_factor_preflight(fixed_sleep_factor_manifest())

    assert first == second
    assert first.final_released is False
    assert first.outer_selection_scored is False
    assert tuple(seed.seed for seed in first.seed_results) == DEVELOPMENT_SEEDS
    for seed in first.seed_results:
        assert seed.final_released is False
        assert seed.role_counts == {
            "a_train": 72,
            "a_inner_guard": 24,
            "a_outer_selection": 24,
            "b_train": 36,
            "b_inner_guard": 12,
            "b_outer_selection": 12,
        }
        assert len(seed.role_hashes) == 6
        arms = {arm.name: arm for arm in seed.arms}
        assert tuple(arms) == ARMS
        assert len({arms[name].initial_parameter_sha256 for name in ARMS[:7]}) == 1
        assert (
            arms["backprop_12"].initial_parameter_sha256 == arms["pc_12"].initial_parameter_sha256
        )
        assert arms["pc_8"].final_parameter_sha256 == arms["neutral_sham"].final_parameter_sha256
        assert (
            arms["gating_sham"].pre_sleep_parameter_sha256
            == arms["gating_reset"].pre_sleep_parameter_sha256
        )
        assert (
            arms["gating_sham"].post_sleep_parameter_sha256
            == arms["gating_reset"].post_sleep_parameter_sha256
        )
        for name, arm in arms.items():
            assert arm.wake_updates == 24
            assert arm.wake_presentations == 1296
            assert arm.replay_updates == 0
            assert arm.latent_inference_loops == (48 if not name.startswith("backprop") else 0)
            assert arm.example_inference_iterations == (
                2592 if not name.startswith("backprop") else 0
            )
            width, parameters = (12, 49) if name.endswith("_12") else (8, 33)
            assert arm.width_initial == arm.width_final == width
            assert arm.parameters_initial == arm.parameters_final == parameters
            if name in SLEEP_ARMS:
                assert arm.sleep is not None
                assert arm.sleep.outcome == "accepted"
                assert arm.sleep.guard_role_hash == seed.role_hashes["a_inner_guard"]
                assert arm.sleep.guard_evaluations == 2
                assert arm.sleep.replay_updates == 0
            else:
                assert arm.sleep is None
        structure = arms["structure_only"]
        assert structure.sleep is not None
        assert len(structure.sleep.proposed_split_pairs) == 1
        assert len(structure.sleep.applied_split_pairs) == 1
        assert len(structure.sleep.proposed_removed_prune_ids) == 1
        assert len(structure.sleep.applied_removed_prune_ids) == 1
        assert structure.width_peak == 9 and structure.parameters_peak == 37
        for name in ("gating_sham", "gating_reset"):
            minimum = arms[name].minimum_a_plasticity
            assert minimum is not None
            assert 0.2 <= minimum < 1.0
        reset = arms["gating_reset"].sleep
        assert reset is not None
        assert 0.0 < reset.chemical_proposed_mean < reset.chemical_before_mean


def test_should_roll_back_rejected_structural_proposal_without_losing_proposal_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CircadianPredictiveCodingNetwork.compute_accuracy
    calls: dict[CircadianPredictiveCodingNetwork, int] = {}

    def reject_structure(
        self: CircadianPredictiveCodingNetwork, inputs: np.ndarray, targets: np.ndarray
    ) -> float:
        if self.config.sleep_enable_split:
            count = calls.get(self, 0)
            calls[self] = count + 1
            return 1.0 if count == 0 else 0.0
        return float(original(self, inputs, targets))

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", reject_structure)
    result = run_sleep_factor_preflight(fixed_sleep_factor_manifest())

    for seed in result.seed_results:
        structural = next(arm for arm in seed.arms if arm.name == "structure_only")
        assert structural.sleep is not None
        assert structural.sleep.outcome == "rolled_back"
        assert len(structural.sleep.proposed_split_pairs) == 1
        assert len(structural.sleep.proposed_removed_prune_ids) == 1
        assert structural.sleep.applied_split_pairs == ()
        assert structural.sleep.applied_removed_prune_ids == ()
        assert structural.pre_sleep_parameter_sha256 == structural.post_sleep_parameter_sha256
        assert structural.width_final == 8
