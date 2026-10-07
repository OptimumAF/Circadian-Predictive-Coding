"""Combined factors remain train-only and charge all rollback executions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_combined_factor_preflight as study
from src.app import continual_shift_benchmark as base
from src.app.continual_combined_factor_manifest import (
    DECISION_ARMS,
    FULL_ARMS,
    PARITY_PAIRS,
    REPLAY_CONTROLS,
    fixed_combined_manifest,
    validate_combined_manifest,
)
from src.app.continual_combined_factor_validation import verify_combined_payload
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("combined gate opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("combined gate opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("combined gate opened outer inputs")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("combined gate opened outer labels")


def _sealed_generator(**kwargs: Any) -> _SealedSource:
    data = generate_two_cluster_dataset_with_transform(**kwargs)
    return _SealedSource(data.train_input, data.train_target)


def test_should_freeze_component_removals_and_budget_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixed_combined_manifest()
    assert validate_combined_manifest(manifest) == 1548
    assert len(manifest.arms) == 17 and manifest.seeds == (263, 269, 271)
    assert not set(manifest.seeds) & set(manifest.confirmation_seeds)
    configs = {
        arm.name: asdict(arm.config)
        for arm in manifest.arms
        if arm.name in FULL_ARMS and arm.config is not None
    }
    changes = {
        "minus_replay": {"sleep_enable_replay"},
        "minus_gating": {"min_plasticity"},
        "minus_structure": {"sleep_enable_split", "sleep_enable_prune"},
        "minus_schedule": {"sleep_mode"},
        "minus_difficulty": {"use_reward_modulated_learning"},
        "minus_homeostasis": {"sleep_enable_homeostasis"},
        "minus_reset": {"sleep_enable_chemical_reset"},
    }
    for name, expected in changes.items():
        assert {
            key for key in configs[name] if configs[name][key] != configs["full"][key]
        } == expected

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("data opened before the frozen gate")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    for changed in (
        replace(manifest, seeds=(263,)),
        replace(manifest, max_optimizer_updates=1599),
        replace(manifest, arms=manifest.arms[:-1]),
    ):
        with pytest.raises(ValueError, match="frozen manifest"):
            study.run_combined_preflight(changed)


def test_should_repeat_all_cells_with_outer_final_and_arrival_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles
    original_progress = study._new_progress
    live: dict[int, study._Progress] = {}

    def capture_progress(*args: Any) -> study._Progress:
        progress = original_progress(*args)
        live[args[1]] = progress
        return progress

    def phase_a(*args: Any) -> Any:
        return replace(original_a(*args), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    def phase_b(*args: Any) -> Any:
        progress = live[args[1]]
        assert len(progress.opportunities) == 12
        assert all(work.wake_updates == 12 for work in progress.work.values())
        assert all(
            model.get_sleep_clocks().wake_batches == 12
            for model in progress.models.values()
            if isinstance(model, CircadianPredictiveCodingNetwork)
        )
        return replace(original_b(*args), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    monkeypatch.setattr(study, "_new_progress", capture_progress)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", phase_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", phase_b)
    manifest = fixed_combined_manifest()
    first = study.run_combined_preflight(manifest)
    second = study.run_combined_preflight(manifest)
    assert first == second
    assert first.final_released is False and first.outer_selection_scored is False
    assert sum(len(seed.methods) for seed in first.seed_results) == 51
    assert sum(len(seed.opportunities) for seed in first.seed_results) == 72
    assert (
        sum(len(row["decisions"]) for seed in first.seed_results for row in seed.opportunities)
        == 648
    )
    for seed in first.seed_results:
        methods = {row["name"]: row for row in seed.methods}
        assert tuple(methods) == tuple(arm.name for arm in manifest.arms)
        for left, right in PARITY_PAIRS:
            for key in (
                "initial_parameter_sha256",
                "after_a_parameter_sha256",
                "final_parameter_sha256",
            ):
                assert methods[left][key] == methods[right][key]
        assert methods["minus_schedule"]["own_sleep_attempts"] == 0
        assert all(
            methods[name]["own_sleep_attempts"] == 6
            for name in DECISION_ARMS
            if name != "minus_schedule"
        )
        assert methods["minus_replay"]["applied_replay_updates"] == 0
        assert methods["minus_structure"]["width_peak"] == 8
        assert methods["backprop_14_off"]["parameters_final"] == 57
        for row in seed.opportunities:
            full = next(decision for decision in row["decisions"] if decision["name"] == "full")
            for name in REPLAY_CONTROLS:
                assert row["applied_control_replay_ids"][name] == (
                    row["selected_ids"] if full["event"]["outcome"] == "accepted" else ()
                )


def test_should_observe_all_rejected_replay_executions_and_restore_complete_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CircadianPredictiveCodingNetwork._run_training_step
    executed_replay = 0
    guards: dict[CircadianPredictiveCodingNetwork, int] = {}

    def count_replay(self: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> float:
        nonlocal executed_replay
        if kwargs.get("update_epoch_state") is False:
            executed_replay += 1
        return original(self, *args, **kwargs)

    def reject(
        self: CircadianPredictiveCodingNetwork, inputs: np.ndarray, targets: np.ndarray
    ) -> float:
        call = guards.get(self, 0)
        guards[self] = call + 1
        return 1.0 if call % 2 == 0 else 0.0

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", count_replay)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", reject)
    result = study.run_combined_preflight(fixed_combined_manifest())
    rejected = sum(
        row["rejected_executed_replay_updates"]
        for seed in result.seed_results
        for row in seed.methods
    )
    assert rejected == executed_replay == 216
    assert sum(seed.executed_optimizer_updates for seed in result.seed_results) == 1440
    assert sum(guards.values()) == 288
    for seed in result.seed_results:
        assert all(row["applied_replay_updates"] == 0 for row in seed.methods)
        for row in seed.opportunities:
            assert all(not ids for ids in row["applied_control_replay_ids"].values())
            for decision in row["decisions"]:
                assert decision["state_sha256_after"] == decision["state_sha256_before"]
                assert decision["lineage_after"] == decision["lineage_before"]
                if row["epoch"] % 4 == 0 and decision["name"] != "minus_schedule":
                    assert decision["event"]["outcome"] == "rolled_back"


def test_should_reject_forged_work_role_replay_or_capacity_facts() -> None:
    manifest = fixed_combined_manifest()
    payload = study.json_value(study.run_combined_preflight(manifest))
    fixed = study.json_value(manifest)
    verify_combined_payload(payload, fixed)
    broken = deepcopy(payload)
    broken["seed_results"][-1]["methods"][-1]["wake_updates"] -= 1
    with pytest.raises(ValueError):
        verify_combined_payload(broken, fixed)
    broken = deepcopy(payload)
    first_seed = broken["seed_results"][0]
    first_seed["opportunities"][3]["decisions"][0]["event"]["guard"]["role_hash"] = first_seed[
        "role_hashes"
    ]["a_outer_selection"]
    with pytest.raises(ValueError, match="guard role"):
        verify_combined_payload(broken, fixed)
    broken = deepcopy(payload)
    broken["seed_results"][0]["opportunities"][3]["decisions"][0]["proposed_replay_ids"] = []
    with pytest.raises(ValueError, match="replay IDs"):
        verify_combined_payload(broken, fixed)
    broken = deepcopy(payload)
    broken["seed_results"][0]["methods"][6]["width_peak"] = 99
    with pytest.raises(ValueError, match="capacity"):
        verify_combined_payload(broken, fixed)
    broken = deepcopy(payload)
    broken["seed_results"][0]["opportunities"][3]["decisions"][0]["event"]["budgets"][
        "split_limit"
    ] = 0
    with pytest.raises(ValueError, match="budget"):
        verify_combined_payload(broken, fixed)


def test_should_bind_rng_chemistry_and_clocks_in_complete_state_fingerprint() -> None:
    model = study._new_models(fixed_combined_manifest(), 263)["full"]
    assert isinstance(model, CircadianPredictiveCodingNetwork)
    snapshot = model.snapshot_state()
    original = study._state_hash(model)
    parameters = study._parameter_hash(model)
    model._rng.random()
    assert study._parameter_hash(model) == parameters and study._state_hash(model) != original
    model.restore_state(snapshot)
    model._hidden_chemical[0] += 0.1
    assert study._parameter_hash(model) == parameters and study._state_hash(model) != original
    model.restore_state(snapshot)
    model._epochs_since_sleep += 1
    assert study._parameter_hash(model) == parameters and study._state_hash(model) != original
    model.restore_state(snapshot)
    assert study._state_hash(model) == original
