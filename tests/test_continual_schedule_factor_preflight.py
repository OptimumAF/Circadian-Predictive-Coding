"""Schedule controls keep outer/final roles sealed and count rollback work."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app import continual_schedule_factor_preflight as study
from src.app.continual_schedule_factor_validation import verify_schedule_preflight_payload
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("schedule preflight opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("schedule preflight opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("schedule preflight opened outer inputs")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("schedule preflight opened outer labels")


def _sealed_generator(**kwargs: object) -> _SealedSource:
    source = generate_two_cluster_dataset_with_transform(**kwargs)  # type: ignore[arg-type]
    return _SealedSource(source.train_input, source.train_target)


def test_should_reject_changed_schedule_manifest_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = study.fixed_schedule_factor_manifest()
    assert study.validate_schedule_factor_manifest(manifest) == 1014

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("schedule source opened before manifest validation")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    with pytest.raises(ValueError, match="frozen manifest"):
        study.run_schedule_factor_preflight(replace(manifest, seeds=(79, 83, 97)))
    with pytest.raises(ValueError, match="frozen manifest"):
        study.run_schedule_factor_preflight(replace(manifest, periodic_interval=2))


def test_should_repeat_all_train_only_cells_with_arrival_outer_and_final_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles
    original_models = study._new_models
    live: dict[int, dict[str, study.Model]] = {}

    def capture_models(manifest: study.ScheduleFactorManifest, seed: int) -> dict[str, study.Model]:
        live[seed] = original_models(manifest, seed)
        return live[seed]

    def phase_b(manifest: arrived.ContinualArrivedRolesConfig, seed: int) -> object:
        assert all(
            model.get_sleep_clocks().wake_batches == 12
            for model in live[seed].values()
            if isinstance(model, CircadianPredictiveCodingNetwork)
        )
        return replace(original_b(manifest, seed), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    monkeypatch.setattr(study, "_new_models", capture_models)
    monkeypatch.setattr(
        arrived,
        "_build_phase_a_roles",
        lambda *args: replace(original_a(*args), outer_selection=_BlockedOuter()),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(arrived, "_build_phase_b_roles", phase_b)
    first = study.run_schedule_factor_preflight(study.fixed_schedule_factor_manifest())
    second = study.run_schedule_factor_preflight(study.fixed_schedule_factor_manifest())
    assert first == second
    assert first.final_released is False and first.outer_selection_scored is False
    assert len(first.seed_results) == 3
    for seed in first.seed_results:
        assert tuple(method.name for method in seed.methods) == study.ARMS
        assert len(seed.opportunities) == 24
        methods = {method.name: method for method in seed.methods}
        for policy in study.POLICIES:
            assert (
                methods[f"pc_{policy}"].after_a_parameter_sha256
                == methods[f"neutral_{policy}"].after_a_parameter_sha256
            )
            assert (
                methods[f"pc_{policy}"].final_parameter_sha256
                == methods[f"neutral_{policy}"].final_parameter_sha256
            )
        assert methods["neutral_no_sleep"].sleep_attempts == 0
        assert methods["neutral_periodic"].sleep_attempts == 6
        assert all(
            method.width_initial == method.width_final == method.width_peak
            for method in seed.methods
        )


def test_should_count_rejected_replay_executions_and_apply_no_baseline_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CircadianPredictiveCodingNetwork._run_training_step
    executed = 0
    guards: dict[CircadianPredictiveCodingNetwork, int] = {}

    def count_replay(self: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> float:
        nonlocal executed
        if kwargs.get("update_epoch_state") is False:
            executed += 1
        return original(self, *args, **kwargs)

    def reject(
        self: CircadianPredictiveCodingNetwork, inputs: np.ndarray, targets: np.ndarray
    ) -> float:
        call = guards.get(self, 0)
        guards[self] = call + 1
        return 1.0 if call % 2 == 0 else 0.0

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", count_replay)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", reject)
    result = study.run_schedule_factor_preflight(study.fixed_schedule_factor_manifest())
    assert executed >= 36
    assert executed == sum(
        method.rejected_executed_replay_updates
        for seed in result.seed_results
        for method in seed.methods
    )
    assert all(
        method.applied_replay_updates == 0
        for seed in result.seed_results
        for method in seed.methods
    )
    for seed in result.seed_results:
        for opportunity in seed.opportunities:
            for decision in opportunity.decisions:
                if decision.attempted:
                    assert decision.outcome == "rolled_back"
                    assert decision.proposed_replay_updates == 2
                    assert decision.applied_ids_by_method == dict.fromkeys(study.METHODS, ())
                    assert decision.parameter_sha256_before == decision.parameter_sha256_after


def test_should_reset_adaptive_spacing_only_on_committed_fixture_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_wake = CircadianPredictiveCodingNetwork.train_epoch
    original_accuracy = CircadianPredictiveCodingNetwork.compute_accuracy

    def fixture_wake(self: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> Any:
        result = original_wake(self, *args, **kwargs)
        if self.config.use_adaptive_sleep_trigger:
            self._energy_history = [0.1] * len(self._energy_history)
            self._hidden_chemical = np.linspace(0.0, 1.0, self.hidden_dim)
            return replace(result, energy=0.1)
        return result

    def accept_fixture(
        self: CircadianPredictiveCodingNetwork, inputs: np.ndarray, targets: np.ndarray
    ) -> float:
        return (
            0.5
            if self.config.use_adaptive_sleep_trigger
            else original_accuracy(self, inputs, targets)
        )

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "train_epoch", fixture_wake)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", accept_fixture)
    result = study.run_schedule_factor_preflight(study.fixed_schedule_factor_manifest())
    for seed in result.seed_results:
        active = [
            row.global_epoch
            for row in seed.opportunities
            for decision in row.decisions
            if decision.policy == "adaptive" and decision.outcome == "accepted"
        ]
        assert active == [10, 20]
        methods = {method.name: method for method in seed.methods}
        assert all(
            methods[f"{method}_adaptive"].applied_replay_updates == 4 for method in study.METHODS
        )


def test_should_reject_forged_readiness_replay_and_rollback_cost_facts() -> None:
    manifest = study.fixed_schedule_factor_manifest()
    result = json.loads(json.dumps(asdict(study.run_schedule_factor_preflight(manifest))))
    expected = json.loads(json.dumps(asdict(manifest)))
    broken = deepcopy(result)
    broken["seed_results"][0]["opportunities"][0]["decisions"][1]["adaptive_due"] = True
    with pytest.raises(ValueError, match="decision, clocks or replay"):
        verify_schedule_preflight_payload(broken, expected)
    broken = deepcopy(result)
    broken["seed_results"][0]["opportunities"][3]["decisions"][0]["applied_ids_by_method"][
        "backprop"
    ] = ["0" * 64]
    with pytest.raises(ValueError, match="replay IDs"):
        verify_schedule_preflight_payload(broken, expected)
    broken = deepcopy(result)
    broken["seed_results"][0]["executed_optimizer_updates"] += 2
    with pytest.raises(ValueError, match="optimizer total"):
        verify_schedule_preflight_payload(broken, expected)
