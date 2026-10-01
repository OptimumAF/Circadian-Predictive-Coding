"""Outer parent-control values remain sealed behind every training/copy check."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_parent_factor_development as development
from src.app import continual_parent_factor_preflight as preflight
from src.app import continual_shift_benchmark as base
from src.app.continual_parent_factor_manifest import fixed_parent_manifest
from src.app.continual_parent_factor_validation import verify_parent_payload
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.continual_metrics import TwoTaskAccuracy
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    # Fresh unscored fixtures keep unit tests independent of ignored local bundles.
    return preflight.json_value(preflight.run_parent_preflight(fixed_parent_manifest()))


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("parent development opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("parent development opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("parent development opened outer input before global gate")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("parent development opened outer label before global gate")


def _sealed_generator(**kwargs: Any) -> _SealedSource:
    data = generate_two_cluster_dataset_with_transform(**kwargs)
    return _SealedSource(data.train_input, data.train_target)


def _block_outer(monkeypatch: pytest.MonkeyPatch) -> None:
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles

    def blocked_a(*args: Any) -> Any:
        return replace(original_a(*args), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    def blocked_b(*args: Any) -> Any:
        return replace(original_b(*args), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    monkeypatch.setattr(arrived, "_build_phase_a_roles", blocked_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", blocked_b)


def test_should_reject_changed_contract_before_any_source(
    monkeypatch: pytest.MonkeyPatch, reference: dict[str, Any]
) -> None:
    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("source opened before contract validation")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    manifest = fixed_parent_manifest()
    with pytest.raises(ValueError, match="frozen manifest"):
        development.run_parent_factor_development(replace(manifest, seeds=(347,)), reference)
    broken = deepcopy(reference)
    broken["outer_selection_scored"] = True
    with pytest.raises(ValueError, match="evaluation seal"):
        development.run_parent_factor_development(manifest, broken)


def test_should_reject_valid_shaped_late_reference_before_any_outer_value(
    monkeypatch: pytest.MonkeyPatch, reference: dict[str, Any]
) -> None:
    broken = deepcopy(reference)
    last = broken["seed_results"][-1]
    last["methods"][-1]["final_parameter_sha256"] = "0" * 64
    last["opportunities"][-1]["wake"][-1]["parameter_sha256"] = "0" * 64
    last["opportunities"][-1]["after_epoch_parameter_sha256"]["pc_13_off"] = "0" * 64
    verify_parent_payload(broken, preflight.json_value(fixed_parent_manifest()))
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    _block_outer(monkeypatch)
    with pytest.raises(ValueError, match="train facts differ"):
        development.run_parent_factor_development(fixed_parent_manifest(), broken)


def test_should_reject_late_supply_fact_before_first_scorer(
    monkeypatch: pytest.MonkeyPatch, reference: dict[str, Any]
) -> None:
    original = development._train_seed

    def changed(*args: Any) -> Any:
        trained = original(*args)
        if trained.facts.seed == 353:
            trained.facts.opportunities[-1]["retained_order_ids"] = ("changed",)
        return trained

    def no_score(*args: object, **kwargs: object) -> None:
        raise AssertionError("first scorer ran before complete supply validation")

    monkeypatch.setattr(development, "_train_seed", changed)
    monkeypatch.setattr(development, "_score_seed", no_score)
    with pytest.raises(ValueError, match="retention|supply"):
        development.run_parent_factor_development(fixed_parent_manifest(), reference)


@pytest.mark.parametrize("checkpoint", ["a", "b"])
@pytest.mark.parametrize("corruption", ["parameters", "noise_rng", "selector_rng", "cursor"])
def test_should_reject_late_complete_checkpoint_before_first_scorer(
    monkeypatch: pytest.MonkeyPatch,
    reference: dict[str, Any],
    checkpoint: str,
    corruption: str,
) -> None:
    original = development._train_seed

    def changed(*args: Any) -> Any:
        trained = original(*args)
        if trained.facts.seed == 353:
            models = trained.models_after_a if checkpoint == "a" else trained.models_after_b
            mode = "scheduled" if corruption == "cursor" else "random"
            model = models[f"{mode}_growth"]
            assert isinstance(model, ParentControlledCircadianNetwork)
            if corruption == "parameters":
                model.weight_input_hidden[0, 0] += 1.0
            elif corruption == "noise_rng":
                model._rng.random()
            elif corruption == "selector_rng":
                model._parent_selection_rng.random()
            else:
                model._parent_selection_cursor += 1
        return trained

    def no_score(*args: object, **kwargs: object) -> None:
        raise AssertionError("first scorer ran before all complete checkpoint checks")

    monkeypatch.setattr(development, "_train_seed", changed)
    monkeypatch.setattr(development, "_score_seed", no_score)
    with pytest.raises(ValueError, match="checkpoint"):
        development.run_parent_factor_development(fixed_parent_manifest(), reference)


def test_should_score_every_cell_after_global_gate_with_arrival_and_final_seals(
    monkeypatch: pytest.MonkeyPatch, reference: dict[str, Any]
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    completed: list[Any] = []
    evaluations: list[int] = []
    progress_by_seed: dict[int, preflight._Progress] = {}
    original_progress = preflight._new_progress
    original_b = arrived._build_phase_b_roles
    original_train, original_score = development._train_seed, development._score_seed

    def capture_progress(*args: Any) -> preflight._Progress:
        progress = original_progress(*args)
        progress_by_seed[args[1]] = progress
        return progress

    def arrived_b(*args: Any) -> Any:
        progress = progress_by_seed[args[1]]
        assert len(progress.opportunities) == 12 and len(progress.work) == 8
        assert all(work.wake_updates == 12 for work in progress.work.values())
        return original_b(*args)

    def count_accuracy(original: Any) -> Any:
        def wrapped(model: Any, inputs: np.ndarray, targets: np.ndarray) -> float:
            if len(completed) == 3:
                evaluations.append(len(inputs))
            return float(original(model, inputs, targets))

        return wrapped

    for model_type in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        monkeypatch.setattr(
            model_type, "compute_accuracy", count_accuracy(model_type.compute_accuracy)
        )

    def record_train(*args: Any) -> Any:
        trained = original_train(*args)
        completed.append(trained)
        return trained

    def require_all_seeds(*args: Any) -> Any:
        assert [item.facts.seed for item in completed] == [347, 349, 353]
        return original_score(*args)

    monkeypatch.setattr(preflight, "_new_progress", capture_progress)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", arrived_b)
    monkeypatch.setattr(development, "_train_seed", record_train)
    monkeypatch.setattr(development, "_score_seed", require_all_seeds)
    result = development.run_parent_factor_development(fixed_parent_manifest(), reference)
    assert preflight.json_value(result.train_facts) == reference
    assert result.final_released is False and result.outer_selection_scored is True
    assert sum(len(seed.arms) for seed in result.scored_seeds) == 24
    assert sum(len(seed.contrasts) for seed in result.scored_seeds) == 60
    assert evaluations == [24, 24, 12] * 24 and sum(evaluations) == 1440
    for trained, seed in zip(completed, result.scored_seeds, strict=True):
        development._require_checkpoint_hashes(trained)
        assert tuple(arm.name for arm in seed.arms) == development.ARMS
        assert tuple((row.left, row.right) for row in seed.contrasts) == development.CONTRASTS
        arms = {arm.name: arm for arm in seed.arms}
        for field in development.SCORE_FIELDS:
            assert getattr(arms["pc_off"], field) == getattr(arms["neutral_off"], field)
        for arm in seed.arms:
            values = TwoTaskAccuracy(arm.a_after_a, arm.a_after_b, arm.b_after_b)
            assert arm.final_mean_task_accuracy == values.final_mean_task_accuracy
            assert arm.signed_forgetting_a == values.signed_forgetting_a
            assert arm.retention_ratio_a == values.retention_ratio_a
        for contrast in seed.contrasts:
            for field in development.CONTRAST_FIELDS:
                assert getattr(contrast, field) == getattr(arms[contrast.left], field) - getattr(
                    arms[contrast.right], field
                )
