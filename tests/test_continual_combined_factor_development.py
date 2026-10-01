"""Combined outer scoring is sealed behind all train facts and model copies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_p63_combined_factor_preflight as c7_adapter
from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app import continual_arrived_benchmark as arrived
from src.app import continual_combined_factor_development as development
from src.app import continual_combined_factor_preflight as preflight
from src.app import continual_shift_benchmark as base
from src.app.continual_combined_factor_manifest import PARITY_PAIRS, fixed_combined_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.continual_metrics import TwoTaskAccuracy
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "artifacts/runs/p63-combined-factor-preflight"


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("combined development opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("combined development opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("combined development opened outer input before global gate")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("combined development opened outer label before global gate")


def _sealed_generator(**kwargs: Any) -> _SealedSource:
    data = generate_two_cluster_dataset_with_transform(**kwargs)
    return _SealedSource(data.train_input, data.train_target)


def _reference() -> dict[str, Any]:
    paths = c7_adapter.artifact_paths(REFERENCE_DIR)
    if not paths["result"].is_file():
        c7_adapter.run_bounded_preflight(REFERENCE_DIR)
    return artifacts.parse_finite_json(paths["result"].read_text(encoding="utf-8"))


def test_should_reject_changed_manifest_and_reference_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("source opened before frozen contract validation")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    manifest = fixed_combined_manifest()
    with pytest.raises(ValueError, match="frozen manifest"):
        development.run_combined_factor_development(replace(manifest, seeds=(263,)), reference)
    reference["outer_selection_scored"] = True
    with pytest.raises(ValueError, match="evaluation seal differs"):
        development.run_combined_factor_development(manifest, reference)


def test_should_fail_valid_shaped_late_fact_mismatch_before_any_outer_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = deepcopy(_reference())
    last = reference["seed_results"][-1]
    last["methods"][-1]["final_parameter_sha256"] = "0" * 64
    last["opportunities"][-1]["wake"][-1]["parameter_sha256"] = "0" * 64
    last["opportunities"][-1]["after_epoch_parameter_sha256"]["pc_14_off"] = "0" * 64
    c7_adapter.verify_result(reference)
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles

    def blocked_a(*args: Any) -> Any:
        return replace(original_a(*args), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    def blocked_b(*args: Any) -> Any:
        return replace(original_b(*args), outer_selection=_BlockedOuter())  # type: ignore[arg-type]

    monkeypatch.setattr(arrived, "_build_phase_a_roles", blocked_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", blocked_b)
    with pytest.raises(ValueError, match="train facts differ"):
        development.run_combined_factor_development(fixed_combined_manifest(), reference)


def test_should_reject_late_replay_fact_mismatch_before_first_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = development._train_seed

    def changed(*args: Any) -> Any:
        trained = original(*args)
        if trained.facts.seed == 271:
            trained.facts.opportunities[0]["selected_ids"] = ("changed",)
        return trained

    def no_score(*args: object, **kwargs: object) -> None:
        raise AssertionError("first scorer ran before all replay facts verified")

    monkeypatch.setattr(development, "_train_seed", changed)
    monkeypatch.setattr(development, "_score_seed", no_score)
    with pytest.raises(ValueError, match="opportunity"):
        development.run_combined_factor_development(fixed_combined_manifest(), _reference())


@pytest.mark.parametrize("corruption", ["parameters", "rng"])
def test_should_reject_late_checkpoint_mismatch_before_first_scorer(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    original = development._train_seed

    def changed_copy(*args: Any) -> Any:
        trained = original(*args)
        if trained.facts.seed == 271:
            model = trained.models_after_a["full"]
            assert isinstance(model, CircadianPredictiveCodingNetwork)
            if corruption == "parameters":
                model.weight_input_hidden[0, 0] += 1.0
            else:
                model._rng.random()
        return trained

    def no_score(*args: object, **kwargs: object) -> None:
        raise AssertionError("first scorer ran before all complete checkpoint checks")

    monkeypatch.setattr(development, "_train_seed", changed_copy)
    monkeypatch.setattr(development, "_score_seed", no_score)
    with pytest.raises(ValueError, match="checkpoint"):
        development.run_combined_factor_development(fixed_combined_manifest(), _reference())


def test_should_score_every_cell_after_global_gate_with_arrival_and_final_seals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()
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
        assert len(progress.opportunities) == 12
        assert len(progress.work) == 17
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
        assert [item.facts.seed for item in completed] == [263, 269, 271]
        return original_score(*args)

    monkeypatch.setattr(preflight, "_new_progress", capture_progress)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", arrived_b)
    monkeypatch.setattr(development, "_train_seed", record_train)
    monkeypatch.setattr(development, "_score_seed", require_all_seeds)
    result = development.run_combined_factor_development(fixed_combined_manifest(), reference)
    assert preflight.json_value(result.train_facts) == reference
    assert result.final_released is False and result.outer_selection_scored is True
    assert sum(len(seed.arms) for seed in result.scored_seeds) == 51
    assert sum(len(seed.contrasts) for seed in result.scored_seeds) == 66
    assert evaluations == [24, 24, 12] * 51
    assert sum(evaluations) == 3060
    for trained, seed in zip(completed, result.scored_seeds, strict=True):
        development._require_checkpoint_hashes(trained)
        assert tuple(arm.name for arm in seed.arms) == development.ARMS
        assert tuple((row.left, row.right) for row in seed.contrasts) == development.CONTRASTS
        arms = {arm.name: arm for arm in seed.arms}
        for left, right in PARITY_PAIRS:
            for field in development.SCORE_FIELDS:
                assert getattr(arms[left], field) == getattr(arms[right], field)
        for arm in seed.arms:
            metrics = TwoTaskAccuracy(arm.a_after_a, arm.a_after_b, arm.b_after_b)
            assert arm.final_mean_task_accuracy == metrics.final_mean_task_accuracy
            assert arm.signed_forgetting_a == metrics.signed_forgetting_a
            assert arm.retention_ratio_a == metrics.retention_ratio_a
        for contrast in seed.contrasts:
            for field in development.CONTRAST_FIELDS:
                assert getattr(contrast, field) == getattr(arms[contrast.left], field) - getattr(
                    arms[contrast.right], field
                )
