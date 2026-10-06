"""No schedule outer value is available until every c5 train fact matches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_schedule_factor_development as development
from src.app import continual_schedule_factor_preflight as preflight
from src.app import continual_shift_benchmark as base
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.continual_metrics import TwoTaskAccuracy
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("schedule development opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("schedule development opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("schedule development opened outer input before global gate")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("schedule development opened outer label before global gate")


def _sealed_generator(**kwargs: Any) -> _SealedSource:
    source = generate_two_cluster_dataset_with_transform(**kwargs)
    return _SealedSource(source.train_input, source.train_target)


@lru_cache(maxsize=1)
def _local_reference() -> dict[str, Any]:
    # Why this: exact boundary checks need a same-environment numerical fixture.
    return preflight._json_value(
        preflight.run_schedule_factor_preflight(preflight.fixed_schedule_factor_manifest())
    )


def _reference() -> dict[str, Any]:
    return deepcopy(_local_reference())


def test_should_reject_changed_manifest_and_reference_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()

    def no_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("source opened before contract validation")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", no_source)
    manifest = preflight.fixed_schedule_factor_manifest()
    with pytest.raises(ValueError, match="frozen manifest"):
        development.run_schedule_factor_development(replace(manifest, seeds=(79,)), reference)
    reference["outer_selection_scored"] = True
    with pytest.raises(ValueError, match="evaluation seal differs"):
        development.run_schedule_factor_development(manifest, reference)


def test_should_fail_late_seed_fact_mismatch_before_any_outer_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = deepcopy(_reference())
    reference["seed_results"][-1]["methods"][-1]["final_parameter_sha256"] = "0" * 64
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
        development.run_schedule_factor_development(
            preflight.fixed_schedule_factor_manifest(), reference
        )


def test_should_reject_changed_replay_fact_before_outer_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()
    original = development._train_seed

    def changed(*args: Any) -> Any:
        trained = original(*args)
        if trained.facts.seed == 89:
            first = trained.facts.opportunities[0]
            trained.facts = replace(
                trained.facts,
                opportunities=(replace(first, selected_ids=("changed",)),)
                + trained.facts.opportunities[1:],
            )
        return trained

    def no_score(*args: object, **kwargs: object) -> None:
        raise AssertionError("outer scorer called before all replay facts verified")

    monkeypatch.setattr(development, "_train_seed", changed)
    monkeypatch.setattr(development, "_score_seed", no_score)
    with pytest.raises(ValueError):
        development.run_schedule_factor_development(
            preflight.fixed_schedule_factor_manifest(), reference
        )


def test_should_score_all_cells_only_after_global_gate_with_final_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    completed: list[Any] = []
    evaluations: list[int] = []
    original_train, original_score = development._train_seed, development._score_seed

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
        assert [item.facts.seed for item in completed] == [79, 83, 89]
        return original_score(*args)

    monkeypatch.setattr(development, "_train_seed", record_train)
    monkeypatch.setattr(development, "_score_seed", require_all_seeds)
    reference = _reference()
    result = development.run_schedule_factor_development(
        preflight.fixed_schedule_factor_manifest(), reference
    )
    assert preflight._json_value(result.train_facts) == reference
    assert result.final_released is False and result.outer_selection_scored is True
    assert sum(len(seed.arms) for seed in result.scored_seeds) == 33
    assert sum(len(seed.contrasts) for seed in result.scored_seeds) == 27
    assert evaluations == [24, 24, 12] * 33
    assert sum(evaluations) == 1980
    for trained, seed in zip(completed, result.scored_seeds, strict=True):
        development._require_checkpoint_hashes(trained)
        assert tuple(arm.name for arm in seed.arms) == preflight.ARMS
        assert tuple((row.left, row.right) for row in seed.contrasts) == development.CONTRASTS
        arms = {arm.name: arm for arm in seed.arms}
        for policy in preflight.POLICIES:
            for field in development.SCORE_FIELDS:
                assert getattr(arms[f"pc_{policy}"], field) == getattr(
                    arms[f"neutral_{policy}"], field
                )
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


def test_should_reject_late_checkpoint_copy_mismatch_before_first_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = development._train_seed

    def changed_copy(*args: Any) -> Any:
        trained = original(*args)
        if trained.facts.seed == 89:
            trained.models_after_a["pc_12_no_sleep"].weight_input_hidden[0, 0] += 1.0
        return trained

    def no_score(*args: object, **kwargs: object) -> None:
        raise AssertionError("first outer scorer ran before later checkpoint verification")

    monkeypatch.setattr(development, "_train_seed", changed_copy)
    monkeypatch.setattr(development, "_score_seed", no_score)
    with pytest.raises(ValueError, match="checkpoint parameters differ"):
        development.run_schedule_factor_development(
            preflight.fixed_schedule_factor_manifest(), _reference()
        )
