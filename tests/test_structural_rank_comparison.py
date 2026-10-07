"""Global-role and factor-isolation gates for the fixed v12 comparison."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.structural_rank_comparison as comparison
from src.app.structural_rank_trial import (
    RankFactors,
    fixed_structural_rank_manifest,
    make_structural_model,
    train_rank_update,
)
from src.infra.datasets import LabeledData
from src.infra.difficulty_streams import DifficultyPhaseSource


@pytest.mark.parametrize("backend", ("numpy", "torch_cpu"))
def test_should_isolate_wake_and_history_factors_from_disabled_core(backend: str) -> None:
    if backend == "torch_cpu":
        pytest.importorskip("torch")
    manifest = fixed_structural_rank_manifest()
    factors = RankFactors(False, False, True)
    isolated = make_structural_model(manifest, backend, 23, factors)
    disabled = make_structural_model(manifest, backend, 23, factors)
    disabled.config = replace(disabled.config, use_reward_modulated_learning=False)
    observed_scales: list[float] = []
    for phase in ("a", "b"):
        source = DifficultyPhaseSource(23, phase)
        batch = LabeledData(source.train_input, source.train_target)
        observed_scales.append(train_rank_update(isolated, backend, factors, batch, manifest))
        if backend == "numpy":
            disabled.train_epoch(
                batch.input,
                batch.target,
                manifest.learning_rate,
                manifest.inference_steps,
                manifest.inference_learning_rate,
            )
            names = ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
        else:
            import torch

            disabled.train_step(
                torch.as_tensor(batch.input, dtype=torch.float32),
                torch.as_tensor(batch.target[:, 0], dtype=torch.long),
                manifest.learning_rate,
                manifest.inference_steps,
                manifest.inference_learning_rate,
            )
            names = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
    assert observed_scales[0] == 1.0
    assert abs(observed_scales[1] - 1.0) > 1e-5
    tolerance = 1e-12 if backend == "numpy" else 1e-6
    for name in names:
        left = getattr(isolated, name)
        right = getattr(disabled, name)
        if backend == "torch_cpu":
            left, right = left.detach().numpy(), right.detach().numpy()
        np.testing.assert_allclose(left, right, atol=tolerance, rtol=tolerance)
    left_importance = isolated._importance_ema
    right_importance = disabled._importance_ema
    if backend == "torch_cpu":
        left_importance = left_importance.detach().numpy()
        right_importance = right_importance.detach().numpy()
    np.testing.assert_allclose(left_importance, right_importance, atol=tolerance, rtol=tolerance)


def test_should_train_all_factor_cells_without_final_access() -> None:
    pytest.importorskip("torch")

    class ForbiddenFinalSource(DifficultyPhaseSource):
        @property
        def test_input(self) -> np.ndarray:
            raise AssertionError("v12 final input opened during train-only preflight")

        @property
        def test_target(self) -> np.ndarray:
            raise AssertionError("v12 final label opened during train-only preflight")

    study = comparison.train_unscored_structural_rank_study(
        fixed_structural_rank_manifest(), source_factory=ForbiddenFinalSource
    )
    assert len(study.trials) == 32
    for trial in study.trials:
        facts = trial.facts
        assert facts.a_work == comparison.RankWork(8, 192, 16, 384, 0, 1)
        assert facts.total_work == comparison.RankWork(16, 384, 32, 768, 0, 1)
        assert len(facts.split_pairs) == len(facts.removed_prune_ids) == 1
        assert facts.initial_width == facts.post_sleep_width == facts.final_width == 8
        assert len(facts.reward_scales) == 16
    assert all(
        role.final_test is None and not role.final_released
        for phases in study.roles_by_seed.values()
        for role in phases.values()
    )


def test_should_release_final_only_after_all_factor_cells_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    trained = 0
    final_reads = 0
    original = comparison.train_structural_rank_trial

    def tracked_train(*args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        assert final_reads == 0
        trial = original(*args, **kwargs)
        trained += 1
        return trial

    class TrackedSource(DifficultyPhaseSource):
        @property
        def test_input(self) -> np.ndarray:
            nonlocal final_reads
            assert trained == 32
            final_reads += 1
            return super().test_input

        @property
        def test_target(self) -> np.ndarray:
            nonlocal final_reads
            assert trained == 32
            final_reads += 1
            return super().test_target

    monkeypatch.setattr(comparison, "train_structural_rank_trial", tracked_train)
    result = comparison.run_structural_rank_comparison(
        fixed_structural_rank_manifest(), source_factory=TrackedSource
    )
    assert trained == 32 and final_reads == 8
    assert len(result.outcomes) == 32
    for seed in (23, 29):
        same_seed = [row for row in result.outcomes if row.train_facts.seed == seed]
        assert len({tuple(row.final_role_hashes.items()) for row in same_seed}) == 1
        assert len({tuple(row.train_facts.role_hashes.items()) for row in same_seed}) == 1
    for row in result.outcomes:
        assert row.forgetting == pytest.approx(row.a_after_a_accuracy - row.a_after_b_accuracy)
        assert all(
            0.0 <= score <= 1.0
            for score in (
                row.a_after_a_accuracy,
                row.a_after_b_accuracy,
                row.b_after_b_accuracy,
            )
        )


@pytest.mark.parametrize("tamper", ("work", "role", "structure", "pre_sleep"))
def test_should_reject_forged_train_fact_before_final_release(
    monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    pytest.importorskip("torch")
    final_reads = 0
    original = comparison.train_structural_rank_trial

    def forged_train(*args: Any, **kwargs: Any) -> Any:
        trial = original(*args, **kwargs)
        facts = trial.facts
        if (
            facts.seed == 23
            and facts.backend == "numpy"
            and facts.factors == RankFactors(False, False, False)
        ):
            if tamper == "work":
                trial.facts = replace(
                    facts, total_work=replace(facts.total_work, wake_examples=385)
                )
            elif tamper == "role":
                trial.facts = replace(facts, role_hashes={**facts.role_hashes, "b/train": "forged"})
            elif tamper == "structure":
                trial.facts = replace(facts, split_pairs=((0, 99),))
            else:
                trial.facts = replace(facts, pre_sleep_parameter_digest="forged")
        return trial

    class TrackedSource(DifficultyPhaseSource):
        @property
        def test_input(self) -> np.ndarray:
            nonlocal final_reads
            final_reads += 1
            return super().test_input

        @property
        def test_target(self) -> np.ndarray:
            nonlocal final_reads
            final_reads += 1
            return super().test_target

    monkeypatch.setattr(comparison, "train_structural_rank_trial", forged_train)
    with pytest.raises(ValueError, match="v12"):
        comparison.run_structural_rank_comparison(
            fixed_structural_rank_manifest(), source_factory=TrackedSource
        )
    assert final_reads == 0


def test_should_reject_changed_manifest_before_source_access() -> None:
    changed = replace(fixed_structural_rank_manifest(), seeds=(23,))

    def forbidden_source(seed: int, phase: str) -> DifficultyPhaseSource:
        raise AssertionError("source must stay unopened")

    with pytest.raises(ValueError, match="predeclared protocol"):
        comparison.run_structural_rank_comparison(changed, source_factory=forbidden_source)


def test_should_repeat_full_outcome_content() -> None:
    pytest.importorskip("torch")
    manifest = fixed_structural_rank_manifest()
    first = asdict(comparison.run_structural_rank_comparison(manifest))
    second = asdict(comparison.run_structural_rank_comparison(manifest))
    assert first == second
