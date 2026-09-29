"""Behavioral gates for the fixed v11 matched difficulty comparison."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.difficulty_matched_benchmark as benchmark
from src.infra.continual_roles import PhaseDecisionRoles, split_phase_decision_roles
from src.infra.difficulty_streams import DifficultyPhaseSource


def _b_roles() -> PhaseDecisionRoles:
    source = DifficultyPhaseSource(17, "b")
    return split_phase_decision_roles(
        source,
        phase="b",
        seed=17,
        split_seed=172,
        inner_guard_fraction=0.2,
        outer_selection_fraction=0.2,
        expected_final_count=40,
    )


def test_should_change_only_one_b_train_field_when_condition_is_corrupted() -> None:
    roles = _b_roles()
    clean = benchmark._effective_train(roles, "clean")
    flipped = benchmark._effective_train(roles, "label_flip")
    outlier = benchmark._effective_train(roles, "feature_outlier")
    np.testing.assert_array_equal(clean.input, flipped.input)
    np.testing.assert_array_equal(clean.target, outlier.target)
    assert np.count_nonzero(clean.target != flipped.target) == 1
    assert np.count_nonzero(clean.input != outlier.input) == 1
    np.testing.assert_array_equal(roles.train.input, clean.input)
    np.testing.assert_array_equal(roles.train.target, clean.target)
    assert roles.final_test is None and not roles.final_released
    assert len(set(roles.sample_ids["train"]) & set(roles.sample_ids["inner_guard"])) == 0
    assert len(set(roles.sample_ids["train"]) & set(roles.sample_ids["outer_selection"])) == 0


def test_should_start_both_modulation_arms_from_same_parameters() -> None:
    pytest.importorskip("torch")
    for backend in ("numpy", "torch_cpu"):
        control = benchmark._make_model(backend, 17, False)
        modulated = benchmark._make_model(backend, 17, True)
        if backend == "numpy":
            for name in (
                "weight_input_hidden",
                "bias_hidden",
                "weight_hidden_output",
                "bias_output",
            ):
                np.testing.assert_array_equal(getattr(control, name), getattr(modulated, name))
        else:
            for name in (
                "weight_feature_hidden",
                "bias_hidden",
                "weight_hidden_output",
                "bias_output",
            ):
                np.testing.assert_array_equal(
                    getattr(control, name).numpy(), getattr(modulated, name).numpy()
                )


def test_should_freeze_every_trial_before_reading_any_final_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    trained = 0
    final_reads = 0
    original = benchmark._train_trial

    def tracked_train(*args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        assert final_reads == 0
        trial = original(*args, **kwargs)
        trained += 1
        return trial

    class TrackedSource:
        def __init__(self, seed: int, phase: str) -> None:
            self.source = DifficultyPhaseSource(seed, phase)

        @property
        def train_input(self) -> np.ndarray:
            return self.source.train_input

        @property
        def train_target(self) -> np.ndarray:
            return self.source.train_target

        @property
        def test_input(self) -> np.ndarray:
            nonlocal final_reads
            assert trained == 24
            final_reads += 1
            return self.source.test_input

        @property
        def test_target(self) -> np.ndarray:
            nonlocal final_reads
            assert trained == 24
            final_reads += 1
            return self.source.test_target

    monkeypatch.setattr(benchmark, "_train_trial", tracked_train)
    result = benchmark.run_difficulty_comparison(
        benchmark.fixed_difficulty_manifest(), source_factory=TrackedSource
    )
    assert trained == 24 and final_reads == 8
    assert len(result.outcomes) == 24
    for seed in (17, 19):
        same_seed = [row for row in result.outcomes if row.seed == seed]
        assert len({tuple(row.final_role_hashes.items()) for row in same_seed}) == 1
        assert len({tuple(row.development_role_hashes.items()) for row in same_seed}) == 1
    for row in result.outcomes:
        assert row.work == benchmark.WakeWork(4, 96, 8, 192, 0, 0)
        assert row.forgetting == pytest.approx(row.a_after_a_accuracy - row.a_after_b_accuracy)
        assert len(row.diagnostics) == 4
        assert row.diagnostics[0].prior_step_loss_improvement is None
        for previous, current in zip(row.diagnostics, row.diagnostics[1:]):
            assert current.prior_step_loss_improvement == pytest.approx(
                previous.train_bce_before - previous.train_bce_after
            )
        if not row.modulated:
            assert all(step.actual_reward_scale == 1.0 for step in row.diagnostics)


@pytest.mark.parametrize("work_boundary", ("total", "after_a", "train_hash"))
def test_should_reject_changed_work_before_final_release(
    monkeypatch: pytest.MonkeyPatch, work_boundary: str
) -> None:
    pytest.importorskip("torch")
    final_reads = 0
    original = benchmark._train_trial

    def changed_work(*args: Any, **kwargs: Any) -> Any:
        trial = original(*args, **kwargs)
        if trial.seed == 17 and trial.condition == "clean" and trial.backend == "numpy":
            if work_boundary == "total":
                trial.work = replace(trial.work, examples=trial.work.examples + 1)
            elif work_boundary == "after_a":
                trial.after_a_work = replace(
                    trial.after_a_work, examples=trial.after_a_work.examples + 1
                )
            else:
                trial.effective_train_hashes["b"] = "forged"
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

    monkeypatch.setattr(benchmark, "_train_trial", changed_work)
    with pytest.raises(ValueError, match="preflight failed"):
        benchmark.run_difficulty_comparison(
            benchmark.fixed_difficulty_manifest(), source_factory=TrackedSource
        )
    assert final_reads == 0


def test_should_reject_changed_manifest_before_source_access() -> None:
    changed = replace(benchmark.fixed_difficulty_manifest(), seeds=(17,))

    def forbidden_source(seed: int, phase: str) -> DifficultyPhaseSource:
        raise AssertionError("source must stay unopened")

    with pytest.raises(ValueError, match="predeclared fixed protocol"):
        benchmark.run_difficulty_comparison(changed, source_factory=forbidden_source)


def test_should_repeat_complete_json_content() -> None:
    pytest.importorskip("torch")
    manifest = benchmark.fixed_difficulty_manifest()
    first = asdict(benchmark.run_difficulty_comparison(manifest))
    second = asdict(benchmark.run_difficulty_comparison(manifest))
    assert first == second
