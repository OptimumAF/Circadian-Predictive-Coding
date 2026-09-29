"""Sealed roles, matched work, and actual timing for the fixed v13 study."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.sleep_trigger_comparison as comparison
from src.app.sleep_trigger_trial import fixed_trigger_manifest
from src.infra.trigger_streams import TriggerPhaseSource


def test_should_pair_stationary_noise_and_axis_shift_without_reusing_final_rows() -> None:
    stationary_a = TriggerPhaseSource(41, "a", "stationary_noise")
    shift_a = TriggerPhaseSource(41, "a", "axis_shift")
    stationary_b = TriggerPhaseSource(41, "b", "stationary_noise")
    shift_b = TriggerPhaseSource(41, "b", "axis_shift")

    np.testing.assert_array_equal(stationary_a.train_input, shift_a.train_input)
    np.testing.assert_array_equal(stationary_a.test_input, shift_a.test_input)
    np.testing.assert_allclose(
        stationary_b.train_input[:, 0] - shift_b.train_input[:, 0],
        1.2 * (2.0 * stationary_b.train_target[:, 0] - 1.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        shift_b.train_input[:, 1] - stationary_b.train_input[:, 1],
        1.2 * (2.0 * stationary_b.train_target[:, 0] - 1.0),
        atol=1e-12,
    )
    assert not np.array_equal(stationary_a.train_input, stationary_a.test_input)
    assert not np.array_equal(stationary_b.train_input, stationary_b.test_input)


def test_should_preflight_all_arms_without_final_data() -> None:
    class ForbiddenFinalSource(TriggerPhaseSource):
        @property
        def test_input(self) -> np.ndarray:
            raise AssertionError("v13 train-only path opened final input")

        @property
        def test_target(self) -> np.ndarray:
            raise AssertionError("v13 train-only path opened final label")

    study = comparison.train_unscored_trigger_study(
        fixed_trigger_manifest(), source_factory=ForbiddenFinalSource
    )
    assert len(study.trials) == 12
    assert all(
        role.final_test is None and not role.final_released
        for phases in study.roles.values()
        for role in phases.values()
    )
    for trial in study.trials:
        facts = trial.facts
        assert len(facts.decisions) == 32
        assert facts.a_work.wake_updates == 16
        assert facts.total_work.wake_updates == 32
        assert facts.total_work.wake_examples == 768
        assert facts.total_work.inference_loops == 64
        assert facts.total_work.example_inference_loops == 1536
        assert facts.total_work.replay_updates == 0
        assert facts.initial_width == facts.post_a_width == facts.final_width == 8
        assert (
            facts.initial_parameter_count
            == facts.post_a_parameter_count
            == facts.final_parameter_count
            == 33
        )
        assert facts.total_work.sleep_events == (4 if facts.arm == "periodic" else 0)
        assert all(
            decision.parameter_digest_before == decision.parameter_digest_after
            for decision in facts.decisions
        )


def test_should_match_adaptive_and_no_sleep_state_when_trigger_never_fires() -> None:
    study = comparison.train_unscored_trigger_study(fixed_trigger_manifest())
    for condition in ("stationary_noise", "axis_shift"):
        for seed in (41, 43):
            by_arm = {
                trial.facts.arm: trial.facts
                for trial in study.trials
                if trial.facts.condition == condition and trial.facts.seed == seed
            }
            adaptive = by_arm["adaptive"]
            no_sleep = by_arm["no_sleep"]
            assert adaptive.total_work.sleep_attempts == no_sleep.total_work.sleep_attempts == 0
            assert adaptive.post_a_parameter_digest == no_sleep.post_a_parameter_digest
            assert adaptive.post_b_parameter_digest == no_sleep.post_b_parameter_digest
            assert [row.energy for row in adaptive.decisions] == [
                row.energy for row in no_sleep.decisions
            ]


def test_should_release_common_final_roles_only_after_all_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = 0
    final_reads = 0
    original = comparison.train_trigger_trial

    def tracked_train(*args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        assert final_reads == 0
        trial = original(*args, **kwargs)
        trained += 1
        return trial

    class TrackedSource(TriggerPhaseSource):
        @property
        def test_input(self) -> np.ndarray:
            nonlocal final_reads
            assert trained == 12
            final_reads += 1
            return super().test_input

        @property
        def test_target(self) -> np.ndarray:
            nonlocal final_reads
            assert trained == 12
            final_reads += 1
            return super().test_target

    monkeypatch.setattr(comparison, "train_trigger_trial", tracked_train)
    result = comparison.run_trigger_comparison(
        fixed_trigger_manifest(), source_factory=TrackedSource
    )
    assert trained == 12
    assert final_reads == 16  # two properties × two phases × four seed/condition pairs
    assert len(result.outcomes) == 12
    for condition in ("stationary_noise", "axis_shift"):
        for seed in (41, 43):
            rows = [
                row
                for row in result.outcomes
                if row.train_facts.condition == condition and row.train_facts.seed == seed
            ]
            assert len({tuple(row.final_role_hashes.items()) for row in rows}) == 1
            assert len({tuple(row.train_facts.role_hashes.items()) for row in rows}) == 1
    for row in result.outcomes:
        assert row.forgetting == pytest.approx(row.a_after_a_accuracy - row.a_after_b_accuracy)
        assert all(
            0.0 <= value <= 1.0
            for value in (
                row.a_after_a_accuracy,
                row.a_after_b_accuracy,
                row.b_after_b_accuracy,
            )
        )
        assert all(
            np.isfinite(value) and value >= 0.0
            for value in (row.a_after_a_bce, row.a_after_b_bce, row.b_after_b_bce)
        )


@pytest.mark.parametrize("tamper", ("role", "batch", "work", "trace", "model"))
def test_should_reject_forged_train_facts_before_final_release(
    monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    final_reads = 0
    original = comparison.train_trigger_trial

    def forged_train(*args: Any, **kwargs: Any) -> Any:
        trial = original(*args, **kwargs)
        facts = trial.facts
        if facts.condition == "stationary_noise" and facts.seed == 41 and facts.arm == "periodic":
            if tamper == "role":
                trial.facts = replace(facts, role_hashes={**facts.role_hashes, "b/train": "forged"})
            elif tamper == "batch":
                trial.facts = replace(
                    facts,
                    train_batch_hashes={
                        **facts.train_batch_hashes,
                        "b": ("forged", *facts.train_batch_hashes["b"][1:]),
                    },
                )
            elif tamper == "work":
                trial.facts = replace(
                    facts, total_work=replace(facts.total_work, wake_examples=769)
                )
            elif tamper == "trace":
                decisions = list(facts.decisions)
                decisions[7] = replace(decisions[7], performed=False)
                trial.facts = replace(facts, decisions=tuple(decisions))
            else:
                trial.post_b_model.weight_hidden_output[0, 0] += 1.0
        return trial

    class TrackedSource(TriggerPhaseSource):
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

    monkeypatch.setattr(comparison, "train_trigger_trial", forged_train)
    with pytest.raises(ValueError, match="v13"):
        comparison.run_trigger_comparison(fixed_trigger_manifest(), source_factory=TrackedSource)
    assert final_reads == 0


def test_should_reject_changed_manifest_before_source_access() -> None:
    changed = replace(fixed_trigger_manifest(), seeds=(41,))

    def forbidden_source(seed: int, phase: str, condition: str) -> TriggerPhaseSource:
        raise AssertionError("source must stay unopened")

    with pytest.raises(ValueError, match="predeclared protocol"):
        comparison.run_trigger_comparison(changed, source_factory=forbidden_source)


def test_should_keep_train_facts_when_final_labels_change() -> None:
    class FlippedFinalSource(TriggerPhaseSource):
        @property
        def test_target(self) -> np.ndarray:
            return 1.0 - super().test_target

    manifest = fixed_trigger_manifest()
    original = comparison.run_trigger_comparison(manifest)
    changed = comparison.run_trigger_comparison(manifest, source_factory=FlippedFinalSource)
    assert [row.train_facts for row in original.outcomes] == [
        row.train_facts for row in changed.outcomes
    ]
    assert any(
        before.a_after_b_accuracy != after.a_after_b_accuracy
        for before, after in zip(original.outcomes, changed.outcomes, strict=True)
    )


def test_should_repeat_full_result_content() -> None:
    manifest = fixed_trigger_manifest()
    first = asdict(comparison.run_trigger_comparison(manifest))
    second = asdict(comparison.run_trigger_comparison(manifest))
    assert first == second
