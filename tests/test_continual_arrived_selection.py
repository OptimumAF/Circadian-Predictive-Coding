"""Outer setting choice must finish before any v7 final-source release."""

from __future__ import annotations

from dataclasses import replace
import pickle
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig
from src.infra import continual_roles
from src.infra.datasets import LabeledData


def _candidates(reverse_order: bool) -> tuple[Any, Any]:
    from src.app import continual_arrived_selection as selection

    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=1,
        phase_b_epochs=1,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=1,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=1,
            replay_memory_size=1,
        ),
        replay_max_examples=4,
        replay_max_bytes=96,
    )
    if reverse_order:
        training = replace(training, model_order=tuple(reversed(training.model_order)))
    first = arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2)
    second = replace(
        first,
        training=replace(
            training,
            backprop_learning_rate=training.backprop_learning_rate * 0.7,
            pc_learning_rate=training.pc_learning_rate * 0.7,
            circadian_learning_rate=training.circadian_learning_rate * 0.7,
        ),
    )
    return (
        selection.ArrivedSelectionCandidate("default", first),
        selection.ArrivedSelectionCandidate("lower_rate", second),
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_freeze_outer_choices_after_all_trials_before_final_source(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.app import continual_arrived_selection as selection

    candidates = _candidates(reverse_order)
    completed: list[tuple[float, int]] = []
    frozen = False
    final_reads: list[tuple[int, str]] = []
    original_train = arrived._train_arrived_seed
    original_freeze = selection._freeze_selection
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source

    class SealedSource:
        def __init__(self, source: Any, seed: int) -> None:
            self.source = source
            self.seed = seed
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            assert len(completed) == 4 and frozen
            final_reads.append((self.seed, "input"))
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            assert len(completed) == 4 and frozen
            final_reads.append((self.seed, "label"))
            return self.source.test_target

    def train(config: Any, seed: int) -> Any:
        pending = original_train(config, seed)
        completed.append((config.training.backprop_learning_rate, seed))
        return pending

    def freeze(*args: Any, **kwargs: Any) -> Any:
        nonlocal frozen
        result = original_freeze(*args, **kwargs)
        frozen = True
        return result

    def source_a(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_a(*args, **kwargs), kwargs["seed"])

    def source_b(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_b(*args, **kwargs), args[1] - 101)

    monkeypatch.setattr(arrived, "_train_arrived_seed", train)
    monkeypatch.setattr(selection, "_freeze_selection", freeze)
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", source_a)
    monkeypatch.setattr(arrived, "_generate_phase_b_source", source_b)

    result = selection.run_arrived_outer_selection(candidates, [17, 19])

    assert len(completed) == 4
    assert len(final_reads) == 8
    assert result.seeds == (17, 19)
    assert result.candidate_ids == ("default", "lower_rate")
    assert len(result.trials) == 12
    assert len(result.selections) == 3
    assert {item.method for item in result.selections} == set(base.CONTINUAL_MODEL_ORDER)
    assert len(result.final_seed_results) == 2
    assert result.freeze.choices == result.selections
    assert all(item.train_updates == 2 and item.outer_examples_scored > 0 for item in result.trials)
    assert all(item.phase_a_outer_hash and item.phase_b_outer_hash for item in result.trials)
    assert all(
        len(item.development_role_ids) == len(item.development_role_hashes) == 6
        for item in result.trials
    )
    assert all(len(item.method_task_information) == 2 for item in result.trials)
    assert all(item.role_accesses[-3:] == item.outer_accesses for item in result.trials)
    assert all(
        item.replay_phase_a is not None and item.replay_phase_b is not None
        if item.method == "circadian_predictive_coding"
        else item.replay_phase_a is None and item.replay_phase_b is None
        for item in result.trials
    )
    assert all(
        event.role == "outer_selection" and event.action == "outer_selection"
        for trial in result.trials
        for event in trial.outer_accesses
    )
    assert all(
        event.role == "final_test" and event.event == "global_freeze"
        for row in result.final_seed_results
        for event in row.final_role_accesses
    )
    for seed in result.seeds:
        for method in base.CONTINUAL_MODEL_ORDER:
            rows = [item for item in result.trials if item.seed == seed and item.method == method]
            assert len(rows) == 2
            assert len({item.train_updates for item in rows}) == 1
            assert len({item.train_examples_seen for item in rows}) == 1
            assert len({item.outer_examples_scored for item in rows}) == 1
            assert rows[0].development_role_ids == rows[1].development_role_ids


def test_should_reject_unequal_or_duplicate_candidate_trials_before_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.app import continual_arrived_selection as selection

    first, second = _candidates(False)

    def no_source(**_: Any) -> Any:
        raise AssertionError("candidate source opened before manifest validation")

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", no_source)
    with pytest.raises(ValueError, match="candidate IDs"):
        selection.run_arrived_outer_selection(
            (first, replace(second, candidate_id="default")), [17]
        )
    with pytest.raises(ValueError, match="fixed work"):
        changed = replace(
            second,
            config=replace(
                second.config, training=replace(second.config.training, phase_b_epochs=2)
            ),
        )
        selection.run_arrived_outer_selection((first, changed), [17])
    with pytest.raises(ValueError, match="distinct learning rates"):
        changed = replace(
            second,
            config=replace(
                second.config,
                training=replace(
                    second.config.training, pc_learning_rate=first.config.training.pc_learning_rate
                ),
            ),
        )
        selection.run_arrived_outer_selection((first, changed), [17])


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_leave_outer_choice_and_models_unchanged_when_final_labels_change(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.app import continual_arrived_selection as selection

    candidates = _candidates(reverse_order)
    trained: list[bytes] = []
    original_train = arrived._train_arrived_seed
    original_source = arrived.generate_two_cluster_dataset_with_transform

    def capture(config: Any, seed: int) -> Any:
        item = original_train(config, seed)
        trained.append(pickle.dumps(item.state, protocol=5))
        return item

    class ChangedFinalLabels:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            return 1.0 - self.source.test_target

    monkeypatch.setattr(arrived, "_train_arrived_seed", capture)
    baseline = selection.run_arrived_outer_selection(candidates, [17, 19])
    baseline_states = tuple(trained)
    trained.clear()

    def changed_a(*args: Any, **kwargs: Any) -> Any:
        source = original_source(*args, **kwargs)
        return ChangedFinalLabels(source) if kwargs["seed"] == 17 else source

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_a)
    changed = selection.run_arrived_outer_selection(candidates, [17, 19])

    assert tuple(trained) == baseline_states
    assert changed.trials == baseline.trials
    assert changed.selections == baseline.selections
    assert changed.freeze == baseline.freeze
    assert changed.final_seed_results[1] == baseline.final_seed_results[1]
    assert (
        changed.final_seed_results[0].role_hashes["phase_a_final_test"]
        != baseline.final_seed_results[0].role_hashes["phase_a_final_test"]
    )


def test_should_use_outer_labels_only_after_training_and_keep_all_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.app import continual_arrived_selection as selection

    candidates = _candidates(False)
    trained: list[bytes] = []
    original_train = arrived._train_arrived_seed
    original_split = arrived.split_phase_decision_roles

    def capture(config: Any, seed: int) -> Any:
        item = original_train(config, seed)
        trained.append(pickle.dumps(item.state, protocol=5))
        return item

    monkeypatch.setattr(arrived, "_train_arrived_seed", capture)
    baseline = selection.run_arrived_outer_selection(candidates, [17, 19])
    baseline_states = tuple(trained)
    trained.clear()

    def changed_outer(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        outer = LabeledData(roles.outer_selection.input, 1.0 - roles.outer_selection.target)
        hashes = dict(roles.split_hashes)
        hashes["outer_selection"] = continual_roles._hash_role(
            roles.phase,
            roles.seed,
            "outer_selection",
            roles.sample_ids["outer_selection"],
            outer,
        )
        return replace(
            roles,
            outer_selection=outer,
            split_hashes=MappingProxyType(hashes),
        )

    monkeypatch.setattr(arrived, "split_phase_decision_roles", changed_outer)
    changed = selection.run_arrived_outer_selection(candidates, [17, 19])

    assert tuple(trained) == baseline_states
    assert len(changed.trials) == len(baseline.trials) == 12
    assert all(
        first.phase_a_outer_hash != second.phase_a_outer_hash
        and first.phase_b_outer_hash != second.phase_b_outer_hash
        for first, second in zip(baseline.trials, changed.trials, strict=True)
    )
    assert any(
        not np.isclose(first.balanced_score, second.balanced_score)
        for first, second in zip(baseline.trials, changed.trials, strict=True)
    )


def test_should_choose_first_declared_candidate_on_exact_outer_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.app import continual_arrived_selection as selection

    candidates = _candidates(False)
    original = selection._score_outer_trial

    def tied(candidate_id: str, pending: Any, method: str) -> Any:
        row = original(candidate_id, pending, method)
        return replace(
            row, phase_a_post_accuracy=0.5, phase_b_post_accuracy=0.5, balanced_score=0.5
        )

    monkeypatch.setattr(selection, "_score_outer_trial", tied)
    result = selection.run_arrived_outer_selection(candidates, [17, 19])
    assert [item.candidate_id for item in result.selections] == ["default"] * 3
    assert all(item.mean_outer_balanced_score == 0.5 for item in result.selections)


def test_should_score_each_method_from_its_own_outer_selected_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.app import continual_arrived_selection as selection

    candidates = _candidates(False)
    trained: dict[tuple[float, int], Any] = {}
    original_train = arrived._train_arrived_seed
    original_outer = selection._score_outer_trial
    original_final = base._score_seed_models
    expected = {
        "backprop": "lower_rate",
        "predictive_coding": "default",
        "circadian_predictive_coding": "lower_rate",
    }
    rates = {
        candidate.candidate_id: candidate.config.training.backprop_learning_rate
        for candidate in candidates
    }

    def capture(config: Any, seed: int) -> Any:
        item = original_train(config, seed)
        trained[config.training.backprop_learning_rate, seed] = item
        return item

    def score_outer(candidate_id: str, pending: Any, method: str) -> Any:
        row = original_outer(candidate_id, pending, method)
        score = 0.9 if candidate_id == expected[method] else 0.1
        return replace(
            row,
            phase_a_post_accuracy=score,
            phase_b_post_accuracy=score,
            balanced_score=score,
        )

    def score_final(config: Any, seed: int, state: Any, *args: Any, **kwargs: Any) -> Any:
        chosen = {
            method: trained[rates[candidate_id], seed].state
            for method, candidate_id in expected.items()
        }
        assert state.backprop_model is chosen["backprop"].backprop_model
        assert state.predictive_model is chosen["predictive_coding"].predictive_model
        assert state.circadian_model is chosen["circadian_predictive_coding"].circadian_model
        return original_final(config, seed, state, *args, **kwargs)

    monkeypatch.setattr(arrived, "_train_arrived_seed", capture)
    monkeypatch.setattr(selection, "_score_outer_trial", score_outer)
    monkeypatch.setattr(base, "_score_seed_models", score_final)
    result = selection.run_arrived_outer_selection(candidates, [17, 19])

    assert {item.method: item.candidate_id for item in result.selections} == expected
    assert all(item.trial_count == 4 for item in result.selections)


def test_should_isolate_model_order_and_repeat_fixed_selection() -> None:
    from src.app import continual_arrived_selection as selection

    forward = selection.run_arrived_outer_selection(_candidates(False), [17, 19])
    repeated = selection.run_arrived_outer_selection(_candidates(False), [17, 19])
    reverse = selection.run_arrived_outer_selection(_candidates(True), [17, 19])

    assert forward == repeated
    for first_trial, second_trial in zip(forward.trials, reverse.trials, strict=True):
        assert replace(first_trial, role_accesses=()) == replace(second_trial, role_accesses=())
    assert forward.selections == reverse.selections
    for first_seed, second_seed in zip(
        forward.final_seed_results, reverse.final_seed_results, strict=True
    ):
        assert first_seed.role_ids == second_seed.role_ids
        assert first_seed.role_hashes == second_seed.role_hashes
        assert first_seed.metrics.backprop == second_seed.metrics.backprop
        assert first_seed.metrics.predictive_coding == second_seed.metrics.predictive_coding
        assert (
            first_seed.metrics.circadian_predictive_coding
            == second_seed.metrics.circadian_predictive_coding
        )
