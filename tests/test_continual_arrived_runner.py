"""The v6 ordinary runner must observe phase and decision-role release gates."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import pickle
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra import continual_roles
from src.infra.datasets import LabeledData


def _config(reverse_order: bool) -> arrived.ContinualArrivedRolesConfig:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
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
    return arrived.ContinualArrivedRolesConfig(
        training=training,
        inner_guard_fraction=0.2,
        outer_selection_fraction=0.2,
        guard_drop_tolerance=0.0,
    )


def test_should_reject_invalid_run_identity_before_any_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_source(**_: Any) -> Any:
        raise AssertionError("source opened before config validation")

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", no_source)
    config = _config(False)
    with pytest.raises(ValueError, match="unique Python integers"):
        arrived.run_continual_arrived_benchmark(config, [17, 17])
    with pytest.raises(ValueError, match="role fractions"):
        arrived.run_continual_arrived_benchmark(replace(config, inner_guard_fraction=0.8), [17])
    with pytest.raises(ValueError, match="matching arrived-roles config"):
        arrived.run_continual_arrived_benchmark(replace(config, protocol_id="wrong"), [17])


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_seal_future_and_final_roles_until_their_arrivals(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(reverse_order)
    phase_a_finished: set[int] = set()
    fully_trained: set[int] = set()
    final_reads: list[tuple[int, str]] = []
    current_seed = -1
    original_a_source = arrived.generate_two_cluster_dataset_with_transform
    original_b_source = arrived._generate_phase_b_source
    original_train_a = base._train_phase_a_models
    original_train_b = base._train_phase_b_models

    class SealedSource:
        def __init__(self, source: Any, seed: int) -> None:
            self.source = source
            self.seed = seed
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            if fully_trained != {17, 19}:
                raise AssertionError("final source input opened before all seeds trained")
            final_reads.append((self.seed, "input"))
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            if fully_trained != {17, 19}:
                raise AssertionError("final source label opened before all seeds trained")
            final_reads.append((self.seed, "label"))
            return self.source.test_target

    def source_a(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_a_source(*args, **kwargs), kwargs["seed"])

    def source_b(*args: Any, **kwargs: Any) -> Any:
        if current_seed not in phase_a_finished:
            raise AssertionError("Phase B source opened before Phase A finished")
        return SealedSource(original_b_source(*args, **kwargs), current_seed)

    def train_a(**kwargs: Any) -> Any:
        nonlocal current_seed
        current_seed = kwargs["seed"]
        state = original_train_a(**kwargs)
        phase_a_finished.add(current_seed)
        return state

    def train_b(**kwargs: Any) -> Any:
        state = original_train_b(**kwargs)
        fully_trained.add(current_seed)
        return state

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", source_a)
    monkeypatch.setattr(arrived, "_generate_phase_b_source", source_b)
    monkeypatch.setattr(base, "_train_phase_a_models", train_a)
    monkeypatch.setattr(base, "_train_phase_b_models", train_b)

    result = arrived.run_continual_arrived_benchmark(config, [17, 19])

    assert phase_a_finished == fully_trained == {17, 19}
    assert len(final_reads) == 8
    assert result.protocol_id == "continual_arrived_roles_v6"
    assert [item.seed for item in result.seed_results] == [17, 19]
    for seed_result in result.seed_results:
        assert set(seed_result.role_ids) == {
            "phase_a_train",
            "phase_a_inner_guard",
            "phase_a_outer_selection",
            "phase_a_final_test",
            "phase_b_train",
            "phase_b_inner_guard",
            "phase_b_outer_selection",
            "phase_b_final_test",
        }
        assert set(seed_result.role_hashes) == set(seed_result.role_ids)
        assert seed_result.guard_decisions
        assert all(item.accepted or item.restored for item in seed_result.guard_decisions)
        assert len(seed_result.method_task_information) == 6
        assert sum(event.action == "train" for event in seed_result.role_accesses) == 12
        assert sum(event.action == "guard" for event in seed_result.role_accesses) == 4
        assert sum(event.action == "label_release" for event in seed_result.role_accesses) == 8
        assert all(
            event.event == "global_freeze"
            for event in seed_result.role_accesses
            if event.role == "final_test"
        )
        assert all(
            event.role == "train" if event.action == "train" else event.role == "inner_guard"
            for event in seed_result.role_accesses
            if event.action in {"train", "guard"}
        )
        assert all(
            info.declared_phases == ("a", "b")
            and info.arrived_phases == (("a",) if info.phase == "a" else ("a", "b"))
            for info in seed_result.method_task_information
        )


def test_should_repeat_reports_and_isolate_model_order() -> None:
    forward = arrived.run_continual_arrived_benchmark(_config(False), [17, 19])
    repeated = arrived.run_continual_arrived_benchmark(_config(False), [17, 19])
    reverse = arrived.run_continual_arrived_benchmark(_config(True), [17, 19])

    assert forward == repeated
    for first, second in zip(forward.seed_results, reverse.seed_results, strict=True):
        assert first.role_ids == second.role_ids
        assert first.role_hashes == second.role_hashes
        assert first.metrics.backprop == second.metrics.backprop
        assert first.metrics.predictive_coding == second.metrics.predictive_coding
        assert (
            first.metrics.circadian_predictive_coding == second.metrics.circadian_predictive_coding
        )
        assert first.metrics.replay_retention == second.metrics.replay_retention
        assert first.guard_decisions == second.guard_decisions


def test_should_ignore_outer_label_changes_during_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(False)
    trained_hashes: list[str] = []
    original_train_b = base._train_phase_b_models

    def capture_state(**kwargs: Any) -> Any:
        state = original_train_b(**kwargs)
        trained_hashes.append(sha256(pickle.dumps(state, protocol=5)).hexdigest())
        return state

    monkeypatch.setattr(base, "_train_phase_b_models", capture_state)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17, 19])
    ordinary_hashes = tuple(trained_hashes)
    trained_hashes.clear()
    original_split = arrived.split_phase_decision_roles

    def changed_outer(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        changed = LabeledData(roles.outer_selection.input, 1.0 - roles.outer_selection.target)
        hashes = dict(roles.split_hashes)
        hashes["outer_selection"] = continual_roles._hash_role(
            roles.phase,
            roles.seed,
            "outer_selection",
            roles.sample_ids["outer_selection"],
            changed,
        )
        return replace(
            roles,
            outer_selection=changed,
            split_hashes=MappingProxyType(hashes),
        )

    monkeypatch.setattr(arrived, "split_phase_decision_roles", changed_outer)
    changed = arrived.run_continual_arrived_benchmark(config, [17, 19])

    assert tuple(trained_hashes) == ordinary_hashes
    for first, second in zip(ordinary.seed_results, changed.seed_results, strict=True):
        assert first.guard_decisions == second.guard_decisions
        assert first.metrics.backprop == second.metrics.backprop
        assert first.metrics.predictive_coding == second.metrics.predictive_coding
        assert (
            first.metrics.circadian_predictive_coding == second.metrics.circadian_predictive_coding
        )
        assert (
            first.role_hashes["phase_a_outer_selection"]
            != second.role_hashes["phase_a_outer_selection"]
        )
        assert (
            first.role_hashes["phase_b_outer_selection"]
            != second.role_hashes["phase_b_outer_selection"]
        )


def test_should_use_changed_inner_labels_only_for_guard_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(False)
    baseline = arrived.run_continual_arrived_benchmark(config, [17])
    original_split = arrived.split_phase_decision_roles
    original_accuracy = CircadianPredictiveCodingNetwork.compute_accuracy
    changed_targets: list[np.ndarray] = []
    observed_guard_targets: list[np.ndarray] = []

    def changed_inner(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        changed = LabeledData(roles.inner_guard.input, 1.0 - roles.inner_guard.target)
        changed_targets.append(changed.target)
        hashes = dict(roles.split_hashes)
        hashes["inner_guard"] = continual_roles._hash_role(
            roles.phase,
            roles.seed,
            "inner_guard",
            roles.sample_ids["inner_guard"],
            changed,
        )
        return replace(roles, inner_guard=changed, split_hashes=MappingProxyType(hashes))

    def record_accuracy(
        self: CircadianPredictiveCodingNetwork, inputs: np.ndarray, targets: np.ndarray
    ) -> float:
        if any(targets is changed for changed in changed_targets):
            observed_guard_targets.append(targets)
        return original_accuracy(self, inputs, targets)

    monkeypatch.setattr(arrived, "split_phase_decision_roles", changed_inner)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", record_accuracy)
    changed = arrived.run_continual_arrived_benchmark(config, [17])

    assert len(changed_targets) == 2
    assert len(observed_guard_targets) == 8
    assert all(
        any(target is changed for changed in changed_targets) for target in observed_guard_targets
    )
    assert baseline.seed_results[0].metrics.backprop == changed.seed_results[0].metrics.backprop
    assert (
        baseline.seed_results[0].metrics.predictive_coding
        == changed.seed_results[0].metrics.predictive_coding
    )
    for phase in ("a", "b"):
        for role in ("train", "outer_selection"):
            key = f"phase_{phase}_{role}"
            assert (
                baseline.seed_results[0].role_hashes[key]
                == changed.seed_results[0].role_hashes[key]
            )
        key = f"phase_{phase}_inner_guard"
        assert baseline.seed_results[0].role_hashes[key] != changed.seed_results[0].role_hashes[key]


def test_should_keep_all_trained_states_when_first_seed_final_labels_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(False)
    trained_hashes: list[str] = []
    original_train_b = base._train_phase_b_models

    def capture_state(**kwargs: Any) -> Any:
        state = original_train_b(**kwargs)
        trained_hashes.append(sha256(pickle.dumps(state, protocol=5)).hexdigest())
        return state

    monkeypatch.setattr(base, "_train_phase_b_models", capture_state)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17, 19])
    ordinary_hashes = tuple(trained_hashes)
    trained_hashes.clear()
    original_source = arrived.generate_two_cluster_dataset_with_transform

    def changed_final(*args: Any, **kwargs: Any) -> Any:
        source = original_source(*args, **kwargs)
        if kwargs["seed"] != 17:
            return source
        return replace(source, test_target=1.0 - source.test_target)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_final)
    changed = arrived.run_continual_arrived_benchmark(config, [17, 19])

    assert tuple(trained_hashes) == ordinary_hashes
    assert ordinary.seed_results[0].guard_decisions == changed.seed_results[0].guard_decisions
    assert ordinary.seed_results[1] == changed.seed_results[1]
    for role in ("train", "inner_guard", "outer_selection"):
        assert (
            ordinary.seed_results[0].role_hashes[f"phase_a_{role}"]
            == changed.seed_results[0].role_hashes[f"phase_a_{role}"]
        )
    assert (
        ordinary.seed_results[0].role_hashes["phase_a_final_test"]
        != changed.seed_results[0].role_hashes["phase_a_final_test"]
    )


def test_should_restore_sleep_when_arrived_inner_guard_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=23,
        circadian_config=CircadianConfig(sleep_mode="components"),
    )
    original_weights = model.weight_input_hidden.copy()
    guard = LabeledData(
        input=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64),
        target=np.array([[0.0], [1.0]], dtype=np.float64),
    )
    scores = iter((1.0, 0.0))
    observed: list[tuple[int, float, float, bool, bool]] = []

    def harmful_sleep(**kwargs: Any) -> Any:
        model.weight_input_hidden += 1.0
        return SimpleNamespace(
            split_indices=(),
            pruned_indices=(),
            old_hidden_dim=4,
            new_hidden_dim=4,
            performed=True,
        )

    monkeypatch.setattr(model, "compute_accuracy", lambda *_: next(scores))
    monkeypatch.setattr(model, "sleep_event", harmful_sleep)
    result = base._apply_scheduled_sleep(
        model=model,
        sleep_interval=1,
        epoch_index=1,
        global_epoch=1,
        total_epochs=1,
        force_sleep=True,
        sleep_event_count=0,
        total_splits=0,
        total_prunes=0,
        guard=guard,
        guard_drop_tolerance=0.0,
        on_guard_decision=lambda *item: observed.append(item),
    )

    assert result == (0, 0, 0)
    np.testing.assert_array_equal(model.weight_input_hidden, original_weights)
    assert observed == [(1, 1.0, 0.0, True, False)]
