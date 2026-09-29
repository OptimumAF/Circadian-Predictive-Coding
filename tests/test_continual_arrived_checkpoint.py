"""A v6 completed-seed checkpoint must keep final fields sealed across resume."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplaySnapshot,
)
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.circadian_checkpoint_files import TrustedLocalArrivedCheckpointStore


class SeedBoundaryInterruption(Exception):
    """Stop after a durable unscored seed transaction."""


class StopAfterFirstSeed:
    def __init__(self, path: Path) -> None:
        self.store = TrustedLocalArrivedCheckpointStore(path)

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if checkpoint.seed_index == 0 and checkpoint.phase == "seed_complete":
            raise SeedBoundaryInterruption()


class ActiveInterruption(Exception):
    """Stop after a durable active A/B model or sleep transaction."""


class StopAtActiveCheckpoint:
    def __init__(
        self,
        path: Path,
        *,
        seed_index: int,
        phase: str,
        epoch: int,
        stage: str,
        next_model_index: int = 0,
    ) -> None:
        self.store = TrustedLocalArrivedCheckpointStore(path)
        self.target = (seed_index, phase, epoch, stage, next_model_index)

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        observed = (
            checkpoint.seed_index,
            checkpoint.phase,
            checkpoint.phase_epoch_completed,
            checkpoint.stage,
            checkpoint.next_model_index,
        )
        if observed == self.target:
            raise ActiveInterruption()


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
    return arrived.ContinualArrivedRolesConfig(training, 0.2, 0.2)


def _assert_same_model_fields(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    left = actual.state if hasattr(actual, "state") else actual.__dict__
    right = expected.state if hasattr(expected, "state") else expected.__dict__
    assert left.keys() == right.keys()
    for name in left:
        if name == "_replay_memory":
            assert len(left[name]) == len(right[name])
            for first, second in zip(left[name], right[name], strict=True):
                np.testing.assert_array_equal(first.input_batch, second.input_batch)
                np.testing.assert_array_equal(first.target_batch, second.target_batch)
                assert first.priority == second.priority
                assert first.positive_fraction == second.positive_fraction
        else:
            assert pickle.dumps(left[name], protocol=5) == pickle.dumps(right[name], protocol=5), (
                name
            )


def _assert_same_unscored_records(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected)
    for first, second in zip(actual, expected, strict=True):
        assert first.seed == second.seed
        assert first.role_ids == second.role_ids
        assert first.role_hashes == second.role_hashes
        assert first.role_accesses == second.role_accesses
        assert first.guard_decisions == second.guard_decisions
        assert first.method_task_information == second.method_task_information
        for field in fields(first.state):
            left = getattr(first.state, field.name)
            right = getattr(second.state, field.name)
            if field.name.startswith(("backprop", "predictive", "circadian")):
                _assert_same_model_fields(left, right)
            else:
                assert left == right


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    ("seed_index", "phase", "epoch", "stage", "next_model_index"),
    [
        (0, "a", 0, "wake", 1),
        (0, "a", 1, "before_sleep", 0),
        (0, "a", 1, "after_sleep", 0),
        (0, "b", 0, "after_sleep", 0),
        (0, "b", 0, "wake", 1),
        (0, "b", 1, "before_sleep", 0),
        (0, "b", 1, "after_sleep", 0),
        (1, "a", 0, "wake", 1),
    ],
)
def test_should_resume_active_v6_transaction_without_repeating_updates(
    reverse_order: bool,
    seed_index: int,
    phase: str,
    epoch: int,
    stage: str,
    next_model_index: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(reverse_order)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17, 19])
    control_store = TrustedLocalArrivedCheckpointStore(tmp_path / "control.ckpt")
    control = arrived.run_continual_arrived_benchmark(
        config, [17, 19], checkpoint_store=control_store
    )
    assert control == ordinary
    updates = {"backprop": 0, "predictive_coding": 0, "circadian_predictive_coding": 0}
    final_reads: list[str] = []
    original_backprop = arrived.base.BackpropMLP.train_epoch
    original_predictive = PredictiveCodingNetwork.train_epoch
    original_circadian = CircadianPredictiveCodingNetwork.train_epoch
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source

    def count_backprop(model: Any, *args: Any, **kwargs: Any) -> Any:
        updates["backprop"] += 1
        return original_backprop(model, *args, **kwargs)

    def count_predictive(model: Any, *args: Any, **kwargs: Any) -> Any:
        updates["predictive_coding"] += 1
        return original_predictive(model, *args, **kwargs)

    def count_circadian(model: Any, *args: Any, **kwargs: Any) -> Any:
        updates["circadian_predictive_coding"] += 1
        return original_circadian(model, *args, **kwargs)

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            if any(count != 8 for count in updates.values()):
                raise AssertionError("active resume opened final input before all training")
            final_reads.append("input")
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            if any(count != 8 for count in updates.values()):
                raise AssertionError("active resume opened final label before all training")
            final_reads.append("label")
            return self.source.test_target

    def checked_b(*args: Any, **kwargs: Any) -> Any:
        required = 2 if args[1] == 17 + 101 else 6
        if any(count < required for count in updates.values()):
            raise AssertionError("Phase B source opened before all Phase A models finished")
        return SealedSource(original_b(*args, **kwargs))

    monkeypatch.setattr(arrived.base.BackpropMLP, "train_epoch", count_backprop)
    monkeypatch.setattr(PredictiveCodingNetwork, "train_epoch", count_predictive)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "train_epoch", count_circadian)
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda *args, **kwargs: SealedSource(original_a(*args, **kwargs)),
    )
    monkeypatch.setattr(
        arrived,
        "_generate_phase_b_source",
        checked_b,
    )
    interrupted = StopAtActiveCheckpoint(
        tmp_path / "active-v6.ckpt",
        seed_index=seed_index,
        phase=phase,
        epoch=epoch,
        stage=stage,
        next_model_index=next_model_index,
    )
    with pytest.raises(ActiveInterruption):
        arrived.run_continual_arrived_benchmark(config, [17, 19], checkpoint_store=interrupted)
    saved = interrupted.store.load()
    assert saved.format_version == 6
    assert saved.active_state is not None
    assert saved.active_circadian is not None
    assert saved.active_position is not None
    assert all("final_test" not in role for role, _ in saved.active_role_hashes)
    assert all(event.role != "final_test" for event in saved.active_role_accesses)
    if phase == "a":
        assert all(role.startswith("phase_a_") for role, _ in saved.active_role_hashes)
    assert final_reads == []

    resumed = arrived.run_continual_arrived_benchmark(
        config, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
    )
    assert updates == {method: 8 for method in updates}
    assert len(final_reads) == 8
    assert resumed == ordinary
    _assert_same_unscored_records(
        interrupted.store.load().unscored_seeds, control_store.load().unscored_seeds
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_resume_later_seed_without_early_final_release(
    reverse_order: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(reverse_order)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17, 19])
    backprop_updates = 0
    final_reads: list[str] = []
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    original_train = arrived.base.BackpropMLP.train_epoch

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            if backprop_updates != 8:
                raise AssertionError("final input opened before every seed trained")
            final_reads.append("input")
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            if backprop_updates != 8:
                raise AssertionError("final label opened before every seed trained")
            final_reads.append("label")
            return self.source.test_target

    def wrap_a(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_a(*args, **kwargs))

    def wrap_b(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_b(*args, **kwargs))

    def count_update(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal backprop_updates
        backprop_updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", wrap_a)
    monkeypatch.setattr(arrived, "_generate_phase_b_source", wrap_b)
    monkeypatch.setattr(arrived.base.BackpropMLP, "train_epoch", count_update)

    interrupted = StopAfterFirstSeed(tmp_path / "arrived-v6.ckpt")
    with pytest.raises(SeedBoundaryInterruption):
        arrived.run_continual_arrived_benchmark(config, [17, 19], checkpoint_store=interrupted)
    saved = interrupted.store.load()
    assert saved.format_version == 6
    assert len(saved.unscored_seeds) == 1
    assert all("final_test" not in key for key, _ in saved.unscored_seeds[0].role_hashes)
    assert backprop_updates == 4
    assert final_reads == []

    resumed = arrived.run_continual_arrived_benchmark(
        config, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
    )
    assert backprop_updates == 8
    assert len(final_reads) == 8
    assert resumed == ordinary


@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_resume_terminal_unscored_checkpoint_without_training(
    reverse_order: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(reverse_order)
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "terminal.ckpt")
    expected = arrived.run_continual_arrived_benchmark(config, [17, 19], checkpoint_store=store)
    assert len(store.load().unscored_seeds) == 2

    def no_training(*_: Any, **__: Any) -> Any:
        raise AssertionError("terminal resume retrained a seed")

    monkeypatch.setattr(arrived.base.BackpropMLP, "train_epoch", no_training)
    actual = arrived.run_continual_arrived_benchmark(
        config, [17, 19], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert actual == expected


def test_should_score_changed_final_labels_only_after_terminal_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(False)
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "final-label.ckpt")
    ordinary = arrived.run_continual_arrived_benchmark(config, [17, 19], checkpoint_store=store)
    checkpoint_bytes = store.path.read_bytes()
    original = arrived.generate_two_cluster_dataset_with_transform

    def changed_final(*args: Any, **kwargs: Any) -> Any:
        source = original(*args, **kwargs)
        if kwargs["seed"] != 17:
            return source
        return replace(source, test_target=1.0 - source.test_target)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_final)
    changed = arrived.run_continual_arrived_benchmark(
        config, [17, 19], checkpoint_store=store, resume_from_checkpoint=True
    )

    assert store.path.read_bytes() == checkpoint_bytes
    assert ordinary.seed_results[0].guard_decisions == changed.seed_results[0].guard_decisions
    assert ordinary.seed_results[1] == changed.seed_results[1]
    assert (
        ordinary.seed_results[0].role_hashes["phase_a_final_test"]
        != changed.seed_results[0].role_hashes["phase_a_final_test"]
    )


def _save_first_seed(config: arrived.ContinualArrivedRolesConfig, path: Path) -> Any:
    interrupted = StopAfterFirstSeed(path)
    with pytest.raises(SeedBoundaryInterruption):
        arrived.run_continual_arrived_benchmark(config, [17, 19], checkpoint_store=interrupted)
    return interrupted.store


def _save_active_seed(
    config: arrived.ContinualArrivedRolesConfig,
    path: Path,
    *,
    phase: str,
    epoch: int,
    stage: str,
    next_model_index: int,
) -> Any:
    interrupted = StopAtActiveCheckpoint(
        path,
        seed_index=0,
        phase=phase,
        epoch=epoch,
        stage=stage,
        next_model_index=next_model_index,
    )
    with pytest.raises(ActiveInterruption):
        arrived.run_continual_arrived_benchmark(config, [17, 19], checkpoint_store=interrupted)
    return interrupted.store


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "tamper",
    [
        "source_a",
        "source_b",
        "replay",
        "unarrived_replay",
        "outer_replay",
        "event_cursor",
        "guard_cursor",
        "baseline_steps",
        "position",
        "future_role",
    ],
)
def test_should_reject_tampered_active_cursor_before_any_update(
    reverse_order: bool,
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(reverse_order)
    phase = "b" if tamper in {"source_b", "replay"} else "a"
    stage = "after_sleep" if tamper == "guard_cursor" else "wake"
    epoch = 1 if stage == "after_sleep" else 0
    next_model_index = 0 if stage == "after_sleep" else 1
    store = _save_active_seed(
        config,
        tmp_path / "tampered-active.ckpt",
        phase=phase,
        epoch=epoch,
        stage=stage,
        next_model_index=next_model_index,
    )
    if tamper == "source_a":
        original_a = arrived.generate_two_cluster_dataset_with_transform

        def changed_a(*args: Any, **kwargs: Any) -> Any:
            source = original_a(*args, **kwargs)
            return replace(source, train_target=1.0 - source.train_target)

        monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_a)
        message = "active development roles"
    elif tamper == "source_b":
        original_b = arrived._generate_phase_b_source

        def changed_b(*args: Any, **kwargs: Any) -> Any:
            source = original_b(*args, **kwargs)
            return replace(source, train_target=1.0 - source.train_target)

        monkeypatch.setattr(arrived, "_generate_phase_b_source", changed_b)
        message = "active development roles"
    else:
        saved = deepcopy(store.load())
        if tamper in {"replay", "unarrived_replay", "outer_replay"}:
            if tamper == "replay":
                future = arrived._build_phase_a_roles(config, 19).train
            elif tamper == "unarrived_replay":
                future = arrived._build_phase_b_roles(config, 17).train

                def no_b_source(*_: Any, **__: Any) -> Any:
                    raise AssertionError("Phase B source opened before Phase A resume")

                monkeypatch.setattr(arrived, "_generate_phase_b_source", no_b_source)
            else:
                future = arrived._build_phase_a_roles(config, 17).outer_selection
            assert saved.active_circadian is not None
            memory = saved.active_circadian.state["_replay_memory"]
            memory.clear()
            memory.append(
                ReplaySnapshot(
                    input_batch=future.input[:1].copy(),
                    target_batch=future.target[:1].copy(),
                    priority=1.0,
                    positive_fraction=float(future.target[0, 0]),
                )
            )
            message = "replay observed training examples"
        elif tamper == "event_cursor":
            saved = replace(saved, active_role_accesses=saved.active_role_accesses[:-1])
            message = "active event digest"
        elif tamper == "guard_cursor":
            saved = replace(
                saved,
                active_role_accesses=tuple(
                    event for event in saved.active_role_accesses if event.action != "guard"
                ),
                active_guard_decisions=(),
            )
            message = "active event digest"
        elif tamper == "baseline_steps":
            assert saved.active_state is not None
            saved.active_state.backprop_model._traffic_steps += 1
            message = "baseline state"
        elif tamper == "position":
            assert saved.active_position is not None
            saved = replace(
                saved,
                active_position=replace(saved.active_position, next_batch_index=2),
            )
            message = "active checkpoint cursor"
        else:
            saved = replace(
                saved,
                active_role_hashes=saved.active_role_hashes + (("phase_b_train", "forged"),),
            )
            message = "active development roles"
        store.save(saved)
    updates = 0
    original_train = arrived.base.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(arrived.base.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match=message):
        arrived.run_continual_arrived_benchmark(
            config, [17, 19], checkpoint_store=store, resume_from_checkpoint=True
        )
    assert updates == 0


@pytest.mark.parametrize("change", ["config", "seeds"])
def test_should_reject_changed_run_identity_before_source_access(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(False)
    store = _save_first_seed(config, tmp_path / "identity.ckpt")

    def no_source(*_: Any, **__: Any) -> Any:
        raise AssertionError("changed run identity opened a source")

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", no_source)
    changed_config = replace(config, guard_drop_tolerance=0.1) if change == "config" else config
    seeds = [17, 19] if change == "config" else [17, 23]
    with pytest.raises(ValueError, match="format, config, or seeds"):
        arrived.run_continual_arrived_benchmark(
            changed_config, seeds, checkpoint_store=store, resume_from_checkpoint=True
        )


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "tamper", ["source_a", "source_b", "replay", "event_cursor", "paired_event_omission"]
)
def test_should_reject_changed_prior_seed_before_later_update(
    reverse_order: bool,
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(reverse_order)
    store = _save_first_seed(config, tmp_path / "prior.ckpt")
    if tamper == "source_a":
        original = arrived.generate_two_cluster_dataset_with_transform

        def changed_source(*args: Any, **kwargs: Any) -> Any:
            source = original(*args, **kwargs)
            if kwargs["seed"] != 17:
                return source
            return replace(source, train_target=1.0 - source.train_target)

        monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_source)
        message = "completed development roles"
    elif tamper == "source_b":
        original_b = arrived._generate_phase_b_source

        def changed_b(*args: Any, **kwargs: Any) -> Any:
            source = original_b(*args, **kwargs)
            if args[1] != 17 + 101:
                return source
            return replace(source, train_target=1.0 - source.train_target)

        monkeypatch.setattr(arrived, "_generate_phase_b_source", changed_b)
        message = "completed development roles"
    else:
        saved = deepcopy(store.load())
        record = saved.unscored_seeds[0]
        if tamper == "replay":
            future = arrived._build_phase_a_roles(config, 19).train
            memory = record.state.circadian_model._replay_memory
            memory.clear()
            memory.append(
                ReplaySnapshot(
                    input_batch=future.input[:1].copy(),
                    target_batch=future.target[:1].copy(),
                    priority=1.0,
                    positive_fraction=float(future.target[0, 0]),
                )
            )
            message = "replay observed training examples"
        elif tamper == "event_cursor":
            record = replace(record, role_accesses=record.role_accesses[:-1])
            saved = replace(saved, unscored_seeds=(record,))
            message = "completed event"
        else:
            record = replace(
                record,
                role_accesses=tuple(
                    event for event in record.role_accesses if event.action != "guard"
                ),
                guard_decisions=(),
            )
            saved = replace(saved, unscored_seeds=(record,))
            message = "completed event digest"
        store.save(saved)
    updates = 0
    original_train = arrived.base.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(arrived.base.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match=message):
        arrived.run_continual_arrived_benchmark(
            config, [17, 19], checkpoint_store=store, resume_from_checkpoint=True
        )
    assert updates == 0
