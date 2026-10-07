"""Run-level final-test release is distinct from the v4 per-seed seal."""

from __future__ import annotations

from dataclasses import fields, replace
from copy import deepcopy
from hashlib import sha256
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_shift_benchmark as continual
from src.app.continual_shift_benchmark import ContinualBoundedReplayConfig
from src.core.circadian_predictive_coding import CircadianConfig, ReplaySnapshot
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore


def _bounded_config(reverse_order: bool) -> ContinualBoundedReplayConfig:
    config = ContinualBoundedReplayConfig(
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
    return (
        replace(config, model_order=tuple(reversed(config.model_order)))
        if reverse_order
        else config
    )


def _global_config(reverse_order: bool) -> continual.ContinualGlobalSealConfig:
    base = _bounded_config(reverse_order)
    return continual.ContinualGlobalSealConfig(
        **{
            item.name: getattr(base, item.name)
            for item in fields(ContinualBoundedReplayConfig)
            if item.name != "protocol_id"
        }
    )


def _seal_source_until_all_seeds_train(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[set[int], list[str]]:
    trained: set[int] = set()
    reads: list[str] = []
    current_seed = -1
    original_generate = continual.generate_two_cluster_dataset_with_transform
    original_train_a = continual._train_phase_a_models
    original_train_b = continual._train_phase_b_models

    class SealedSource:
        def __init__(self, actual: Any) -> None:
            self.actual = actual
            self.train_input = actual.train_input
            self.train_target = actual.train_target

        @property
        def test_input(self) -> Any:
            if trained != {17, 19}:
                raise AssertionError("final-test source input opened before run freeze")
            reads.append("input")
            return self.actual.test_input

        @property
        def test_target(self) -> Any:
            if trained != {17, 19}:
                raise AssertionError("final-test source label opened before run freeze")
            reads.append("target")
            return self.actual.test_target

    def sealed_source(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_generate(*args, **kwargs))

    def train_a(**kwargs: Any) -> Any:
        nonlocal current_seed
        current_seed = kwargs["seed"]
        return original_train_a(**kwargs)

    def train_b(**kwargs: Any) -> Any:
        state = original_train_b(**kwargs)
        trained.add(current_seed)
        return state

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", sealed_source)
    monkeypatch.setattr(continual, "_train_phase_a_models", train_a)
    monkeypatch.setattr(continual, "_train_phase_b_models", train_b)
    return trained, reads


def _seal_checkpoint_source_until_all_seeds_train(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[set[int], list[str]]:
    trained: set[int] = set()
    reads: list[str] = []
    original_generate = continual.generate_two_cluster_dataset_with_transform
    original_train = continual._train_checkpoint_phase

    class SealedSource:
        def __init__(self, actual: Any) -> None:
            self.actual = actual
            self.train_input = actual.train_input
            self.train_target = actual.train_target

        @property
        def test_input(self) -> Any:
            if trained != {17, 19}:
                raise AssertionError("checkpoint final input opened before run freeze")
            reads.append("input")
            return self.actual.test_input

        @property
        def test_target(self) -> Any:
            if trained != {17, 19}:
                raise AssertionError("checkpoint final label opened before run freeze")
            reads.append("target")
            return self.actual.test_target

    def sealed_source(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_generate(*args, **kwargs))

    def train_phase(*args: Any, **kwargs: Any) -> Any:
        completed = original_train(*args, **kwargs)
        if kwargs["phase"] == "b":
            context = args[0]
            trained.add(context.seeds[context.seed_index])
        return completed

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", sealed_source)
    monkeypatch.setattr(continual, "_train_checkpoint_phase", train_phase)
    return trained, reads


class GlobalInterruption(Exception):
    """Stop immediately after a durable format-5 transaction."""


class StopAtGlobalCheckpoint:
    def __init__(
        self,
        path: Path,
        *,
        seed_index: int,
        phase: str,
        epoch: int,
        stage: str = "after_sleep",
        next_model_index: int = 0,
    ) -> None:
        self.store = TrustedLocalContinualCheckpointStore(path)
        self.seed_index = seed_index
        self.phase = phase
        self.epoch = epoch
        self.stage = stage
        self.next_model_index = next_model_index

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if (
            checkpoint.seed_index == self.seed_index
            and checkpoint.phase == self.phase
            and checkpoint.phase_epoch_completed == self.epoch
            and checkpoint.combined.position.stage == self.stage
            and checkpoint.combined.position.next_batch_index == self.next_model_index
        ):
            raise GlobalInterruption()


def _assert_same_trained_snapshot(actual: Any, expected: Any) -> None:
    assert actual.format_version == expected.format_version
    assert actual.input_dim == expected.input_dim
    assert actual.initial_hidden_dims == expected.initial_hidden_dims
    assert actual.min_hidden_dim == expected.min_hidden_dim
    assert actual.max_hidden_dim == expected.max_hidden_dim
    assert actual.config == expected.config
    assert actual.state.keys() == expected.state.keys()
    for name in actual.state:
        if name == "_replay_memory":
            left_memory = actual.state[name]
            right_memory = expected.state[name]
            assert left_memory.maxlen == right_memory.maxlen
            assert len(left_memory) == len(right_memory)
            for left, right in zip(left_memory, right_memory, strict=True):
                for role in ("input_batch", "target_batch"):
                    left_array = getattr(left, role)
                    right_array = getattr(right, role)
                    assert left_array.dtype == right_array.dtype
                    assert left_array.shape == right_array.shape
                    np.testing.assert_array_equal(left_array, right_array)
                assert left.priority == right.priority
                assert left.positive_fraction == right.positive_fraction
            continue
        # Why this: whole-dict pickle bytes encode incidental object aliases;
        # each saved field must still match exactly after an interrupted run.
        assert pickle.dumps(actual.state[name], protocol=5) == pickle.dumps(
            expected.state[name], protocol=5
        ), name


def _assert_same_unscored_training(actual: Any, expected: Any) -> None:
    assert actual.seed == expected.seed
    assert actual.data_digest == expected.data_digest
    assert actual.split_hashes == expected.split_hashes
    for field in fields(actual.state):
        left = getattr(actual.state, field.name)
        right = getattr(expected.state, field.name)
        if field.name == "circadian_after_a":
            _assert_same_trained_snapshot(left, right)
        else:
            assert pickle.dumps(left, protocol=5) == pickle.dumps(right, protocol=5), field.name
    _assert_same_trained_snapshot(actual.circadian_final, expected.circadian_final)


def test_v4_two_seed_final_test_opens_before_last_seed_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, reads = _seal_source_until_all_seeds_train(monkeypatch)

    # Why this: retain the measured v4 gap until a separate run-level route
    # proves that no earlier seed can expose labels to later decisions.
    with pytest.raises(AssertionError, match="before run freeze"):
        continual.run_continual_shift_benchmark(_bounded_config(False), [17, 19])
    assert trained == {17}
    assert not reads


@pytest.mark.parametrize("reverse_order", [False, True])
def test_global_seal_trains_all_seeds_before_any_final_test_source_read(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(reverse_order)
    trained, reads = _seal_source_until_all_seeds_train(monkeypatch)

    result = continual.run_continual_shift_benchmark(config, [17, 19])

    assert trained == {17, 19}
    assert "input" in reads and "target" in reads
    assert [item.seed for item in result.seed_results] == [17, 19]
    assert all(
        {"phase_a_test", "phase_b_test"}.issubset(item.split_hashes) for item in result.seed_results
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_global_seal_preserves_each_seed_report_from_bounded_route(reverse_order: bool) -> None:
    old = continual.run_continual_shift_benchmark(_bounded_config(reverse_order), [17, 19])
    sealed = continual.run_continual_shift_benchmark(_global_config(reverse_order), [17, 19])

    assert sealed.seed_results == old.seed_results
    assert sealed.aggregate == old.aggregate


@pytest.mark.parametrize("reverse_order", [False, True])
def test_global_seal_test_label_change_cannot_change_any_trained_seed(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(reverse_order)
    original_generate = continual.generate_two_cluster_dataset_with_transform
    original_train_a = continual._train_phase_a_models
    original_train_b = continual._train_phase_b_models
    perturb_first_test = False
    current_seed = -1
    trained_hashes: list[tuple[int, str]] = []

    def generated(*args: Any, **kwargs: Any) -> Any:
        source = original_generate(*args, **kwargs)
        if perturb_first_test and kwargs["seed"] in {17, 118}:
            return replace(source, test_target=1.0 - source.test_target)
        return source

    def train_a(**kwargs: Any) -> Any:
        nonlocal current_seed
        current_seed = kwargs["seed"]
        return original_train_a(**kwargs)

    def train_b(**kwargs: Any) -> Any:
        state = original_train_b(**kwargs)
        trained_hashes.append((current_seed, sha256(pickle.dumps(state, protocol=5)).hexdigest()))
        return state

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", generated)
    monkeypatch.setattr(continual, "_train_phase_a_models", train_a)
    monkeypatch.setattr(continual, "_train_phase_b_models", train_b)

    normal = continual.run_continual_shift_benchmark(config, [17, 19])
    normal_hashes = tuple(trained_hashes)
    trained_hashes.clear()
    perturb_first_test = True
    changed = continual.run_continual_shift_benchmark(config, [17, 19])

    assert tuple(trained_hashes) == normal_hashes
    assert [seed for seed, _ in normal_hashes] == [17, 19]
    assert (
        normal.seed_results[0].split_hashes["phase_a_test"]
        != changed.seed_results[0].split_hashes["phase_a_test"]
    )
    assert normal.seed_results[1] == changed.seed_results[1]


@pytest.mark.parametrize("reverse_order", [False, True])
def test_global_seal_checkpoint_holds_all_tests_until_last_seed_trains(
    reverse_order: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(reverse_order)
    expected = continual.run_continual_shift_benchmark(config, [17, 19])
    trained, reads = _seal_checkpoint_source_until_all_seeds_train(monkeypatch)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "global.ckpt")
    actual = continual.run_continual_shift_benchmark(config, [17, 19], checkpoint_store=store)

    assert actual == expected
    assert trained == {17, 19}
    assert "input" in reads and "target" in reads
    checkpoint = store.load()
    assert checkpoint.format_version == 5
    assert checkpoint.completed_results == []
    assert checkpoint.completed_test_digests == ()
    assert len(checkpoint.unscored_seeds) == 2


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    ("seed_index", "phase", "epoch", "stage", "next_model_index", "remaining_updates"),
    [
        (0, "a", 0, "wake", 1, None),
        (0, "a", 1, "after_sleep", 0, 7),
        (0, "b", 1, "after_sleep", 0, 5),
        (0, "seed_complete", 2, "after_sleep", 0, 4),
        (1, "a", 1, "after_sleep", 0, 3),
        (1, "b", 1, "before_sleep", 0, 1),
        (1, "b", 1, "after_sleep", 0, 1),
        (1, "seed_complete", 2, "after_sleep", 0, 0),
    ],
)
def test_global_checkpoint_resume_keeps_prior_final_tests_sealed(
    reverse_order: bool,
    seed_index: int,
    phase: str,
    epoch: int,
    stage: str,
    next_model_index: int,
    remaining_updates: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _global_config(reverse_order)
    expected = continual.run_continual_shift_benchmark(config, [17, 19])
    control_store = TrustedLocalContinualCheckpointStore(tmp_path / "control.ckpt")
    control = continual.run_continual_shift_benchmark(
        config, [17, 19], checkpoint_store=control_store
    )
    assert control == expected
    trained, reads = _seal_checkpoint_source_until_all_seeds_train(monkeypatch)
    interrupted = StopAtGlobalCheckpoint(
        tmp_path / "resumed.ckpt",
        seed_index=seed_index,
        phase=phase,
        epoch=epoch,
        stage=stage,
        next_model_index=next_model_index,
    )

    with pytest.raises(GlobalInterruption):
        continual.run_continual_shift_benchmark(config, [17, 19], checkpoint_store=interrupted)
    saved = interrupted.store.load()
    assert saved.format_version == 5
    assert saved.completed_results == []
    assert saved.completed_test_digests == ()
    assert all("test" not in name for name, _ in saved.split_hashes)
    assert all(
        "test" not in name for record in saved.unscored_seeds for name, _ in record.split_hashes
    )
    assert not reads

    resumed_updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal resumed_updates
        resumed_updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    actual = continual.run_continual_shift_benchmark(
        config, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
    )
    assert actual == expected
    assert resumed_updates == (
        (7 if not reverse_order else 8) if remaining_updates is None else remaining_updates
    )
    assert trained == {17, 19}
    assert "input" in reads and "target" in reads
    final_saved = interrupted.store.load()
    control_saved = control_store.load()
    assert len(final_saved.unscored_seeds) == len(control_saved.unscored_seeds) == 2
    for left, right in zip(final_saved.unscored_seeds, control_saved.unscored_seeds, strict=True):
        _assert_same_unscored_training(left, right)


@pytest.mark.parametrize("tamper", ["development_digest", "trained_steps"])
def test_global_resume_rejects_tampered_unscored_seed_before_updates(
    tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(False)
    interrupted = StopAtGlobalCheckpoint(
        tmp_path / "tampered.ckpt", seed_index=0, phase="seed_complete", epoch=2
    )
    with pytest.raises(GlobalInterruption):
        continual.run_continual_shift_benchmark(config, [17, 19], checkpoint_store=interrupted)
    saved = deepcopy(interrupted.store.load())
    assert len(saved.unscored_seeds) == 1
    if tamper == "development_digest":
        damaged = replace(saved.unscored_seeds[0], data_digest="0" * 64)
        saved = replace(saved, unscored_seeds=(damaged,))
    else:
        saved.unscored_seeds[0].state.backprop_model._traffic_steps += 1
    interrupted.store.save(saved)

    updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match="incompatible"):
        continual.run_continual_shift_benchmark(
            config, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
        )
    assert updates == 0


@pytest.mark.parametrize("change", ["settings", "seed_list"])
def test_global_resume_rejects_changed_run_identity_before_source_access(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(False)
    interrupted = StopAtGlobalCheckpoint(
        tmp_path / "identity.ckpt", seed_index=0, phase="seed_complete", epoch=2
    )
    with pytest.raises(GlobalInterruption):
        continual.run_continual_shift_benchmark(config, [17, 19], checkpoint_store=interrupted)

    def forbidden_source(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("changed run identity opened a data source")

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", forbidden_source)
    resumed_config = replace(config, replay_max_examples=5) if change == "settings" else config
    resumed_seeds = [17, 23] if change == "seed_list" else [17, 19]
    with pytest.raises(ValueError, match="config or seed list"):
        continual.run_continual_shift_benchmark(
            resumed_config,
            resumed_seeds,
            checkpoint_store=interrupted.store,
            resume_from_checkpoint=True,
        )


def test_global_resume_rejects_changed_prior_development_role_before_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(False)
    interrupted = StopAtGlobalCheckpoint(
        tmp_path / "changed-source.ckpt", seed_index=1, phase="a", epoch=1
    )
    with pytest.raises(GlobalInterruption):
        continual.run_continual_shift_benchmark(config, [17, 19], checkpoint_store=interrupted)
    original_generate = continual.generate_two_cluster_dataset_with_transform

    def changed_source(*args: Any, **kwargs: Any) -> Any:
        source = original_generate(*args, **kwargs)
        if kwargs["seed"] != 17:
            return source
        changed = source.train_target.copy()
        changed[0, 0] = 1.0 - changed[0, 0]
        return replace(source, train_target=changed)

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", changed_source)
    updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match="unscored development data"):
        continual.run_continual_shift_benchmark(
            config, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
        )
    assert updates == 0


def test_global_resume_rejects_future_seed_replay_in_prior_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _global_config(False)
    interrupted = StopAtGlobalCheckpoint(
        tmp_path / "future-replay.ckpt", seed_index=0, phase="seed_complete", epoch=2
    )
    with pytest.raises(GlobalInterruption):
        continual.run_continual_shift_benchmark(config, [17, 19], checkpoint_store=interrupted)
    saved = deepcopy(interrupted.store.load())
    future = continual._build_checkpoint_phase_a_data(config, 19).phase_a_train
    memory = saved.unscored_seeds[0].circadian_final.state["_replay_memory"]
    memory.clear()
    memory.append(
        ReplaySnapshot(
            input_batch=future.input[:1].copy(),
            target_batch=future.target[:1].copy(),
            priority=1.0,
            positive_fraction=float(future.target[0, 0]),
        )
    )
    interrupted.store.save(saved)
    updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match="replay observed training examples"):
        continual.run_continual_shift_benchmark(
            config, [17, 19], checkpoint_store=interrupted.store, resume_from_checkpoint=True
        )
    assert updates == 0
