"""Final-test labels are unavailable to bounded continual training decisions."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import pickle
from typing import Any, cast

import pytest

from src.app import continual_shift_benchmark as continual
from src.app.continual_shift_benchmark import ContinualBoundedReplayConfig
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    replay_sample_id,
)
from src.infra import datasets as dataset_roles
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore
from src.infra.datasets import LabeledData


def _config(reverse_order: bool) -> ContinualBoundedReplayConfig:
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


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_bounded_route_opens_final_test_roles_only_after_both_phases_train(
    checkpointed: bool,
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    training_complete = False
    final_reads = 0
    hash_test_flags: list[bool] = []
    original_split = continual.split_training_validation
    original_hash = dataset_roles._split_hash

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("final-test inputs opened before Phase B finished")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("final-test labels opened before Phase B finished")
            final_reads += 1
            return self.actual.target

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        hash_test_flags.append(kwargs.get("hash_test", True))
        roles = original_split(*args, **kwargs)
        return replace(roles, test=cast(LabeledData, SealedFinalTest(roles.test)))

    def sealed_hash(role: str, samples: Any) -> str:
        if role == "test" and not training_complete:
            raise AssertionError("final-test role hashed before Phase B finished")
        return original_hash(role, samples)

    monkeypatch.setattr(continual, "split_training_validation", sealed_split)
    monkeypatch.setattr(dataset_roles, "_split_hash", sealed_hash)
    config = _config(reverse_order)
    if checkpointed:
        original_train = continual._train_checkpoint_phase

        def finished_checkpoint_phase(*args: Any, **kwargs: Any) -> Any:
            nonlocal training_complete
            trained = original_train(*args, **kwargs)
            if kwargs["phase"] == "b":
                training_complete = True
            return trained

        monkeypatch.setattr(continual, "_train_checkpoint_phase", finished_checkpoint_phase)
        store = TrustedLocalContinualCheckpointStore(tmp_path / "sealed.ckpt")
        result = continual.run_continual_shift_benchmark(config, [17], checkpoint_store=store)
    else:
        original_train_b = continual._train_phase_b_models

        def finished_ordinary_phase(**kwargs: Any) -> Any:
            nonlocal training_complete
            trained = original_train_b(**kwargs)
            training_complete = True
            return trained

        monkeypatch.setattr(continual, "_train_phase_b_models", finished_ordinary_phase)
        result = continual.run_continual_shift_benchmark(config, [17])

    assert training_complete
    assert hash_test_flags == [False, False]
    assert final_reads > 0
    assert {"phase_a_test", "phase_b_test"}.issubset(result.seed_results[0].split_hashes)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_bounded_route_releases_source_test_fields_only_after_training(
    checkpointed: bool,
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    training_complete = False
    source_input_reads = 0
    source_label_reads = 0
    original_generate = continual.generate_two_cluster_dataset_with_transform

    class SealedSource:
        def __init__(self, actual: Any) -> None:
            self.actual = actual
            self.train_input = actual.train_input
            self.train_target = actual.train_target

        @property
        def test_input(self) -> Any:
            nonlocal source_input_reads
            if not training_complete:
                raise AssertionError("source test inputs released before Phase B finished")
            source_input_reads += 1
            return self.actual.test_input

        @property
        def test_target(self) -> Any:
            nonlocal source_label_reads
            if not training_complete:
                raise AssertionError("source test labels released before Phase B finished")
            source_label_reads += 1
            return self.actual.test_target

    def sealed_source(*args: Any, **kwargs: Any) -> Any:
        return SealedSource(original_generate(*args, **kwargs))

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", sealed_source)
    config = _config(reverse_order)
    if checkpointed:
        original_train = continual._train_checkpoint_phase

        def finished_checkpoint_phase(*args: Any, **kwargs: Any) -> Any:
            nonlocal training_complete
            trained = original_train(*args, **kwargs)
            if kwargs["phase"] == "b":
                training_complete = True
            return trained

        monkeypatch.setattr(continual, "_train_checkpoint_phase", finished_checkpoint_phase)
        store = TrustedLocalContinualCheckpointStore(tmp_path / "source-sealed.ckpt")
        result = continual.run_continual_shift_benchmark(config, [17], checkpoint_store=store)
    else:
        original_train_b = continual._train_phase_b_models

        def finished_ordinary_phase(**kwargs: Any) -> Any:
            nonlocal training_complete
            trained = original_train_b(**kwargs)
            training_complete = True
            return trained

        monkeypatch.setattr(continual, "_train_phase_b_models", finished_ordinary_phase)
        result = continual.run_continual_shift_benchmark(config, [17])

    assert training_complete
    assert source_input_reads >= 2
    assert source_label_reads >= 2
    assert {"phase_a_test", "phase_b_test"}.issubset(result.seed_results[0].split_hashes)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_final_test_label_perturbation_changes_only_reporting(
    checkpointed: bool,
    reverse_order: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(reverse_order)
    original_generate = continual.generate_two_cluster_dataset_with_transform
    original_score = continual._score_seed_models
    original_sleep = CircadianPredictiveCodingNetwork.sleep_event
    original_select = CircadianPredictiveCodingNetwork._select_replay_snapshots
    perturb_labels = False
    trained_states: list[tuple[str, ...]] = []
    sleep_decisions: list[tuple[Any, ...]] = []
    replay_choices: list[tuple[str, ...]] = []

    def generated(*args: Any, **kwargs: Any) -> Any:
        source = original_generate(*args, **kwargs)
        return replace(source, test_target=1.0 - source.test_target) if perturb_labels else source

    def score(*args: Any, **kwargs: Any) -> Any:
        state = args[2]
        trained_states.append(
            tuple(
                sha256(pickle.dumps(value, protocol=5)).hexdigest()
                for value in (
                    state.backprop_model,
                    state.predictive_model,
                    state.circadian_model.snapshot_state(),
                    state.backprop_after_a,
                    state.predictive_after_a,
                    state.circadian_after_a.snapshot_state(),
                )
            )
        )
        return original_score(*args, **kwargs)

    def sleep(model: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> Any:
        result = original_sleep(model, *args, **kwargs)
        progress = kwargs["epoch_progress"]
        sleep_decisions.append(
            (
                progress.completed_epochs,
                progress.total_epochs,
                result.split_indices,
                result.pruned_indices,
            )
        )
        return result

    def select(model: CircadianPredictiveCodingNetwork, replay_count: int) -> Any:
        chosen = original_select(model, replay_count)
        replay_choices.append(
            tuple(replay_sample_id(item.input_batch, item.target_batch) for item in chosen)
        )
        return chosen

    monkeypatch.setattr(continual, "generate_two_cluster_dataset_with_transform", generated)
    monkeypatch.setattr(continual, "_score_seed_models", score)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", sleep)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_select_replay_snapshots", select)

    def run_once(name: str) -> tuple[Any, tuple[str, ...], tuple[Any, ...], tuple[Any, ...]]:
        trained_states.clear()
        sleep_decisions.clear()
        replay_choices.clear()
        store = TrustedLocalContinualCheckpointStore(tmp_path / f"{name}.ckpt")
        result = continual.run_continual_shift_benchmark(
            config, [17], checkpoint_store=store if checkpointed else None
        )
        return result, trained_states[0], tuple(sleep_decisions), tuple(replay_choices)

    original = run_once("original")
    perturb_labels = True
    changed = run_once("perturbed")

    assert original[1:] == changed[1:]
    assert original[3]
    left, right = original[0].seed_results[0], changed[0].seed_results[0]
    for role in ("phase_a_train", "phase_a_validation", "phase_b_train", "phase_b_validation"):
        assert left.split_hashes[role] == right.split_hashes[role]
    for role in ("phase_a_test", "phase_b_test"):
        assert left.split_hashes[role] != right.split_hashes[role]
    assert left.backprop.phase_a_pre_accuracy != right.backprop.phase_a_pre_accuracy
