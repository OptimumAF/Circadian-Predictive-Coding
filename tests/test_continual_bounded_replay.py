"""Observed-example replay budgets for the opt-in continual route."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_shift_benchmark as continual
from src.app.continual_shift_benchmark import (
    CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
    ContinualBoundedReplayConfig,
    ContinualBoundedReplaySeedResult,
    ContinualShiftConfig,
    format_continual_shift_benchmark,
)
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    ReplaySnapshot,
    replay_sample_id,
)
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore
from src.infra.datasets import LabeledData


class ReplayInterruption(Exception):
    """Stop at a saved transaction before later examples can arrive."""


class InterruptingStore:
    def __init__(self, path: Path, *, phase: str, epoch: int, seed_index: int = 0) -> None:
        self.store = TrustedLocalContinualCheckpointStore(path)
        self.phase = phase
        self.epoch = epoch
        self.seed_index = seed_index

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if (
            checkpoint.seed_index == self.seed_index
            and checkpoint.phase == self.phase
            and checkpoint.phase_epoch_completed == self.epoch
            and checkpoint.combined.position.stage == "after_sleep"
        ):
            raise ReplayInterruption()


def _tiny_config() -> ContinualBoundedReplayConfig:
    return ContinualBoundedReplayConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
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


def _role_ids(role: LabeledData) -> set[str]:
    return {
        replay_sample_id(role.input[index : index + 1], role.target[index : index + 1])
        for index in range(role.input.shape[0])
    }


@pytest.mark.parametrize(
    ("max_examples", "max_bytes", "expected_examples"),
    [(4, 240, 4), (10, 48, 2), (4, 96, 4)],
)
def test_bounded_core_replay_retains_unique_observed_examples_within_both_caps(
    max_examples: int, max_bytes: int, expected_examples: int
) -> None:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=19,
        circadian_config=CircadianConfig(replay_steps=1, replay_memory_size=1),
    )
    model.configure_replay_retention(
        ReplayRetentionBudget(max_examples=max_examples, max_bytes=max_bytes)
    )
    inputs = np.arange(20, dtype=np.float64).reshape(10, 2) / 10.0
    targets = (np.arange(10) % 2).astype(np.float64).reshape(-1, 1)
    for _ in range(2):
        model.train_epoch(inputs, targets, 0.02, 2, 0.1)

    retained = model.get_replay_retention()
    allowed = _role_ids(LabeledData(inputs, targets))
    assert retained.example_count == expected_examples
    assert retained.retained_bytes == expected_examples * 24
    assert set(retained.sample_ids).issubset(allowed)
    assert len(set(retained.sample_ids)) == expected_examples
    assert len(model._replay_memory) == expected_examples
    assert all(snapshot.input_batch.shape == (1, 2) for snapshot in model._replay_memory)


def test_legacy_core_replay_keeps_batch_snapshot_limit() -> None:
    inputs = np.arange(20, dtype=np.float64).reshape(10, 2) / 10.0
    targets = (np.arange(10) % 2).astype(np.float64).reshape(-1, 1)
    legacy = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=19,
        circadian_config=CircadianConfig(replay_steps=1, replay_memory_size=1),
    )
    legacy.train_epoch(inputs, targets, 0.02, 2, 0.1)
    assert len(legacy._replay_memory) == 1
    assert legacy._replay_memory[0].input_batch.shape == (10, 2)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_bounded_continual_replay_selects_only_arrived_training_examples(
    reverse_order: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    if reverse_order:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    allowed: set[str] = set()
    selected: list[tuple[str, ...]] = []
    original_a = continual._train_phase_a_models
    original_b = continual._train_phase_b_models
    original_select = CircadianPredictiveCodingNetwork._select_replay_snapshots

    def train_a(**kwargs: Any) -> Any:
        allowed.update(_role_ids(kwargs["phase_a_train"]))
        return original_a(**kwargs)

    def train_b(**kwargs: Any) -> Any:
        allowed.update(_role_ids(kwargs["phase_b_train"]))
        return original_b(**kwargs)

    def select(model: CircadianPredictiveCodingNetwork, replay_count: int) -> Any:
        chosen = original_select(model, replay_count)
        ids = tuple(replay_sample_id(item.input_batch, item.target_batch) for item in chosen)
        assert set(ids).issubset(allowed)
        selected.append(ids)
        return chosen

    monkeypatch.setattr(continual, "_train_phase_a_models", train_a)
    monkeypatch.setattr(continual, "_train_phase_b_models", train_b)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_select_replay_snapshots", select)
    result = continual.run_continual_shift_benchmark(config, [17])

    assert selected
    report = result.seed_results[0]
    assert isinstance(report, ContinualBoundedReplaySeedResult)
    assert report.replay_retention.budget_examples == 4
    assert report.replay_retention.budget_bytes == 96
    for snapshot in (report.replay_retention.phase_a, report.replay_retention.phase_b):
        assert snapshot.example_count == len(snapshot.sample_ids) <= 4
        assert snapshot.retained_bytes <= 96
        assert set(snapshot.sample_ids).issubset(allowed)
    assert "Replay retained" in format_continual_shift_benchmark(result)


@pytest.mark.parametrize("phase,epoch", [("a", 1), ("b", 1)])
def test_bounded_replay_checkpoint_resume_matches_uninterrupted_report(
    phase: str, epoch: int, tmp_path: Path
) -> None:
    config = _tiny_config()
    ordinary = continual.run_continual_shift_benchmark(config, [17])
    expected_store = TrustedLocalContinualCheckpointStore(tmp_path / "expected.ckpt")
    expected = continual.run_continual_shift_benchmark(
        config, [17], checkpoint_store=expected_store
    )
    interrupted = InterruptingStore(tmp_path / "resumed.ckpt", phase=phase, epoch=epoch)
    with pytest.raises(ReplayInterruption):
        continual.run_continual_shift_benchmark(config, [17], checkpoint_store=interrupted)
    checkpoint = interrupted.store.load()
    assert checkpoint.format_version == 4
    assert checkpoint.phase == phase
    if phase == "a":
        assert {role for role, _ in checkpoint.split_hashes} == {
            "phase_a_train",
            "phase_a_validation",
        }

    actual = continual.run_continual_shift_benchmark(
        config, [17], checkpoint_store=interrupted.store, resume_from_checkpoint=True
    )
    assert actual == expected == ordinary
    result = actual.seed_results[0]
    assert isinstance(result, ContinualBoundedReplaySeedResult)
    assert result.replay_retention.phase_b.example_count <= 4


def test_phase_a_resume_rejects_future_phase_replay_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _tiny_config()
    interrupted = InterruptingStore(tmp_path / "tampered.ckpt", phase="a", epoch=1)
    with pytest.raises(ReplayInterruption):
        continual.run_continual_shift_benchmark(config, [17], checkpoint_store=interrupted)
    checkpoint = deepcopy(interrupted.store.load())
    future = continual._build_phase_b_roles(config, 17 + 101).train
    snapshot = checkpoint.combined.model_state
    assert isinstance(snapshot, CircadianNetworkSnapshot)
    replay = snapshot.state["_replay_memory"]
    replay.clear()
    replay.append(
        ReplaySnapshot(
            input_batch=future.input[:1].copy(),
            target_batch=future.target[:1].copy(),
            priority=1.0,
            positive_fraction=float(future.target[0, 0]),
        )
    )
    interrupted.store.save(checkpoint)
    updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match="replay.*observed"):
        continual.run_continual_shift_benchmark(
            config, [17], checkpoint_store=interrupted.store, resume_from_checkpoint=True
        )
    assert updates == 0


@pytest.mark.parametrize("tamper_frozen_a", [False, True])
def test_phase_b_resume_rejects_unobserved_active_or_frozen_replay(
    tamper_frozen_a: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _tiny_config()
    interrupted = InterruptingStore(tmp_path / "tampered-b.ckpt", phase="b", epoch=1)
    with pytest.raises(ReplayInterruption):
        continual.run_continual_shift_benchmark(config, [17], checkpoint_store=interrupted)
    checkpoint = deepcopy(interrupted.store.load())
    phase_b_roles = continual._build_phase_b_roles(config, 17 + 101)
    snapshot: CircadianNetworkSnapshot | dict[str, Any] | None
    if tamper_frozen_a:
        snapshot = checkpoint.state.circadian_after_a
        sample = phase_b_roles.train
    else:
        snapshot = checkpoint.combined.model_state
        sample = phase_b_roles.validation
    assert isinstance(snapshot, CircadianNetworkSnapshot)
    replay = snapshot.state["_replay_memory"]
    replay.clear()
    replay.append(
        ReplaySnapshot(
            input_batch=sample.input[:1].copy(),
            target_batch=sample.target[:1].copy(),
            priority=1.0,
            positive_fraction=float(sample.target[0, 0]),
        )
    )
    interrupted.store.save(checkpoint)
    updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match="replay.*observed"):
        continual.run_continual_shift_benchmark(
            config, [17], checkpoint_store=interrupted.store, resume_from_checkpoint=True
        )
    assert updates == 0


def test_bounded_replay_requires_explicit_positive_limits() -> None:
    config = _tiny_config()
    for invalid in (replace(config, replay_max_examples=0), replace(config, replay_max_bytes=0)):
        with pytest.raises(ValueError, match="replay.*budget"):
            continual.run_continual_shift_benchmark(invalid, [17])
    with pytest.raises(ValueError, match="components sleep mode"):
        continual.run_continual_shift_benchmark(
            replace(config, circadian_config=replace(config.circadian_config, sleep_mode="legacy")),
            [17],
        )
    with pytest.raises(ValueError, match="bounded replay config"):
        continual.run_continual_shift_benchmark(
            ContinualShiftConfig(protocol_id=CONTINUAL_BOUNDED_REPLAY_PROTOCOL), [17]
        )


def test_bounded_replay_resume_validates_prior_seed_retention_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _tiny_config()
    seeds = [17, 19]
    expected_store = TrustedLocalContinualCheckpointStore(tmp_path / "two-expected.ckpt")
    expected = continual.run_continual_shift_benchmark(
        config, seeds, checkpoint_store=expected_store
    )
    interrupted = InterruptingStore(tmp_path / "two-resumed.ckpt", phase="a", epoch=1, seed_index=1)
    with pytest.raises(ReplayInterruption):
        continual.run_continual_shift_benchmark(config, seeds, checkpoint_store=interrupted)
    checkpoint = deepcopy(interrupted.store.load())
    assert len(checkpoint.completed_results) == 1
    first = checkpoint.completed_results[0]
    assert isinstance(first, ContinualBoundedReplaySeedResult)
    future_validation = continual._build_phase_b_roles(config, 17 + 101).validation
    forbidden_id = replay_sample_id(future_validation.input[:1], future_validation.target[:1])
    retained = first.replay_retention.phase_a
    checkpoint.completed_results[0] = replace(
        first,
        replay_retention=replace(
            first.replay_retention,
            phase_a=replace(
                retained,
                sample_ids=(forbidden_id, *retained.sample_ids[1:]),
            ),
        ),
    )
    interrupted.store.save(checkpoint)
    updates = 0
    original_train = continual.BackpropMLP.train_epoch

    def count_updates(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal updates
        updates += 1
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", count_updates)
    with pytest.raises(ValueError, match="replay observed completed seed report"):
        continual.run_continual_shift_benchmark(
            config, seeds, checkpoint_store=interrupted.store, resume_from_checkpoint=True
        )
    assert updates == 0

    good = InterruptingStore(tmp_path / "two-good.ckpt", phase="a", epoch=1, seed_index=1)
    with pytest.raises(ReplayInterruption):
        continual.run_continual_shift_benchmark(config, seeds, checkpoint_store=good)
    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", original_train)
    actual = continual.run_continual_shift_benchmark(
        config, seeds, checkpoint_store=good.store, resume_from_checkpoint=True
    )
    assert actual == expected
