"""A format-9 completed-policy checkpoint keeps all final roles sealed."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.run_continual_replay_policy_smoke import _fixed_manifest
from src.app import continual_arrived_benchmark as arrived
from src.app import continual_replay_policy_comparison as comparison
from src.app import continual_shift_benchmark as base
from src.app import continual_arrived_transactions as transactions
from src.app.continual_arrived_checkpoint import arrived_event_digest, arrived_sleep_history_digest
from src.app.continual_replay_policy_checkpoint import replay_exposure_digest
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.circadian_predictive_coding import ReplaySnapshot
from src.infra.circadian_checkpoint_files import (
    TrustedLocalArrivedCheckpointStore,
    TrustedLocalReplayPolicyCheckpointStore,
)


class InterruptedAfterTrial(Exception):
    """Stop immediately after a durable, unscored policy/seed."""


class StopAfterTrial:
    def __init__(self, path: Path, trial_count: int = 1) -> None:
        self.store = TrustedLocalReplayPolicyCheckpointStore(path)
        self.trial_count = trial_count

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if checkpoint.next_trial_index == self.trial_count:
            raise InterruptedAfterTrial()


class InterruptedAtActiveCursor(Exception):
    """Stop after a durable wake or sleep transaction within one trial."""


class StopAtActiveCursor:
    def __init__(self, path: Path, target: tuple[int, int, str, int, str, int]) -> None:
        self.store = TrustedLocalReplayPolicyCheckpointStore(path)
        self.target = target

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        active = checkpoint.active
        if active is None:
            return
        observed = (
            active.policy_index,
            active.seed_index,
            active.phase,
            active.phase_epoch_completed,
            active.stage,
            active.next_model_index,
        )
        if observed == self.target:
            raise InterruptedAtActiveCursor()


def _save_first_seed(path: Path) -> TrustedLocalReplayPolicyCheckpointStore:
    with pytest.raises(InterruptedAfterTrial):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=StopAfterTrial(path)
        )
    return TrustedLocalReplayPolicyCheckpointStore(path)


def _save_active_cursor(
    path: Path, target: tuple[int, int, str, int, str, int]
) -> TrustedLocalReplayPolicyCheckpointStore:
    with pytest.raises(InterruptedAtActiveCursor):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=StopAtActiveCursor(path, target)
        )
    return TrustedLocalReplayPolicyCheckpointStore(path)


def _comparable_state(value: Any) -> Any:
    if isinstance(value, np.random.Generator):
        return _comparable_state(value.bit_generator.state)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, _comparable_state(value.tolist()))
    if is_dataclass(value):
        return (
            type(value).__name__,
            tuple(
                (field.name, _comparable_state(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, dict):
        return {key: _comparable_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return tuple(_comparable_state(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_comparable_state(item) for item in value))
    if hasattr(value, "__dict__"):
        return (type(value).__name__, _comparable_state(vars(value)))
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(f"unsupported state value {type(value)!r}")


def test_should_resume_completed_policy_seed_without_retraining_or_early_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixed_manifest()
    ordinary = comparison.run_replay_policy_comparison(manifest)
    store = _save_first_seed(tmp_path / "policy.ckpt")
    saved = store.load()
    assert saved.format_version == 9
    with pytest.raises(ValueError, match="header"):
        TrustedLocalArrivedCheckpointStore(store.path).load()
    assert saved.next_trial_index == 1
    assert len(saved.unscored_seeds) == 1
    assert all(
        "final_test" not in role
        for record in saved.unscored_seeds
        for role, _ in record.arrived.role_ids
    )

    trained: list[tuple[str, int]] = []
    original_train = transactions.train_or_resume_arrived_seed

    def train(*args: Any, **kwargs: Any) -> Any:
        pending = original_train(*args, **kwargs)
        trained.append((kwargs["retention_policy"].name, pending.seed))
        return pending

    monkeypatch.setattr(transactions, "train_or_resume_arrived_seed", train)
    resumed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed == ordinary
    assert trained == [
        ("recent_fifo", 19),
        ("seeded_reservoir", 17),
        ("seeded_reservoir", 19),
    ]
    assert store.load().next_trial_index == 4


def test_should_resume_after_policy_boundary_without_retraining_reservoir_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixed_manifest()
    ordinary = comparison.run_replay_policy_comparison(manifest)
    path = tmp_path / "policy.ckpt"
    with pytest.raises(InterruptedAfterTrial):
        comparison.run_replay_policy_comparison(
            manifest, checkpoint_store=StopAfterTrial(path, trial_count=3)
        )
    store = TrustedLocalReplayPolicyCheckpointStore(path)
    assert store.load().next_trial_index == 3
    trained: list[tuple[str, int]] = []
    original_train = transactions.train_or_resume_arrived_seed

    def train(*args: Any, **kwargs: Any) -> Any:
        item = original_train(*args, **kwargs)
        trained.append((kwargs["retention_policy"].name, item.seed))
        return item

    monkeypatch.setattr(transactions, "train_or_resume_arrived_seed", train)
    resumed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed == ordinary
    assert trained == [("seeded_reservoir", 19)]


@pytest.mark.parametrize("change", ["policy_seed", "replay_cap", "seed_order"])
def test_should_reject_changed_policy_manifest_before_source_access(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_first_seed(tmp_path / "policy.ckpt")
    manifest = _fixed_manifest()
    if change == "policy_seed":
        changed = replace(
            manifest,
            policies=(
                ReplayRetentionPolicy("recent_fifo"),
                ReplayRetentionPolicy("seeded_reservoir", 54),
            ),
        )
    elif change == "replay_cap":
        changed_training = replace(manifest.arrived.training, replay_max_examples=3)
        changed = replace(manifest, arrived=replace(manifest.arrived, training=changed_training))
    else:
        changed = replace(manifest, seeds=tuple(reversed(manifest.seeds)))
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="manifest"):
        comparison.run_replay_policy_comparison(
            changed, checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_forged_exposure_before_new_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_first_seed(tmp_path / "policy.ckpt")
    saved = deepcopy(store.load())
    saved.unscored_seeds[0].arrived.state.circadian_model._replay_exposed_ids.add("f" * 64)
    store.save(saved)
    monkeypatch.setattr(
        transactions,
        "train_or_resume_arrived_seed",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("trained before preflight")),
    )
    with pytest.raises(ValueError, match="replay exposure"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_recompute_duplicate_counts_even_when_digest_is_reforged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_first_seed(tmp_path / "policy.ckpt")
    saved = deepcopy(store.load())
    record = saved.unscored_seeds[0]
    record.arrived.state.circadian_model._replay_duplicate_occurrences += 1
    state = record.arrived.state
    forged_digest = replay_exposure_digest(
        state.circadian_after_a.get_replay_exposure(),
        state.circadian_model.get_replay_exposure(),
    )
    store.save(
        replace(
            saved,
            unscored_seeds=(replace(record, exposure_digest=forged_digest),),
        )
    )
    monkeypatch.setattr(
        transactions,
        "train_or_resume_arrived_seed",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("trained before preflight")),
    )
    with pytest.raises(ValueError, match="replay exposure"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_nonprefix_cursor_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_first_seed(tmp_path / "policy.ckpt")
    store.save(replace(store.load(), next_trial_index=2))
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="trial cursor"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_changed_arrived_training_role_before_new_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_first_seed(tmp_path / "policy.ckpt")
    original_source = arrived.generate_two_cluster_dataset_with_transform

    def changed_source(**kwargs: Any) -> Any:
        source = original_source(**kwargs)
        if kwargs["seed"] != 17:
            return source
        shifted = source.train_input.copy()
        shifted[0, 0] += 0.5
        return replace(source, train_input=shifted)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_source)
    monkeypatch.setattr(
        transactions,
        "train_or_resume_arrived_seed",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("trained before preflight")),
    )
    with pytest.raises(ValueError, match="development roles"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_seal_both_final_sources_until_all_policy_records_are_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TrustedLocalReplayPolicyCheckpointStore(tmp_path / "policy.ckpt")
    original_a = arrived.generate_two_cluster_dataset_with_transform
    original_b = arrived._generate_phase_b_source
    final_reads: list[str] = []

    class SealedSource:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.train_input = source.train_input
            self.train_target = source.train_target

        @property
        def test_input(self) -> Any:
            assert store.load().next_trial_index == 4
            final_reads.append("input")
            return self.source.test_input

        @property
        def test_target(self) -> Any:
            assert store.load().next_trial_index == 4
            final_reads.append("target")
            return self.source.test_target

    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **kwargs: SealedSource(original_a(**kwargs)),
    )
    monkeypatch.setattr(
        arrived,
        "_generate_phase_b_source",
        lambda *args, **kwargs: SealedSource(original_b(*args, **kwargs)),
    )
    manifest = _fixed_manifest()
    control = comparison.run_replay_policy_comparison(manifest, checkpoint_store=store)
    assert final_reads == ["input", "target"] * 8
    monkeypatch.setattr(
        transactions,
        "train_or_resume_arrived_seed",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("retrained terminal checkpoint")),
    )
    final_reads.clear()
    resumed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed == control
    assert final_reads == ["input", "target"] * 8


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    ("epoch", "stage", "model_index"),
    [(0, "wake", 1), (1, "before_sleep", 0)],
)
def test_should_resume_active_phase_a_without_repeating_wake_or_opening_b(
    reverse_order: bool,
    epoch: int,
    stage: str,
    model_index: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _fixed_manifest()
    if reverse_order:
        training = replace(
            manifest.arrived.training,
            model_order=tuple(reversed(manifest.arrived.training.model_order)),
        )
        manifest = replace(manifest, arrived=replace(manifest.arrived, training=training))
    ordinary = comparison.run_replay_policy_comparison(manifest)
    path = tmp_path / "active.ckpt"
    b_calls: list[int] = []
    original_b = arrived._generate_phase_b_source

    def source_b(*args: Any, **kwargs: Any) -> Any:
        b_calls.append(args[1] if len(args) > 1 else kwargs["seed"])
        return original_b(*args, **kwargs)

    monkeypatch.setattr(arrived, "_generate_phase_b_source", source_b)
    with monkeypatch.context() as sealed:
        sealed.setattr(
            arrived,
            "release_final_test",
            lambda *_: (_ for _ in ()).throw(AssertionError("final source opened")),
        )
        target = (0, 0, "a", epoch, stage, model_index)
        with pytest.raises(InterruptedAtActiveCursor):
            comparison.run_replay_policy_comparison(
                manifest, checkpoint_store=StopAtActiveCursor(path, target)
            )
    assert b_calls == []
    store = TrustedLocalReplayPolicyCheckpointStore(path)
    saved = store.load()
    assert saved.next_trial_index == 0
    assert saved.unscored_seeds == ()
    assert saved.active is not None and saved.active.format_version == 9

    updates: dict[str, int] = dict.fromkeys(manifest.arrived.training.model_order, 0)
    original_update = base._train_named_model_epoch

    def count_update(*args: Any, **kwargs: Any) -> Any:
        method = args[-1]
        updates[method] += 1
        return original_update(*args, **kwargs)

    monkeypatch.setattr(base, "_train_named_model_epoch", count_update)
    resumed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed == ordinary
    already_updated = set(manifest.arrived.training.model_order[:model_index])
    if stage == "before_sleep":
        already_updated = set(manifest.arrived.training.model_order)
    assert updates == {
        method: 16 - int(method in already_updated)
        for method in manifest.arrived.training.model_order
    }


@pytest.mark.parametrize(
    ("policy_index", "phase", "epoch", "stage", "model_index", "arrived_b_count"),
    [
        (0, "a", 1, "after_sleep", 0, 0),
        (0, "b", 0, "wake", 1, 1),
        (0, "b", 1, "before_sleep", 0, 1),
        (1, "a", 0, "wake", 1, 2),
        (1, "b", 1, "after_sleep", 0, 3),
    ],
)
@pytest.mark.parametrize("reverse_order", [False, True])
def test_should_resume_phase_and_policy_boundaries_with_exact_wake_work(
    reverse_order: bool,
    policy_index: int,
    phase: str,
    epoch: int,
    stage: str,
    model_index: int,
    arrived_b_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _fixed_manifest()
    if reverse_order:
        training = replace(
            manifest.arrived.training,
            model_order=tuple(reversed(manifest.arrived.training.model_order)),
        )
        manifest = replace(manifest, arrived=replace(manifest.arrived, training=training))
    ordinary = comparison.run_replay_policy_comparison(manifest)
    path = tmp_path / "active.ckpt"
    original_b = arrived._generate_phase_b_source
    b_calls: list[int] = []

    def source_b(*args: Any, **kwargs: Any) -> Any:
        b_calls.append(args[1] if len(args) > 1 else kwargs["seed"])
        return original_b(*args, **kwargs)

    monkeypatch.setattr(arrived, "_generate_phase_b_source", source_b)
    with monkeypatch.context() as sealed:
        sealed.setattr(
            arrived,
            "release_final_test",
            lambda *_: (_ for _ in ()).throw(AssertionError("final source opened")),
        )
        target = (policy_index, 0, phase, epoch, stage, model_index)
        with pytest.raises(InterruptedAtActiveCursor):
            comparison.run_replay_policy_comparison(
                manifest, checkpoint_store=StopAtActiveCursor(path, target)
            )
    assert len(b_calls) == arrived_b_count
    store = TrustedLocalReplayPolicyCheckpointStore(path)
    saved = store.load()
    assert saved.next_trial_index == policy_index * len(manifest.seeds)
    assert saved.active is not None
    assert saved.active.policy_index == policy_index
    assert saved.active.phase == phase

    updates: dict[str, int] = dict.fromkeys(manifest.arrived.training.model_order, 0)
    original_update = base._train_named_model_epoch

    def count_update(*args: Any, **kwargs: Any) -> Any:
        method = args[-1]
        updates[method] += 1
        return original_update(*args, **kwargs)

    monkeypatch.setattr(base, "_train_named_model_epoch", count_update)
    resumed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed == ordinary
    prior_trials = policy_index * len(manifest.seeds)
    earlier_phase_epochs = manifest.arrived.training.phase_a_epochs if phase == "b" else 0
    prefix = set(manifest.arrived.training.model_order[:model_index]) if stage == "wake" else set()
    assert updates == {
        method: 16 - (prior_trials * 4 + earlier_phase_epochs + epoch + int(method in prefix))
        for method in manifest.arrived.training.model_order
    }


@pytest.mark.parametrize("change", ["policy_seed", "replay_cap", "seed_order"])
def test_should_reject_active_changed_manifest_before_source_access(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "a", 0, "wake", 1))
    manifest = _fixed_manifest()
    if change == "policy_seed":
        changed = replace(
            manifest,
            policies=(
                ReplayRetentionPolicy("recent_fifo"),
                ReplayRetentionPolicy("seeded_reservoir", 54),
            ),
        )
    elif change == "replay_cap":
        training = replace(manifest.arrived.training, replay_max_bytes=72)
        changed = replace(manifest, arrived=replace(manifest.arrived, training=training))
    else:
        changed = replace(manifest, seeds=tuple(reversed(manifest.seeds)))
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(ValueError, match="manifest"):
        comparison.run_replay_policy_comparison(
            changed, checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_active_duplicate_ledger_before_another_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "a", 1, "after_sleep", 0))
    saved = deepcopy(store.load())
    assert saved.active is not None and saved.active.active_circadian is not None
    saved.active.active_circadian.state["_replay_duplicate_occurrences"] += 1
    store.save(saved)
    monkeypatch.setattr(
        base,
        "_train_named_model_epoch",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("updated before preflight")),
    )
    with pytest.raises(ValueError, match="replay exposure"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_active_saved_policy_mismatch_before_another_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "a", 1, "after_sleep", 0))
    saved = deepcopy(store.load())
    assert saved.active is not None and saved.active.active_circadian is not None
    saved.active.active_circadian.state["_replay_retention_policy"] = ReplayRetentionPolicy(
        "seeded_reservoir", 53
    )
    store.save(saved)
    monkeypatch.setattr(
        base,
        "_train_named_model_epoch",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("updated before preflight")),
    )
    with pytest.raises(ValueError, match="replay or RNG"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_active_reordered_sleep_history_before_another_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "b", 1, "after_sleep", 0))
    saved = store.load()
    active = saved.active
    assert active is not None and len(active.active_sleep_events) >= 2
    swapped = (
        active.active_sleep_events[1],
        active.active_sleep_events[0],
        *active.active_sleep_events[2:],
    )
    store.save(
        replace(
            saved,
            active=replace(
                active,
                active_sleep_events=swapped,
                active_sleep_history_digest=arrived_sleep_history_digest(
                    swapped, active.active_event_digest
                ),
            ),
        )
    )
    monkeypatch.setattr(
        base,
        "_train_named_model_epoch",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("updated before preflight")),
    )
    with pytest.raises(ValueError, match="sleep"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_future_b_replay_in_active_a_without_b_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixed_manifest()
    phase_b = arrived._build_phase_b_roles(manifest.arrived, 17)
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "a", 1, "after_sleep", 0))
    saved = deepcopy(store.load())
    assert saved.active is not None and saved.active.active_circadian is not None
    memory = saved.active.active_circadian.state["_replay_memory"]
    memory[0] = ReplaySnapshot(
        input_batch=phase_b.train.input[:1].copy(),
        target_batch=phase_b.train.target[:1].copy(),
        priority=0.0,
        positive_fraction=float(phase_b.train.target[0, 0]),
    )
    store.save(saved)
    monkeypatch.setattr(
        arrived,
        "_generate_phase_b_source",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("B arrived early")),
    )
    monkeypatch.setattr(
        base,
        "_train_named_model_epoch",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("updated before preflight")),
    )
    with pytest.raises(ValueError, match="replay"):
        comparison.run_replay_policy_comparison(
            manifest, checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_forged_phase_b_cursor_before_b_source_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "a", 0, "wake", 1))
    saved = store.load()
    assert saved.active is not None
    store.save(replace(saved, active=replace(saved.active, phase="b")))
    monkeypatch.setattr(
        arrived,
        "_generate_phase_b_source",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("B arrived early")),
    )
    with pytest.raises(ValueError, match="frozen Phase A state"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_reject_active_guard_role_even_with_recomputed_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "b", 1, "after_sleep", 0))
    saved = deepcopy(store.load())
    active = saved.active
    assert active is not None and active.active_guard_decisions
    forged_decisions = (
        replace(active.active_guard_decisions[0], role_hash="f" * 64),
        *active.active_guard_decisions[1:],
    )
    event_digest = arrived_event_digest(
        active.active_role_accesses, forged_decisions, active.active_task_information
    )
    forged = replace(
        active,
        active_guard_decisions=forged_decisions,
        active_event_digest=event_digest,
        active_sleep_history_digest=arrived_sleep_history_digest(
            active.active_sleep_events, event_digest
        ),
    )
    store.save(replace(saved, active=forged))
    monkeypatch.setattr(
        base,
        "_train_named_model_epoch",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("updated before preflight")),
    )
    with pytest.raises(ValueError, match="guard role"):
        comparison.run_replay_policy_comparison(
            _fixed_manifest(), checkpoint_store=store, resume_from_checkpoint=True
        )


def test_should_preserve_active_training_when_final_labels_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixed_manifest()
    control = comparison.run_replay_policy_comparison(manifest)
    store = _save_active_cursor(tmp_path / "active.ckpt", (0, 0, "b", 1, "before_sleep", 0))
    original_a = arrived.generate_two_cluster_dataset_with_transform

    def changed_final(**kwargs: Any) -> Any:
        source = original_a(**kwargs)
        if kwargs["seed"] != 17:
            return source
        return replace(source, test_target=1.0 - source.test_target)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", changed_final)
    changed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=store, resume_from_checkpoint=True
    )
    for before_policy, after_policy in zip(control.policies, changed.policies, strict=True):
        for before, after in zip(before_policy.seeds, after_policy.seeds, strict=True):
            assert before.exposure == after.exposure
            assert before.baseline_state_digests == after.baseline_state_digests
            assert before.arrived.guard_decisions == after.arrived.guard_decisions
            assert before.arrived.metrics.replay_retention == after.arrived.metrics.replay_retention
            for phase in ("a", "b"):
                for role in ("train", "inner_guard", "outer_selection"):
                    key = f"phase_{phase}_{role}"
                    assert before.arrived.role_hashes[key] == after.arrived.role_hashes[key]
            if before.arrived.seed == 17:
                assert (
                    before.arrived.role_hashes["phase_a_final_test"]
                    != after.arrived.role_hashes["phase_a_final_test"]
                )
            else:
                assert before.arrived == after.arrived


def test_should_match_full_trained_state_and_final_seal_after_active_reservoir_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixed_manifest()
    fresh_store = TrustedLocalReplayPolicyCheckpointStore(tmp_path / "fresh.ckpt")
    fresh = comparison.run_replay_policy_comparison(manifest, checkpoint_store=fresh_store)
    resumed_store = _save_active_cursor(tmp_path / "active.ckpt", (1, 0, "b", 1, "before_sleep", 0))
    final_reads: list[str] = []
    original_release = arrived.release_final_test

    def sealed_release(*args: Any, **kwargs: Any) -> Any:
        saved = resumed_store.load()
        assert saved.next_trial_index == 4 and saved.active is None
        final_reads.append("released")
        return original_release(*args, **kwargs)

    monkeypatch.setattr(arrived, "release_final_test", sealed_release)
    resumed = comparison.run_replay_policy_comparison(
        manifest, checkpoint_store=resumed_store, resume_from_checkpoint=True
    )
    assert resumed == fresh
    assert final_reads
    fresh_records = fresh_store.load().unscored_seeds
    resumed_records = resumed_store.load().unscored_seeds
    assert len(fresh_records) == len(resumed_records) == 4
    for expected, actual in zip(fresh_records, resumed_records, strict=True):
        for name in ("circadian_after_a", "circadian_model"):
            expected_state = getattr(expected.arrived.state, name).snapshot_state()
            actual_state = getattr(actual.arrived.state, name).snapshot_state()
            assert _comparable_state(expected_state) == _comparable_state(actual_state)
