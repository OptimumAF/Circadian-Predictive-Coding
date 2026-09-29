"""Arrived guard decisions retain truthful scores and rejected core proposals."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_shift_benchmark as base
from src.app.continual_arrived_checkpoint import arrived_sleep_history_digest
from src.app.continual_shift_benchmark import ContinualGlobalSealConfig
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.infra.datasets import LabeledData
from src.infra.circadian_checkpoint_files import TrustedLocalArrivedCheckpointStore


def _config(reverse_order: bool = False) -> arrived.ContinualArrivedRolesConfig:
    training = ContinualGlobalSealConfig(
        sample_count_phase_a=40,
        sample_count_phase_b=40,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=2,
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


@pytest.mark.parametrize("reverse_order", [False, True])
def test_ordinary_v6_records_phase_local_guard_and_skip_facts(reverse_order: bool) -> None:
    result = arrived.run_continual_arrived_benchmark(_config(reverse_order), [17, 19])

    for seed in result.seed_results:
        events = seed.metrics.circadian_predictive_coding.sleep_events
        assert len(events) == 4
        assert [event.completed_epoch for event in events] == [1, 2, 3, 4]
        assert [event.trigger_reason for event in events] == ["not_due", "periodic"] * 2
        for phase, event in (("a", events[1]), ("b", events[3])):
            assert event.guard is not None
            assert event.guard.role == "inner_guard"
            assert event.guard.role_hash == seed.role_hashes[f"phase_{phase}_inner_guard"]
            assert event.guard.metric_name == "accuracy"
            assert event.guard.pre_cross_entropy is None
            assert event.guard.post_cross_entropy is None
            assert event.guard.examples_scored == 2 * len(
                seed.role_ids[f"phase_{phase}_inner_guard"]
            )
            assert event.durations.attempt_seconds >= event.durations.core_seconds
            assert event.outcome in {"accepted", "rolled_back"}
        assert events[0].guard is None and events[2].guard is None
        json.dumps([asdict(event) for event in events], allow_nan=False)


def test_rejected_guard_keeps_full_core_proposal_and_restores_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=True,
        sleep_enable_homeostasis=False,
        sleep_enable_split=True,
        sleep_enable_prune=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        replay_steps=1,
        replay_memory_size=2,
        split_threshold=0.8,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        split_noise_scale=0.0,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2, hidden_dim=4, seed=1261, max_hidden_dim=6, circadian_config=config
    )
    model.train_epoch(
        np.array([[0.3, -0.2], [-0.1, 0.4]]),
        np.array([[1.0], [0.0]]),
        0.03,
        2,
        0.2,
    )
    model.set_chemical_state(np.array([0.95, 0.5, 0.5, 0.5]))
    before_width = model.hidden_dim
    guard = LabeledData(np.array([[0.3, -0.2], [-0.1, 0.4]]), np.array([[1.0], [0.0]]))
    scores = iter((0.8, 0.3))
    monkeypatch.setattr(model, "compute_accuracy", lambda *_: next(scores))
    events: list[Any] = []

    counts = base._apply_scheduled_sleep(
        model=model,
        sleep_interval=1,
        epoch_index=1,
        global_epoch=1,
        total_epochs=2,
        force_sleep=True,
        sleep_event_count=0,
        total_splits=0,
        total_prunes=0,
        guard=guard,
        guard_role_hash="a" * 64,
        guard_drop_tolerance=0.1,
        on_sleep_event=events.append,
    )

    assert counts == (0, 0, 0)
    assert model.hidden_dim == before_width
    event = events[0]
    assert event.outcome == "rolled_back"
    assert event.reason == "guard_rejected"
    assert event.changes.proposed_split_pairs == ((0, 4),)
    assert event.changes.applied_split_pairs == ()
    assert event.proposed_width == 5 and event.final_width == 4
    assert event.chemistry_final == event.chemistry_before
    assert event.guard is not None
    assert event.guard.role_hash == "a" * 64
    assert event.guard.delta == pytest.approx(0.5)
    assert event.guard.tolerance == 0.1
    assert event.replay.proposed_examples == 2
    assert event.replay.proposed_updates == 1
    assert event.replay.applied_examples == event.replay.applied_updates == 0


@pytest.mark.parametrize("failure_stage", ["core", "post_guard"])
def test_failed_guarded_attempt_emits_error_and_restores_model(
    failure_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_replay=True,
        sleep_enable_split=True,
        sleep_enable_prune=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=0,
        replay_steps=1,
        replay_memory_size=2,
        split_threshold=0.8,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        split_noise_scale=0.0,
    )
    model = CircadianPredictiveCodingNetwork(
        input_dim=2, hidden_dim=4, seed=1261, max_hidden_dim=6, circadian_config=config
    )
    guard = LabeledData(np.array([[0.3, -0.2], [-0.1, 0.4]]), np.array([[1.0], [0.0]]))
    model.train_epoch(guard.input, guard.target, 0.03, 2, 0.2)
    model.set_chemical_state(np.array([0.95, 0.5, 0.5, 0.5]))
    before_clocks = model.get_sleep_clocks()
    events: list[Any] = []
    original_sleep = CircadianPredictiveCodingNetwork.sleep_event
    scores = iter((0.8, RuntimeError("post guard failed")))

    def score_or_fail(*_: Any) -> float:
        value = next(scores)
        if isinstance(value, Exception):
            raise value
        return value

    def sleep_or_fail(self: Any, **kwargs: Any) -> Any:
        if failure_stage == "core":
            raise RuntimeError("core failed")
        return original_sleep(self, **kwargs)

    with monkeypatch.context() as patch_guard:
        patch_guard.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", score_or_fail)
        patch_guard.setattr(CircadianPredictiveCodingNetwork, "sleep_event", sleep_or_fail)
        with pytest.raises(RuntimeError, match="core failed|post guard failed"):
            base._apply_scheduled_sleep(
                model=model,
                sleep_interval=1,
                epoch_index=1,
                global_epoch=1,
                total_epochs=2,
                force_sleep=True,
                sleep_event_count=0,
                total_splits=0,
                total_prunes=0,
                guard=guard,
                guard_role_hash="a" * 64,
                on_sleep_event=events.append,
            )

    assert len(events) == 1
    event = events[0]
    assert event.outcome == "error"
    assert event.reason == (
        "sleep_core_exception" if failure_stage == "core" else "inner_guard_post_exception"
    )
    assert event.guard is not None
    assert event.guard.role_hash == "a" * 64
    assert event.guard.pre_accuracy == 0.8
    assert event.guard.post_accuracy is None and event.guard.delta is None
    assert event.chemistry_final == event.chemistry_before
    assert event.durations.attempt_seconds >= event.durations.core_seconds
    assert model.hidden_dim == 4
    assert model.get_sleep_clocks() == before_clocks
    if failure_stage == "post_guard":
        assert event.changes.proposed_split_pairs == ((0, 4),)
        assert event.changes.applied_split_pairs == ()
        assert event.replay.proposed_updates == 1
        assert event.replay.applied_updates == 0


@pytest.mark.parametrize("failure_stage", ["pre_exception", "pre_nonfinite", "post_nonfinite"])
def test_invalid_guard_score_emits_truthful_partial_error(
    failure_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=1261,
        circadian_config=CircadianConfig(sleep_mode="components", replay_steps=0),
    )
    guard = LabeledData(np.array([[0.3, -0.2]]), np.array([[1.0]]))
    values: list[Any] = (
        [RuntimeError("pre guard failed")]
        if failure_stage == "pre_exception"
        else [float("nan")]
        if failure_stage == "pre_nonfinite"
        else [0.8, float("nan")]
    )

    def score_or_fail(*_: Any) -> float:
        value = values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", score_or_fail)
    events: list[Any] = []
    with pytest.raises(
        (RuntimeError, ValueError), match="pre guard failed|accuracy must be finite"
    ):
        base._apply_scheduled_sleep(
            model=model,
            sleep_interval=1,
            epoch_index=1,
            global_epoch=1,
            total_epochs=2,
            force_sleep=True,
            sleep_event_count=0,
            total_splits=0,
            total_prunes=0,
            guard=guard,
            guard_role_hash="a" * 64,
            on_sleep_event=events.append,
        )

    assert len(events) == 1
    event = events[0]
    assert event.outcome == "error"
    assert event.reason == f"inner_guard_{failure_stage}"
    assert event.guard is not None
    assert event.guard.pre_accuracy == (0.8 if failure_stage == "post_nonfinite" else None)
    assert event.guard.post_accuracy is None and event.guard.delta is None
    assert event.guard.examples_scored == (1 if failure_stage == "post_nonfinite" else 0)


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("error_epoch", [2, 4])
@pytest.mark.parametrize("failure_stage", ["core", "post_guard"])
def test_checkpointed_error_attempt_is_kept_before_retry_and_final_scoring(
    reverse_order: bool,
    error_epoch: int,
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(reverse_order)
    control_store = TrustedLocalArrivedCheckpointStore(tmp_path / "control.ckpt")
    control = arrived.run_continual_arrived_benchmark(config, [17], checkpoint_store=control_store)
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "error.ckpt")
    original_sleep = base._apply_scheduled_sleep
    fail_once = True

    def raise_core_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("core failed")

    def inject_sleep_error(**kwargs: Any) -> tuple[int, int, int]:
        nonlocal fail_once
        if kwargs["global_epoch"] != error_epoch or not fail_once:
            return original_sleep(**kwargs)
        fail_once = False
        if failure_stage == "core":
            with monkeypatch.context() as patch_core:
                patch_core.setattr(
                    CircadianPredictiveCodingNetwork, "sleep_event", raise_core_error
                )
                return original_sleep(**kwargs)
        calls = 0

        def score_or_fail(*_: Any) -> float:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0.8
            raise RuntimeError("post guard failed")

        with monkeypatch.context() as patch_guard:
            patch_guard.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", score_or_fail)
            return original_sleep(**kwargs)

    monkeypatch.setattr(base, "_apply_scheduled_sleep", inject_sleep_error)
    final_releases = 0
    original_release = arrived.release_final_test

    def count_final_release(*args: Any, **kwargs: Any) -> Any:
        nonlocal final_releases
        final_releases += 1
        return original_release(*args, **kwargs)

    monkeypatch.setattr(arrived, "release_final_test", count_final_release)
    with pytest.raises(RuntimeError, match="core failed|post guard failed"):
        arrived.run_continual_arrived_benchmark(config, [17], checkpoint_store=store)
    assert final_releases == 0
    saved = store.load()
    phase = "a" if error_epoch == 2 else "b"
    assert (saved.phase, saved.phase_epoch_completed, saved.stage) == (phase, 2, "before_sleep")
    assert [event.completed_epoch for event in saved.active_sleep_events] == list(
        range(1, error_epoch + 1)
    )
    assert saved.active_sleep_events[-1].outcome == "error"
    assert saved.active_sleep_events[-1].reason == (
        "sleep_core_exception" if failure_stage == "core" else "inner_guard_post_exception"
    )
    assert len(saved.active_guard_decisions) == (0 if phase == "a" else 1)

    resumed = arrived.run_continual_arrived_benchmark(
        config, [17], checkpoint_store=store, resume_from_checkpoint=True
    )
    events = resumed.seed_results[0].metrics.circadian_predictive_coding.sleep_events
    assert resumed == control
    assert [event.completed_epoch for event in events] == (
        [1, 2, 2, 3, 4] if phase == "a" else [1, 2, 3, 4, 4]
    )
    assert [event.outcome for event in events if event.completed_epoch == error_epoch] == [
        "error",
        "accepted",
    ]
    assert len(resumed.seed_results[0].guard_decisions) == len(
        control.seed_results[0].guard_decisions
    )
    assert final_releases == 2
    reloaded = arrived.run_continual_arrived_benchmark(
        config, [17], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert _semantic_events(
        reloaded.seed_results[0].metrics.circadian_predictive_coding.sleep_events
    ) == (_semantic_events(events))
    actual_state = store.load().unscored_seeds[0].state
    control_state = control_store.load().unscored_seeds[0].state
    for name in ("backprop_model", "predictive_model", "circadian_model"):
        actual_model = getattr(actual_state, name)
        control_model = getattr(control_state, name)
        assert actual_model.__dict__.keys() == control_model.__dict__.keys()
        for field_name, actual_value in actual_model.__dict__.items():
            control_value = control_model.__dict__[field_name]
            if field_name == "_replay_memory":
                assert len(actual_value) == len(control_value)
                for actual_snapshot, control_snapshot in zip(
                    actual_value, control_value, strict=True
                ):
                    np.testing.assert_array_equal(
                        actual_snapshot.input_batch, control_snapshot.input_batch
                    )
                    np.testing.assert_array_equal(
                        actual_snapshot.target_batch, control_snapshot.target_batch
                    )
                    assert actual_snapshot.priority == control_snapshot.priority
                    assert actual_snapshot.positive_fraction == control_snapshot.positive_fraction
            else:
                assert pickle.dumps(actual_value, protocol=5) == pickle.dumps(
                    control_value, protocol=5
                ), field_name


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("error_epoch", [2, 4])
@pytest.mark.parametrize("failure_stage", ["core", "post_guard"])
def test_ordinary_v6_can_explicitly_retry_one_failed_guard_attempt(
    reverse_order: bool,
    error_epoch: int,
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(reverse_order)
    control = arrived.run_continual_arrived_benchmark(config, [17, 19])
    original_sleep = base._apply_scheduled_sleep
    fail_once = True

    def raise_core_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("core failed")

    def inject_sleep_error(**kwargs: Any) -> tuple[int, int, int]:
        nonlocal fail_once
        if kwargs["global_epoch"] != error_epoch or not fail_once:
            return original_sleep(**kwargs)
        fail_once = False
        if failure_stage == "core":
            with monkeypatch.context() as patch_core:
                patch_core.setattr(
                    CircadianPredictiveCodingNetwork, "sleep_event", raise_core_error
                )
                return original_sleep(**kwargs)
        calls = 0

        def score_or_fail(*_: Any) -> float:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0.8
            raise RuntimeError("post guard failed")

        with monkeypatch.context() as patch_guard:
            patch_guard.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", score_or_fail)
            return original_sleep(**kwargs)

    monkeypatch.setattr(base, "_apply_scheduled_sleep", inject_sleep_error)
    result = arrived.run_continual_arrived_benchmark(config, [17, 19], sleep_error_retries=1)

    assert result == control
    first = result.seed_results[0]
    second = result.seed_results[1]
    events = first.metrics.circadian_predictive_coding.sleep_events
    assert [event.completed_epoch for event in events] == (
        [1, 2, 2, 3, 4] if error_epoch == 2 else [1, 2, 3, 4, 4]
    )
    assert [event.outcome for event in events if event.completed_epoch == error_epoch] == [
        "error",
        "accepted",
    ]
    assert len(second.metrics.circadian_predictive_coding.sleep_events) == 4
    assert first.guard_decisions == control.seed_results[0].guard_decisions
    json.dumps([asdict(event) for event in events], allow_nan=False)


def test_two_error_attempts_keep_one_retryable_epoch_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    control = arrived.run_continual_arrived_benchmark(config, [17])
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "two-errors.ckpt")
    original_sleep = base._apply_scheduled_sleep
    failures_left = 2

    def inject_post_guard_error(**kwargs: Any) -> tuple[int, int, int]:
        nonlocal failures_left
        if kwargs["global_epoch"] != 2 or failures_left == 0:
            return original_sleep(**kwargs)
        failures_left -= 1
        calls = 0

        def score_or_fail(*_: Any) -> float:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0.8
            raise RuntimeError("post guard failed")

        with monkeypatch.context() as patch_guard:
            patch_guard.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", score_or_fail)
            return original_sleep(**kwargs)

    monkeypatch.setattr(base, "_apply_scheduled_sleep", inject_post_guard_error)
    for retry, expected_errors in ((False, 1), (True, 2)):
        with pytest.raises(RuntimeError, match="post guard failed"):
            arrived.run_continual_arrived_benchmark(
                config, [17], checkpoint_store=store, resume_from_checkpoint=retry
            )
        checkpoint = store.load()
        assert checkpoint.stage == "before_sleep"
        assert [event.outcome for event in checkpoint.active_sleep_events[1:]] == (
            ["error"] * expected_errors
        )

    result = arrived.run_continual_arrived_benchmark(
        config, [17], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert result == control
    events = result.seed_results[0].metrics.circadian_predictive_coding.sleep_events
    assert [event.completed_epoch for event in events] == [1, 2, 2, 2, 3, 4]
    assert [event.outcome for event in events[1:4]] == ["error", "error", "accepted"]
    assert result.seed_results[0].guard_decisions == control.seed_results[0].guard_decisions


def test_retry_limit_rejects_invalid_values_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arrived,
        "generate_two_cluster_dataset_with_transform",
        lambda **_: pytest.fail("source opened before retry validation"),
    )
    with pytest.raises(ValueError, match="nonnegative"):
        arrived.run_continual_arrived_benchmark(_config(), [17], sleep_error_retries=-1)
    with pytest.raises(ValueError, match="explicit checkpoint resume"):
        arrived.run_continual_arrived_benchmark(
            _config(),
            [17],
            checkpoint_store=TrustedLocalArrivedCheckpointStore(Path("unused.ckpt")),
            sleep_error_retries=1,
        )


@pytest.mark.parametrize("damage", ["role_hash", "reason"])
def test_error_attempt_tamper_rejects_before_model_restore(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "error-tamper.ckpt")
    original_sleep = base._apply_scheduled_sleep
    failed = False

    def raise_core_error(*_: Any, **__: Any) -> Any:
        raise RuntimeError("core failed")

    def inject_core_error(**kwargs: Any) -> tuple[int, int, int]:
        nonlocal failed
        if kwargs["global_epoch"] == 2 and not failed:
            failed = True
            with monkeypatch.context() as patch_core:
                patch_core.setattr(
                    CircadianPredictiveCodingNetwork,
                    "sleep_event",
                    raise_core_error,
                )
                return original_sleep(**kwargs)
        return original_sleep(**kwargs)

    monkeypatch.setattr(base, "_apply_scheduled_sleep", inject_core_error)
    with pytest.raises(RuntimeError, match="core failed"):
        arrived.run_continual_arrived_benchmark(config, [17], checkpoint_store=store)
    checkpoint = store.load()
    error_event = checkpoint.active_sleep_events[-1]
    assert error_event.guard is not None
    changed = (
        replace(error_event, guard=replace(error_event.guard, role_hash="0" * 64))
        if damage == "role_hash"
        else replace(error_event, reason="inner_guard_pre_exception")
    )
    events = (*checkpoint.active_sleep_events[:-1], changed)
    store.save(
        replace(
            checkpoint,
            active_sleep_events=events,
            active_sleep_history_digest=arrived_sleep_history_digest(
                events, checkpoint.active_event_digest
            ),
        )
    )
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "restore_state",
        lambda *_: pytest.fail("model restored before error history validation"),
    )

    with pytest.raises(ValueError, match="sleep error guard facts"):
        arrived.run_continual_arrived_benchmark(
            config, [17], checkpoint_store=store, resume_from_checkpoint=True
        )


class _StopAtCursor(Exception):
    pass


class _InterruptingStore:
    def __init__(self, path: Path, phase: str, epoch: int, stage: str) -> None:
        self.store = TrustedLocalArrivedCheckpointStore(path)
        self.target = (phase, epoch, stage)

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if (
            checkpoint.seed_index == 0
            and (checkpoint.phase, checkpoint.phase_epoch_completed, checkpoint.stage)
            == self.target
        ):
            raise _StopAtCursor()


def _semantic_events(events: Any) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in asdict(event).items() if key != "durations"}
        for event in events
    ]


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    ("phase", "epoch", "stage"),
    [
        ("a", 1, "before_sleep"),
        ("a", 2, "after_sleep"),
        ("b", 1, "before_sleep"),
        ("b", 2, "after_sleep"),
    ],
)
def test_checkpointed_v6_preserves_guarded_history_across_both_phases(
    reverse_order: bool, phase: str, epoch: int, stage: str, tmp_path: Path
) -> None:
    config = _config(reverse_order)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17, 19])
    path = tmp_path / "arrived.ckpt"
    with pytest.raises(_StopAtCursor):
        arrived.run_continual_arrived_benchmark(
            config, [17, 19], checkpoint_store=_InterruptingStore(path, phase, epoch, stage)
        )
    resumed = arrived.run_continual_arrived_benchmark(
        config,
        [17, 19],
        checkpoint_store=TrustedLocalArrivedCheckpointStore(path),
        resume_from_checkpoint=True,
    )

    assert resumed == ordinary
    for actual, expected in zip(resumed.seed_results, ordinary.seed_results, strict=True):
        assert _semantic_events(actual.metrics.circadian_predictive_coding.sleep_events) == (
            _semantic_events(expected.metrics.circadian_predictive_coding.sleep_events)
        )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_rejected_v6_guard_history_survives_phase_b_resume(
    reverse_order: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_sleep = base._apply_scheduled_sleep

    def controlled_sleep(**kwargs: Any) -> tuple[int, int, int]:
        if kwargs["epoch_index"] != 2:
            return original_sleep(**kwargs)
        scores = iter((0.8, 0.2))
        with monkeypatch.context() as patch_guard:
            patch_guard.setattr(
                CircadianPredictiveCodingNetwork,
                "compute_accuracy",
                lambda *_: next(scores),
            )
            return original_sleep(**kwargs)

    monkeypatch.setattr(base, "_apply_scheduled_sleep", controlled_sleep)
    config = _config(reverse_order)
    ordinary = arrived.run_continual_arrived_benchmark(config, [17])
    path = tmp_path / "rejected.ckpt"
    with pytest.raises(_StopAtCursor):
        arrived.run_continual_arrived_benchmark(
            config, [17], checkpoint_store=_InterruptingStore(path, "b", 2, "before_sleep")
        )
    resumed = arrived.run_continual_arrived_benchmark(
        config,
        [17],
        checkpoint_store=TrustedLocalArrivedCheckpointStore(path),
        resume_from_checkpoint=True,
    )

    expected = ordinary.seed_results[0]
    actual = resumed.seed_results[0]
    assert _semantic_events(actual.metrics.circadian_predictive_coding.sleep_events) == (
        _semantic_events(expected.metrics.circadian_predictive_coding.sleep_events)
    )
    assert len(actual.guard_decisions) == 2
    assert all(not decision.accepted and decision.restored for decision in actual.guard_decisions)
    assert actual.metrics.circadian_predictive_coding.sleep_event_count == 0
    for event in actual.metrics.circadian_predictive_coding.sleep_events[1::2]:
        assert event.outcome == "rolled_back"
        assert event.guard is not None and event.guard.delta == pytest.approx(0.6)
        assert event.changes.applied_split_pairs == ()
        assert event.replay.applied_updates == 0


@pytest.mark.parametrize("damage", ["version", "missing", "digest", "role_hash"])
def test_active_v6_rejects_incompatible_sleep_history_before_training(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    path = tmp_path / "active.ckpt"
    with pytest.raises(_StopAtCursor):
        arrived.run_continual_arrived_benchmark(
            config, [17], checkpoint_store=_InterruptingStore(path, "a", 2, "after_sleep")
        )
    store = TrustedLocalArrivedCheckpointStore(path)
    checkpoint = store.load()
    assert len(checkpoint.active_sleep_events) == 2
    if damage == "version":
        checkpoint = replace(checkpoint, sleep_event_history_version=0)
    elif damage == "missing":
        checkpoint = replace(checkpoint, active_sleep_events=checkpoint.active_sleep_events[:-1])
    elif damage == "digest":
        checkpoint = replace(checkpoint, active_sleep_history_digest="0" * 64)
    else:
        event = checkpoint.active_sleep_events[1]
        assert event.guard is not None
        changed = replace(event, guard=replace(event.guard, role_hash="0" * 64))
        events = (checkpoint.active_sleep_events[0], changed)
        checkpoint = replace(
            checkpoint,
            active_sleep_events=events,
            active_sleep_history_digest=arrived_sleep_history_digest(
                events, checkpoint.active_event_digest
            ),
        )
    store.save(checkpoint)
    monkeypatch.setattr(
        base,
        "_train_named_model_epoch",
        lambda *_: pytest.fail("training started before sleep history validation"),
    )

    with pytest.raises(ValueError, match="sleep"):
        arrived.run_continual_arrived_benchmark(
            config, [17], checkpoint_store=store, resume_from_checkpoint=True
        )


@pytest.mark.parametrize("damage", ["version", "missing", "digest"])
def test_completed_v6_rejects_incompatible_sleep_history_before_scoring(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = TrustedLocalArrivedCheckpointStore(tmp_path / "completed.ckpt")
    arrived.run_continual_arrived_benchmark(config, [17], checkpoint_store=store)
    checkpoint = store.load()
    record = checkpoint.unscored_seeds[0]
    if damage == "version":
        record = replace(record, sleep_event_history_version=0)
    elif damage == "missing":
        record = replace(record, sleep_events=record.sleep_events[:-1])
    else:
        record = replace(record, sleep_history_digest="0" * 64)
    store.save(replace(checkpoint, unscored_seeds=(record,)))
    monkeypatch.setattr(
        arrived,
        "_score_arrived_seed",
        lambda *_: pytest.fail("final scoring started before sleep history validation"),
    )

    with pytest.raises(ValueError, match="sleep"):
        arrived.run_continual_arrived_benchmark(
            config, [17], checkpoint_store=store, resume_from_checkpoint=True
        )
