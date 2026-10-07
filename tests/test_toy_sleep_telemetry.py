"""Toy runner decisions survive every checkpoint boundary and local JSON export."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import sys
from typing import Any

import pytest

from src.adapters import cli
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore


class _Interrupted(Exception):
    pass


class _InterruptingStore:
    def __init__(self, store: TrustedLocalToyCheckpointStore, stage: str) -> None:
        self.store = store
        self.stage = stage

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        position = checkpoint.combined.position
        if position.stage == self.stage and position.completed_epoch == 2:
            raise _Interrupted()


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=4,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=2,
        random_seed=29,
        circadian_config=CircadianConfig(
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=2,
            replay_memory_size=4,
        ),
    )


def _without_time(event: SleepEventTelemetry) -> dict[str, Any]:
    record = asdict(event)
    record.pop("durations")
    return record


def test_toy_periodic_and_skipped_schedule_is_complete_and_typed() -> None:
    result = run_experiment(_config())
    events = result.circadian_sleep.events
    assert len(events) == 4
    assert all(isinstance(event, SleepEventTelemetry) for event in events)
    assert [event.completed_epoch for event in events] == [1, 2, 3, 4]
    assert [event.trigger_reason for event in events] == [
        "not_due",
        "periodic",
        "not_due",
        "periodic",
    ]
    assert [event.outcome for event in events][::2] == ["skipped", "skipped"]
    assert all(event.guard is None for event in events)
    assert all(event.durations.attempt_seconds >= event.durations.core_seconds for event in events)
    assert result.circadian_sleep.event_count == 1
    first_attempt = events[1]
    assert first_attempt.outcome == "applied"
    assert first_attempt.reason == "core_executed"
    assert first_attempt.budgets.split_limit == 1
    assert len(first_attempt.changes.applied_split_pairs) == 1
    assert first_attempt.before_width == 4
    assert first_attempt.final_width == 5
    assert first_attempt.chemistry_before.count == 4
    assert first_attempt.chemistry_final.count == 5
    assert first_attempt.replay.proposed_examples >= first_attempt.replay.applied_examples
    json.dumps([asdict(event) for event in events], allow_nan=False)


@pytest.mark.parametrize("stage", ["wake", "before_sleep", "after_sleep"])
@pytest.mark.parametrize("reverse", [False, True])
def test_toy_sleep_history_matches_resume_at_each_stage(
    tmp_path: Any, stage: str, reverse: bool
) -> None:
    config = _config()
    if reverse:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    expected = run_experiment(config)
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.checkpoint")
    with pytest.raises(_Interrupted):
        run_experiment(config, checkpoint_store=_InterruptingStore(store, stage))
    checkpoint = store.load()
    assert len(checkpoint.sleep_events) == (1 if stage == "before_sleep" else 2)
    actual = run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert [_without_time(event) for event in actual.circadian_sleep.events] == [
        _without_time(event) for event in expected.circadian_sleep.events
    ]
    assert tuple(actual.circadian_sleep.events) == store.load().sleep_events
    assert actual.circadian_sleep.event_count == expected.circadian_sleep.event_count


def test_toy_disabled_and_adaptive_decisions_are_recorded() -> None:
    disabled = run_experiment(
        replace(_config(), circadian_config=CircadianConfig(sleep_mode="disabled"))
    )
    assert [event.trigger_reason for event in disabled.circadian_sleep.events] == ["disabled"] * 4
    assert all(event.outcome == "skipped" for event in disabled.circadian_sleep.events)

    adaptive_config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_chemical_reset=True,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        replay_steps=0,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        use_adaptive_sleep_trigger=True,
        min_epochs_between_sleep=2,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
    )
    adaptive = run_experiment(
        replace(
            _config(),
            epoch_count=3,
            circadian_sleep_interval=0,
            circadian_force_sleep=False,
            circadian_config=adaptive_config,
        )
    )
    assert "adaptive" in {event.trigger_reason for event in adaptive.circadian_sleep.events}
    assert adaptive.circadian_sleep.event_count == 1

    combined_due = run_experiment(replace(_config(), circadian_config=adaptive_config))
    assert "periodic_and_adaptive" in {
        event.trigger_reason for event in combined_due.circadian_sleep.events
    }


def test_toy_checkpoint_rejects_old_and_incomplete_event_histories_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.app import experiment_runner

    config = _config()
    store = TrustedLocalToyCheckpointStore(tmp_path / "toy.checkpoint")
    run_experiment(config, checkpoint_store=store)
    checkpoint = store.load()
    trained = 0
    original_train = experiment_runner.BackpropMLP.train_epoch

    def tracked_train(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        trained += 1
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(experiment_runner.BackpropMLP, "train_epoch", tracked_train)
    for changed, message in (
        (replace(checkpoint, format_version=1), "format"),
        (replace(checkpoint, sleep_events=checkpoint.sleep_events[:-1]), "event history"),
        (
            replace(
                checkpoint,
                sleep_events=(
                    replace(checkpoint.sleep_events[0], completed_epoch=99),
                    *checkpoint.sleep_events[1:],
                ),
            ),
            "event history",
        ),
    ):
        store.save(changed)
        with pytest.raises(ValueError, match=message):
            run_experiment(config, checkpoint_store=store, resume_from_checkpoint=True)
    assert trained == 0


def test_toy_cli_writes_json_result_with_typed_sleep_events(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy",
            "--samples",
            "80",
            "--epochs",
            "2",
            "--sleep-interval",
            "1",
            "--json-result",
            str(path),
        ],
    )
    cli.main()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol_id"] == "toy_validation_v1"
    assert len(payload["circadian_sleep"]["events"]) == 2
    assert payload["circadian_sleep"]["events"][0]["trigger_reason"] == "periodic"
    assert payload["circadian_sleep"]["events"][0]["format_version"] == 1
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_bytes() == original
