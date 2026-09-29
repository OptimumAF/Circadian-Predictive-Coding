"""Historical continual protocols retain complete phase-local sleep history."""

from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
import json
import sys
from typing import Any

import pytest

from src.app.continual_shift_benchmark import (
    CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
    CONTINUAL_GLOBAL_SEAL_PROTOCOL,
    CONTINUAL_LEGACY_PROTOCOL,
    CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
    CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
    CONTINUAL_VALIDATION_PROTOCOL,
    ContinualBoundedReplayConfig,
    ContinualGlobalSealConfig,
    ContinualShiftConfig,
    run_continual_shift_benchmark,
)
from src.core.circadian_predictive_coding import CircadianConfig
from src.core.sleep_telemetry import SleepEventTelemetry
from src.infra.circadian_checkpoint_files import TrustedLocalContinualCheckpointStore
from src.infra.local_result_json import write_local_result_json


class _Interrupted(Exception):
    pass


class _InterruptingStore:
    def __init__(
        self, store: TrustedLocalContinualCheckpointStore, *, phase: str, stage: str, local: int
    ) -> None:
        self.store = store
        self.phase = phase
        self.stage = stage
        self.local = local

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if (
            checkpoint.seed_index == 0
            and checkpoint.phase == self.phase
            and checkpoint.phase_epoch_completed == self.local
            and checkpoint.combined.position.stage == self.stage
        ):
            raise _Interrupted()


def _config(protocol: str) -> ContinualShiftConfig:
    base = ContinualShiftConfig(
        sample_count_phase_a=80,
        sample_count_phase_b=80,
        phase_b_train_fraction=0.5,
        hidden_dim=4,
        phase_a_epochs=2,
        phase_b_epochs=2,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval_phase_a=2,
        circadian_sleep_interval_phase_b=1,
        circadian_config=CircadianConfig(
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=2,
            replay_memory_size=4,
        ),
    )
    if protocol in {CONTINUAL_BOUNDED_REPLAY_PROTOCOL, CONTINUAL_GLOBAL_SEAL_PROTOCOL}:
        cls = (
            ContinualGlobalSealConfig
            if protocol == CONTINUAL_GLOBAL_SEAL_PROTOCOL
            else ContinualBoundedReplayConfig
        )
        return cls(
            **{
                **base.__dict__,
                "protocol_id": protocol,
                "circadian_config": CircadianConfig(
                    sleep_mode="components",
                    max_split_per_sleep=0,
                    max_prune_per_sleep=0,
                    replay_steps=1,
                    replay_memory_size=1,
                ),
            },
            replay_max_examples=4,
            replay_max_bytes=96,
        )
    return replace(base, protocol_id=protocol)


def _semantic(event: SleepEventTelemetry) -> dict[str, Any]:
    payload = asdict(event)
    payload.pop("durations")
    return payload


@pytest.mark.parametrize(
    "protocol",
    [
        CONTINUAL_LEGACY_PROTOCOL,
        CONTINUAL_VALIDATION_PROTOCOL,
        CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
        CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
        CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
        CONTINUAL_GLOBAL_SEAL_PROTOCOL,
    ],
)
def test_historical_continual_reports_complete_unguarded_phase_history(
    protocol: str, tmp_path: Any
) -> None:
    result = run_continual_shift_benchmark(_config(protocol), [13, 17])
    assert len(result.seed_results) == 2
    for seed in result.seed_results:
        events = seed.circadian_predictive_coding.sleep_events
        assert len(events) == 4
        assert all(isinstance(event, SleepEventTelemetry) for event in events)
        assert [event.completed_epoch for event in events] == [1, 2, 3, 4]
        assert [event.trigger_reason for event in events] == [
            "not_due",
            "periodic",
            "periodic",
            "periodic",
        ]
        assert all(event.guard is None for event in events)
        assert all(
            event.durations.attempt_seconds >= event.durations.core_seconds for event in events
        )
        assert seed.circadian_predictive_coding.sleep_event_count <= sum(
            event.outcome == "applied" for event in events
        )
    path = tmp_path / "result.json"
    write_local_result_json(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["seed_results"]) == 2
    assert all(
        len(item["circadian_predictive_coding"]["sleep_events"]) == 4
        for item in payload["seed_results"]
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("phase", "stage", "local", "expected_count"),
    [
        ("a", "wake", 0, 0),
        ("a", "before_sleep", 2, 1),
        ("a", "after_sleep", 2, 2),
        ("b", "after_sleep", 0, 2),
        ("b", "before_sleep", 1, 2),
        ("b", "after_sleep", 1, 3),
    ],
)
def test_historical_continual_event_history_matches_resume(
    tmp_path: Any, reverse: bool, phase: str, stage: str, local: int, expected_count: int
) -> None:
    config = _config(CONTINUAL_VALIDATION_PROTOCOL)
    if reverse:
        config = replace(config, model_order=tuple(reversed(config.model_order)))
    expected = run_continual_shift_benchmark(config, [13, 17])
    store = TrustedLocalContinualCheckpointStore(tmp_path / "continual.ckpt")
    with pytest.raises(_Interrupted):
        run_continual_shift_benchmark(
            config,
            [13, 17],
            checkpoint_store=_InterruptingStore(store, phase=phase, stage=stage, local=local),
        )
    assert len(store.load().sleep_events) == expected_count
    actual = run_continual_shift_benchmark(
        config, [13, 17], checkpoint_store=store, resume_from_checkpoint=True
    )
    for actual_seed, expected_seed in zip(actual.seed_results, expected.seed_results, strict=True):
        assert [
            _semantic(item) for item in actual_seed.circadian_predictive_coding.sleep_events
        ] == [_semantic(item) for item in expected_seed.circadian_predictive_coding.sleep_events]
        assert actual_seed.circadian_predictive_coding.sleep_event_count == (
            expected_seed.circadian_predictive_coding.sleep_event_count
        )


@pytest.mark.parametrize(
    "protocol",
    [
        CONTINUAL_LEGACY_PROTOCOL,
        CONTINUAL_VALIDATION_PROTOCOL,
        CONTINUAL_PHASE_ARRIVAL_PROTOCOL,
        CONTINUAL_PHASE_LOCAL_SCHEDULE_PROTOCOL,
        CONTINUAL_BOUNDED_REPLAY_PROTOCOL,
        CONTINUAL_GLOBAL_SEAL_PROTOCOL,
    ],
)
def test_historical_continual_protocols_preserve_history_through_phase_b_resume(
    protocol: str, tmp_path: Any
) -> None:
    config = _config(protocol)
    expected = run_continual_shift_benchmark(config, [13, 17])
    store = TrustedLocalContinualCheckpointStore(tmp_path / "continual.ckpt")
    with pytest.raises(_Interrupted):
        run_continual_shift_benchmark(
            config,
            [13, 17],
            checkpoint_store=_InterruptingStore(store, phase="b", stage="before_sleep", local=1),
        )
    assert [event.completed_epoch for event in store.load().sleep_events] == [1, 2]
    actual = run_continual_shift_benchmark(
        config, [13, 17], checkpoint_store=store, resume_from_checkpoint=True
    )
    for actual_seed, expected_seed in zip(actual.seed_results, expected.seed_results, strict=True):
        assert [
            _semantic(item) for item in actual_seed.circadian_predictive_coding.sleep_events
        ] == [_semantic(item) for item in expected_seed.circadian_predictive_coding.sleep_events]


@pytest.mark.parametrize(
    "protocol", [CONTINUAL_VALIDATION_PROTOCOL, CONTINUAL_GLOBAL_SEAL_PROTOCOL]
)
def test_historical_continual_committed_seed_history_survives_later_seed_resume(
    protocol: str, tmp_path: Any
) -> None:
    config = _config(protocol)
    expected = run_continual_shift_benchmark(config, [13, 17])
    store = TrustedLocalContinualCheckpointStore(tmp_path / "continual.ckpt")
    with pytest.raises(_Interrupted):
        run_continual_shift_benchmark(
            config,
            [13, 17],
            checkpoint_store=_InterruptingStore(
                store, phase="seed_complete", stage="after_sleep", local=2
            ),
        )
    checkpoint = store.load()
    if protocol == CONTINUAL_GLOBAL_SEAL_PROTOCOL:
        assert len(checkpoint.unscored_seeds) == 1
        assert len(checkpoint.unscored_seeds[0].sleep_events) == 4
    else:
        assert len(checkpoint.completed_results) == 1
        assert len(checkpoint.completed_results[0].circadian_predictive_coding.sleep_events) == 4
    actual = run_continual_shift_benchmark(
        config, [13, 17], checkpoint_store=store, resume_from_checkpoint=True
    )
    for actual_seed, expected_seed in zip(actual.seed_results, expected.seed_results, strict=True):
        assert [
            _semantic(item) for item in actual_seed.circadian_predictive_coding.sleep_events
        ] == [_semantic(item) for item in expected_seed.circadian_predictive_coding.sleep_events]


def test_historical_continual_rejects_old_and_incomplete_history_before_training(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.app import continual_shift_benchmark as continual

    config = _config(CONTINUAL_VALIDATION_PROTOCOL)
    store = TrustedLocalContinualCheckpointStore(tmp_path / "continual.ckpt")
    with pytest.raises(_Interrupted):
        run_continual_shift_benchmark(
            config,
            [13],
            checkpoint_store=_InterruptingStore(store, phase="b", stage="before_sleep", local=1),
        )
    checkpoint = store.load()
    trained = 0
    original = continual.BackpropMLP.train_epoch

    def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal trained
        trained += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(continual.BackpropMLP, "train_epoch", tracked)
    incomplete = replace(checkpoint, sleep_events=checkpoint.sleep_events[:-1])
    wrong_epoch = replace(
        checkpoint,
        sleep_events=(
            replace(checkpoint.sleep_events[0], completed_epoch=99),
            *checkpoint.sleep_events[1:],
        ),
    )
    old_payload = deepcopy(checkpoint)
    object.__delattr__(old_payload, "sleep_event_history_version")
    for changed, message in (
        (old_payload, "format"),
        (replace(checkpoint, sleep_event_history_version=0), "format"),
        (incomplete, "event history"),
        (wrong_epoch, "event history"),
    ):
        store.save(changed)
        with pytest.raises(ValueError, match=message):
            run_continual_shift_benchmark(
                config, [13], checkpoint_store=store, resume_from_checkpoint=True
            )
    assert trained == 0
    store.save(checkpoint)
    valid = run_continual_shift_benchmark(
        config, [13], checkpoint_store=store, resume_from_checkpoint=True
    )
    assert len(valid.seed_results[0].circadian_predictive_coding.sleep_events) == 4


def test_historical_continual_records_adaptive_and_disabled_skips() -> None:
    disabled = run_continual_shift_benchmark(
        replace(
            _config(CONTINUAL_VALIDATION_PROTOCOL),
            circadian_config=CircadianConfig(sleep_mode="disabled"),
        ),
        [13],
    )
    assert [
        event.trigger_reason
        for event in disabled.seed_results[0].circadian_predictive_coding.sleep_events
    ] == ["disabled"] * 4
    adaptive_config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_chemical_reset=True,
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_split=False,
        sleep_enable_prune=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=0,
        replay_steps=0,
        use_adaptive_sleep_trigger=True,
        min_epochs_between_sleep=2,
        sleep_energy_window=2,
        sleep_plateau_delta=1e9,
        sleep_chemical_variance_threshold=0.0,
    )
    adaptive = run_continual_shift_benchmark(
        replace(
            _config(CONTINUAL_VALIDATION_PROTOCOL),
            phase_a_epochs=3,
            phase_b_epochs=3,
            circadian_sleep_interval_phase_a=0,
            circadian_sleep_interval_phase_b=0,
            circadian_config=adaptive_config,
        ),
        [13],
    )
    assert "adaptive" in {
        event.trigger_reason
        for event in adaptive.seed_results[0].circadian_predictive_coding.sleep_events
    }


def test_continual_cli_writes_complete_json_history(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_continual_shift_benchmark as cli

    path = tmp_path / "continual.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continual",
            "--profile",
            "baseline",
            "--seeds",
            "13,17",
            "--sample-count-phase-a",
            "80",
            "--sample-count-phase-b",
            "80",
            "--phase-b-train-fraction",
            "0.5",
            "--phase-a-epochs",
            "2",
            "--phase-b-epochs",
            "2",
            "--hidden-dim",
            "4",
            "--sleep-interval-phase-a",
            "2",
            "--sleep-interval-phase-b",
            "1",
            "--json-result",
            str(path),
        ],
    )
    cli.main()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["seed_results"]) == 2
    assert all(
        len(item["circadian_predictive_coding"]["sleep_events"]) == 4
        for item in payload["seed_results"]
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_bytes() == original
