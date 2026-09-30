"""A measured process RSS cap stops toy work at checked boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from src.app import experiment_runner
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.app.toy_execution_budget import (
    ToyExecutionBudget,
    ToyExecutionProgress,
    ToyExecutionStopped,
)
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore
from src.shared.process_memory import ProcessRssSampler


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=1,
        random_seed=29,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=0,
    )


def _sampler(readings: Sequence[int | None]) -> ProcessRssSampler:
    remaining = iter(readings)
    last = readings[-1]

    def read() -> int | None:
        nonlocal last
        last = next(remaining, last)
        return last

    return ProcessRssSampler(interval_seconds=60.0, read_rss_bytes=read)


def test_should_reject_initial_high_water_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "generate_two_cluster_dataset",
        lambda **_kwargs: pytest.fail("RSS cap reached dataset construction"),
    )
    progress = ToyExecutionProgress()

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            _config(),
            execution_budget=ToyExecutionBudget(max_process_rss_bytes=110),
            execution_progress=progress,
            execution_rss_sampler=_sampler([100, 120]),
        )

    assert caught.value.stop.reason == "max_process_rss_bytes"
    assert caught.value.stop.updates_completed == 0
    assert not caught.value.stop.resumable
    assert progress.process_rss_segment is not None
    assert progress.process_rss_segment.start_bytes == 100
    assert progress.process_rss_segment.peak_bytes == 120
    assert progress.process_rss_segment.sample_count >= 2


@pytest.mark.parametrize(
    ("readings", "stage", "updates"),
    [
        ([100, 100, 100, 140], "wake", 1),
        ([100, 100, 100, 100, 100, 140], "before_sleep", 3),
        ([100, 100, 100, 100, 100, 100, 140], "after_sleep", 3),
    ],
)
def test_should_stop_at_checked_wake_sleep_or_final_boundary(
    tmp_path: Path,
    readings: list[int],
    stage: str,
    updates: int,
) -> None:
    store = TrustedLocalToyCheckpointStore(tmp_path / "rss.checkpoint")
    progress = ToyExecutionProgress()

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            _config(),
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_process_rss_bytes=120),
            execution_progress=progress,
            execution_rss_sampler=_sampler(readings),
        )

    stop = caught.value.stop
    assert stop.reason == "max_process_rss_bytes"
    assert stop.updates_completed == updates
    assert stop.checkpoint_position == store.load().combined.position
    assert stop.checkpoint_position.stage == stage
    assert stop.process_rss_segment is not None
    assert stop.process_rss_segment.peak_bytes == 140
    assert progress.process_rss_segment is not None
    assert progress.process_rss_segment.peak_bytes == 140


def test_should_use_observed_peak_even_after_current_rss_falls(tmp_path: Path) -> None:
    store = TrustedLocalToyCheckpointStore(tmp_path / "rss.checkpoint")
    sampler = _sampler([100, 100, 100, 150, 100])
    original_train = experiment_runner.BackpropMLP.train_epoch

    def train_with_transient_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_train(self, *args, **kwargs)
        sampler.sample()
        return result

    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr(experiment_runner.BackpropMLP, "train_epoch", train_with_transient_sample)
        with pytest.raises(ToyExecutionStopped) as caught:
            run_experiment(
                _config(),
                checkpoint_store=store,
                execution_budget=ToyExecutionBudget(max_process_rss_bytes=120),
                execution_rss_sampler=sampler,
            )

    assert caught.value.stop.reason == "max_process_rss_bytes"
    assert caught.value.stop.process_rss_segment is not None
    assert caught.value.stop.process_rss_segment.peak_bytes == 150


def test_should_leave_final_test_sealed_when_rss_exceeds_final_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_split = experiment_runner.split_training_validation
    final_reads = 0

    class SealedFinal:
        @property
        def input(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("final input opened")

        @property
        def target(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("final target opened")

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinal(),
            split_hashes=roles.split_hashes,
        )

    monkeypatch.setattr(experiment_runner, "split_training_validation", sealed_split)
    store = TrustedLocalToyCheckpointStore(tmp_path / "rss.checkpoint")

    with pytest.raises(ToyExecutionStopped) as caught:
        run_experiment(
            _config(),
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_process_rss_bytes=120),
            execution_rss_sampler=_sampler([100, 100, 100, 100, 100, 100, 140]),
        )

    assert caught.value.stop.checkpoint_position is not None
    assert caught.value.stop.checkpoint_position.stage == "after_sleep"
    assert final_reads == 0


def test_should_fail_clearly_before_dataset_when_process_rss_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_runner,
        "generate_two_cluster_dataset",
        lambda **_kwargs: pytest.fail("unsupported RSS reached dataset construction"),
    )

    with pytest.raises(RuntimeError, match="process RSS.*unavailable|unsupported"):
        run_experiment(
            _config(),
            execution_budget=ToyExecutionBudget(max_process_rss_bytes=1),
            execution_rss_sampler=_sampler([None]),
        )


def test_should_reject_invalid_rss_limit_and_unbudgeted_sampler() -> None:
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_process_rss_bytes"):
            ToyExecutionBudget(max_process_rss_bytes=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="execution_rss_sampler"):
        run_experiment(_config(), execution_rss_sampler=_sampler([100]))
