"""CPU RSS reports for a fixed-width checkpoint resumed in another process."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from os import getpid
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import matched_head_benchmark as benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.core.resnet50_variants import CircadianPredictiveCodingHead  # noqa: E402
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalCircadianCheckpointStore,
)


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=8,
        guard_samples=8,
        validation_samples=8,
        test_samples=8,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=16,
        predictive_inference_steps=2,
        circadian_head_hidden_dim=16,
        circadian_min_hidden_dim=16,
        circadian_max_hidden_dim=16,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
        circadian_enable_sleep_rollback=True,
    )


@pytest.fixture(autouse=True)
def preserve_random_streams() -> Any:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


class InterruptedMemoryRun(Exception):
    pass


class InterruptingMemoryStore:
    def __init__(self, store: TrustedLocalCircadianCheckpointStore, stage: str) -> None:
        self.store = store
        self.stage = stage

    def load(self) -> Any:
        return self.store.load()

    def save(self, checkpoint: Any) -> None:
        self.store.save(checkpoint)
        if checkpoint.combined.position.stage == self.stage:
            raise InterruptedMemoryRun()


_WORKER = """
import json
import os
from pathlib import Path
import sys

from src.app.matched_head_benchmark import run_three_head_fixed_width_capacity_benchmark
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig
from src.infra.circadian_checkpoint_files import TrustedLocalCircadianCheckpointStore

mode, config_path, checkpoint_path = sys.argv[1:]
config = ResNet50BenchmarkConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
store = TrustedLocalCircadianCheckpointStore(Path(checkpoint_path))

class StopAfterCheckpoint(Exception):
    pass

class InterruptingStore:
    def load(self):
        return store.load()

    def save(self, checkpoint):
        store.save(checkpoint)
        if checkpoint.combined.position.stage == "before_sleep":
            raise StopAfterCheckpoint()

if mode == "interrupt":
    try:
        run_three_head_fixed_width_capacity_benchmark(
            config, checkpoint_store=InterruptingStore(), checkpoint_memory=True
        )
    except StopAfterCheckpoint:
        saved = store.load()
        segment = saved.memory_segments[-1]
        print(json.dumps({"pid": os.getpid(), "stage": saved.combined.position.stage,
                          "segment": segment.__dict__}))
    else:
        raise AssertionError("The first process did not stop at the checkpoint")
else:
    result = run_three_head_fixed_width_capacity_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True, checkpoint_memory=True
    )
    report = result.circadian
    print(json.dumps({"pid": os.getpid(), "protocol": result.protocol_id,
                      "scope": result.memory_observation_scope,
                      "segments": [segment.__dict__ for segment in report.process_rss_segments],
                      "baseline_segments": {
                          "backprop": [segment.__dict__ for segment in result.backprop.process_rss_segments],
                          "predictive": [segment.__dict__ for segment in result.predictive_coding.process_rss_segments]},
                      "start": report.process_rss_start_bytes,
                      "peak": report.process_rss_peak_observed_bytes,
                      "samples": report.process_rss_samples,
                      "sleep_attempts": report.sleep_attempts,
                      "rollbacks": report.total_rollbacks,
                      "head_hash": result.trained_head_hashes["circadian_predictive_coding"]}))
"""


def _run_worker(mode: str, config_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, mode, str(config_path), str(checkpoint_path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_capacity_memory_checkpoint_reports_both_process_segments(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "capacity.ckpt"
    config_path.write_text(json.dumps(asdict(_config())), encoding="utf-8")

    first = _run_worker("interrupt", config_path, checkpoint_path)
    second = _run_worker("resume", config_path, checkpoint_path)

    assert first["stage"] == "before_sleep"
    assert first["segment"]["pid"] == first["pid"]
    assert first["pid"] != getpid()
    assert second["pid"] != getpid()
    assert second["protocol"] == "vision_three_head_fixed_width_capacity_checkpoint_memory_v1"
    assert second["scope"] == "committed_head_training_segments_absolute_process_rss"
    assert len(second["segments"]) == 2
    assert [segment["pid"] for segment in second["segments"]] == [first["pid"], second["pid"]]
    assert second["segments"][0] == first["segment"]
    for baseline_segments in second["baseline_segments"].values():
        assert len(baseline_segments) == 1
        assert baseline_segments[0]["pid"] == second["pid"]
    assert all(
        segment["peak_bytes"] >= segment["start_bytes"] > 0
        and segment["sample_count"] >= 1
        and segment["interval_seconds"] == 0.005
        for segment in second["segments"]
    )
    assert second["start"] is None
    assert second["peak"] == max(segment["peak_bytes"] for segment in second["segments"])
    assert second["samples"] == sum(segment["sample_count"] for segment in second["segments"])
    assert second["sleep_attempts"] == 1
    assert len(second["head_hash"]) == 64


@pytest.mark.parametrize(
    ("stage", "reject"),
    [("wake", False), ("after_sleep", False), ("after_sleep", True)],
)
def test_capacity_memory_resume_preserves_learning_and_final_test_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, reject: bool
) -> None:
    config = replace(_config(), circadian_sleep_rollback_eval_batches=1)
    original_evaluate = benchmark._evaluate_head
    guard_calls = 0

    def evaluate(
        torch_module: Any, device: Any, predict: Any, batches: Any, limit: Any, **kwargs: Any
    ) -> tuple[float, float]:
        nonlocal guard_calls
        if limit == 1 and isinstance(predict.__self__, CircadianPredictiveCodingHead):
            guard_calls += 1
            return (0.7, 1.0) if reject and guard_calls % 2 == 0 else (0.8, 0.5)
        return original_evaluate(torch_module, device, predict, batches, limit, **kwargs)

    monkeypatch.setattr(benchmark, "_evaluate_head", evaluate)
    random.seed(831)
    np.random.seed(832)
    torch.manual_seed(833)
    control = benchmark.run_three_head_fixed_width_capacity_benchmark(
        config,
        checkpoint_store=TrustedLocalCircadianCheckpointStore(tmp_path / "control.ckpt"),
        checkpoint_memory=True,
    )
    expected_draw = (random.random(), float(np.random.random()), float(torch.rand(())))

    original_build = benchmark._build_benchmark_loaders
    original_verify = benchmark._verify_fixed_width_capacity
    final_test_open = False

    class SealedTestLoader:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __iter__(self) -> Any:
            if not final_test_open:
                raise AssertionError("checkpoint memory opened final test before capacity check")
            return iter(self.wrapped)

    def build(config: ResNet50BenchmarkConfig) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader(loaders.test_loader))

    def verify(*args: Any) -> Any:
        nonlocal final_test_open
        capacity = original_verify(*args)
        final_test_open = True
        return capacity

    monkeypatch.setattr(benchmark, "_build_benchmark_loaders", build)
    monkeypatch.setattr(benchmark, "_verify_fixed_width_capacity", verify)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "interrupted.ckpt")
    guard_calls = 0
    random.seed(831)
    np.random.seed(832)
    torch.manual_seed(833)
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config,
            checkpoint_store=InterruptingMemoryStore(store, stage),
            checkpoint_memory=True,
        )
    assert final_test_open is False
    resumed = benchmark.run_three_head_fixed_width_capacity_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True, checkpoint_memory=True
    )
    assert final_test_open is True
    assert resumed.trained_head_hashes == control.trained_head_hashes
    assert resumed.initial_head_hashes == control.initial_head_hashes
    assert resumed.capacity_control == control.capacity_control
    assert resumed.circadian.total_rollbacks == int(reject)
    for actual, expected in zip(
        (resumed.backprop, resumed.predictive_coding, resumed.circadian),
        (control.backprop, control.predictive_coding, control.circadian),
        strict=True,
    ):
        for field in fields(actual):
            if field.name == "sleep_events":
                left = [asdict(event) for event in actual.sleep_events]
                right = [asdict(event) for event in expected.sleep_events]
                for event in (*left, *right):
                    event.pop("durations")
                assert left == right
                continue
            if field.name not in {
                "train_seconds",
                "process_rss_start_bytes",
                "process_rss_peak_observed_bytes",
                "process_rss_samples",
                "process_rss_segments",
            }:
                assert getattr(actual, field.name) == getattr(expected, field.name)
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected_draw


def test_checkpoint_memory_rejects_bad_segments_before_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "bad-segments.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config,
            checkpoint_store=InterruptingMemoryStore(store, "wake"),
            checkpoint_memory=True,
        )
    saved = store.load()
    segment = saved.memory_segments[0]
    monkeypatch.setattr(
        benchmark,
        "restore_circadian_checkpoint",
        lambda *args, **kwargs: pytest.fail("bad memory segment reached head restoration"),
    )
    for bad_segments in (
        (),
        (replace(segment, peak_bytes=segment.start_bytes - 1),),
        (replace(segment, sample_count=0),),
        (replace(segment, interval_seconds=0.01),),
    ):
        store.save(replace(saved, memory_segments=bad_segments))
        with pytest.raises(ValueError, match="memory segments"):
            benchmark.run_three_head_fixed_width_capacity_benchmark(
                config,
                checkpoint_store=store,
                resume_from_checkpoint=True,
                checkpoint_memory=True,
            )


def test_checkpoint_memory_rejects_capacity_only_file_before_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    plain_store = TrustedLocalCircadianCheckpointStore(tmp_path / "capacity-only.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config, checkpoint_store=InterruptingMemoryStore(plain_store, "wake")
        )
    monkeypatch.setattr(
        benchmark,
        "restore_circadian_checkpoint",
        lambda *args, **kwargs: pytest.fail("cross-protocol file reached head restoration"),
    )
    with pytest.raises(ValueError, match="checkpoint config"):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config,
            checkpoint_store=plain_store,
            resume_from_checkpoint=True,
            checkpoint_memory=True,
        )

    memory_store = TrustedLocalCircadianCheckpointStore(tmp_path / "memory.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config,
            checkpoint_store=InterruptingMemoryStore(memory_store, "wake"),
            checkpoint_memory=True,
        )
    with pytest.raises(ValueError, match="checkpoint config"):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config, checkpoint_store=memory_store, resume_from_checkpoint=True
        )


def test_older_capacity_checkpoint_without_segment_field_still_resumes(tmp_path: Path) -> None:
    config = _config()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "older-capacity.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_width_capacity_benchmark(
            config, checkpoint_store=InterruptingMemoryStore(store, "wake")
        )
    older_payload = store.load()
    object.__delattr__(older_payload, "memory_segments")
    store.save(older_payload)

    resumed = benchmark.run_three_head_fixed_width_capacity_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True
    )
    assert resumed.protocol_id == benchmark.THREE_HEAD_FIXED_WIDTH_CAPACITY_CHECKPOINT_PROTOCOL
    assert resumed.circadian.sleep_attempts == 1


def test_fixed_feature_checkpoint_memory_uses_distinct_protocol(tmp_path: Path) -> None:
    config = _config()
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "fixed-feature.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_feature_benchmark(
            config,
            checkpoint_store=InterruptingMemoryStore(store, "wake"),
            measure_memory=True,
        )
    assert store.load().protocol_id == benchmark.THREE_HEAD_FIXED_FEATURE_CHECKPOINT_MEMORY_PROTOCOL
    resumed = benchmark.run_three_head_fixed_feature_benchmark(
        config, checkpoint_store=store, resume_from_checkpoint=True, measure_memory=True
    )
    assert resumed.protocol_id == benchmark.THREE_HEAD_FIXED_FEATURE_CHECKPOINT_MEMORY_PROTOCOL
    assert resumed.memory_observation_scope == benchmark.CHECKPOINT_MEMORY_SCOPE
    assert len(resumed.circadian.process_rss_segments) == 2


def test_wall_time_checkpoint_memory_uses_distinct_protocol(tmp_path: Path) -> None:
    config = replace(_config(), epochs=1000)
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall-time.ckpt")
    result = benchmark.run_three_head_fixed_feature_wall_time_benchmark(
        config,
        wall_time_budget_seconds=0.001,
        measure_memory=True,
        checkpoint_store=store,
    )
    assert (
        result.protocol_id
        == benchmark.THREE_HEAD_FIXED_FEATURE_WALL_TIME_CHECKPOINT_MEMORY_PROTOCOL
    )
    assert result.memory_observation_scope == benchmark.CHECKPOINT_MEMORY_SCOPE
    assert len(result.circadian.process_rss_segments) == 1


def test_wall_time_memory_resume_keeps_cumulative_active_budget(tmp_path: Path) -> None:
    class TrainingClock:
        def __init__(self) -> None:
            self.seconds = 0.0

        def read(self) -> float:
            return self.seconds

        def advance(self, seconds: float) -> None:
            self.seconds += seconds

    class TimedStore:
        def __init__(
            self, store: TrustedLocalCircadianCheckpointStore, clock: TrainingClock, interrupt: bool
        ) -> None:
            self.store = store
            self.clock = clock
            self.interrupt = interrupt

        def load(self) -> Any:
            return self.store.load()

        def save(self, checkpoint: Any) -> None:
            self.store.save(checkpoint)
            self.clock.advance(0.25)
            if self.interrupt and checkpoint.combined.position.stage == "before_sleep":
                raise InterruptedMemoryRun()

    config = replace(_config(), epochs=4, circadian_sleep_interval=1)
    device = torch.device("cpu")
    train = (
        (torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]]), torch.tensor([1, 0])),
        (torch.tensor([[0.1, 0.2, -0.4], [0.3, -0.2, 0.1]]), torch.tensor([0, 1])),
    )
    guard = ((torch.tensor([[0.2, 0.1, -0.1]]), torch.tensor([1])),)
    validation = ((torch.tensor([[0.1, -0.2, 0.3]]), torch.tensor([0])),)

    def run(
        clock: TrainingClock, store: TimedStore, resume: bool
    ) -> tuple[Any, CircadianPredictiveCodingHead]:
        head = CircadianPredictiveCodingHead(
            feature_dim=3,
            hidden_dim=16,
            num_classes=2,
            device=device,
            seed=251,
            config=benchmark._build_circadian_head_config(config),
            min_hidden_dim=16,
            max_hidden_dim=16,
        )
        original_step = head.train_step

        def timed_step(**kwargs: Any) -> Any:
            result = original_step(**kwargs)
            clock.advance(1.0)
            return result

        setattr(head, "train_step", timed_step)
        outcome = benchmark._train_with_checkpoint_memory_telemetry(
            torch,
            device,
            lambda sampler, _cuda_start: benchmark._train_circadian_head(
                torch,
                device,
                head,
                train,
                guard,
                validation,
                config,
                time_budget_seconds=5.0,
                clock=clock.read,
                checkpoint_store=store,
                resume_from_checkpoint=resume,
                checkpoint_protocol_id=benchmark.THREE_HEAD_FIXED_FEATURE_WALL_TIME_CHECKPOINT_MEMORY_PROTOCOL,
                memory_sampler=sampler,
            ),
        )
        return outcome, head

    control_clock = TrainingClock()
    control_store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall-control.ckpt")
    control, control_head = run(
        control_clock, TimedStore(control_store, control_clock, False), False
    )
    assert control.stop_reason == "deadline"
    assert control.wake_batches == 5

    interrupted_clock = TrainingClock()
    interrupted_store = TrustedLocalCircadianCheckpointStore(tmp_path / "wall-resume.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        run(interrupted_clock, TimedStore(interrupted_store, interrupted_clock, True), False)
    assert interrupted_store.load().combined.position.stage == "before_sleep"
    resumed_clock = TrainingClock()
    resumed, resumed_head = run(
        resumed_clock, TimedStore(interrupted_store, resumed_clock, False), True
    )
    assert resumed.stop_reason == "deadline"
    assert resumed.wake_batches == control.wake_batches
    assert resumed.train_seconds == control.train_seconds == 5.0
    assert benchmark._hash_trained_head(resumed_head) == benchmark._hash_trained_head(control_head)
    assert len(control.process_rss_segments) == 1
    assert len(resumed.process_rss_segments) == 2
    assert resumed.process_rss_peak_observed_bytes == max(
        segment.peak_bytes for segment in resumed.process_rss_segments
    )
