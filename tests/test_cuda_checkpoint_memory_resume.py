"""Actual-device CUDA allocator segments survive a trusted-file restart."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")

from src.app import matched_head_benchmark as benchmark  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
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
        device="cuda:0",
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


_WORKER = """
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch
from src.app.matched_head_benchmark import (
    run_three_head_fixed_feature_benchmark,
    run_three_head_fixed_feature_wall_time_benchmark,
    run_three_head_fixed_width_capacity_benchmark,
)
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig
from src.infra.circadian_checkpoint_files import TrustedLocalCircadianCheckpointStore

route, mode, config_path, checkpoint_path = sys.argv[1:]
torch.use_deterministic_algorithms(True)
config = ResNet50BenchmarkConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
store = TrustedLocalCircadianCheckpointStore(Path(checkpoint_path))

def run(checkpoint_store, resume=False):
    if route == "capacity":
        return run_three_head_fixed_width_capacity_benchmark(
            config, checkpoint_store=checkpoint_store,
            resume_from_checkpoint=resume, checkpoint_memory=True,
        )
    if route == "wall":
        return run_three_head_fixed_feature_wall_time_benchmark(
            config, wall_time_budget_seconds=1.0, measure_memory=True,
            checkpoint_store=checkpoint_store, resume_from_checkpoint=resume,
        )
    return run_three_head_fixed_feature_benchmark(
        config, measure_memory=True, checkpoint_store=checkpoint_store,
        resume_from_checkpoint=resume,
    )

class StopAfterCheckpoint(Exception):
    pass

class InterruptingStore:
    def load(self):
        return store.load()

    def save(self, checkpoint):
        store.save(checkpoint)
        stage = "wake" if route == "wall" else "before_sleep"
        if checkpoint.combined.position.stage == stage:
            raise StopAfterCheckpoint()

if mode == "interrupt":
    try:
        run(InterruptingStore())
    except StopAfterCheckpoint:
        saved = store.load()
        print(json.dumps({"pid": os.getpid(), "stage": saved.combined.position.stage,
                          "rss": saved.memory_segments[-1].__dict__,
                          "cuda": saved.cuda_allocator_segments[-1].__dict__}))
    else:
        raise AssertionError("The first process did not stop at the checkpoint")
else:
    result = run(store, resume=mode == "resume")
    reports = {name: getattr(result, name) for name in
               ("backprop", "predictive_coding", "circadian")}
    def non_timing_state(report):
        state = asdict(report)
        # Why this: checkpoint and process boundaries change elapsed seconds,
        # while every semantic sleep fact must remain exact.
        for event in state.get("sleep_events", []):
            event.pop("durations")
        for field in (
            "train_seconds", "deadline_overshoot_seconds", "process_rss_start_bytes",
            "process_rss_peak_observed_bytes", "process_rss_samples", "process_rss_segments",
            "cuda_allocated_start_bytes", "cuda_allocated_peak_bytes",
            "cuda_reserved_peak_bytes", "cuda_allocator_segments",
        ):
            state.pop(field)
        return state
    print(json.dumps({"pid": os.getpid(), "protocol": result.protocol_id,
                      "scope": result.memory_observation_scope,
                      "capacity": asdict(result.capacity_control) if result.capacity_control else None,
                      "hashes": result.trained_head_hashes,
                      "reports": {name: {"rss": [segment.__dict__ for segment in report.process_rss_segments],
                                         "cuda": [segment.__dict__ for segment in report.cuda_allocator_segments],
                                         "state": non_timing_state(report),
                                         "allocated_start": report.cuda_allocated_start_bytes,
                                         "allocated_peak": report.cuda_allocated_peak_bytes,
                                         "reserved_peak": report.cuda_reserved_peak_bytes,
                                         "sleep_attempts": report.sleep_attempts,
                                         "rollbacks": report.total_rollbacks,
                                         "stop_reason": report.stop_reason,
                                         "train_seconds": report.train_seconds,
                                         "test_accuracy": report.test_accuracy}
                                  for name, report in reports.items()}}))
"""


def _run_worker(
    mode: str, config_path: Path, checkpoint_path: Path, *, route: str = "fixed"
) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, route, mode, str(config_path), str(checkpoint_path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=35,
        check=False,
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_cuda_checkpoint_memory_keeps_two_process_allocator_segments(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(_config())), encoding="utf-8")
    checkpoint_path = tmp_path / "capacity.ckpt"

    interrupted = _run_worker("interrupt", config_path, checkpoint_path)
    resumed = _run_worker("resume", config_path, checkpoint_path)
    control = _run_worker("control", config_path, tmp_path / "control.ckpt")

    assert interrupted["stage"] == "before_sleep"
    assert interrupted["rss"]["pid"] == interrupted["pid"]
    assert interrupted["cuda"]["pid"] == interrupted["pid"]
    assert len({interrupted["pid"], resumed["pid"], control["pid"]}) == 3
    assert resumed["protocol"] == "vision_three_head_fixed_feature_cuda_checkpoint_memory_v2"
    assert resumed["scope"] == "committed_head_training_segments_absolute_rss_cuda_allocator"
    assert resumed["hashes"] == control["hashes"]
    circadian = resumed["reports"]["circadian"]
    assert circadian["cuda"][0] == interrupted["cuda"]
    assert [segment["pid"] for segment in circadian["cuda"]] == [
        interrupted["pid"],
        resumed["pid"],
    ]
    assert [segment["pid"] for segment in circadian["rss"]] == [
        interrupted["pid"],
        resumed["pid"],
    ]
    assert circadian["allocated_start"] is None
    assert circadian["allocated_peak"] == max(
        segment["allocated_peak_bytes"] for segment in circadian["cuda"]
    )
    assert circadian["reserved_peak"] == max(
        segment["reserved_peak_bytes"] for segment in circadian["cuda"]
    )
    assert len(circadian["state"]["sleep_events"]) == 1
    assert circadian["state"]["sleep_events"][0]["guard"]["role"] == "inner_guard"
    assert circadian["state"]["sleep_events"][0]["guard"]["examples_scored"] == 16
    assert resumed["reports"]["backprop"]["state"]["sleep_events"] == []
    assert resumed["reports"]["predictive_coding"]["state"]["sleep_events"] == []
    for name, report in resumed["reports"].items():
        assert report["state"] == control["reports"][name]["state"]
        assert report["sleep_attempts"] == control["reports"][name]["sleep_attempts"]
        assert report["rollbacks"] == control["reports"][name]["rollbacks"]
        assert report["test_accuracy"] == control["reports"][name]["test_accuracy"]
        assert len(report["cuda"]) == (2 if name == "circadian" else 1)
        for segment in report["cuda"]:
            assert segment["device"] == "cuda:0"
            assert 0 <= segment["allocated_start_bytes"] <= segment["allocated_peak_bytes"]
            assert 0 <= segment["reserved_start_bytes"] <= segment["reserved_peak_bytes"]
            assert segment["reserved_peak_bytes"] >= segment["allocated_peak_bytes"]


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


def test_cuda_capacity_checkpoint_memory_keeps_fixed_width_across_processes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "capacity-config.json"
    config_path.write_text(json.dumps(asdict(_config())), encoding="utf-8")
    checkpoint_path = tmp_path / "capacity.ckpt"
    interrupted = _run_worker("interrupt", config_path, checkpoint_path, route="capacity")
    resumed = _run_worker("resume", config_path, checkpoint_path, route="capacity")
    control = _run_worker("control", config_path, tmp_path / "control.ckpt", route="capacity")

    assert (
        resumed["protocol"]
        == benchmark.THREE_HEAD_FIXED_WIDTH_CAPACITY_CUDA_CHECKPOINT_MEMORY_PROTOCOL
    )
    assert resumed["capacity"] == control["capacity"]
    assert resumed["capacity"] is not None
    assert (
        resumed["capacity"]["initial_head_parameters"]
        == resumed["capacity"]["final_head_parameters"]
    )
    assert resumed["scope"] == benchmark.CHECKPOINT_CUDA_MEMORY_SCOPE
    assert resumed["hashes"] == control["hashes"]
    for name, report in resumed["reports"].items():
        assert report["state"] == control["reports"][name]["state"]
    assert [segment["pid"] for segment in resumed["reports"]["circadian"]["cuda"]] == [
        interrupted["pid"],
        resumed["pid"],
    ]
    assert resumed["reports"]["circadian"]["sleep_attempts"] >= 1


def test_cuda_wall_time_checkpoint_memory_uses_remaining_deadline(tmp_path: Path) -> None:
    config = replace(_config(), epochs=1000)
    config_path = tmp_path / "wall-config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    checkpoint_path = tmp_path / "wall-time.ckpt"
    budget = 1.0
    interrupted = _run_worker("interrupt", config_path, checkpoint_path, route="wall")
    resumed = _run_worker("resume", config_path, checkpoint_path, route="wall")
    assert interrupted["stage"] == "wake"
    assert (
        resumed["protocol"]
        == benchmark.THREE_HEAD_FIXED_FEATURE_WALL_TIME_CUDA_CHECKPOINT_MEMORY_PROTOCOL
    )
    assert resumed["reports"]["circadian"]["stop_reason"] == "deadline"
    assert resumed["reports"]["circadian"]["train_seconds"] >= budget
    assert [segment["pid"] for segment in resumed["reports"]["circadian"]["cuda"]] == [
        interrupted["pid"],
        resumed["pid"],
    ]
    assert len(resumed["reports"]["circadian"]["rss"]) == 2
    assert all(report["stop_reason"] == "deadline" for report in resumed["reports"].values())


def test_bad_cuda_allocator_segments_reject_before_head_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TrustedLocalCircadianCheckpointStore(tmp_path / "bad-segments.ckpt")
    with pytest.raises(InterruptedMemoryRun):
        benchmark.run_three_head_fixed_feature_benchmark(
            _config(),
            checkpoint_store=InterruptingMemoryStore(store, "wake"),
            measure_memory=True,
        )
    saved = store.load()
    segment = saved.cuda_allocator_segments[0]
    monkeypatch.setattr(
        benchmark,
        "restore_circadian_checkpoint",
        lambda *args, **kwargs: pytest.fail("bad CUDA segment reached head restoration"),
    )
    bad_segments = (
        (),
        (replace(segment, allocated_peak_bytes=segment.allocated_start_bytes - 1),),
        (replace(segment, reserved_peak_bytes=segment.allocated_peak_bytes - 1),),
        (replace(segment, device="cuda:9"),),
        (replace(segment, pid=segment.pid + 1),),
    )
    for changed in bad_segments:
        store.save(replace(saved, cuda_allocator_segments=changed))
        with pytest.raises(ValueError, match="CUDA allocator segments"):
            benchmark.run_three_head_fixed_feature_benchmark(
                _config(), checkpoint_store=store, resume_from_checkpoint=True, measure_memory=True
            )
