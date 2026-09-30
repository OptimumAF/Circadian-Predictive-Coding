"""Bounded CPU readback for public fixed-feature, capacity, and checkpoint APIs."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.resnet50_benchmark import ResNet50BenchmarkConfig  # noqa: E402
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalCircadianCheckpointStore,
)
from src.shared.process_memory import read_process_rss_bytes  # noqa: E402


HEADS = {"backprop_mlp", "predictive_coding", "circadian_predictive_coding"}
ROLES = {"train", "guard", "validation", "test"}


def _config() -> ResNet50BenchmarkConfig:
    return ResNet50BenchmarkConfig(
        train_samples=4,
        guard_samples=4,
        validation_samples=4,
        test_samples=4,
        num_classes=3,
        image_size=32,
        batch_size=4,
        epochs=1,
        seed=47,
        device="cpu",
        dataset_download=False,
        target_accuracy=None,
        backprop_freeze_backbone=True,
        backbone_weights="none",
        predictive_head_hidden_dim=8,
        predictive_inference_steps=1,
        circadian_head_hidden_dim=8,
        circadian_min_hidden_dim=8,
        circadian_max_hidden_dim=8,
        circadian_inference_steps=1,
        circadian_sleep_interval=1,
        circadian_force_sleep=True,
        circadian_sleep_warmup_steps=0,
        circadian_use_adaptive_sleep_trigger=False,
        circadian_enable_sleep_rollback=True,
    )


def _read_result(result: matched.ThreeHeadFixedFeatureResult) -> dict[str, Any]:
    # Why this: these public APIs return dataclasses; JSON round-trip checks their
    # complete portable report surface without inventing a new file producer.
    payload = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert set(payload["initial_head_hashes"]) == HEADS
    assert set(payload["trained_head_hashes"]) == HEADS
    assert len(set(payload["initial_head_hashes"].values())) == 1
    assert set(payload["split_hashes"]) == set(payload["feature_hashes"]) == ROLES
    assert all(len(value) == 64 for value in payload["feature_hashes"].values())
    assert len(payload["backbone_hash"]) == 64
    assert payload["benchmark_track"] == "frozen_shared_representation"
    assert payload["backbone_weights"] == payload["backbone_pretraining"] == "none"
    assert payload["training_order"] == [
        "backprop_mlp",
        "predictive_coding",
        "circadian_predictive_coding",
    ]
    assert payload["feature_bytes"] > 0
    for name in ("backprop", "predictive_coding", "circadian"):
        report = payload[name]
        assert report["epochs_ran"] == 1
        assert report["seen_samples"] == 4
        assert report["wake_batches"] == 1
        assert report["benchmark_track"] == "frozen_shared_representation"
        assert report["trainable_parameters"] > 0
        assert report["replay_examples"] == 0
        assert report["cuda_allocated_peak_bytes"] is None
        assert report["cuda_reserved_peak_bytes"] is None
    assert payload["backprop"]["sleep_events"] == []
    assert payload["predictive_coding"]["sleep_events"] == []
    return payload


def test_should_read_fixed_feature_control_capacity_and_trusted_cpu_checkpoint(
    tmp_path: Path,
) -> None:
    capacity_config = _config()
    config = replace(capacity_config, circadian_max_hidden_dim=16)
    forced = _read_result(matched.run_three_head_fixed_feature_benchmark(config))
    disabled = _read_result(
        matched.run_three_head_fixed_feature_benchmark(
            replace(config, circadian_force_sleep=False, circadian_sleep_mode="disabled")
        )
    )
    assert (
        forced["protocol_id"] == disabled["protocol_id"] == ("vision_three_head_fixed_feature_v1")
    )
    assert forced["split_hashes"] == disabled["split_hashes"]
    assert forced["feature_hashes"] == disabled["feature_hashes"]
    assert forced["backbone_hash"] == disabled["backbone_hash"]
    assert forced["initial_head_hashes"] == disabled["initial_head_hashes"]
    assert forced["circadian"]["sleep_attempts"] == 1
    assert disabled["circadian"]["sleep_attempts"] == 0
    forced_event = forced["circadian"]["sleep_events"][0]
    disabled_event = disabled["circadian"]["sleep_events"][0]
    assert (forced_event["trigger_reason"], forced_event["outcome"]) == ("periodic", "accepted")
    assert forced_event["guard"]["role"] == "inner_guard"
    assert disabled_event["reason"] == "sleep_disabled"
    assert disabled_event["outcome"] == "skipped"

    capacity = _read_result(matched.run_three_head_fixed_width_capacity_benchmark(capacity_config))
    assert capacity["protocol_id"] == "vision_three_head_fixed_width_capacity_memory_v1"
    assert capacity["feature_hashes"] == forced["feature_hashes"]
    assert capacity["initial_head_hashes"] == forced["initial_head_hashes"]
    assert capacity["capacity_control"]["mode"] == "fixed_width_parameter_matched"
    assert len(set(capacity["capacity_control"]["initial_head_parameters"].values())) == 1
    assert len(set(capacity["capacity_control"]["final_head_parameters"].values())) == 1
    assert capacity["circadian"]["sleep_attempts"] == 1
    assert capacity["circadian"]["total_splits"] == 0
    assert capacity["circadian"]["total_prunes"] == 0
    if read_process_rss_bytes() is not None:
        for name in ("backprop", "predictive_coding", "circadian"):
            report = capacity[name]
            assert report["process_rss_start_bytes"] > 0
            assert report["process_rss_peak_observed_bytes"] >= report["process_rss_start_bytes"]

    checkpoint_path = tmp_path / "fixed-feature.ckpt"
    store = TrustedLocalCircadianCheckpointStore(checkpoint_path)
    checkpointed = _read_result(
        matched.run_three_head_fixed_feature_benchmark(config, checkpoint_store=store)
    )
    checkpoint = store.load()  # Only the checkpoint just created by this test is trusted.
    assert checkpoint_path.is_file()
    assert checkpoint.format_version == 2
    assert checkpoint.protocol_id == checkpointed["protocol_id"]
    assert (
        checkpoint.initial_head_hash
        == checkpointed["initial_head_hashes"]["circadian_predictive_coding"]
    )
    # Why this: the trusted training cursor seals development features only;
    # final-test features appear in the completed public result after training.
    assert set(dict(checkpoint.feature_hashes)) == {"train", "guard", "validation"}
    assert dict(checkpoint.feature_hashes) == {
        role: checkpointed["feature_hashes"][role] for role in ("train", "guard", "validation")
    }
    assert checkpoint.combined.position.stage == "after_sleep"
    assert checkpoint.combined.position.completed_epoch == 1
    assert checkpoint.progress.seen_samples == 4
    assert checkpoint.progress.sleep_attempts == 1
    assert len(checkpoint.sleep_events) == 1
    resumed = _read_result(
        matched.run_three_head_fixed_feature_benchmark(
            config, checkpoint_store=store, resume_from_checkpoint=True
        )
    )
    assert resumed["feature_hashes"] == checkpointed["feature_hashes"]
    assert resumed["trained_head_hashes"] == checkpointed["trained_head_hashes"]
    assert resumed["circadian"]["sleep_attempts"] == 1
    with pytest.raises(ValueError, match="checkpoint config or features"):
        matched.run_three_head_fixed_feature_benchmark(
            replace(config, seed=48), checkpoint_store=store, resume_from_checkpoint=True
        )

    corrupt_path = tmp_path / "corrupt.ckpt"
    raw = checkpoint_path.read_bytes()
    corrupt_path.write_bytes(raw[:-1] + bytes((raw[-1] ^ 1,)))
    with pytest.raises(ValueError, match="checksum"):
        TrustedLocalCircadianCheckpointStore(corrupt_path).load()
