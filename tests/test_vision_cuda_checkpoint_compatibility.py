"""Older CPU vision payloads remain valid after optional CUDA fields arrive."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.app.resnet50_benchmark import (  # noqa: E402
    ResNet50BenchmarkConfig,
    run_resnet50_benchmark,
)
from src.app.vision_checkpoint import (  # noqa: E402
    capture_vision_checkpoint,
    validate_vision_checkpoint,
)
from src.infra.circadian_checkpoint_files import (  # noqa: E402
    TrustedLocalVisionCheckpointStore,
)


def test_cpu_vision_checkpoint_without_new_cuda_fields_still_validates() -> None:
    config = ResNet50BenchmarkConfig(device="cpu")
    order = ("backprop", "predictive", "circadian")
    checkpoint = capture_vision_checkpoint(
        config=config,
        data_digest="development-roles",
        training_order=order,
        completed_outcomes=(),
        completed_hashes=(),
        torch=torch,
    )
    object.__delattr__(checkpoint, "torch_cuda_device")
    object.__delattr__(checkpoint, "torch_cuda_random_state")
    older_payload = pickle.loads(pickle.dumps(checkpoint))

    validate_vision_checkpoint(
        older_payload,
        config=config,
        data_digest="development-roles",
        training_order=order,
        torch=torch,
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only runtime required")
def test_cuda_vision_checkpoint_reports_unavailable_device_before_data(tmp_path: Any) -> None:
    config = ResNet50BenchmarkConfig(device="cuda:0")
    store = TrustedLocalVisionCheckpointStore(tmp_path / "unavailable.ckpt")

    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        run_resnet50_benchmark(config, checkpoint_store=store)
