"""Runtime helpers for optional torch/torchvision dependencies."""

from __future__ import annotations

import importlib
from typing import Any


def require_torch() -> Any:
    """Load torch lazily and fail with a clear installation message."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for ResNet benchmarks. Install requirements-resnet.txt first."
        ) from exc


def require_torchvision_models() -> Any:
    """Load torchvision models lazily."""
    try:
        return importlib.import_module("torchvision.models")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required for ResNet benchmarks. Install requirements-resnet.txt first."
        ) from exc


def require_torchvision_datasets() -> Any:
    """Load torchvision datasets lazily."""
    try:
        return importlib.import_module("torchvision.datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required for dataset benchmarks. Install requirements-resnet.txt first."
        ) from exc


def require_torchvision_transforms() -> Any:
    """Load torchvision transforms lazily."""
    try:
        return importlib.import_module("torchvision.transforms")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required for dataset benchmarks. Install requirements-resnet.txt first."
        ) from exc


def sync_device(torch_module: Any, device: Any) -> None:
    """Synchronize CUDA device timing when available."""
    if str(device).startswith("cuda"):
        torch_module.cuda.synchronize(device=device)
