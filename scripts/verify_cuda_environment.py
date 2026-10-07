"""Run P1.7f's seeded, synthetic-only CUDA device smoke.

This checks the isolated CUDA runtime and a torchvision model path. It does
not load datasets, train a benchmark head, or read final-test examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUEST_PATH = REPO_ROOT / "data" / "cuda-env-seed109-retry1-request.json"
RESULT_PATH = REPO_ROOT / "data" / "cuda-env-seed109-retry1-result.json"
FAILURE_PATH = REPO_ROOT / "data" / "cuda-env-seed109-retry1-failure.json"
SEED = 109
LIMIT_SECONDS = 60
EXPECTED_TORCH = "2.14.0+cu130"
EXPECTED_TORCHVISION = "0.29.0+cu130"


@dataclass(frozen=True)
class SmokeRequest:
    seed: int
    limit_seconds: int
    expected_torch: str
    expected_torchvision: str
    device_index: int
    input_shape: tuple[int, int, int, int]
    operations: tuple[str, ...]
    dataset_access: bool


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _tensor_digest(value: Any) -> str:
    return sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    if any(path.exists() for path in (REQUEST_PATH, RESULT_PATH, FAILURE_PATH)):
        raise FileExistsError("CUDA environment smoke already has an artifact")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("P1.7f requires an isolated Python 3.11 environment")

    import torch
    import torchvision

    request = SmokeRequest(
        seed=SEED,
        limit_seconds=LIMIT_SECONDS,
        expected_torch=EXPECTED_TORCH,
        expected_torchvision=EXPECTED_TORCHVISION,
        device_index=0,
        input_shape=(2, 3, 32, 32),
        operations=("conv2d_forward_backward", "untrained_resnet50_forward"),
        dataset_access=False,
    )
    _save_json(REQUEST_PATH, asdict(request))
    started = monotonic()
    try:
        if torch.__version__ != EXPECTED_TORCH or torchvision.__version__ != EXPECTED_TORCHVISION:
            raise RuntimeError(
                "CUDA Torch and torchvision versions differ from the predeclared pair"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the isolated Torch environment")
        device = torch.device("cuda:0")
        # Why this: initialize the selected context before querying allocator telemetry.
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(0)
        free_before, total_bytes = torch.cuda.mem_get_info(0)

        convolution = torch.nn.Conv2d(3, 4, kernel_size=3).to(device)
        images = torch.randn(request.input_shape, device=device)
        convolution_output = convolution(images)
        loss = convolution_output.square().mean()
        loss.backward()
        if convolution.weight.grad is None or not torch.isfinite(convolution.weight.grad).all():
            raise AssertionError("CUDA convolution produced missing or nonfinite gradients")

        model = torchvision.models.resnet50(weights=None).to(device).eval()
        with torch.no_grad():
            logits = model(images)
        torch.cuda.synchronize(device)
        if logits.shape != (2, 1000) or not torch.isfinite(logits).all():
            raise AssertionError("CUDA ResNet-50 produced invalid synthetic logits")
        elapsed = monotonic() - started
        if elapsed > LIMIT_SECONDS:
            raise TimeoutError(f"CUDA environment smoke exceeded {LIMIT_SECONDS} s")
        free_after, _ = torch.cuda.mem_get_info(0)
        result = {
            "request": asdict(request),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "device_total_bytes": total_bytes,
            "device_free_before_bytes": free_before,
            "device_free_after_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "convolution_output_sha256": _tensor_digest(convolution_output),
            "convolution_gradient_sha256": _tensor_digest(convolution.weight.grad),
            "resnet_logits_sha256": _tensor_digest(logits),
            "convolution_loss": loss.item(),
            "elapsed_seconds": round(elapsed, 3),
        }
        _save_json(RESULT_PATH, result)
    except Exception as error:
        _save_json(
            FAILURE_PATH,
            {
                "request": asdict(request),
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(monotonic() - started, 3),
            },
        )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
