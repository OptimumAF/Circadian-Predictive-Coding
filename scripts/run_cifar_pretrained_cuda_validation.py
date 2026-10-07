"""Freeze P1.8m's validation-only CUDA matched-head selection.

The CPU study's role sizes and equal candidate grid are reused as declared.
Only device and independent seed change. Final test is sealed throughout.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  # noqa: E402
import torchvision  # noqa: E402

from scripts import run_cifar_pretrained_validation as cpu_study  # noqa: E402
from src.app import matched_head_benchmark as matched  # noqa: E402
from src.app.matched_head_tuning import MatchedHeadTuningError, run_matched_head_tuning  # noqa: E402
from src.app.repeated_head_confirmation import (  # noqa: E402
    create_confirmation_manifest,
    restore_confirmation_manifest,
)

OUTPUT_DIR = REPO_ROOT / "artifacts"
PREFIX = "benchmark_cifar_pretrained_cuda_v1"
SELECTION_SEEDS = (151,)
CONFIRMATION_SEEDS = (157, 163, 167)
WALL_TIME_BUDGET_SECONDS = 0.5
WALL_TIME_EPOCH_CAP = 1000
SELECTION_LIMIT_SECONDS = 120


def _path(name: str) -> Path:
    return OUTPUT_DIR / f"{PREFIX}_{name}_smoke.json"


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    if any(_path(name).exists() for name in ("request", "selection", "manifest", "failure")):
        raise FileExistsError("CUDA selection already has an artifact")
    if torch.__version__ != "2.14.0+cu130" or torchvision.__version__ != "0.29.0+cu130":
        raise RuntimeError("P1.8m requires the verified isolated CUDA package pair")
    if not torch.cuda.is_available():
        raise RuntimeError("P1.8m requires an actual CUDA device")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("Predeclared CUBLAS workspace setting changed")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    inputs = cpu_study._verify_local_inputs()
    # Why this: copy the previous study's fixed data/head controls verbatim.
    base = replace(cpu_study._base_config(), seed=SELECTION_SEEDS[0], device="cuda")
    candidates = cpu_study._candidates(base)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(
        _path("request"),
        {
            **inputs,
            "selection_seeds": SELECTION_SEEDS,
            "confirmation_seeds": CONFIRMATION_SEEDS,
            "candidates_per_head": 2,
            "wall_time_budget_seconds": WALL_TIME_BUDGET_SECONDS,
            "wall_time_epoch_cap": WALL_TIME_EPOCH_CAP,
            "selection_limit_seconds": SELECTION_LIMIT_SECONDS,
            "confirmation_limit_seconds": 480,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "base_config": asdict(base),
            "candidates": {
                name: [asdict(candidate) for candidate in options]
                for name, options in candidates.items()
            },
        },
    )

    original_build = matched._build_benchmark_loaders
    test_iterations = [0]

    class SealedTestLoader:
        def __iter__(self) -> Any:
            test_iterations[0] += 1
            raise AssertionError("CUDA final test opened during validation selection")

    def build_sealed_loaders(config: Any) -> Any:
        loaders = original_build(config)
        return replace(loaders, test_loader=SealedTestLoader())

    started = monotonic()
    try:
        with patch.object(matched, "_build_benchmark_loaders", build_sealed_loaders):
            selection = run_matched_head_tuning(
                base,
                candidates,
                seeds=SELECTION_SEEDS,
                candidates_per_head=2,
                confirm_test=False,
            )
        elapsed = monotonic() - started
        if elapsed > SELECTION_LIMIT_SECONDS:
            raise TimeoutError(f"CUDA selection exceeded {SELECTION_LIMIT_SECONDS} s")
        if test_iterations[0] != 0:
            raise AssertionError("CUDA final test was iterated during selection")
        cpu_study._verify_equal_inputs(selection)
        _save_json(_path("selection"), asdict(selection))
        manifest = create_confirmation_manifest(
            selection,
            confirmation_seeds=CONFIRMATION_SEEDS,
            wall_time_budget_seconds=WALL_TIME_BUDGET_SECONDS,
            wall_time_epoch_cap=WALL_TIME_EPOCH_CAP,
        )
        _save_json(_path("manifest"), asdict(manifest))
        if restore_confirmation_manifest(json.loads(_path("manifest").read_text())) != manifest:
            raise AssertionError("Saved CUDA manifest did not round-trip")
    except Exception as error:
        failure: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": round(monotonic() - started, 3),
            "final_test_iterations": test_iterations[0],
        }
        if isinstance(error, MatchedHeadTuningError):
            failure["attempts"] = [asdict(item) for item in error.attempts]
            failure["trials"] = [asdict(item) for item in error.trials]
        _save_json(_path("failure"), failure)
        raise
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "attempts": len(selection.attempts),
                "selection_seeds": SELECTION_SEEDS,
                "confirmation_seeds": CONFIRMATION_SEEDS,
                "selections": {item.head_name: item.candidate_id for item in selection.selections},
                "selection_seconds": round(elapsed, 3),
                "final_test_iterations": test_iterations[0],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
