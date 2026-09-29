"""Confirm the saved P1.8k pretrained-CIFAR selection without retuning.

Every seed, scope, metric, and candidate comes from the digest-checked
validation manifest. This is the first stage allowed to read final test.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import md5, sha256
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torchvision.models import ResNet50_Weights  # noqa: E402

from src.app.repeated_head_confirmation import (  # noqa: E402
    RepeatedConfirmationManifest,
    RepeatedConfirmationResult,
    restore_confirmation_manifest,
    run_repeated_confirmation,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREFIX = "benchmark_cifar_pretrained_v1"
RESULT_PATH = ARTIFACT_DIR / f"{PREFIX}_result_smoke.json"
FAILURE_PATH = ARTIFACT_DIR / f"{PREFIX}_failure_smoke.json"
CONFIRMATION_LIMIT_SECONDS = 480
EXPECTED_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
EXPECTED_WEIGHT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
EXPECTED_WEIGHT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


def _read_json(name: str) -> dict[str, Any]:
    path = ARTIFACT_DIR / f"{PREFIX}_{name}_smoke.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _file_digest(path: Path, algorithm: Any) -> str:
    digest = algorithm()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verify_inputs(
    request: dict[str, Any],
    selection: dict[str, Any],
    manifest: RepeatedConfirmationManifest,
) -> None:
    archive = REPO_ROOT / "data" / "cifar-10-python.tar.gz"
    if (
        request["archive"] != str(archive)
        or request["archive_bytes"] != 170_498_071
        or request["archive_md5"] != EXPECTED_ARCHIVE_MD5
        or not archive.is_file()
        or archive.stat().st_size != request["archive_bytes"]
        or _file_digest(archive, md5) != EXPECTED_ARCHIVE_MD5
    ):
        raise ValueError("Saved CIFAR archive provenance is missing or changed")

    weight_url = ResNet50_Weights.IMAGENET1K_V2.url
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / weight_url.rsplit("/", 1)[-1]
    if (
        weight_url != EXPECTED_WEIGHT_URL
        or request["weight_url"] != weight_url
        or request["weight_file"] != str(weight_file)
        or request["weight_bytes"] != 102_540_417
        or request["weight_sha256"] != EXPECTED_WEIGHT_SHA256
        or not weight_file.is_file()
        or weight_file.stat().st_size != request["weight_bytes"]
        or _file_digest(weight_file, sha256) != EXPECTED_WEIGHT_SHA256
    ):
        raise ValueError("Saved ImageNet V2 weight provenance is missing or changed")

    source_digest = sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        source_digest != manifest.source_selection_digest
        or request["selection_seeds"] != list(manifest.selection_seeds)
        or request["confirmation_seeds"] != list(manifest.confirmation_seeds)
        or manifest.selection_seeds != (113,)
        or manifest.confirmation_seeds != (127, 131, 137)
        or request["base_config"] != asdict(manifest.base_config)
        or selection["base_config"] != request["base_config"]
        or selection["seeds"] != request["selection_seeds"]
        or selection["candidates_per_head"] != request["candidates_per_head"]
        or selection["confirmations"]
        or request["wall_time_budget_seconds"] != manifest.wall_time_budget_seconds
        or request["wall_time_epoch_cap"] != manifest.wall_time_epoch_cap
        or request["confirmation_limit_seconds"] != CONFIRMATION_LIMIT_SECONDS
    ):
        raise ValueError("Saved request, validation selection, and manifest disagree")
    if len(selection["attempts"]) != 6 or any(
        attempt["status"] != "complete" for attempt in selection["attempts"]
    ):
        raise ValueError("Saved validation trial ledger is incomplete")
    selected_ids = {item["head_name"]: item["candidate_id"] for item in selection["selections"]}
    if len(selected_ids) != 3 or len(selection["selections"]) != 3:
        raise ValueError("Saved validation selections are incomplete")
    for item in manifest.selected_heads:
        options = request["candidates"][item.head_name]
        matches = [option for option in options if option["candidate_id"] == item.candidate_id]
        if (
            selected_ids[item.head_name] != item.candidate_id
            or len(options) != 2
            or len(matches) != 1
            or matches[0]["config"] != asdict(item.config)
        ):
            raise ValueError(f"Saved selected candidate changed for {item.head_name}")


def _verify_result(
    result: RepeatedConfirmationResult,
    manifest: RepeatedConfirmationManifest,
) -> None:
    if result.manifest != manifest or len(result.fixed_data.confirmations) != 9:
        raise AssertionError("Fixed-data confirmation is incomplete or changed manifest")
    if len(result.wall_time) != 3 or len(result.capacity_memory) != 3:
        raise AssertionError("One declared wall-time or memory seed is missing")
    if any(len(report.reports) != 3 for report in result.capacity_memory):
        raise AssertionError("A process-memory report lacks one matched head")
    for scope in (result.fixed_data_accuracy, result.wall_time_accuracy, result.observed_train_rss):
        if set(scope) != {"backprop_mlp", "predictive_coding", "circadian_predictive_coding"}:
            raise AssertionError("Confirmation summary lacks a matched head")
        if any(summary.seeds != manifest.confirmation_seeds for summary in scope.values()):
            raise AssertionError("Confirmation summary changed the declared seeds")


def main() -> None:
    if RESULT_PATH.exists() or FAILURE_PATH.exists():
        raise FileExistsError("Pretrained-CIFAR confirmation already has a result or failure")
    request = _read_json("request")
    selection = _read_json("selection")
    manifest = restore_confirmation_manifest(_read_json("manifest"))
    _verify_inputs(request, selection, manifest)

    started = monotonic()
    try:
        result = run_repeated_confirmation(manifest)
        elapsed = monotonic() - started
        if elapsed > CONFIRMATION_LIMIT_SECONDS:
            raise TimeoutError(f"Confirmation exceeded {CONFIRMATION_LIMIT_SECONDS} s")
        _verify_result(result, manifest)
        _save_json(RESULT_PATH, asdict(result))
    except Exception as error:
        _save_json(
            FAILURE_PATH,
            {
                "manifest_digest": manifest.manifest_digest,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(monotonic() - started, 2),
            },
        )
        raise
    print(
        json.dumps(
            {
                "manifest_digest": manifest.manifest_digest,
                "confirmation_seeds": manifest.confirmation_seeds,
                "elapsed_seconds": round(elapsed, 2),
                "result_path": str(RESULT_PATH),
                "fixed_data_accuracy": {
                    name: asdict(summary) for name, summary in result.fixed_data_accuracy.items()
                },
                "wall_time_accuracy": {
                    name: asdict(summary) for name, summary in result.wall_time_accuracy.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
