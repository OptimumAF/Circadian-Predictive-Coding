"""Run the saved, validation-selected real-CIFAR matched-head confirmation.

Inputs are P1.8h's request, selection, and manifest JSON files plus the local
verified CIFAR-10 archive. Output is one complete JSON result or a failure
record. Seeds, metrics, and scopes come only from the saved manifest.
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

from src.app.repeated_head_confirmation import (  # noqa: E402
    restore_confirmation_manifest,
    run_repeated_confirmation,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREFIX = "benchmark_cifar_v2"
RESULT_PATH = ARTIFACT_DIR / f"{PREFIX}_result_retry2_smoke.json"
FAILURE_PATH = ARTIFACT_DIR / f"{PREFIX}_failure_retry2_smoke.json"
WALL_BUDGET_SECONDS = 240


def _read_json(name: str) -> dict[str, Any]:
    path = ARTIFACT_DIR / f"{PREFIX}_{name}_smoke.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _verify_archive(request: dict[str, Any]) -> None:
    archive = Path(request["archive"])
    if not archive.is_file() or archive.stat().st_size != request["archive_bytes"]:
        raise FileNotFoundError("Predeclared CIFAR archive is missing or changed size")
    digest = md5()
    with archive.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != request["archive_md5"]:
        raise ValueError("Predeclared CIFAR archive content changed")


def main() -> None:
    if RESULT_PATH.exists() or FAILURE_PATH.exists():
        raise FileExistsError("CIFAR confirmation already has a result or failure record")
    request = _read_json("request")
    selection = _read_json("selection")
    manifest = restore_confirmation_manifest(_read_json("manifest"))
    if (
        tuple(request["selection_seeds"]) != manifest.selection_seeds
        or tuple(request["confirmation_seeds"]) != manifest.confirmation_seeds
        or manifest.confirmation_seeds != (83, 89, 97)
        or selection["confirmations"]
    ):
        raise ValueError("Saved CIFAR request, selection, and manifest disagree")
    source_digest = sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if source_digest != manifest.source_selection_digest:
        raise ValueError("Saved CIFAR selection changed after manifest creation")
    _verify_archive(request)

    started = monotonic()
    try:
        result = run_repeated_confirmation(manifest)
        elapsed = monotonic() - started
        if elapsed > WALL_BUDGET_SECONDS:
            raise TimeoutError(f"CIFAR confirmation exceeded {WALL_BUDGET_SECONDS} s")
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
