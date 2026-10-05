"""Bind original inventory/training evidence and prospective retention sources.

Inputs are complete current original artifact/source/request identities and
an unchanged full training reader port. Output is a complete repeated owned
retention ledger with fresh original reader authority. No source/model/train,
scored reader, final view, resource profiling or original parent completion.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_retention_costs import INVENTORY_ID
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_resource_artifacts import artifact_paths as inventory_paths
from src.infra.continual_confirmation_resource_bindings import (
    checked_resource_request,
    current_resource_sources,
)
from src.infra.continual_confirmation_retention_references import read_retention_references
from src.infra.continual_confirmation_training_references import (
    CompletedTrainingReader,
    stream_file_identity,
)


PROTOCOL_ID = "p610_original_checkpoint_owned_retention_costs_v1"
# Why this: fresh complete training readers and whole decoded-byte checks
# cost more than inventory preservation. Prior complete cost/report operations
# took 75/178 seconds. Declare this local derivative budget before fixtures.
VALIDATION_SECONDS = 180
INVENTORY_SOURCE_MAP_SHA256 = "3b16adb6caab2d76d537217fad6792ef8899dc431b955001fd24a85b4322b4f5"
APP_SOURCES = {
    "src/app/continual_confirmation_retention_checkpoints.py": "969655375a0f95cef30678bd1073616e4a3efbf53444bc47d1bb93bb0029db54",
    "src/app/continual_confirmation_retention_costs.py": "b7d8306560a71193975728249f2781e2260cd3f249a0fc065cd98606e15bd3b8",
    "src/app/continual_confirmation_retention_rendering.py": "e2417370dfccc7749621e8ec3114aaf66f2e9d6e823a273775cf58abdffe4d6a",
}
OWN_SOURCES = (
    "src/infra/continual_confirmation_retention_references.py",
    "src/infra/continual_confirmation_retention_bindings.py",
    "src/infra/continual_confirmation_retention_artifacts.py",
    "scripts/run_p610_retention_costs.py",
)
HANDOFF_FILE = "artifacts/runs/p610-resource-inventory-handoff-validation.json"
HANDOFF_ID = {
    "byte_count": 45746,
    "sha256": "b70b9b2d9db8fe9a208485aae6a8868230b617f062fe88f6fcc367f3c8616932",
}


def current_retention_sources(root: Path) -> dict[str, str]:
    sources = current_resource_sources(root)
    same_json(
        digest_json(sources), INVENTORY_SOURCE_MAP_SHA256, "retention original inventory sources"
    )
    require(all(len(value) == 64 for value in APP_SOURCES.values()), "retention sources not frozen")
    sources.update(verify_source_files(root, APP_SOURCES))
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    require(len(sources) == 121, "retention complete source scope differs")
    return sources


def _inputs(root: Path, scope_file: Path) -> dict[str, Any]:
    same_json(
        stream_file_identity(root / HANDOFF_FILE),
        HANDOFF_ID,
        "retention prior inventory handoff bytes",
    )
    handoff = read_json(root / HANDOFF_FILE)
    same_json(
        digest_json(handoff["source_sha256"]),
        INVENTORY_SOURCE_MAP_SHA256,
        "retention handoff original source map",
    )
    requests, files = [], []
    for directory in (
        "artifacts/runs/p610-resource-inventory",
        "artifacts/runs/p610-resource-inventory-repeat",
    ):
        paths = inventory_paths(root / directory)
        require(
            not paths["claim"].exists() and not paths["failure"].exists(),
            "retention original inventory has failure/claim",
        )
        expected = handoff["inventory_files"][str((root / directory).resolve())]
        for name in ("request", "result", "markdown", "audit"):
            same_json(
                stream_file_identity(paths[name]),
                expected[name],
                f"retention original inventory {name}",
            )
        same_json(
            expected["result"], INVENTORY_ID, "retention original complete inventory identity"
        )
        request = checked_resource_request(root, paths, scope_file)
        same_json(
            request["source_sha256"],
            handoff["source_sha256"],
            "retention original current inventory sources",
        )
        requests.append(request)
        files.append({"directory": directory, "files": deepcopy(expected)})
    same_json(
        requests[0]["inputs"], requests[1]["inputs"], "retention original repeated inventory inputs"
    )
    return {
        "inventory_files": files,
        "inventory_handoff": dict(HANDOFF_ID),
        "original_inventory_inputs": deepcopy(requests[0]["inputs"]),
    }


def retention_request(
    root: Path, output_dir: Path, scope_file: Path, started_utc: str
) -> dict[str, Any]:
    require(type(started_utc) is str, "retention request time must be a string")
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except ValueError as error:
        raise ValueError("retention request time malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "retention request requires UTC",
    )
    sources = current_retention_sources(root)
    return {
        "schema_id": "p610_retention_costs_request_v1",
        "protocol_id": PROTOCOL_ID,
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "inputs": _inputs(root, scope_file),
        "scope": {
            "cells": 560,
            "checkpoints": 1680,
            "contexts": 60,
            "training_readers": 2,
            "projection_gaps": 300,
        },
        "validation_budget_seconds": VALIDATION_SECONDS,
        "command": [
            sys.executable,
            "-m",
            "scripts.run_p610_retention_costs",
            "--publish",
            "--output-dir",
            str(output_dir.resolve()),
        ],
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "started_utc": started_utc,
        "new_measurement_training_or_final_access_authorized": False,
        "original_P6_10_acceptance_complete": False,
    }


def checked_retention_request(
    root: Path, paths: dict[str, Path], scope_file: Path
) -> dict[str, Any]:
    before = stream_file_identity(paths["request"])
    request = read_json(paths["request"])
    same_json(canonical_body_identity(request), before, "retention canonical request bytes")
    same_json(
        request,
        retention_request(
            root, paths["request"].parent, scope_file, request.get("started_utc", "")
        ),
        "retention complete current request/source/input bindings",
    )
    same_json(
        stream_file_identity(paths["request"]), before, "retention request changed during bindings"
    )
    return request


def read_retention_inputs(
    root: Path, paths: dict[str, Path], scope_file: Path, reader: CompletedTrainingReader
) -> dict[str, Any]:
    request = checked_retention_request(root, paths, scope_file)
    inventory = read_json(
        root / "artifacts/runs/p610-resource-inventory/resource-inventory.result.json"
    )
    references = read_retention_references(root, reader, inventory)
    del inventory
    body = references["projection"]
    body["provenance"] = {
        "validation_scope": "current_source_bound_two_complete_unchanged_original_training_readers_and_full_retention_projection",
        "source_sha256": deepcopy(request["source_sha256"]),
        "source_map_sha256": request["source_map_sha256"],
        "inputs": deepcopy(request["inputs"]),
        "training_references": references["training_references"],
        "complete_original_training_readers": 2,
        "new_measurement_training_or_final_access": False,
    }
    same_json(
        checked_retention_request(root, paths, scope_file),
        request,
        "retention bindings changed during complete readers",
    )
    return body
