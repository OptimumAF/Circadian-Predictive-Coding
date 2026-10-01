"""Bind current original resource evidence and reconstruct its full inventory.

Inputs are fixed complete saved report/cost/audit references and current
sources. Output proves current byte preservation and inventory consistency,
with recorded original complete readback authority. This is not another
scientific reader execution, model/data access or resource measurement.
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
from src.app.continual_confirmation_resources import build_resource_inventory, REPORT_ID
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_report_artifacts import artifact_paths as report_paths
from src.infra.continual_confirmation_report_bindings import (
    checked_report_request,
    current_report_sources,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


PROTOCOL_ID = "p610_original_resource_field_scope_inventory_v1"
VALIDATION_SECONDS = 120
REPORT_SOURCE_MAP_SHA256 = "67063e2b13329d8fb86654cbb4acf4c4b348db0d98e1f9e7abeca5e265058517"
APP_SOURCES: dict[str, str] = {
    "src/app/continual_confirmation_matrix_inputs.py": "f5649c2639a611a59a10a6a96bd3ab2e489014e33d5eab33204bbfb09a8cb100",
    "src/app/continual_confirmation_resource_contexts.py": "ea92a26f4d229ce6686acad4b16ec0a7003e08e8f53d6117fa7f10b1fc7141d7",
    "src/app/continual_confirmation_resource_fields.py": "4cb9f82ad9b7f7786d6c766cc0aab88c241c62aa164cda521de645f80a68453d",
    "src/app/continual_confirmation_resources.py": "ff25c71bea23ce9854c280a65ff460a21786d5f34b4fcea062abe6e4b27447c2",
    "src/app/continual_confirmation_resource_rendering.py": "0b7757336640695b63e31ceafc8b02b33b1d822839031960da75d4e4eb8e65cf",
}
OWN_SOURCES = (
    "src/infra/continual_confirmation_resource_bindings.py",
    "src/infra/continual_confirmation_resource_artifacts.py",
    "scripts/run_p610_resource_inventory.py",
)
REPORT_FILES = (
    (
        "artifacts/runs/p611-confirmation-report",
        {
            "request": {
                "byte_count": 26104,
                "sha256": "57db44ded2fdc9a0011dac61ac8f0b11e55f6e831df21215ba2f98003054aff7",
            },
            "result": REPORT_ID,
            "markdown": {
                "byte_count": 2958568,
                "sha256": "c989df16853868e910bf99303470927d17c6f01ed86e5dd004d35692bf03fddc",
            },
            "audit": {
                "byte_count": 4474,
                "sha256": "5bd1f8d641b7658ef661ded7a633e1ce5a04026b64380215e91b1832f3585486",
            },
        },
    ),
    (
        "artifacts/runs/p611-confirmation-report-repeat",
        {
            "request": {
                "byte_count": 26111,
                "sha256": "4e5bb5956f4a8d35274054bf08ec5cf32349ca43e999e93c1fa6ec4d9fa1733e",
            },
            "result": REPORT_ID,
            "markdown": {
                "byte_count": 2958568,
                "sha256": "c989df16853868e910bf99303470927d17c6f01ed86e5dd004d35692bf03fddc",
            },
            "audit": {
                "byte_count": 4475,
                "sha256": "0d6b2fd124c7029b52d86623533c6bef5ecac51cbdffae448c8a1fe777d0a31a",
            },
        },
    ),
)
RECORDED_VALIDATIONS = {
    "artifacts/runs/p611-confirmation-cost-validation.json": {
        "byte_count": 15682,
        "sha256": "eac94f8e95144d52e389cacf61ad1351ce5b71c07eadb972cfdc8b2cec889e37",
    },
    "artifacts/runs/p611-confirmation-report-validation.json": {
        "byte_count": 17011,
        "sha256": "031f398f03115cc2967be7b4c6a571ac5d8c822910bc0bd55ce74f500495a9d8",
    },
}


def current_resource_sources(root: Path) -> dict[str, str]:
    sources = current_report_sources(root)
    same_json(
        digest_json(sources), REPORT_SOURCE_MAP_SHA256, "resource original complete report sources"
    )
    require(all(len(value) == 64 for value in APP_SOURCES.values()), "resource sources not frozen")
    sources.update(verify_source_files(root, APP_SOURCES))
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    require(len(sources) == 114, "resource complete source scope differs")
    return sources


def _inputs(root: Path, scope_file: Path) -> dict[str, Any]:
    references, requests = [], []
    for directory, identities in REPORT_FILES:
        paths = report_paths(root / directory)
        require(
            not paths["claim"].exists() and not paths["failure"].exists(),
            "resource input report has claim/failure",
        )
        for name, identity in identities.items():
            same_json(
                stream_file_identity(paths[name]), identity, f"resource original report {name}"
            )
        requests.append(checked_report_request(root, paths, scope_file))
        references.append({"directory": directory, "files": deepcopy(identities)})
    same_json(requests[0]["inputs"], requests[1]["inputs"], "resource original repeated inputs")
    for name, identity in RECORDED_VALIDATIONS.items():
        same_json(
            stream_file_identity(root / name),
            identity,
            "resource original recorded complete reader validation",
        )
    return {
        "report_files": references,
        "original_inputs": deepcopy(requests[0]["inputs"]),
        "recorded_complete_reader_validations": deepcopy(RECORDED_VALIDATIONS),
        "scope_identity": requests[0]["scope_identity"],
        "scoring_manifest_sha256": requests[0]["scoring_manifest_sha256"],
        "analysis_contract_sha256": requests[0]["analysis_contract_sha256"],
    }


def resource_request(
    root: Path, output_dir: Path, scope_file: Path, started_utc: str
) -> dict[str, Any]:
    require(type(started_utc) is str, "resource request time must be a string")
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except ValueError as error:
        raise ValueError("resource request time malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "resource request requires UTC",
    )
    sources = current_resource_sources(root)
    return {
        "schema_id": "p610_resource_inventory_request_v1",
        "protocol_id": PROTOCOL_ID,
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "inputs": _inputs(root, scope_file),
        "inventory": {
            "cells": 560,
            "checkpoint_capacities": 1680,
            "shared_contexts": 60,
            "original_run_audits": 4,
            "complete_saved_cost_reads": 2,
        },
        "validation_budget_seconds": VALIDATION_SECONDS,
        "command": [
            sys.executable,
            "-m",
            "scripts.run_p610_resource_inventory",
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
        "new_complete_scientific_readback_claimed": False,
        "original_P6_10_acceptance_complete": False,
    }


def checked_resource_request(
    root: Path, paths: dict[str, Path], scope_file: Path
) -> dict[str, Any]:
    before = stream_file_identity(paths["request"])
    request = read_json(paths["request"])
    same_json(canonical_body_identity(request), before, "resource canonical request bytes")
    same_json(
        request,
        resource_request(root, paths["request"].parent, scope_file, request.get("started_utc", "")),
        "resource complete current request/source/input binding",
    )
    same_json(
        stream_file_identity(paths["request"]), before, "resource request changed during bindings"
    )
    return request


def read_resource_inventory_inputs(
    root: Path, paths: dict[str, Path], scope_file: Path
) -> dict[str, Any]:
    request = checked_resource_request(root, paths, scope_file)
    outputs: list[dict[str, Any]] = []
    for repeat in (False, True):
        suffix = "-repeat" if repeat else ""
        report = read_json(
            root
            / f"artifacts/runs/p611-confirmation-report{suffix}/confirmation-report.result.json"
        )
        inspection = read_json(root / f"artifacts/runs/p611-confirmation-costs{suffix}.json")
        audits = tuple(
            read_json(
                root / f"artifacts/runs/p67-confirmation-{kind}{rep}/confirmation-{kind}.audit.json"
            )
            for kind in ("train", "scored")
            for rep in ("", "-repeat")
        )
        inventory = build_resource_inventory(inspection, report, audits)
        # Only the small derived inventory survives to the next complete read.
        del inspection, report, audits
        if outputs:
            same_json(inventory, outputs[0], "resource full original inventory repetition")
        else:
            outputs.append(inventory)
    same_json(
        checked_resource_request(root, paths, scope_file),
        request,
        "resource inputs changed during complete saved reads",
    )
    inventory = outputs[0]
    inventory["provenance"] = {
        "validation_scope": "current_source_bound_complete_saved_cost_report_audit_preservation_and_inventory_reconstruction_with_recorded_original_reader_authority_not_new_scientific_readback",
        "source_sha256": deepcopy(request["source_sha256"]),
        "source_map_sha256": request["source_map_sha256"],
        "inputs": deepcopy(request["inputs"]),
        "complete_saved_cost_inspections_read": 2,
        "new_measurement_training_or_final_access": False,
    }
    return inventory
