"""Bind current matrix sources/inputs and the unchanged complete report reader.

Inputs are fixed original artifact references and one injected complete reader.
Output retains the verified matrix plus current original source/input facts.
No dataset/model, training, final access, scientific override or publication.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_matrix import build_confirmation_matrix
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_report_artifacts import artifact_paths as report_paths
from src.infra.continual_confirmation_report_bindings import (
    checked_report_request,
    current_report_sources,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


PROTOCOL_ID = "p69_stored_confirmation_stage_task_matrix_v1"
VALIDATION_SECONDS = 180
REPORT_SOURCE_MAP_SHA256 = "67063e2b13329d8fb86654cbb4acf4c4b348db0d98e1f9e7abeca5e265058517"
REPORT_RESULT_ID = {
    "byte_count": 7537678,
    "sha256": "363ed97dba808281d13224526b4b36614ae452188cade256992b530b6ab03088",
}
REPORT_MARKDOWN_ID = {
    "byte_count": 2958568,
    "sha256": "c989df16853868e910bf99303470927d17c6f01ed86e5dd004d35692bf03fddc",
}
REPORT_FILES = (
    (
        "artifacts/runs/p611-confirmation-report",
        {
            "request": {
                "byte_count": 26104,
                "sha256": "57db44ded2fdc9a0011dac61ac8f0b11e55f6e831df21215ba2f98003054aff7",
            },
            "result": REPORT_RESULT_ID,
            "markdown": REPORT_MARKDOWN_ID,
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
            "result": REPORT_RESULT_ID,
            "markdown": REPORT_MARKDOWN_ID,
            "audit": {
                "byte_count": 4475,
                "sha256": "0d6b2fd124c7029b52d86623533c6bef5ecac51cbdffae448c8a1fe777d0a31a",
            },
        },
    ),
)
MATRIX_APP_SHA256 = {
    "src/app/continual_confirmation_matrix_inputs.py": "f5649c2639a611a59a10a6a96bd3ab2e489014e33d5eab33204bbfb09a8cb100",
    "src/app/continual_confirmation_matrix.py": "412ff5579832bcd4f259a1685232d8a8032fdb828064d2e206327816d305275d",
    "src/app/continual_confirmation_matrix_rendering.py": "04f86e8b3664fd64fc868d77e200e277b2f444de83c1e7ed6b37be83f456f7c6",
}
OWN_SOURCES = (
    "src/infra/continual_confirmation_matrix_bindings.py",
    "src/infra/continual_confirmation_matrix_artifacts.py",
    "scripts/run_p69_confirmation_matrix.py",
)
ReportReader = Callable[[], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]


def current_matrix_sources(root: Path) -> dict[str, str]:
    sources = current_report_sources(root)
    same_json(
        digest_json(sources), REPORT_SOURCE_MAP_SHA256, "matrix original complete report sources"
    )
    require(
        all(len(value) == 64 for value in MATRIX_APP_SHA256.values()), "matrix sources not frozen"
    )
    sources.update(verify_source_files(root, MATRIX_APP_SHA256))
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    require(len(sources) == 112, "matrix full source closure differs")
    return sources


def _input_files(root: Path, scope_file: Path) -> dict[str, Any]:
    references = []
    originals = []
    for directory, expected in REPORT_FILES:
        paths = report_paths(root / directory)
        require(
            not paths["claim"].exists() and not paths["failure"].exists(),
            "matrix input report has claim/failure",
        )
        for name, identity in expected.items():
            same_json(
                stream_file_identity(paths[name]), identity, f"matrix input report {name} bytes"
            )
        request = checked_report_request(root, paths, scope_file)
        originals.append(request)
        references.append({"directory": directory, "files": deepcopy(expected)})
    same_json(
        originals[0]["inputs"], originals[1]["inputs"], "matrix original repeated report inputs"
    )
    return {
        "report_files": references,
        "original_inputs": deepcopy(originals[0]["inputs"]),
        "scope_identity": originals[0]["scope_identity"],
        "scoring_manifest_sha256": originals[0]["scoring_manifest_sha256"],
        "analysis_contract_sha256": originals[0]["analysis_contract_sha256"],
    }


def _environment() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def matrix_request(
    root: Path, output_dir: Path, scope_file: Path, started_utc: str
) -> dict[str, Any]:
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except (TypeError, ValueError) as error:
        raise ValueError("matrix request time malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "matrix request requires UTC",
    )
    sources = current_matrix_sources(root)
    return {
        "schema_id": "p69_confirmation_matrix_request_v1",
        "protocol_id": PROTOCOL_ID,
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "inputs": _input_files(root, scope_file),
        "inventory": {
            "rows": 560,
            "matrix_slots": 2240,
            "original_endpoints": 1680,
            "unmeasured_slots": 560,
            "complete_report_readers": 1,
        },
        "validation_budget_seconds": VALIDATION_SECONDS,
        "command": [
            sys.executable,
            "-m",
            "scripts.run_p69_confirmation_matrix",
            "--publish",
            "--output-dir",
            str(output_dir.resolve()),
        ],
        "environment": _environment(),
        "started_utc": started_utc,
        "new_training_or_final_source_authorized": False,
        "new_primary_metric_or_interval_family": False,
        "original_fully_measured_matrix_acceptance_complete": False,
    }


def checked_matrix_request(root: Path, paths: dict[str, Path], scope_file: Path) -> dict[str, Any]:
    before = stream_file_identity(paths["request"])
    request = read_json(paths["request"])
    same_json(canonical_body_identity(request), before, "matrix canonical request bytes")
    expected = matrix_request(
        root, paths["request"].parent, scope_file, request.get("started_utc", "")
    )
    same_json(request, expected, "matrix complete current request/source/input binding")
    same_json(
        stream_file_identity(paths["request"]), before, "matrix request changed during bindings"
    )
    return request


def verified_matrix_body(
    root: Path, paths: dict[str, Path], scope_file: Path, reader: ReportReader
) -> dict[str, Any]:
    request = checked_matrix_request(root, paths, scope_file)
    parts = reader()
    require(type(parts) is tuple and len(parts) == 3, "matrix requires one complete report reader")
    expected = REPORT_FILES[0][1]
    for name, body in zip(("request", "result", "audit"), parts, strict=True):
        same_json(
            canonical_body_identity(body), expected[name], f"matrix complete returned report {name}"
        )
    same_json(
        checked_matrix_request(root, paths, scope_file),
        request,
        "matrix inputs changed after complete report reader",
    )
    matrix = build_confirmation_matrix(parts[1])
    matrix["provenance"] = {
        "validation_scope": "current_source_bound_complete_original_report_readback_and_stored_matrix",
        "source_sha256": deepcopy(request["source_sha256"]),
        "source_map_sha256": request["source_map_sha256"],
        "inputs": deepcopy(request["inputs"]),
        "new_training_or_final_source_access": False,
    }
    return matrix
