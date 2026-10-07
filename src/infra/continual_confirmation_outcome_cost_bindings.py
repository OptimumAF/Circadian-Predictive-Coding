"""Bind all accepted inputs and compose one whole unchanged official reader.

Inputs are current sources, complete original/companion bundles and a no-argument
report reader port. Output is the unchanged complete pure presentation with fresh
reader provenance. No model/data/train/score, final view, profiling or selection.
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
from src.app.continual_confirmation_outcome_costs import (
    INPUT_FILES,
    INPUT_IDENTITIES,
    build_outcome_cost_presentation,
)
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_report_artifacts import artifact_paths as report_paths
from src.infra.continual_confirmation_report_bindings import checked_report_request
from src.infra.continual_confirmation_resource_artifacts import artifact_paths as inventory_paths
from src.infra.continual_confirmation_resource_bindings import (
    REPORT_FILES,
    checked_resource_request,
)
from src.infra.continual_confirmation_retention_artifacts import artifact_paths as retention_paths
from src.infra.continual_confirmation_retention_bindings import (
    checked_retention_request,
    current_retention_sources,
)
from src.infra.continual_confirmation_training_references import stream_file_identity


PROTOCOL_ID = "p610_complete_original_outcome_cost_publication_v1"
# Why this: declared before b2 fixtures, based on the unchanged 176-178 s
# whole report operations plus 31 s preservation and complete metadata joins.
# The original inner report's 180 s gate and all scientific caps stay intact.
VALIDATION_SECONDS = 240
RETENTION_SOURCE_MAP_SHA256 = "5ab335ea6959542f99cdd610b742ed8ee739d230ea540dca41abcab00645424c"
PURE_SOURCE_MAP_SHA256 = "d2df05e5cdf192f573e5564b25d336ce993166d4cbead40d29cea0082c24a9d9"
APP_SOURCES = {
    "src/app/continual_confirmation_outcome_cost_inputs.py": "e705158e2d50eacc95b2384844a27dc7f26910c7ff89db5c692567c7a1f17d4f",
    "src/app/continual_confirmation_outcome_costs.py": "4f3859c76f2210cd12202c4dbc89e3e32717545c5f8c849b7a49ad10ec7eff4b",
    "src/app/continual_confirmation_outcome_cost_rendering.py": "70693673fdc245339d207e8bb46e8360877c9c462a597466a94cf7948e78aaea",
}
OWN_SOURCES = (
    "src/infra/continual_confirmation_outcome_cost_bindings.py",
    "src/infra/continual_confirmation_outcome_cost_artifacts.py",
    "scripts/run_p610_outcome_costs.py",
)
HANDOFFS: dict[str, Any] = {
    "retention": {
        "file": "artifacts/runs/p610-retention-costs-handoff-validation.json",
        "identity": {
            "byte_count": 71965,
            "sha256": "27c172131fd960deeddc1d6cd179cf4b46d3b3dd81759846865237133bcbafb3",
        },
    },
    "pure": {
        "file": "artifacts/runs/p610-outcome-cost-presentation-handoff-validation.json",
        "identity": {
            "byte_count": 138278,
            "sha256": "573b665124aba168125ffb209820952ed10a2fdc518bf1a7faf9d09c1c2ae550",
        },
    },
}
PURE_RESULT_ID = {
    "byte_count": 39777631,
    "sha256": "73f5892ef7d152a9a1ada89d59f877483e720026ca160a7afb146c53f4a7f4e8",
}
CompletedReportReader = Callable[[], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]


def current_outcome_cost_sources(root: Path) -> dict[str, str]:
    sources = current_retention_sources(root)
    same_json(digest_json(sources), RETENTION_SOURCE_MAP_SHA256, "outcome costs retention sources")
    sources.update(verify_source_files(root, APP_SOURCES))
    same_json(digest_json(sources), PURE_SOURCE_MAP_SHA256, "outcome costs unchanged pure sources")
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    require(len(sources) == 127, "outcome costs complete source scope differs")
    return sources


def _handoffs(root: Path) -> dict[str, Any]:
    records = {}
    for name, reference in HANDOFFS.items():
        path = root / reference["file"]
        same_json(stream_file_identity(path), reference["identity"], "outcome costs prior " + name)
        body = read_json(path)
        same_json(canonical_body_identity(body), reference["identity"], "canonical prior " + name)
        records[name] = body
    same_json(
        digest_json(records["retention"]["source_sha256"]),
        RETENTION_SOURCE_MAP_SHA256,
        "outcome costs retained source contract",
    )
    same_json(
        digest_json(records["pure"]["source_sha256"]),
        PURE_SOURCE_MAP_SHA256,
        "outcome costs pure source contract",
    )
    return records


def _bundle_files(paths: dict[str, Path], expected: dict[str, Any]) -> None:
    require(
        not paths["claim"].exists() and not paths["failure"].exists(),
        "outcome costs upstream bundle has failure/claim",
    )
    same_json(sorted(expected), ["audit", "markdown", "request", "result"], "upstream parts")
    for name, identity in expected.items():
        same_json(stream_file_identity(paths[name]), identity, "outcome costs upstream " + name)


def _reports(root: Path, scope_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files, requests = [], {}
    for index, (directory, expected) in enumerate(REPORT_FILES):
        paths = report_paths(root / directory)
        _bundle_files(paths, expected)
        requests["report" + ("-repeat" if index else "")] = checked_report_request(
            root,
            paths,
            scope_file,
        )
        files.append({"directory": directory, "files": deepcopy(expected)})
    same_json(
        requests["report"]["inputs"], requests["report-repeat"]["inputs"], "report repetition"
    )
    return files, requests


def _companions(
    root: Path,
    scope_file: Path,
    records: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    files: dict[str, Any] = {"inventory": [], "retention": []}
    requests = {}
    for kind, path_builder, checker in (
        ("inventory", inventory_paths, checked_resource_request),
        ("retention", retention_paths, checked_retention_request),
    ):
        expected_files = (
            records["pure"]["current_complete_upstream_bindings"][
                "previous_inventory_and_matrix_files"
            ]["inventory_files"]
            if kind == "inventory"
            else records["retention"]["retention_files"]
        )
        for suffix in ("", "-repeat"):
            directory = f"artifacts/runs/p610-{'resource-inventory' if kind == 'inventory' else 'retention-costs'}{suffix}"
            paths = path_builder(root / directory)
            expected = expected_files[str((root / directory).resolve())]
            _bundle_files(paths, expected)
            same_json(expected["result"], INPUT_IDENTITIES[kind], "whole companion identity")
            requests[kind + suffix] = checker(root, paths, scope_file)
            files[kind].append({"directory": directory, "files": deepcopy(expected)})
        same_json(
            requests[kind]["inputs"], requests[kind + "-repeat"]["inputs"], kind + " repetition"
        )
    return files, requests


def _pure_files(root: Path, records: dict[str, Any]) -> list[dict[str, Any]]:
    files = []
    for suffix in ("", "-repeat"):
        directory = "artifacts/runs/p610-outcome-cost-pure" + suffix
        expected = records["pure"]["pure_output_files"][str((root / directory).resolve())]
        require(
            not (root / directory / "pure-validation.failure.json").exists(),
            "outcome costs pure evidence failed",
        )
        for name, filename in (
            ("result", "outcome-costs.result.json"),
            ("markdown", "outcome-costs.md"),
        ):
            same_json(
                stream_file_identity(root / directory / filename), expected[name], "pure " + name
            )
        same_json(expected["result"], PURE_RESULT_ID, "accepted whole pure presentation")
        files.append({"directory": directory, "files": deepcopy(expected)})
    same_json(files[0]["files"], files[1]["files"], "accepted complete pure repetition")
    return files


def _inputs(root: Path, scope_file: Path) -> dict[str, Any]:
    records = _handoffs(root)
    reports, requests = _reports(root, scope_file)
    companions, companion_requests = _companions(root, scope_file, records)
    requests.update(companion_requests)
    previous = records["pure"]["current_complete_upstream_bindings"][
        "original_and_companion_requests"
    ]
    same_json(
        {name: requests[name] for name in previous}, previous, "current accepted upstream requests"
    )
    scope = stream_file_identity(scope_file)
    same_json(scope, requests["report"]["scope_identity"], "outcome costs original scope")
    return {
        "report_files": reports,
        "inventory_files": companions["inventory"],
        "retention_files": companions["retention"],
        "current_upstream_requests": requests,
        "prior_handoffs": deepcopy(HANDOFFS),
        "pure_files": _pure_files(root, records),
        "whole_input_identities": deepcopy(INPUT_IDENTITIES),
        "scope_identity": scope,
    }


def outcome_cost_request(
    root: Path,
    output_dir: Path,
    scope_file: Path,
    started_utc: str,
) -> dict[str, Any]:
    require(type(started_utc) is str, "outcome costs request time must be a string")
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except ValueError as error:
        raise ValueError("outcome costs request time malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "outcome costs request requires UTC",
    )
    sources = current_outcome_cost_sources(root)
    return {
        "schema_id": "p610_outcome_costs_request_v1",
        "protocol_id": PROTOCOL_ID,
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "inputs": _inputs(root, scope_file),
        "scope": {
            "cells": 560,
            "contexts": 60,
            "checkpoints": 1680,
            "cell_metrics": 3360,
            "metric_vectors": 626,
            "primary_statements": 116,
            "complete_report_readers": 1,
        },
        "validation_budget_seconds": VALIDATION_SECONDS,
        "command": [
            sys.executable,
            "-m",
            "scripts.run_p610_outcome_costs",
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


def checked_outcome_cost_request(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
) -> dict[str, Any]:
    before = stream_file_identity(paths["request"])
    request = read_json(paths["request"])
    same_json(canonical_body_identity(request), before, "outcome costs canonical request bytes")
    same_json(
        request,
        outcome_cost_request(
            root, paths["request"].parent, scope_file, request.get("started_utc", "")
        ),
        "outcome costs complete current request/source/input/environment bindings",
    )
    same_json(
        stream_file_identity(paths["request"]),
        before,
        "outcome costs request changed during bindings",
    )
    return request


def _read_report(reader: CompletedReportReader) -> tuple[dict[str, Any], dict[str, Any]]:
    parts = reader()
    require(
        type(parts) is tuple and len(parts) == 3 and all(type(part) is dict for part in parts),
        "outcome costs complete official report dispatch differs",
    )
    identities = {}
    for name, body in zip(("request", "result", "audit"), parts, strict=True):
        identity = canonical_body_identity(body)
        same_json(identity, REPORT_FILES[0][1][name], "complete official report " + name)
        identities[name] = identity
    return parts[1], identities


def _read_bound_outcome_cost_inputs(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    reader: CompletedReportReader,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Internal handoff of a just-validated request; never a narrower reader.

    Why this: immediately repeating the same whole binding traversal made the
    first derivative exceed its fixed budget. The caller validates current
    bindings immediately before this handoff; bytes are checked here and every
    complete current binding is still reconstructed after the whole derivation.
    No cached values or checks are shared between independent operations.
    """
    same_json(
        canonical_body_identity(request),
        stream_file_identity(paths["request"]),
        "outcome costs freshly validated request snapshot bytes",
    )
    report, report_parts = _read_report(reader)
    # Why this: load the companion metadata after the unchanged report reader,
    # keeping its original memory and time gates independent of this derivative.
    inventory, retention = (
        read_json(root / INPUT_FILES[name]) for name in ("inventory", "retention")
    )
    body = build_outcome_cost_presentation(report, inventory, retention)
    same_json(
        canonical_body_identity(body), PURE_RESULT_ID, "unchanged accepted complete pure result"
    )
    del report, inventory, retention
    body["provenance"] = {
        "validation_scope": "current_source_bound_unchanged_complete_official_report_reader_and_all_original_outcomes_against_scoped_costs",
        "source_sha256": deepcopy(request["source_sha256"]),
        "source_map_sha256": request["source_map_sha256"],
        "inputs": deepcopy(request["inputs"]),
        "complete_original_report_parts": report_parts,
        "complete_original_report_readers": 1,
        "new_measurement_training_or_final_access": False,
    }
    same_json(
        checked_outcome_cost_request(root, paths, scope_file),
        request,
        "outcome costs bindings changed after reader and whole derivation",
    )
    return body


def read_outcome_cost_inputs(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    reader: CompletedReportReader,
) -> dict[str, Any]:
    request = checked_outcome_cost_request(root, paths, scope_file)
    return _read_bound_outcome_cost_inputs(root, paths, scope_file, reader, request)
