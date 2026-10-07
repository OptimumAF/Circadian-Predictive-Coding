"""Bind every current fixed findings input, source, environment and marker.

Inputs are the fixed publication catalog and current repository files. Outputs
are complete request snapshots. No model, data, reader dispatch, selection or
publication belongs here; old request checkers and all original gates remain.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from time import monotonic
from typing import Any

import numpy as np

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_matrix_artifacts import artifact_paths as matrix_paths
from src.infra.continual_confirmation_matrix_bindings import checked_matrix_request
from src.infra.continual_confirmation_outcome_cost_artifacts import artifact_paths as outcome_paths
from src.infra.continual_confirmation_outcome_cost_bindings import checked_outcome_cost_request
from src.infra.continual_confirmation_training_references import stream_file_identity


PROTOCOL_ID = "p612_complete_current_findings_publication_v1"
CATALOG_FILE = "src/config/p612_findings_publication.json"
CATALOG_ID = {
    "byte_count": 20731,
    "sha256": "adecc576bc94d1fd88d39fa8fbeec5c93c81170e4763f1d4e0fe9f7a88be30a2",
}
BUDGETS = {
    "outcome_cost_reader": 240,
    "matrix_reader": 180,
    "development_reader": 120,
    "pure_synthesis_and_presentation": 180,
    "current_bindings_and_artifacts_allowance": 120,
    "outer_operation": 840,
}
OWN_SOURCES = (
    "src/app/continual_findings_readers.py",
    "src/infra/continual_findings_current_bindings.py",
    "src/infra/continual_findings_current_inputs.py",
    "src/infra/continual_findings_artifacts.py",
    "scripts/run_p612_complete_findings.py",
    CATALOG_FILE,
)
OWN_TEST_FILES = (
    "tests/test_continual_findings_current_bindings.py",
    "tests/test_continual_findings_current_inputs.py",
    "tests/test_continual_findings_artifacts.py",
    "tests/test_p612_complete_findings_cli.py",
    "tests/findings_publication_fixtures.py",
)
CURRENT_BUNDLE_CHECKERS = {
    "outcome_costs": (outcome_paths, checked_outcome_cost_request),
    "matrix": (matrix_paths, checked_matrix_request),
}


def read_bound_json(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    same_json(stream_file_identity(path), expected, "findings whole input " + str(path))
    body = read_json(path)
    same_json(canonical_body_identity(body), expected, "findings canonical input " + str(path))
    same_json(
        stream_file_identity(path), expected, "findings input changed during read " + str(path)
    )
    return body


def publication_declarations(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = read_bound_json(root / CATALOG_FILE, CATALOG_ID)
    same_json(catalog["budget_seconds"], BUDGETS, "findings all prospective operation budgets")
    prior = read_bound_json(
        root / catalog["prior_b2_handoff"]["path"], catalog["prior_b2_handoff"]["identity"]
    )
    require(len(prior["source_sha256"]) == 144, "findings all prior accepted source pins")
    same_json(
        digest_json(prior["source_sha256"]), prior["source_map_sha256"], "findings prior source map"
    )
    return catalog, prior


def _sources_and_tests(root: Path, prior: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    sources = verify_source_files(root, prior["source_sha256"])
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    require(len(sources) == 150, "findings complete old and new source scope")
    tests = prior["test_files"].copy()
    for name, expected in tests.items():
        same_json(
            stream_file_identity(root / name), expected, "findings prior test evidence " + name
        )
    for name in OWN_TEST_FILES:
        tests[name] = stream_file_identity(root / name)
    return sources, tests


def _file_bindings(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    for name, identity in expected.items():
        same_json(
            stream_file_identity(root / name), identity, "findings complete current file " + name
        )
    return expected.copy()


def _development_markers(root: Path, catalog: dict[str, Any]) -> None:
    for name in catalog["development_input_files"]:
        if not name.endswith(".request.json"):
            continue
        prefix = name.removesuffix(".request.json")
        require(
            not (root / (prefix + ".claim")).exists()
            and not (root / (prefix + ".failure.json")).exists(),
            "findings development failure/claim marker",
        )


def _current_bundles(root: Path, scope_file: Path, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    bundles = []
    require(len(catalog["current_bundles"]) == 4, "findings all current original/repeated bundles")
    for reference in catalog["current_bundles"]:
        builder, checker = CURRENT_BUNDLE_CHECKERS[reference["kind"]]
        paths = builder(root / reference["directory"])
        require(
            not paths["claim"].exists() and not paths["failure"].exists(),
            "findings current upstream failure/claim marker",
        )
        for entry in reference["parts"].values():
            same_json(
                stream_file_identity(root / entry["path"]),
                entry["identity"],
                "findings whole current upstream part",
            )
        request = checker(root, paths, scope_file)
        bundles.append(
            {
                "kind": reference["kind"],
                "directory": reference["directory"],
                "parts": reference["parts"],
                "current_complete_request": request,
            }
        )
    return bundles


def current_findings_bindings(root: Path, scope_file: Path) -> dict[str, Any]:
    """Recheck the whole physical scope; never replace a complete reader dispatch."""
    started = monotonic()
    catalog, prior = publication_declarations(root)
    sources, tests = _sources_and_tests(root, prior)
    files = _file_bindings(root, prior["input_files"])
    files.update(_file_bindings(root, catalog["development_input_files"]))
    for reference in catalog["upstream_handoffs"].values():
        read_bound_json(root / reference["path"], reference["identity"])
        files[reference["path"]] = reference["identity"]
    for name, identity in prior["current_output_files"].items():
        same_json(
            stream_file_identity(root / name), identity, "findings accepted whole pure output"
        )
        files[name] = identity
    _development_markers(root, catalog)
    bundles = _current_bundles(root, scope_file, catalog)
    require(
        (root / "artifacts/runs/p610-outcome-costs/outcome-costs.claim").is_file(),
        "findings preserved timeout claim disappeared",
    )
    failed = root / "artifacts/runs/p612-complete-findings-pure"
    require(
        failed.is_dir() and not any(failed.iterdir()), "findings failed synthesis directory changed"
    )
    elapsed_seconds(
        monotonic() - started,
        "findings complete current binding",
        BUDGETS["current_bindings_and_artifacts_allowance"],
    )
    return {
        "publication_catalog_identity": CATALOG_ID.copy(),
        "prior_b2_handoff": catalog["prior_b2_handoff"],
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "test_files": tests,
        "input_files": files,
        "current_upstream_bundles": bundles,
        "scope_identity": stream_file_identity(scope_file),
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "preserved_failed_synthesis_directory_empty": True,
        "preserved_timeout_claim_present": True,
    }


def findings_request(
    root: Path, output_dir: Path, scope_file: Path, started_utc: str
) -> dict[str, Any]:
    require(type(started_utc) is str, "findings request timestamp must be a string")
    timestamp = datetime.fromisoformat(started_utc)
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "findings request timestamp must be UTC",
    )
    bindings = current_findings_bindings(root, scope_file)
    return {
        "schema_id": "p612_complete_findings_request_v1",
        "protocol_id": PROTOCOL_ID,
        "bindings": bindings,
        "source_map_sha256": bindings["source_map_sha256"],
        "validation_budget_seconds": BUDGETS["outer_operation"],
        "inner_budget_seconds": BUDGETS.copy(),
        "command": [
            sys.executable,
            "-m",
            "scripts.run_p612_complete_findings",
            "--publish",
            "--output-dir",
            str(output_dir.resolve()),
        ],
        "started_utc": started_utc,
        "new_scientific_operation_authorized": False,
        "original_P6_12_acceptance_complete": False,
    }


def checked_findings_request(
    root: Path, paths: dict[str, Path], scope_file: Path
) -> dict[str, Any]:
    before = stream_file_identity(paths["request"])
    request = read_json(paths["request"])
    same_json(canonical_body_identity(request), before, "findings canonical owned request")
    same_json(
        request,
        findings_request(root, paths["request"].parent, scope_file, request.get("started_utc", "")),
        "findings complete current request/source/input/environment binding",
    )
    same_json(
        stream_file_identity(paths["request"]), before, "findings request changed during bindings"
    )
    return request
