"""Verify complete report inputs and current source/request identities.

Inputs are fixed local artifact references and unchanged complete-reader ports.
Output is the complete declared report plus source-bound provenance/resources.
Readers are sequential and large shared proof graphs are discarded between
calls. No source/model/train/final/scoring/selection or publication belongs here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np

from src.app.continual_confirmation_execution import digest_json, json_value
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_report import build_confirmation_report
from src.app.continual_confirmation_report_cost_binding import (
    BASE_SOURCE_MAP_SHA256,
    COST_FILES,
    COST_INSPECTION_ID,
    REPORT_COST_ID,
    compact_report_costs,
)
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
)
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_scoring_bindings import current_sources
from src.infra.continual_confirmation_training_references import (
    stream_file_identity,
    verify_training_reference_bytes,
)


PROTOCOL_ID = "p611_exhaustive_confirmation_report_v1"
VALIDATION_SECONDS = 180
SCORED_FILES = (
    (
        "artifacts/runs/p67-confirmation-scored",
        {
            "request": {
                "sha256": "a89efeff39467c1d4c55a45707ee5f700242c26dbfd47dc100912202921e1334",
                "byte_count": 183577,
            },
            "result": {
                "sha256": "2fae14cc615b2f2bf50716ea3e930514551283b662cded84befab91e39f698a7",
                "byte_count": 1116254,
            },
            "audit": {
                "sha256": "b53bfea96cdcf8c903311f94ac714fe4f2b4d19fa8b0775cfbd0f2d257fcd01f",
                "byte_count": 1039853,
            },
        },
    ),
    (
        "artifacts/runs/p67-confirmation-scored-repeat",
        {
            "request": {
                "sha256": "c34829231ca2d1c85d0211865d920c3747a3d4ccf1121bf9d90f7ad098bf609a",
                "byte_count": 183584,
            },
            "result": {
                "sha256": "2fae14cc615b2f2bf50716ea3e930514551283b662cded84befab91e39f698a7",
                "byte_count": 1116254,
            },
            "audit": {
                "sha256": "76688d9eff2018dda2918192f6b7aeb8efc82c806d75d94abdf5d4a53a71c36d",
                "byte_count": 1039852,
            },
        },
    ),
)
B1_SOURCE_SHA256 = {
    "src/app/continual_confirmation_report_costs.py": "7881c74a462550f407b16e8f2ccbf761f0faa0ed81a46bee14f3d2d099164298",
    "src/infra/continual_confirmation_report_cost_references.py": "7c92d58e1e9c5a168a22afa85e2bfdfdf8ae98ab36f5264c1d121c6bc1a26133",
    "scripts/inspect_p611_confirmation_costs.py": "3c3add12dd9de8b5226d4ba7c70dffa4b8b5185a7107d53a860dfd97f8a6b73e",
}
REPORT_APP_SHA256 = {
    "src/app/continual_confirmation_report_cost_binding.py": "bcdefe01975d7eea334a43391d7a43ec10caec299456301ebf21f404871f7975",
    "src/app/continual_confirmation_report.py": "2fa3428d1d10a271bb501a5c51c9303545c034e6c4b25d6a8c44f39b7d9ab619",
    "src/app/continual_confirmation_report_rendering.py": "d18de2e75bd1ca60c6a8d51b474a687b895df85cec75790cffdc83a7fbace7f2",
}
OWN_SOURCES = (
    "src/infra/continual_confirmation_report_bindings.py",
    "src/infra/continual_confirmation_report_artifacts.py",
    "scripts/run_p611_confirmation_report.py",
)
CostReader = Callable[[], dict[str, Any]]
ScoredReader = Callable[
    [Path, Callable[[], dict[str, Any]]], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
]


def current_report_sources(root: Path) -> dict[str, str]:
    sources = current_sources(root)
    same_json(
        digest_json(sources),
        "e0cfd897867786271974bcc971b647084f7fcf34b28e8db8a140a2d8c3f7ff8c",
        "report original scored sources",
    )
    sources.update(verify_source_files(root, B1_SOURCE_SHA256))
    same_json(digest_json(sources), BASE_SOURCE_MAP_SHA256, "report complete original cost sources")
    require(
        all(len(value) == 64 for value in REPORT_APP_SHA256.values()), "report sources not frozen"
    )
    sources.update(verify_source_files(root, REPORT_APP_SHA256))
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    require(len(sources) == 106, "report full source closure differs")
    return sources


def _input_files(root: Path) -> dict[str, Any]:
    manifest = fixed_scoring_manifest()
    files = {"training_bundles": verify_training_reference_bytes(root, manifest)}
    for path in COST_FILES:
        same_json(
            stream_file_identity(root / path),
            COST_INSPECTION_ID,
            "report complete cost inspection bytes",
        )
        files[path] = deepcopy(COST_INSPECTION_ID)
    for directory, expected in SCORED_FILES:
        for marker in ("confirmation-scored.claim", "confirmation-scored.failure.json"):
            require(
                not (root / directory / marker).exists(), "report scored input has failure/claim"
            )
        for kind, identity in expected.items():
            path = f"{directory}/confirmation-scored.{kind}.json"
            same_json(
                stream_file_identity(root / path), identity, f"report scored input {kind} bytes"
            )
            files[path] = deepcopy(identity)
    return files


def _environment() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def report_request(
    root: Path, output_dir: Path, scope_file: Path, started_utc: str
) -> dict[str, Any]:
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except (TypeError, ValueError) as error:
        raise ValueError("report request time malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "report request requires UTC",
    )
    manifest = fixed_scoring_manifest()
    scope = stream_file_identity(scope_file)
    same_json(scope["sha256"], manifest.scope_record_sha256, "report original scope bytes")
    sources = current_report_sources(root)
    return {
        "schema_id": "p611_confirmation_report_request_v1",
        "protocol_id": PROTOCOL_ID,
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "inputs": _input_files(root),
        "scope_identity": scope,
        "analysis_contract": json_value(asdict(manifest.analysis_contract)),
        "analysis_contract_sha256": manifest.analysis_contract_sha256,
        "scoring_manifest_sha256": scoring_manifest_digest(manifest),
        "cost_vector_identity": deepcopy(REPORT_COST_ID),
        "inventory": {
            "cells": 560,
            "metric_vectors": 626,
            "seed_observations": 6260,
            "primary_statements": 116,
            "scored_runs": 2,
            "seeds_per_vector": 10,
            "distinct_source_seeds": 50,
        },
        "validation_budget_seconds": VALIDATION_SECONDS,
        "command": [
            sys.executable,
            "-m",
            "scripts.run_p611_confirmation_report",
            "--publish",
            "--output-dir",
            str(output_dir.resolve()),
        ],
        "environment": _environment(),
        "started_utc": started_utc,
        "new_training_or_final_source_authorized": False,
        "outer_selection_scored": False,
    }


def checked_report_request(root: Path, paths: dict[str, Path], scope_file: Path) -> dict[str, Any]:
    before = stream_file_identity(paths["request"])
    request = read_json(paths["request"])
    same_json(canonical_body_identity(request), before, "report canonical request bytes")
    expected = report_request(
        root, paths["request"].parent, scope_file, request.get("started_utc", "")
    )
    same_json(request, expected, "report complete current request/source/input binding")
    same_json(
        stream_file_identity(paths["request"]), before, "report request changed during bindings"
    )
    return request


def _run_facts(
    directory: str, identities: dict[str, Any], request: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "work",
        "observed_updates",
        "final_observation",
        "process_rss",
        "worker_elapsed_seconds",
        "elapsed_seconds",
        "validation_scope",
    )
    return {
        "directory": directory,
        "files": deepcopy(identities),
        "command": request["command"],
        "environment": request["environment"],
        "started_utc": request["started_utc"],
        "original_source_map_sha256": audit["source_map_sha256"],
        **{key: deepcopy(audit[key]) for key in fields},
    }


def _collect_inputs(
    root: Path, scored_reader: ScoredReader, cost_reader: CostReader
) -> tuple[tuple[dict[str, Any], dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Private port seam; actual source-bound adapter supplies unchanged readers."""
    payloads: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    facts = []
    for directory, identities in SCORED_FILES:
        reads = 0

        def references() -> dict[str, Any]:
            nonlocal reads
            reads += 1
            inspection = cost_reader()
            compact = compact_report_costs(inspection)
            if costs:
                same_json(compact, costs[0], "report repeated original cost join")
            else:
                costs.append(compact)
            return inspection["cost_references"]["training_references"]

        parts = scored_reader(root / directory, references)
        require(
            reads == 1 and type(parts) is tuple and len(parts) == 3,
            "report complete reader dispatch differs",
        )
        for name, body in zip(("request", "result", "audit"), parts, strict=True):
            same_json(
                canonical_body_identity(body),
                identities[name],
                f"report complete returned scored {name}",
            )
        request, result, audit = parts
        payloads.append(result)
        facts.append(_run_facts(directory, identities, request, audit))
    require(len(payloads) == 2 and len(costs) == 1, "report whole input scope differs")
    return (payloads[0], payloads[1]), costs[0], facts


def verified_report_body(
    root: Path,
    paths: dict[str, Path],
    scope_file: Path,
    scored_reader: ScoredReader,
    cost_reader: CostReader,
) -> dict[str, Any]:
    request = checked_report_request(root, paths, scope_file)
    payloads, costs, facts = _collect_inputs(root, scored_reader, cost_reader)
    same_json(
        checked_report_request(root, paths, scope_file),
        request,
        "report inputs/request changed after complete readers",
    )
    report = build_confirmation_report(payloads, costs)
    report["scored_run_facts"] = facts
    report["provenance"] = {
        "validation_scope": "current_source_bound_complete_scored_and_original_cost_readbacks",
        "source_sha256": deepcopy(request["source_sha256"]),
        "source_map_sha256": request["source_map_sha256"],
        "inputs": deepcopy(request["inputs"]),
        "scope_identity": request["scope_identity"],
        "scoring_manifest_sha256": request["scoring_manifest_sha256"],
        "analysis_contract_sha256": request["analysis_contract_sha256"],
        "original_training_references": json_value(
            [asdict(item) for item in fixed_scoring_manifest().training_bundles]
        ),
        "new_training_or_final_source_access": False,
    }
    return report
