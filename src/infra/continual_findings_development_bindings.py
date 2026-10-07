"""Bind complete current development inputs through unchanged validator ports.

Inputs are the fixed catalog, prior source pins, whole bundles and validator
ports. Output retains every raw part plus current before/after byte bindings.
No experiment, model, data, score, final view or confirmation reader is owned.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Callable

from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_json import require, same_json
from src.app.continual_confirmation_manifest import ConfirmationFamily, fixed_confirmation_manifest
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_scoring_execution import elapsed_seconds
from src.app.continual_findings_development import CATALOG_ID, FAMILIES, PREFIXES
from src.infra.continual_confirmation_io import read_json, verify_source_files
from src.infra.continual_confirmation_training_references import stream_file_identity


VALIDATION_SECONDS = 120
CATALOG_FILE = "src/config/p612_development_inputs.json"
OWN_SOURCES = (
    "src/app/continual_findings_development.py",
    "src/infra/continual_findings_development_bindings.py",
    "scripts/inspect_p612_development_findings.py",
    CATALOG_FILE,
)
DevelopmentVerifier = Callable[[Path, ConfirmationFamily], dict[str, Any]]
PreflightVerifier = Callable[[str, dict[str, Any]], None]


def _bound_json(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    same_json(stream_file_identity(path), expected, "findings whole input " + str(path))
    body = read_json(path)
    same_json(
        canonical_body_identity(body), expected, "findings canonical whole input " + str(path)
    )
    same_json(
        stream_file_identity(path), expected, "findings input changed during read " + str(path)
    )
    return body


def _declarations(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    catalog = _bound_json(root / CATALOG_FILE, CATALOG_ID)
    same_json(
        catalog["validation_seconds"], VALIDATION_SECONDS, "findings prospective development budget"
    )
    prior = _bound_json(
        root / catalog["prior_sources"]["file"], catalog["prior_sources"]["identity"]
    )
    sources = verify_source_files(root, prior["source_sha256"])
    require(len(sources) == 129, "findings all prior accepted source pins")
    same_json(
        digest_json(sources), prior["source_map_sha256"], "findings prior complete source map"
    )
    for name in OWN_SOURCES:
        sources[name] = stream_file_identity(root / name)["sha256"]
    return catalog, sources


def _read_bundle(
    root: Path, family: str, directory: str, prefix: str, files: dict[str, Any]
) -> dict[str, Any]:
    location = root / directory
    require(
        not (location / (prefix + ".failure.json")).exists()
        and not (location / (prefix + ".claim")).exists(),
        "development input has a failure/claim marker",
    )
    require(set(files) == {"request", "result", "audit"}, "development input complete file scope")
    parts = {
        name: _bound_json(location / f"{prefix}.{name}.json", identity)
        for name, identity in files.items()
    }
    audit, request = parts["audit"], parts["request"]
    require(audit["status"] == "completed", "development input unsuccessful audit")
    for name in ("request", "result"):
        same_json(
            audit[name + "_sha256"], files[name]["sha256"], "development input audit identity"
        )
    elapsed_seconds(
        audit["elapsed_seconds"], "original development operation", request["wall_limit_seconds"]
    )
    same_json(request["wall_limit_seconds"], 120, "original development wall gate")
    verify_source_files(root, request["source_sha256"])
    if "process_rss" in audit:
        memory = audit["process_rss"]
        require(
            type(memory["pid"]) is int
            and memory["pid"] > 0
            and type(memory["sample_count"]) is int
            and memory["sample_count"] >= 2
            and type(memory["start_bytes"]) is int
            and type(memory["peak_bytes"]) is int
            and 0
            < memory["start_bytes"]
            <= memory["peak_bytes"]
            <= request["max_process_rss_bytes"]
            and memory["interval_seconds"] == 0.005,
            "original development sampled RSS gate",
        )
        same_json(
            request["max_process_rss_bytes"], 256 * 1024 * 1024, "original development RSS cap"
        )
    return {"family": family, "directory": directory, "files": files, "parts": parts}


def _development(
    root: Path, scope: dict[str, Any], verifiers: dict[str, DevelopmentVerifier]
) -> list[dict[str, Any]]:
    same_json(set(verifiers), set(FAMILIES), "all unchanged development validator ports")
    bundles = []
    manifest = fixed_confirmation_manifest()
    for family, reference in zip(manifest.families, scope["development_references"], strict=True):
        same_json(reference["family"], family.name, "development original scope order")
        require(len(reference["bundles"]) == 2, "both original development bundles required")
        for suffix, expected in zip(("", "-repeat"), reference["bundles"], strict=True):
            prefix = PREFIXES[family.name]
            directory = f"artifacts/runs/p63-{prefix}{suffix}"
            same_json(
                expected["directory"].replace("\\", "/"),
                directory,
                "development original directory",
            )
            observed = verifiers[family.name](root / directory, family)
            same_json(observed, expected, "unchanged whole development validator result")
            files = {
                name: {
                    "sha256": digest,
                    "byte_count": (root / directory / f"{prefix}.{name}.json").stat().st_size,
                }
                for name, digest in expected["file_sha256"].items()
            }
            bundles.append(_read_bundle(root, family.name, directory, prefix, files))
    return bundles


def _preflights(
    root: Path, catalog: dict[str, Any], verifier: PreflightVerifier
) -> list[dict[str, Any]]:
    same_json(
        [(b["family"], b["directory"]) for b in catalog["preflight_bundles"]],
        [
            (name, f"artifacts/runs/p63-{name}-factor-preflight{suffix}")
            for name in FAMILIES[2:]
            for suffix in ("", "-repeat")
        ],
        "all original preflight references",
    )
    bundles = []
    for reference in catalog["preflight_bundles"]:
        bundle = _read_bundle(
            root,
            reference["family"],
            reference["directory"],
            reference["prefix"],
            reference["files"],
        )
        request = bundle["parts"]["request"]
        same_json(
            request["adapter_sha256"],
            stream_file_identity(
                root / f"scripts/run_p63_{reference['family']}_factor_preflight.py"
            )["sha256"],
            "preflight current adapter identity",
        )
        verifier(reference["family"], bundle["parts"]["result"])
        bundles.append(bundle)
    return bundles


def _late_bundle(root: Path, bundle: dict[str, Any], prefix: str) -> None:
    location = root / bundle["directory"]
    require(
        not (location / (prefix + ".failure.json")).exists()
        and not (location / (prefix + ".claim")).exists(),
        "development late failure/claim marker",
    )
    for name, identity in bundle["files"].items():
        same_json(
            stream_file_identity(location / f"{prefix}.{name}.json"),
            identity,
            "development late whole part",
        )


def _collect(
    root: Path,
    catalog: dict[str, Any],
    sources: dict[str, str],
    verifiers: dict[str, DevelopmentVerifier],
    preflight_verifier: PreflightVerifier,
) -> dict[str, Any]:
    scopes = [
        _bound_json(root / name, identity) for name, identity in catalog["scope_files"].items()
    ]
    require(len(scopes) == 2, "both whole prospective scope records required")
    same_json(scopes[0], scopes[1], "whole prospective scope repetition")
    scope = scopes[0]
    development = _development(root, scope, verifiers)
    preflights = _preflights(root, catalog, preflight_verifier)
    for bundles, training_only in ((development, False), (preflights, True)):
        for bundle in bundles:
            prefix = (
                bundle["family"] + "-factor-preflight"
                if training_only
                else PREFIXES[bundle["family"]]
            )
            _late_bundle(root, bundle, prefix)
    for name, identity in catalog["scope_files"].items():
        same_json(
            stream_file_identity(root / name), identity, "development late whole prospective scope"
        )
    verify_source_files(root, sources)
    return {
        "schema_id": "p612_complete_current_development_inputs_v1",
        "input_catalog": catalog,
        "original_scope": scope,
        "scope_files": catalog["scope_files"],
        "source_sha256": sources,
        "source_map_sha256": digest_json(sources),
        "development_bundles": development,
        "preflight_bundles": preflights,
        "validation_facts": {
            "unchanged_development_bundle_validations": 12,
            "unchanged_preflight_result_validations": 8,
            "complete_bundle_files": 60,
            "current_before_and_after_byte_source_bindings": True,
            "new_training_scoring_or_final_access": False,
            "fresh_confirmation_reader_authority": False,
        },
    }


def read_development_inputs(
    root: Path, verifiers: dict[str, DevelopmentVerifier], preflight_verifier: PreflightVerifier
) -> dict[str, Any]:
    """Read and revalidate every fixed development part; no fresh confirmation proof."""
    started = monotonic()
    catalog, sources = _declarations(root)
    body = _collect(root, catalog, sources, verifiers, preflight_verifier)
    same_json(_declarations(root), (catalog, sources), "development final current declarations")
    elapsed_seconds(
        monotonic() - started, "complete development input validation", VALIDATION_SECONDS
    )
    return body


def verify_development_input_bindings(root: Path, inputs: dict[str, Any]) -> None:
    """Recheck all whole bytes/markers/sources after projection or publication."""
    catalog, sources = _declarations(root)
    same_json(catalog, inputs["input_catalog"], "development late catalog")
    same_json(sources, inputs["source_sha256"], "development late complete source scope")
    for bundles, training_only in (
        (inputs["development_bundles"], False),
        (inputs["preflight_bundles"], True),
    ):
        for bundle in bundles:
            prefix = (
                bundle["family"] + "-factor-preflight"
                if training_only
                else PREFIXES[bundle["family"]]
            )
            _late_bundle(root, bundle, prefix)
    for name, identity in catalog["scope_files"].items():
        same_json(stream_file_identity(root / name), identity, "development late whole scope")
