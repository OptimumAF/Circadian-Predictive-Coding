"""Validate one versioned local run manifest without touching files.

Inputs are JSON-compatible run facts supplied by an app use case. Outputs
are validation errors or canonical bytes. This module does not collect
machine metadata, train models, choose results, or write artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, cast


RUN_MANIFEST_SCHEMA_ID = "circadian_run_manifest_v1"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json\Z")
_HEX = re.compile(r"[0-9a-f]+\Z")
_STATUSES = {"completed", "incomplete", "failed", "canceled"}
_TOP_FIELDS = {
    "schema_id",
    "run_id",
    "status",
    "protocol_versions",
    "algorithm_versions",
    "source",
    "resolved_config",
    "config_sha256",
    "seed_map",
    "dataset_role_names",
    "dataset_split_hashes",
    "pretrained_weights",
    "dependency_versions",
    "hardware",
    "precision",
    "determinism",
    "timing_scope",
    "files",
}


@dataclass(frozen=True)
class RunEnvironment:
    """Execution facts collected outside the pure run-schema layer."""

    source: dict[str, object]
    dependency_versions: dict[str, str]
    hardware: dict[str, object]
    python_hash_seed: str | None


def _object(value: object, label: str, fields: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    result = cast(dict[str, Any], value)
    if fields is not None and set(result) != fields:
        raise ValueError(f"{label} fields differ: expected {sorted(fields)}")
    return result


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _digest(value: object, label: str, *, lengths: tuple[int, ...] = (64,)) -> str:
    if type(value) is not str or len(value) not in lengths or _HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _versions(value: object, label: str) -> dict[str, str]:
    records = _object(value, label)
    if not records:
        raise ValueError(f"{label} must contain at least one version")
    for name, version in records.items():
        _text(name, f"{label} key")
        _text(version, f"{label}.{name}")
    return records


def _validate_source(value: object) -> None:
    source = _object(
        value,
        "source",
        {
            "commit_sha",
            "dirty",
            "status_sha256",
            "tracked_diff_sha256",
            "workspace_sha256",
            "untracked_file_count",
            "unavailable_reason",
        },
    )
    if source["unavailable_reason"] is None:
        _digest(source["commit_sha"], "Git commit", lengths=(40, 64))
        if type(source["dirty"]) is not bool:
            raise ValueError("Git dirty state must be a boolean")
        _digest(source["status_sha256"], "Git status SHA-256")
        _digest(source["tracked_diff_sha256"], "Git tracked diff SHA-256")
        _digest(source["workspace_sha256"], "Git workspace SHA-256")
        if type(source["untracked_file_count"]) is not int or source["untracked_file_count"] < 0:
            raise ValueError("Git untracked file count must be nonnegative")
        empty_digest = sha256(b"").hexdigest()
        if source["dirty"] is False and (
            source["status_sha256"] != empty_digest
            or source["tracked_diff_sha256"] != empty_digest
            or source["untracked_file_count"] != 0
        ):
            raise ValueError("clean Git source cannot have changes or untracked files")
        if source["dirty"] is True and source["status_sha256"] == empty_digest:
            raise ValueError("dirty Git source requires a nonempty status")
    elif (
        source["commit_sha"] is not None
        or source["dirty"] is not None
        or source["status_sha256"] is not None
        or source["tracked_diff_sha256"] is not None
        or source["workspace_sha256"] is not None
        or source["untracked_file_count"] is not None
    ):
        raise ValueError("Git provenance must be entirely known or explicitly unavailable")
    else:
        _text(source["unavailable_reason"], "Git unavailable reason")


def _validate_pretrained(value: object) -> None:
    record = _object(value, "pretrained_weights", {"state", "sha256", "reason"})
    if record["state"] == "known":
        _digest(record["sha256"], "pretrained weight SHA-256")
        if record["reason"] is not None:
            raise ValueError("known pretrained weights cannot have an unavailable reason")
    elif record["state"] in {"not_applicable", "unavailable"}:
        if record["sha256"] is not None:
            raise ValueError("unavailable pretrained weights cannot have a SHA-256")
        _text(record["reason"], "pretrained weight reason")
    else:
        raise ValueError("pretrained weight state is unsupported")


def _validate_seeds_and_roles(manifest: dict[str, Any]) -> None:
    config = _object(manifest["resolved_config"], "resolved_config")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("resolved_config.seeds must be nonempty Python integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("resolved_config.seeds must be distinct")
    expected_seeds = {str(seed) for seed in seeds}
    seed_map = _object(manifest["seed_map"], "seed_map")
    if set(seed_map) != expected_seeds:
        raise ValueError("seed map keys differ from resolved seeds")
    for seed, entries in seed_map.items():
        derived = _object(entries, f"seed_map.{seed}")
        if not derived or any(type(item) is not int for item in derived.values()):
            raise ValueError("seed map values must be nonempty integer derivations")
    roles = manifest["dataset_role_names"]
    if (
        not isinstance(roles, list)
        or not roles
        or any(type(role) is not str or not role for role in roles)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("dataset role names must be distinct nonempty text")
    hashes = _object(manifest["dataset_split_hashes"], "dataset_split_hashes")
    if set(hashes) != expected_seeds:
        raise ValueError("dataset/split hash seed keys differ from resolved seeds")
    for seed, phases in hashes.items():
        phase_records = _object(phases, f"dataset_split_hashes.{seed}")
        if not phase_records:
            raise ValueError("dataset/split hashes require at least one phase")
        for phase, values in phase_records.items():
            role_hashes = _object(values, f"dataset_split_hashes.{seed}.{phase}")
            if set(role_hashes) != set(roles):
                raise ValueError("dataset/split role hashes differ from declared role names")
            for role, digest in role_hashes.items():
                _digest(digest, f"dataset/split hash {seed}.{phase}.{role}")


def _validate_execution_facts(manifest: dict[str, Any]) -> None:
    _validate_source(manifest["source"])
    _validate_pretrained(manifest["pretrained_weights"])
    dependencies = _versions(manifest["dependency_versions"], "dependency_versions")
    if "python" not in dependencies or "numpy" not in dependencies:
        raise ValueError("dependency versions require Python and NumPy")
    hardware = _object(
        manifest["hardware"],
        "hardware",
        {
            "system",
            "release",
            "machine",
            "processor",
            "processor_unavailable_reason",
            "logical_cpu_count",
            "compute_device",
        },
    )
    for name in ("system", "release", "machine", "compute_device"):
        _text(hardware[name], f"hardware.{name}")
    count = hardware["logical_cpu_count"]
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("hardware.logical_cpu_count must be positive or unknown")
    if hardware["processor"] is None:
        _text(hardware["processor_unavailable_reason"], "processor unavailable reason")
    else:
        _text(hardware["processor"], "hardware.processor")
        if hardware["processor_unavailable_reason"] is not None:
            raise ValueError("known processor cannot have an unavailable reason")
    precision = _object(manifest["precision"], "precision", {"inputs", "parameters", "metrics"})
    for name, value in precision.items():
        _text(value, f"precision.{name}")
    determinism = _object(
        manifest["determinism"], "determinism", {"numpy_rng", "python_hash_seed", "scope"}
    )
    _text(determinism["numpy_rng"], "determinism.numpy_rng")
    _text(determinism["scope"], "determinism.scope")
    if determinism["python_hash_seed"] is not None:
        _text(determinism["python_hash_seed"], "determinism.python_hash_seed")
    timing = _object(manifest["timing_scope"], "timing_scope", {"mode", "includes", "excludes"})
    if timing["mode"] not in {"not_measured", "wall_clock"}:
        raise ValueError("timing scope mode is unsupported")
    for name in ("includes", "excludes"):
        values = timing[name]
        if not isinstance(values, list) or any(
            type(item) is not str or not item for item in values
        ):
            raise ValueError(f"timing_scope.{name} must be a text list")
    if timing["mode"] == "not_measured" and timing["includes"]:
        raise ValueError("unmeasured timing cannot include measured work")


def _validate_files(manifest: dict[str, Any]) -> None:
    files = _object(manifest["files"], "files")
    if manifest["status"] == "completed" and not files:
        raise ValueError("completed run requires at least one output file")
    seen_paths: set[str] = set()
    versions = _object(manifest["protocol_versions"], "protocol_versions")
    for name, value in files.items():
        record = _object(value, f"files.{name}", {"path", "sha256", "protocol_id"})
        path = record["path"]
        if type(path) is not str or _OUTPUT_NAME.fullmatch(path) is None:
            raise ValueError("output path must be a local JSON basename")
        if path in seen_paths:
            raise ValueError("output paths must be distinct")
        seen_paths.add(path)
        _digest(record["sha256"], f"files.{name}.sha256")
        if name not in versions or record["protocol_id"] != versions[name]:
            raise ValueError("output protocol ID differs from declared version")


def validate_run_id(value: object) -> str:
    """Return a safe local run slug before starting expensive work."""
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id must be a lowercase local slug")
    return value


def validate_run_manifest(value: object) -> None:
    """Reject incomplete or contradictory required provenance fields."""
    manifest = _object(value, "run manifest", _TOP_FIELDS)
    if manifest["schema_id"] != RUN_MANIFEST_SCHEMA_ID:
        raise ValueError("run manifest schema ID is incompatible")
    validate_run_id(manifest["run_id"])
    if manifest["status"] not in _STATUSES:
        raise ValueError("run status is unsupported")
    protocols = _versions(manifest["protocol_versions"], "protocol_versions")
    _versions(manifest["algorithm_versions"], "algorithm_versions")
    config = _object(manifest["resolved_config"], "resolved_config")
    if config.get("protocol_id") != protocols.get("source"):
        raise ValueError("resolved configuration and source protocol ID differ")
    _digest(manifest["config_sha256"], "config SHA-256")
    _validate_seeds_and_roles(manifest)
    _validate_execution_facts(manifest)
    _validate_files(manifest)
    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("run manifest must contain only finite JSON values") from exc


def serialize_run_manifest(manifest: Mapping[str, object]) -> bytes:
    """Return stable UTF-8/LF bytes after full schema validation."""
    validate_run_manifest(dict(manifest))
    return (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
