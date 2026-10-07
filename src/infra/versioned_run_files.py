"""Write and verify one completed v14 run bundle on local disk.

Inputs are a validated P5.1 manifest and fixed v14 payload bytes. Outputs
are exclusive local JSON files or a verified manifest. This boundary does
not train models, score data, choose an arm, or resume an interrupted run.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.comparison_scope import (
    NUMPY_BACKPROP_ALGORITHM_ID,
    NUMPY_CIRCADIAN_ALGORITHM_ID,
    NUMPY_PC_ALGORITHM_ID,
)
from src.app.continual_trigger_replay_outcomes import TRIGGER_REPLAY_OUTCOMES_PROTOCOL
from src.app.continual_trigger_replay_schedule import TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL
from src.app.continual_trigger_replay_training_study import TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL
from src.core.run_manifest import serialize_run_manifest, validate_run_manifest
from src.infra.atomic_artifact_directory import publish_artifact_directory


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"nonfinite JSON token is invalid: {token}")


def _read_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        record = json.loads(payload, parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    return record


def _verify_v14_roles(
    manifest: dict[str, Any], training: dict[str, Any], outcomes: dict[str, Any]
) -> None:
    expected = manifest["dataset_split_hashes"]
    config = manifest["resolved_config"]
    cells = [(seed, arm) for seed in config["seeds"] for arm in config["arms"]]
    training_rows = training.get("rows")
    outcome_rows = outcomes.get("outcomes")
    if not isinstance(training_rows, list) or not isinstance(outcome_rows, list):
        raise ValueError("v14 run bundle lacks training or outcome cells")
    try:
        if [(row["seed"], row["arm"]) for row in training_rows] != cells or [
            (row["seed"], row["arm"]) for row in outcome_rows
        ] != cells:
            raise ValueError("v14 run bundle has missing or reordered cells")
        for row in training_rows:
            seed = str(row["seed"])
            for phase in ("a", "b"):
                prefix = "phase_a" if phase == "a" else "phase_b"
                if row[f"{prefix}_development_role_hashes"] != {
                    role: expected[seed][phase][role]
                    for role in ("train", "inner_guard", "outer_selection")
                }:
                    raise ValueError("v14 training source-role hashes differ from run manifest")
        for row in outcome_rows:
            seed = str(row["seed"])
            if dict(row["final_role_hashes"]) != {
                phase: expected[seed][phase]["final_test"] for phase in ("a", "b")
            }:
                raise ValueError("v14 final source-role hashes differ from run manifest")
            if [method["method"] for method in row["methods"]] != [
                "backprop",
                "predictive_coding",
                "circadian_predictive_coding",
            ]:
                raise ValueError("v14 outcome methods differ from run manifest")
        contrasts = outcomes["contrasts"]
        if not isinstance(contrasts, list) or len(contrasts) != 18:
            raise ValueError("v14 outcome contrasts are incomplete")
    except (KeyError, TypeError) as exc:
        raise ValueError("v14 run bundle role or method fields are malformed") from exc


def _verify_bundle_bytes(
    manifest: dict[str, Any], training_bytes: bytes, outcomes_bytes: bytes
) -> None:
    validate_run_manifest(manifest)
    if manifest["status"] != "completed" or set(manifest["files"]) != {"training", "outcomes"}:
        raise ValueError("v14 run bundle requires both completed output files")
    if manifest["protocol_versions"] != {
        "source": TRIGGER_REPLAY_OPPORTUNITIES_PROTOCOL,
        "training": TRIGGER_REPLAY_TRAINING_STUDY_PROTOCOL,
        "outcomes": TRIGGER_REPLAY_OUTCOMES_PROTOCOL,
    } or manifest["algorithm_versions"] != {
        "backprop": NUMPY_BACKPROP_ALGORITHM_ID,
        "predictive_coding": NUMPY_PC_ALGORITHM_ID,
        "circadian_predictive_coding": NUMPY_CIRCADIAN_ALGORITHM_ID,
    }:
        raise ValueError("v14 run bundle protocol or algorithm versions are incompatible")
    for name, payload in (("training", training_bytes), ("outcomes", outcomes_bytes)):
        if sha256(payload).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"{name} SHA-256 differs from run manifest")
    training = _read_json(training_bytes, "training payload")
    outcomes = _read_json(outcomes_bytes, "outcome payload")
    for name, record in (("training", training), ("outcomes", outcomes)):
        if (
            record.get("protocol_id") != manifest["protocol_versions"][name]
            or record.get("manifest_digest") != manifest["config_sha256"]
            or record.get("resolved_manifest" if name == "training" else "manifest")
            != manifest["resolved_config"]
        ):
            raise ValueError(f"{name} protocol or configuration differs from run manifest")
    _verify_v14_roles(manifest, training, outcomes)


def write_run_bundle(
    output_root: str | Path,
    manifest: dict[str, Any],
    training_bytes: bytes,
    outcomes_bytes: bytes,
) -> Path:
    """Write complete v14 files once, with the completed manifest last."""
    _verify_bundle_bytes(manifest, training_bytes, outcomes_bytes)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / manifest["run_id"]
    return publish_artifact_directory(
        destination,
        {
            "training.json": training_bytes,
            "outcomes.json": outcomes_bytes,
            "manifest.json": serialize_run_manifest(manifest),
        },
    )


def verify_run_bundle(run_path: str | Path) -> dict[str, Any]:
    """Reject partial, changed, or mismatched v14 files before consumption."""
    directory = Path(run_path)
    try:
        manifest = _read_json((directory / "manifest.json").read_bytes(), "run manifest")
        validate_run_manifest(manifest)
        if manifest["run_id"] != directory.name:
            raise ValueError("run manifest ID differs from its directory")
        training = (directory / manifest["files"]["training"]["path"]).read_bytes()
        outcomes = (directory / manifest["files"]["outcomes"]["path"]).read_bytes()
    except (KeyError, OSError) as exc:
        raise ValueError("run manifest or required output file is missing") from exc
    _verify_bundle_bytes(manifest, training, outcomes)
    return manifest
