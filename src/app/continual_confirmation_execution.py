"""Bind frozen execution metadata and compare observed work with pure facts.

Inputs are the fixed manifest and boundary-supplied identities. Outputs are
strict request metadata and independently derived summaries. No IO, source
construction, training, measurement or evaluation belongs to this module.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_json import (
    hash_value,
    integer,
    object_fields,
    require,
    same_json,
)
from src.app.continual_confirmation_manifest import (
    ConfirmationManifest,
    fixed_confirmation_manifest,
    validate_confirmation_manifest,
)
from src.app.continual_confirmation_training import PROTOCOL_ID
from src.app.continual_confirmation_work_validation import SeedWork


REQUEST_SCHEMA = "p67_confirmation_train_request_v1"
AUDIT_SCHEMA = "p67_confirmation_train_audit_v1"
FAILURE_SCHEMA = "p67_confirmation_train_failure_v1"


def json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def execution_request(
    manifest: ConfirmationManifest,
    scope_sha256: str,
    source_sha256: dict[str, str],
    adapter_sha256: str,
    command: list[str],
    environment: dict[str, str],
    started_utc: str,
) -> dict[str, Any]:
    """Resolve a request without accepting a partial scientific scope."""
    summary = validate_confirmation_manifest(manifest)
    same_json(
        json_value(asdict(manifest)),
        json_value(asdict(fixed_confirmation_manifest())),
        "execution manifest",
    )
    hash_value(scope_sha256, "scope record")
    hash_value(adapter_sha256, "execution adapter")
    require(type(source_sha256) is dict and bool(source_sha256), "execution sources empty")
    for name, digest in source_sha256.items():
        require(type(name) is str and bool(name), "execution source name differs")
        hash_value(digest, f"source {name}")
    require(type(command) is list and bool(command), "worker command empty")
    require(all(type(part) is str and bool(part) for part in command), "worker command differs")
    object_fields(
        environment, {"python_version", "numpy_version", "platform", "processor"}, "environment"
    )
    require(all(type(value) is str for value in environment.values()), "environment types differ")
    require(type(started_utc) is str, "request time type differs")
    try:
        timestamp = datetime.fromisoformat(started_utc)
    except ValueError as error:
        raise ValueError("confirmation request time is malformed") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp),
        "request time must be UTC",
    )
    resolved = json_value(asdict(manifest))
    return {
        "schema_id": REQUEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "manifest": resolved,
        "manifest_sha256": digest_json(resolved),
        "scope_record_sha256": scope_sha256,
        "source_sha256": dict(source_sha256),
        "adapter_sha256": adapter_sha256,
        "summary": summary,
        "limits": {
            "max_optimizer_updates": manifest.max_optimizer_updates,
            "wall_limit_seconds": manifest.wall_limit_seconds,
            "max_process_rss_bytes": manifest.max_process_rss_bytes,
            "rss_interval_seconds": manifest.rss_interval_seconds,
        },
        "command": list(command),
        "environment": dict(environment),
        "started_utc": started_utc,
        "outer_selection_scored": False,
        "final_released": False,
    }


def verify_execution_request(payload: dict[str, Any], **bindings: Any) -> None:
    """Compare every field with independently supplied current identities."""
    require(type(payload) is dict, "request must be an object")
    started_utc = payload.get("started_utc")
    if not isinstance(started_utc, str):
        raise ValueError("confirmation request time type differs")
    expected = execution_request(started_utc=started_utc, **bindings)
    same_json(payload, expected, "execution request")


def work_summary(work: tuple[SeedWork, ...]) -> dict[str, Any]:
    rows = [
        dict(asdict(item), executed_optimizer_updates=item.executed_optimizer_updates)
        for item in work
    ]
    additive = (
        "cells",
        "wake_updates",
        "applied_replay_updates",
        "rejected_executed_replay_updates",
        "executed_optimizer_updates",
        "guarded_attempts",
        "guard_evaluations",
        "guard_examples",
        "retained_array_bytes_before_copies",
    )
    return {
        "by_seed": rows,
        "totals": {name: sum(row[name] for row in rows) for name in additive},
        "maximum_transient_width": max(item.maximum_transient_width for item in work),
    }


def verify_observed_updates(
    observed: dict[str, Any], work: tuple[SeedWork, ...], payload: dict[str, Any]
) -> None:
    object_fields(
        observed, {"attempted_updates", "executed_updates", "by_model_kind"}, "observed updates"
    )
    expected = sum(item.executed_optimizer_updates for item in work)
    same_json(observed["executed_updates"], expected, "live executed/derived work")
    same_json(observed["attempted_updates"], expected, "live attempted/derived work")
    counts = object_fields(
        observed["by_model_kind"], {"backprop", "pc", "circadian"}, "observed model kinds"
    )
    require(
        sum(integer(value, "observed model count") for value in counts.values()) == expected,
        "observed model count sum differs",
    )
    same_json(counts, _kind_work(payload), "live model kind/derived work")


def _kind_work(payload: dict[str, Any]) -> dict[str, int]:
    """Partition already validated raw method work by held model type."""
    kinds = {
        "src.core.backprop_mlp.BackpropMLP": "backprop",
        "src.core.predictive_coding.PredictiveCodingNetwork": "pc",
        "src.core.circadian_predictive_coding.CircadianPredictiveCodingNetwork": "circadian",
        "src.core.controlled_parent_selection.ParentControlledCircadianNetwork": "circadian",
    }
    expected = {"backprop": 0, "pc": 0, "circadian": 0}
    for row in payload["seed_results"]:
        simple = row["family"] in {"gating", "replay", "sleep"}
        methods = row["legacy_train_facts"]["arms" if row["family"] == "sleep" else "methods"]
        for method in methods:
            name = method["method" if row["family"] in {"gating", "replay"} else "name"]
            kind = kinds[row["after_b"][name]["model_type"]]
            replay = method["replay_updates" if simple else "applied_replay_updates"]
            expected[kind] += (
                method["wake_updates"] + replay + method.get("rejected_executed_replay_updates", 0)
            )
    return expected


def verify_process_memory(memory: dict[str, Any], manifest: ConfirmationManifest) -> None:
    object_fields(
        memory,
        {"pid", "start_bytes", "peak_bytes", "sample_count", "interval_seconds"},
        "process RSS",
    )
    integer(memory["pid"], "RSS pid", 1)
    start = integer(memory["start_bytes"], "RSS start", 1)
    peak = integer(memory["peak_bytes"], "RSS peak", 1)
    integer(memory["sample_count"], "RSS samples", 2)
    same_json(memory["interval_seconds"], manifest.rss_interval_seconds, "RSS interval")
    require(start <= peak <= manifest.max_process_rss_bytes, "RSS exceeds cap or start")
