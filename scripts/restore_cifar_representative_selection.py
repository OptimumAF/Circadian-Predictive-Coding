"""Read the exact P1.8p1 freeze before representative final-test access.

This module only reads saved evidence and source file hashes. It never
constructs a CIFAR dataset, trains a head, or scores a final example.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_cifar_representative_selection as selection  # noqa: E402
from src.app import matched_head_tuning as tuning  # noqa: E402
from src.app.repeated_head_confirmation import (  # noqa: E402
    RepeatedConfirmationManifest,
    restore_confirmation_manifest,
)

RESULT_SHA256 = "3746e32459201c47198fa8eb529e03049a08da0bfa406e0a2dc678305927b1c6"
MANIFEST_SHA256 = "6566678860366ee0435e70436fad1cd7fad667917823ce29fda079fcd60b8a6a"
JOURNAL_SHA256 = "49afd5c236354aa8b55ef3d413ababc1258c38acb57edefa0bb30e9746cc7741"


@dataclass(frozen=True)
class RestoredRepresentativeSelection:
    request: dict[str, Any]
    result: dict[str, Any]
    freeze: dict[str, Any]
    attempt_journal: tuple[dict[str, Any], ...]
    confirmation_manifest: RepeatedConfirmationManifest


def _read_exact(path: Path, expected_digest: str, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"frozen {label} is missing: {path}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_digest:
        raise ValueError(f"frozen {label} digest changed")
    return raw


def _parse_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"frozen {label} must be a JSON object")
    return value


def _parse_journal(raw: bytes) -> tuple[dict[str, Any], ...]:
    try:
        events = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen attempt journal is malformed") from error
    if len(events) != 12 or any(not isinstance(event, dict) for event in events):
        raise ValueError("frozen attempt journal must contain 12 events")
    return events


def _verify_journal(events: tuple[dict[str, Any], ...], result: dict[str, Any]) -> None:
    attempts = result["selection"]["attempts"]
    trials = result["selection"]["trials"]
    for index, (attempt, trial) in enumerate(zip(attempts, trials, strict=True)):
        started, completed = events[2 * index : 2 * index + 2]
        expected_start = {**attempt, "status": "started", "error": None}
        if (
            set(started) != {"attempt", "trial"}
            or started["attempt"] != expected_start
            or started["trial"] is not None
            or completed != {"attempt": attempt, "trial": trial}
        ):
            raise ValueError("frozen attempt journal disagrees with trial ledger")


def _verify_freeze(
    request: dict[str, Any],
    result: dict[str, Any],
    freeze: dict[str, Any],
    journal_raw: bytes,
    events: tuple[dict[str, Any], ...],
) -> RepeatedConfirmationManifest:
    if (
        result.get("schema") != "cifar_representative_validation_selection_result_v1"
        or result.get("request_sha256") != selection.REQUEST_SHA256
        or result.get("source_hashes") != request["source"]
        or result.get("attempt_journal_sha256") != sha256(journal_raw).hexdigest()
        or result.get("final_test_source_constructions") != 0
        or result.get("final_test_iterations") != 0
        or result.get("worker_elapsed_seconds", float("inf")) > request["selection_limit_seconds"]
    ):
        raise ValueError("frozen selection result fails provenance or final-source gates")
    quiet = result.get("quiet_window", [])
    gate = request["quiet_window"]
    if len(quiet) != gate["readings"] or any(
        row["utilization_percent"] > gate["max_utilization_percent"]
        or row["free_mib"] < gate["min_free_mib"]
        for row in quiet
    ):
        raise ValueError("frozen selection missed its quiet CUDA gate")
    selected = result["selection"]
    if (
        selected.get("protocol_id") != tuning.MATCHED_HEAD_VALIDATION_SELECTION_PROTOCOL
        or selected.get("base_config") != request["base_config"]
        or selected.get("seeds") != request["selection_seeds"]
        or selected.get("candidates_per_head") != request["candidates_per_head"]
        or selected.get("trials_per_head") != request["candidates_per_head"]
    ):
        raise ValueError("frozen selection changed its declared protocol or grid")
    selection._validate_worker_payload(
        {
            "status": "complete",
            "final_test_source_constructions": 0,
            "selection": selected,
            "confirmation_manifest": freeze["typed_manifest"],
        },
        request,
    )
    if freeze != selection._freeze_manifest(request, result, freeze["typed_manifest"]):
        raise ValueError("frozen selection envelope or digest changed")
    _verify_journal(events, result)
    typed = restore_confirmation_manifest(freeze["typed_manifest"])
    if asdict(typed.base_config) != request["base_config"]:
        raise ValueError("confirmation base config differs from frozen request")
    choices = {row["head_name"]: row["candidate_id"] for row in selected["selections"]}
    for head in typed.selected_heads:
        declared = next(
            row["config"]
            for row in request["candidate_grid"][head.head_name]
            if row["candidate_id"] == choices[head.head_name]
        )
        if head.candidate_id != choices[head.head_name] or asdict(head.config) != declared:
            raise ValueError("confirmation head differs from frozen outer choice")
    return typed


def read_saved_selection(
    request_path: Path = selection.REQUEST_PATH,
    result_path: Path = selection.RESULT_PATH,
    manifest_path: Path = selection.MANIFEST_PATH,
    journal_path: Path = selection.JOURNAL_PATH,
) -> RestoredRepresentativeSelection:
    """Restore exact saved bytes and validate every pre-final decision."""
    # Why this order: corrupt selection evidence fails before even source
    # hashing; no CIFAR dataset or final label is ever constructed here.
    result_raw = _read_exact(result_path, RESULT_SHA256, "selection result")
    manifest_raw = _read_exact(manifest_path, MANIFEST_SHA256, "selection manifest")
    journal_raw = _read_exact(journal_path, JOURNAL_SHA256, "attempt journal")
    result = _parse_object(result_raw, "selection result")
    freeze = _parse_object(manifest_raw, "selection manifest")
    events = _parse_journal(journal_raw)
    request, _ = selection._verify_saved_request(request_path)
    typed = _verify_freeze(request, result, freeze, journal_raw, events)
    return RestoredRepresentativeSelection(request, result, freeze, events, typed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.parse_args()
    restored = read_saved_selection()
    print(
        json.dumps(
            {
                "request_sha256": selection.REQUEST_SHA256,
                "selection_sha256": restored.freeze["selection_sha256"],
                "manifest_digest": restored.confirmation_manifest.manifest_digest,
                "freeze_digest": restored.freeze["freeze_digest"],
                "complete_trials": len(restored.result["selection"]["trials"]),
                "final_test_iterations": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
