"""Frozen selection evidence must reject tampering before final access."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import restore_cifar_representative_selection as restore


def test_should_reject_changed_selection_bytes_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    journal = tmp_path / "journal.jsonl"
    result.write_text('{"changed":true}\n', encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    journal.write_text("", encoding="utf-8")

    def unexpected(*args: Any) -> Any:
        raise AssertionError("source accessed before frozen artifact check")

    monkeypatch.setattr(restore.selection, "_verify_saved_request", unexpected)

    with pytest.raises(ValueError, match="selection result digest changed"):
        restore.read_saved_selection(tmp_path / "request.json", result, manifest, journal)


def test_should_reject_missing_journal_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    result.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(restore, "RESULT_SHA256", sha256(result.read_bytes()).hexdigest())
    monkeypatch.setattr(restore, "MANIFEST_SHA256", sha256(manifest.read_bytes()).hexdigest())

    def unexpected(*args: Any) -> Any:
        raise AssertionError("source accessed before missing journal check")

    monkeypatch.setattr(restore.selection, "_verify_saved_request", unexpected)
    with pytest.raises(FileNotFoundError, match="attempt journal is missing"):
        restore.read_saved_selection(
            tmp_path / "request.json", result, manifest, tmp_path / "missing"
        )


def test_should_reject_changed_manifest_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    journal = tmp_path / "journal.jsonl"
    result.write_text("{}\n", encoding="utf-8")
    manifest.write_text('{"freeze_digest":"changed"}\n', encoding="utf-8")
    journal.write_text("", encoding="utf-8")
    monkeypatch.setattr(restore, "RESULT_SHA256", sha256(result.read_bytes()).hexdigest())

    def unexpected(*args: Any) -> Any:
        raise AssertionError("source accessed before frozen manifest check")

    monkeypatch.setattr(restore.selection, "_verify_saved_request", unexpected)
    with pytest.raises(ValueError, match="selection manifest digest changed"):
        restore.read_saved_selection(tmp_path / "request.json", result, manifest, journal)


def test_should_reject_changed_final_iteration_without_using_manifest() -> None:
    request = {"source": {}, "selection_limit_seconds": 180}
    result = {
        "schema": "cifar_representative_validation_selection_result_v1",
        "request_sha256": restore.selection.REQUEST_SHA256,
        "source_hashes": {},
        "attempt_journal_sha256": sha256(b"").hexdigest(),
        "final_test_source_constructions": 0,
        "final_test_iterations": 1,
        "worker_elapsed_seconds": 1.0,
    }
    with pytest.raises(ValueError, match="final-source gates"):
        restore._verify_freeze(request, result, {}, b"", ())


def test_should_reject_journal_that_changes_a_completed_trial() -> None:
    attempt = {"head_name": "backprop_mlp", "status": "complete", "error": None}
    trial = {"head_name": "backprop_mlp", "validation_accuracy": 0.5}
    events = (
        {"attempt": {**attempt, "status": "started"}, "trial": None},
        {"attempt": attempt, "trial": {**trial, "validation_accuracy": 0.9}},
    )
    result = {"selection": {"attempts": [attempt], "trials": [trial]}}

    with pytest.raises(ValueError, match="journal disagrees"):
        restore._verify_journal(events, result)


def test_should_reject_malformed_journal_before_source_access() -> None:
    with pytest.raises(ValueError, match="malformed"):
        restore._parse_journal(b"{broken\n")
    with pytest.raises(ValueError, match="12 events"):
        restore._parse_journal((json.dumps({"attempt": {}}) + "\n").encode())
