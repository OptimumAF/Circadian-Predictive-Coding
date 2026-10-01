"""Scope IO rejects used reservations and protects prior evidence outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts import inspect_p67_confirmation_scope as adapter
from scripts import run_p63_sleep_factor_preflight as artifacts


@pytest.mark.parametrize("location", ["seed_results", "scored_seeds", "train_facts"])
def test_should_reject_a_reserved_seed_in_any_actual_result_rows(
    tmp_path: Path, location: str
) -> None:
    directory = tmp_path / "artifacts/runs/p63-example"
    directory.mkdir(parents=True)
    payload: dict[str, Any] = {location: [{"seed": 101}]}
    if location == "train_facts":
        payload = {"train_facts": {"seed_results": [{"seed": 101}]}}
    artifacts.write_exclusive(directory / "example.result.json", payload)
    with pytest.raises(ValueError, match="reservation already appears"):
        adapter._verify_unused_reservations(tmp_path, {101})


def test_should_distinguish_declarations_from_actual_source_usage(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts/runs/p63-example"
    directory.mkdir(parents=True)
    payload = {"manifest": {"confirmation_seeds": [101]}, "seed_results": [{"seed": 41}]}
    artifacts.write_exclusive(directory / "example.result.json", payload)
    result = adapter._verify_unused_reservations(tmp_path, {101})
    assert result["observed_source_seeds"] == [41]
    assert len(result["result_files_checked"]) == 1


def test_should_fail_malformed_usage_rows_loudly(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts/runs/p63-example"
    directory.mkdir(parents=True)
    artifacts.write_exclusive(
        directory / "example.result.json", {"seed_results": [{"seed": "101"}]}
    )
    with pytest.raises(ValueError, match="malformed seed row"):
        adapter._verify_unused_reservations(tmp_path, {101})


def test_should_refuse_occupied_output_before_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scope.json"
    path.write_bytes(b"existing evidence")

    def no_inspection() -> dict[str, Any]:
        raise AssertionError("occupied scope output entered inspection")

    monkeypatch.setattr(adapter, "inspect_confirmation_scope", no_inspection)
    with pytest.raises(FileExistsError, match="already exists"):
        adapter.publish_scope_inspection(path)
    assert path.read_bytes() == b"existing evidence"


def test_should_record_complete_finite_scope_and_saved_byte_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"summary": {"cell_count": 560}, "confirmation_scored": False, "final_released": False}
    monkeypatch.setattr(adapter, "inspect_confirmation_scope", lambda: report)
    path = tmp_path / "nested/scope.json"
    identity = adapter.publish_scope_inspection(path)
    from hashlib import sha256

    assert identity["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert artifacts.parse_finite_json(path.read_text(encoding="utf-8")) == report
