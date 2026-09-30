"""The P6.3 pilot adapter budgets, verifies, and protects local artifacts."""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from scripts import run_p63_gating_pilot as study
from src.app.continual_gating_pilot import fixed_gating_pilot_manifest, run_gating_pilot


def complete_payload() -> dict[str, Any]:
    return study.parse_finite_json(
        json.dumps(asdict(run_gating_pilot(fixed_gating_pilot_manifest())))
    )


def test_should_save_frozen_request_before_worker_and_audit_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = study.artifact_paths(tmp_path)
    payload = complete_payload()
    calls = 0

    def fake_worker(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        request = study.parse_finite_json(paths["request"].read_text(encoding="utf-8"))
        assert request["manifest"] == study.manifest_payload()
        assert request["source_sha256"] == study.SOURCE_SHA256
        assert request["command"] == command
        assert request["planned_wake_updates"] == 216
        assert request["wall_limit_seconds"] == kwargs["timeout"] == 120
        assert not paths["result"].exists()
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(study.subprocess, "run", fake_worker)

    audit = study.run_bounded_pilot(tmp_path)

    assert calls == 1
    assert audit["status"] == "completed"
    assert audit["cell_count"] == 9
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert study.parse_finite_json(paths["result"].read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        study.run_bounded_pilot(tmp_path)
    assert calls == 1


def test_should_record_timeout_without_publishing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(study.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        study.run_bounded_pilot(tmp_path)
    paths = study.artifact_paths(tmp_path)
    failure = study.parse_finite_json(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == "wall_limit"
    assert paths["request"].exists()
    assert not paths["result"].exists()
    assert not paths["audit"].exists()


def test_should_reject_tampered_work_and_final_release() -> None:
    payload = complete_payload()
    study.verify_result(payload)

    wrong_work = deepcopy(payload)
    wrong_work["seed_results"][0]["methods"][0]["wake_updates"] = 23
    with pytest.raises(ValueError, match="work or capacity"):
        study.verify_result(wrong_work)

    opened_final = deepcopy(payload)
    opened_final["final_released"] = True
    with pytest.raises(ValueError, match="final seal"):
        study.verify_result(opened_final)


def test_should_reject_overflowed_finite_parser_number() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        study.parse_finite_json('{"score": 1e309}')
