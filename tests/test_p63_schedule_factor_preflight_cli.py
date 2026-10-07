"""The public schedule preflight preserves bounded exclusive train-only files."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_p63_schedule_factor_preflight as adapter
from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app.continual_schedule_factor_preflight import (
    fixed_schedule_factor_manifest,
    run_schedule_factor_preflight,
)


def test_should_write_and_read_all_public_schedule_preflight_artifacts(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_p63_schedule_factor_preflight",
        "--output-dir",
        str(tmp_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stderr
    paths = adapter.artifact_paths(tmp_path)
    assert not paths["failure"].exists()
    request = artifacts.parse_finite_json(paths["request"].read_text(encoding="utf-8"))
    result = artifacts.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = artifacts.parse_finite_json(paths["audit"].read_text(encoding="utf-8"))
    adapter.verify_result(result)
    artifacts.verify_memory(audit["process_rss"], 256 * 1024 * 1024)
    assert request["manifest_sha256"] == artifacts.digest_json(request["manifest"])
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert request["planned_wake_optimizer_updates"] == 792
    assert request["maximum_executed_optimizer_updates"] == 1014
    assert audit["cell_count"] == 33 and audit["decision_count"] == 216
    assert audit["work"] == adapter._work_summary(result)
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert result["final_released"] is False and result["outer_selection_scored"] is False
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    occupied = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert occupied.returncode != 0 and "already exists" in occupied.stderr
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


def test_should_reject_changed_source_before_request_and_forged_method_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setitem(
            adapter.ADDITIONAL_SOURCE_SHA256,
            "src/app/continual_schedule_factor_preflight.py",
            "0" * 64,
        )
        with pytest.raises(ValueError, match="frozen source changed"):
            adapter.run_bounded_preflight(tmp_path)
    assert list(tmp_path.iterdir()) == []
    result = json.loads(
        json.dumps(asdict(run_schedule_factor_preflight(fixed_schedule_factor_manifest())))
    )
    result["seed_results"][0]["methods"][0]["applied_replay_updates"] += 2
    with pytest.raises(ValueError, match="work/capacity differs"):
        adapter.verify_result(result)


def test_should_save_timeout_failure_without_completed_schedule_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=120)

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.run_bounded_preflight(tmp_path)
    paths = adapter.artifact_paths(tmp_path)
    assert paths["request"].exists() and paths["failure"].exists()
    assert not paths["result"].exists() and not paths["audit"].exists()
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == "wall_limit"
