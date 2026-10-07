"""The public train-only sleep-factor writer verifies budgeted artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_p63_sleep_factor_preflight as adapter
from src.app.continual_sleep_factor_preflight import (
    fixed_sleep_factor_manifest,
    run_sleep_factor_preflight,
)


def test_should_run_public_train_only_preflight_and_read_complete_artifacts(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_p63_sleep_factor_preflight",
        "--output-dir",
        str(tmp_path),
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert first.returncode == 0, first.stderr
    paths = adapter.artifact_paths(tmp_path)
    assert not paths["failure"].exists()
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    result = adapter.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    adapter.verify_result(result)
    adapter.verify_memory(audit["process_rss"], 256 * 1024 * 1024)
    assert request["manifest_sha256"] == adapter.digest_json(request["manifest"])
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert request["planned_optimizer_updates"] == 648
    assert request["wall_limit_seconds"] == 120
    assert audit["cell_count"] == 27
    assert audit["guarded_sleep_attempts"] == 15
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert result["outer_selection_scored"] is False
    assert result["final_released"] is False

    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    occupied = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert occupied.returncode != 0
    assert "already exists" in occupied.stderr
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


def test_should_reject_changed_source_and_tampered_train_only_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "src/app/continual_sleep_factor_preflight.py"
    with monkeypatch.context() as patch:
        patch.setitem(adapter.SOURCE_SHA256, name, "0" * 64)
        with pytest.raises(ValueError, match="frozen source changed"):
            adapter.run_bounded_preflight(tmp_path)
    assert list(tmp_path.iterdir()) == []

    result = json.loads(
        json.dumps(asdict(run_sleep_factor_preflight(fixed_sleep_factor_manifest())))
    )
    adapter.verify_result(result)
    broken = deepcopy(result)
    broken["seed_results"][0]["arms"][3]["sleep"]["applied_split_pairs"] = []
    with pytest.raises(ValueError, match="structural proposal"):
        adapter.verify_result(broken)
    broken = deepcopy(result)
    broken["outer_selection_scored"] = True
    with pytest.raises(ValueError, match="evaluation seal differs"):
        adapter.verify_result(broken)
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.parse_finite_json('{"x": NaN}')


def test_should_save_failure_after_child_timeout_without_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=120)

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.run_bounded_preflight(tmp_path)
    paths = adapter.artifact_paths(tmp_path)
    assert paths["request"].exists()
    assert not paths["result"].exists()
    assert not paths["audit"].exists()
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == "wall_limit"
