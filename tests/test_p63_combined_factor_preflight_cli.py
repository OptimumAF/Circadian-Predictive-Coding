"""The combined public gate saves complete bounded exclusive train facts."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_p63_combined_factor_preflight as adapter
from scripts import run_p63_sleep_factor_preflight as artifacts


def test_should_publish_all_combined_cells_and_verify_exclusive_artifacts(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_p63_combined_factor_preflight",
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
    assert request["manifest"] == adapter.manifest_payload()
    assert request["manifest_sha256"] == artifacts.digest_json(request["manifest"])
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert request["adapter_sha256"] == sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    assert request["planned_wake_optimizer_updates"] == 1224
    assert request["maximum_executed_optimizer_updates"] == 1548
    assert request["planned_guarded_attempts"] == audit["work"]["guarded_attempts"] == 144
    assert request["planned_guard_evaluations"] == audit["work"]["guard_evaluations"] == 288
    assert audit["cell_count"] == 51 and audit["decision_count"] == 648
    assert audit["work"] == adapter._work_summary(result)
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert result["outer_selection_scored"] is False and result["final_released"] is False
    assert all(len(seed["methods"]) == 17 for seed in result["seed_results"])
    assert all(len(seed["opportunities"]) == 24 for seed in result["seed_results"])

    broken = deepcopy(result)
    broken["seed_results"][-1]["executed_optimizer_updates"] += 1
    with pytest.raises(ValueError, match="optimizer total"):
        adapter.verify_result(broken)
    broken = deepcopy(result)
    broken["seed_results"][0]["opportunities"][0]["decisions"][0]["state_sha256_after"] = "0" * 64
    with pytest.raises(ValueError):
        adapter.verify_result(broken)
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    occupied = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert occupied.returncode != 0 and "already exists" in occupied.stderr
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


def test_should_reject_changed_source_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        adapter.ADDITIONAL_SOURCE_SHA256, "src/app/continual_combined_factor_manifest.py", "0" * 64
    )
    with pytest.raises(ValueError, match="frozen source changed"):
        adapter.run_bounded_preflight(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["timeout", "nonfinite"])
def test_should_save_failure_without_unscored_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(cmd="worker", timeout=120)
        return SimpleNamespace(returncode=0, stdout='{"result": NaN}', stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", failed)
    with pytest.raises((subprocess.TimeoutExpired, ValueError)):
        adapter.run_bounded_preflight(tmp_path)
    paths = adapter.artifact_paths(tmp_path)
    assert paths["request"].exists() and paths["failure"].exists()
    assert not paths["result"].exists() and not paths["audit"].exists()
    failure = artifacts.parse_finite_json(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == ("wall_limit" if failure_kind == "timeout" else "worker_or_audit")
