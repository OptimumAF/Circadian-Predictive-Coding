"""Bounded parent CLI binds all facts and preserves exclusive/failure artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any

import pytest

from scripts import run_p63_parent_factor_preflight as adapter
from scripts import run_p63_sleep_factor_preflight as artifacts


def _read(path: Path) -> dict[str, Any]:
    return artifacts.parse_finite_json(path.read_text(encoding="utf-8"))


def test_should_repeat_complete_parent_bundles_and_refuse_occupied_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = adapter.run_bounded_preflight(tmp_path / "first")
    second = adapter.run_bounded_preflight(tmp_path / "second")
    paths = adapter.artifact_paths(tmp_path / "first")
    duplicate = adapter.artifact_paths(tmp_path / "second")
    assert first["result_sha256"] == second["result_sha256"]
    assert paths["result"].read_bytes() == duplicate["result"].read_bytes()
    request, result = _read(paths["request"]), _read(paths["result"])
    adapter.verify_result(result)
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert request["manifest"] == adapter.manifest_payload()
    assert request["manifest_sha256"] == artifacts.digest_json(request["manifest"])
    assert request["adapter_sha256"] == sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    assert (
        request["maximum_executed_optimizer_updates"]
        == request["planned_wake_optimizer_updates"]
        == 576
    )
    assert request["persistent_labeled_array_bytes_per_seed"] == 960
    assert first["seed_count"] == 3 and first["cell_count"] == 24 and first["decision_count"] == 216
    assert first["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert first["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert first["work"] == adapter._work_summary(result)
    assert first["work"]["executed_optimizer_updates"] == 576
    assert first["work"]["guarded_attempts"] == 54 and first["work"]["guard_evaluations"] == 108
    assert (
        first["work"]["guard_examples"] == 1944 and first["work"]["zero_add_guarded_attempts"] == 9
    )
    assert 0 <= first["elapsed_seconds"] <= request["wall_limit_seconds"] == 120
    artifacts.verify_memory(first["process_rss"], request["max_process_rss_bytes"])
    assert not paths["failure"].exists() and not duplicate["failure"].exists()
    saved = {key: path.read_bytes() for key, path in paths.items() if path.exists()}

    def forbidden_launch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("occupied parent outputs launched work")

    monkeypatch.setattr(adapter.subprocess, "run", forbidden_launch)
    with pytest.raises(FileExistsError, match="already exists"):
        adapter.run_bounded_preflight(tmp_path / "first")
    assert saved == {key: path.read_bytes() for key, path in paths.items() if path.exists()}
    forged = deepcopy(result)
    forged["seed_results"][-1]["methods"][-1]["wake_updates"] = 23
    with pytest.raises(ValueError, match="parent"):
        adapter.verify_result(forged)


@pytest.mark.parametrize("case", ["source", "manifest"])
def test_should_reject_changed_source_or_manifest_before_launch(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_launch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid parent request launched work")

    monkeypatch.setattr(adapter.subprocess, "run", forbidden_launch)
    if case == "source":
        monkeypatch.setitem(
            adapter.ADDITIONAL_SOURCE_SHA256, "src/core/controlled_parent_selection.py", "0" * 64
        )
    else:
        manifest = replace(adapter.fixed_parent_manifest(), selector_seed_offset=5002)
        monkeypatch.setattr(adapter, "fixed_parent_manifest", lambda: manifest)
    with pytest.raises(ValueError, match="frozen"):
        adapter.run_bounded_preflight(tmp_path / "invalid")
    assert not any((tmp_path / "invalid").glob("*.json"))


@pytest.mark.parametrize("case", ["timeout", "nonfinite", "exit", "incomplete"])
def test_should_record_failure_without_false_completed_result(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(command: list[str], **kwargs: Any) -> Any:
        assert kwargs["timeout"] == 120 and command[-1] == "--worker"
        if case == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if case == "exit":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="injected failure")
        stdout = '{"result": NaN}' if case == "nonfinite" else '{"result": {}}'
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fail)
    output = tmp_path / case
    with pytest.raises((subprocess.TimeoutExpired, ValueError, RuntimeError)):
        adapter.run_bounded_preflight(output)
    paths = adapter.artifact_paths(output)
    assert paths["request"].is_file() and paths["failure"].is_file()
    assert not paths["result"].exists() and not paths["audit"].exists()
    assert _read(paths["failure"])["reason"] == (
        "wall_limit" if case == "timeout" else "worker_or_audit"
    )
