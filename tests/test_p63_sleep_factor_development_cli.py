"""Public scored sleep-factor artifacts bind c3 and all development cells."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import run_p63_sleep_factor_development as adapter
from scripts import run_p63_sleep_factor_preflight as c3_adapter


from factor_cli_value_fixtures import factor_cli_command, select_factor_reference


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("sleep-train-reference")
    c3_adapter.run_bounded_preflight(directory)
    return directory


@pytest.fixture(autouse=True)
def local_reference(monkeypatch: pytest.MonkeyPatch, reference_dir: Path) -> None:
    # Why this: boundary cases need exact local pins, never ignored historic paths.
    select_factor_reference(monkeypatch, adapter, reference_dir)


def test_should_reject_changed_c3_result_bytes_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_paths = c3_adapter.artifact_paths(adapter.REFERENCE_DIR)
    target_paths = c3_adapter.artifact_paths(tmp_path)
    for name in ("request", "result", "audit"):
        shutil.copyfile(source_paths[name], target_paths[name])
    monkeypatch.setattr(adapter, "REFERENCE_DIR", tmp_path)
    assert adapter.read_reference()["final_released"] is False

    target_paths["result"].write_bytes(target_paths["result"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="result bytes differ"):
        adapter.read_reference()
    shutil.copyfile(source_paths["result"], target_paths["result"])
    audit = json.loads(target_paths["audit"].read_text(encoding="utf-8"))
    audit["result_sha256"] = "0" * 64
    target_paths["audit"].write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="request or audit differs"):
        adapter.read_reference()


def test_should_write_complete_development_result_and_reject_tampered_metrics(
    tmp_path: Path,
) -> None:
    command = factor_cli_command(
        adapter.__name__, adapter.REFERENCE_DIR, "--output-dir", str(tmp_path)
    )
    process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stderr
    paths = adapter.artifact_paths(tmp_path)
    assert not paths["failure"].exists()
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    result = c3_adapter.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    reference = adapter.read_reference()
    adapter.verify_result(result, reference)
    c3_adapter.verify_memory(audit["process_rss"], 256 * 1024 * 1024)
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert request["reference_sha256"] == adapter.REFERENCE_SHA256
    assert request["planned_optimizer_updates"] == 648
    assert request["planned_outer_evaluations"] == 81
    assert request["planned_outer_examples"] == 1620
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert audit["cell_count"] == 27 and audit["outer_evaluations"] == 81

    broken = deepcopy(result)
    broken["scored_seeds"][0]["arms"][0]["final_mean_task_accuracy"] = 0.123
    with pytest.raises(ValueError, match="derived metrics differ"):
        adapter.verify_result(broken, reference)
    broken = deepcopy(result)
    broken["scored_seeds"][1]["contrasts"][0]["signed_forgetting_a"] = 0.123
    with pytest.raises(ValueError, match="contrast differs"):
        adapter.verify_result(broken, reference)
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    occupied = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert occupied.returncode != 0
    assert "already exists" in occupied.stderr
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


def test_should_reject_changed_scored_source_before_writing_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setitem(
            adapter.ADDITIONAL_SOURCE_SHA256,
            "src/app/continual_sleep_factor_development.py",
            "0" * 64,
        )
        with pytest.raises(ValueError, match="frozen source changed"):
            adapter.run_bounded_development(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_should_save_failure_after_child_timeout_without_scored_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=120)

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.run_bounded_development(tmp_path)
    paths = adapter.artifact_paths(tmp_path)
    assert paths["request"].exists()
    assert not paths["result"].exists()
    assert not paths["audit"].exists()
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == "wall_limit"
