"""Scored combined artifacts bind every c7 fact, cost, cell and contrast."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_p63_combined_factor_development as adapter
from scripts import run_p63_combined_factor_preflight as c7_adapter
from scripts import run_p63_sleep_factor_preflight as artifacts


from factor_cli_value_fixtures import factor_cli_command, select_factor_reference


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("combined-train-reference")
    c7_adapter.run_bounded_preflight(directory)
    return directory


@pytest.fixture(autouse=True)
def local_reference(monkeypatch: pytest.MonkeyPatch, reference_dir: Path) -> None:
    # Why this: boundary cases need exact local pins, never ignored historic paths.
    select_factor_reference(monkeypatch, adapter, reference_dir)


def test_should_reject_changed_reference_bytes_or_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_paths = c7_adapter.artifact_paths(adapter.REFERENCE_DIR)
    target_paths = c7_adapter.artifact_paths(tmp_path)
    for name in ("request", "result", "audit"):
        shutil.copyfile(source_paths[name], target_paths[name])
    monkeypatch.setattr(adapter, "REFERENCE_DIR", tmp_path)
    assert adapter.read_reference()["outer_selection_scored"] is False
    target_paths["result"].write_bytes(target_paths["result"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="result bytes differ"):
        adapter.read_reference()
    shutil.copyfile(source_paths["result"], target_paths["result"])
    audit = json.loads(target_paths["audit"].read_text(encoding="utf-8"))
    audit["work"]["executed_optimizer_updates"] += 1
    target_paths["audit"].write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="request or audit differs"):
        adapter.read_reference()


def test_should_publish_all_score_cells_and_reject_metric_contrast_or_fact_tampering(
    tmp_path: Path,
) -> None:
    command = factor_cli_command(
        adapter.__name__, adapter.REFERENCE_DIR, "--output-dir", str(tmp_path)
    )
    process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stderr
    paths = adapter.artifact_paths(tmp_path)
    assert not paths["failure"].exists()
    request = artifacts.parse_finite_json(paths["request"].read_text(encoding="utf-8"))
    result = artifacts.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = artifacts.parse_finite_json(paths["audit"].read_text(encoding="utf-8"))
    reference = adapter.read_reference()
    adapter.verify_result(result, reference)
    artifacts.verify_memory(audit["process_rss"], 256 * 1024 * 1024)
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert len(request["source_sha256"]) == 29
    assert request["manifest"] == c7_adapter.manifest_payload()
    assert request["manifest_sha256"] == artifacts.digest_json(request["manifest"])
    assert request["adapter_sha256"] == sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    assert request["planned_guarded_attempts"] == audit["work"]["guarded_attempts"] == 144
    assert request["planned_guard_evaluations"] == audit["work"]["guard_evaluations"] == 288
    assert audit["contrast_count"] == 66
    assert request["reference_sha256"] == adapter.REFERENCE_SHA256
    assert request["paired_contrasts"] == [list(pair) for pair in adapter.CONTRASTS]
    assert request["planned_wake_optimizer_updates"] == 1224
    assert request["maximum_executed_optimizer_updates"] == 1548
    assert request["planned_outer_evaluations"] == audit["outer_evaluations"] == 153
    assert request["planned_outer_examples"] == audit["outer_examples"] == 3060
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert audit["work"] == c7_adapter._work_summary(reference)
    assert audit["cell_count"] == 51 and audit["decision_count"] == 648
    assert len(result["scored_seeds"]) == 3
    assert sum(len(seed["contrasts"]) for seed in result["scored_seeds"]) == 66

    broken = deepcopy(result)
    broken["scored_seeds"][0]["arms"][0]["final_mean_task_accuracy"] = 0.123
    with pytest.raises(ValueError, match="derived metrics differ"):
        adapter.verify_result(broken, reference)
    broken = deepcopy(result)
    broken["scored_seeds"][-1]["contrasts"][-1]["signed_forgetting_a"] = 0.123
    with pytest.raises(ValueError, match="contrast differs"):
        adapter.verify_result(broken, reference)
    broken = deepcopy(result)
    broken["train_facts"]["seed_results"][-1]["executed_optimizer_updates"] += 1
    with pytest.raises(ValueError, match="global train gate differs"):
        adapter.verify_result(broken, reference)
    broken = deepcopy(result)
    broken["scored_seeds"][-1]["arms"].pop()
    with pytest.raises(ValueError, match="model cells differ"):
        adapter.verify_result(broken, reference)
    broken = deepcopy(result)
    broken["scored_seeds"][-1]["contrasts"].pop()
    with pytest.raises(ValueError, match="paired contrasts differ"):
        adapter.verify_result(broken, reference)

    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    occupied = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert occupied.returncode != 0 and "already exists" in occupied.stderr
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


def test_should_reject_changed_source_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        adapter.ADDITIONAL_SOURCE_SHA256,
        "src/app/continual_combined_factor_development.py",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="frozen source changed"):
        adapter.run_bounded_development(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_kind", ["timeout", "nonfinite"])
def test_should_save_failure_without_result_or_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:

    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(cmd="worker", timeout=120)
        return SimpleNamespace(returncode=0, stdout='{"result": NaN}', stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", failed)
    with pytest.raises((subprocess.TimeoutExpired, ValueError)):
        adapter.run_bounded_development(tmp_path)
    paths = adapter.artifact_paths(tmp_path)
    assert paths["request"].exists()
    assert not paths["result"].exists() and not paths["audit"].exists()
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == ("wall_limit" if failure_kind == "timeout" else "worker_or_audit")
