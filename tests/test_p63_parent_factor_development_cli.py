"""Parent scoring boundaries use fresh isolated bundles, never ignored goldens."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_p63_parent_factor_development as adapter
from scripts import run_p63_parent_factor_preflight as c9b_adapter
from scripts import run_p63_sleep_factor_preflight as artifacts
from src.app import continual_parent_factor_development as development


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("parent-train-reference")
    c9b_adapter.run_bounded_preflight(directory)
    return directory


def _fixture_hashes(directory: Path) -> dict[str, str]:
    paths = c9b_adapter.artifact_paths(directory)
    return {
        name: sha256(paths[name].read_bytes()).hexdigest()
        for name in ("request", "result", "audit")
    }


def _select_fixture_reference(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    # Why this: local numerical result identity also differs across environments.
    monkeypatch.setattr(adapter, "REFERENCE_DIR", directory)
    monkeypatch.setattr(adapter, "REFERENCE_FILE_SHA256", _fixture_hashes(directory))
    monkeypatch.setattr(adapter, "REFERENCE_SHA256", _fixture_hashes(directory)["result"])
    monkeypatch.setattr(development, "REFERENCE_SHA256", _fixture_hashes(directory)["result"])


@pytest.fixture(scope="module")
def worker_pair(reference_dir: Path) -> tuple[subprocess.CompletedProcess[str], ...]:
    # A real child exercises the worker/main boundary without requiring ignored
    # canonical artifacts in a clean clone. Fixture identities stay test-local.
    bootstrap = """
from hashlib import sha256
from pathlib import Path
import sys
from scripts import run_p63_parent_factor_development as adapter
from scripts import run_p63_parent_factor_preflight as c9b
from src.app import continual_parent_factor_development as development
adapter.REFERENCE_DIR = Path(sys.argv[1])
paths = c9b.artifact_paths(adapter.REFERENCE_DIR)
adapter.REFERENCE_FILE_SHA256 = {
    name: sha256(paths[name].read_bytes()).hexdigest()
    for name in ('request', 'result', 'audit')
}
adapter.REFERENCE_SHA256 = adapter.REFERENCE_FILE_SHA256['result']
development.REFERENCE_SHA256 = adapter.REFERENCE_SHA256
sys.argv = [sys.argv[0], '--worker']
adapter.main()
"""
    command = [sys.executable, "-c", bootstrap, str(reference_dir)]
    results = tuple(
        subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        for _ in range(2)
    )
    for result in results:
        assert result.returncode == 0, result.stderr
    return results


@pytest.mark.parametrize("name", ["request", "result", "audit"])
def test_should_reject_changed_reference_bytes_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_dir: Path, name: str
) -> None:
    source = c9b_adapter.artifact_paths(reference_dir)
    target = c9b_adapter.artifact_paths(tmp_path)
    for file_name in ("request", "result", "audit"):
        shutil.copyfile(source[file_name], target[file_name])
    _select_fixture_reference(monkeypatch, tmp_path)
    assert adapter.read_reference()["outer_selection_scored"] is False
    target[name].write_bytes(target[name].read_bytes() + b" ")
    with pytest.raises(ValueError, match=f"{name} bytes differ"):
        adapter.read_reference()


def test_should_reject_rehashed_audit_and_incomplete_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_dir: Path
) -> None:
    source = c9b_adapter.artifact_paths(reference_dir)
    target = c9b_adapter.artifact_paths(tmp_path)
    for name in ("request", "result", "audit"):
        shutil.copyfile(source[name], target[name])
    audit = json.loads(target["audit"].read_text(encoding="utf-8"))
    audit["work"]["executed_optimizer_updates"] += 1
    target["audit"].write_text(json.dumps(audit), encoding="utf-8")
    _select_fixture_reference(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="request or audit differs"):
        adapter.read_reference()
    target["failure"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="complete c9b reference"):
        adapter.read_reference()
    target["failure"].unlink()
    target["audit"].unlink()
    with pytest.raises(ValueError, match="complete c9b reference"):
        adapter.read_reference()


def test_should_repeat_real_workers_publish_complete_artifacts_and_refuse_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_dir: Path,
    worker_pair: tuple[subprocess.CompletedProcess[str], ...],
) -> None:
    _select_fixture_reference(monkeypatch, reference_dir)
    first, repeat = (artifacts.parse_finite_json(item.stdout) for item in worker_pair)
    assert first["result"] == repeat["result"]
    for payload in (first, repeat):
        artifacts.verify_memory(payload["process_rss"], 256 * 1024 * 1024)
    launched: list[Any] = []

    def completed(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        launched.append((command, kwargs))
        return worker_pair[0]

    monkeypatch.setattr(adapter.subprocess, "run", completed)
    adapter.run_bounded_development(tmp_path)
    assert len(launched) == 1 and launched[0][1]["timeout"] == 120
    assert launched[0][0][-3:] == ["-m", "scripts.run_p63_parent_factor_development", "--worker"]
    paths = adapter.artifact_paths(tmp_path)
    assert not paths["failure"].exists()
    request = artifacts.parse_finite_json(paths["request"].read_text(encoding="utf-8"))
    result = artifacts.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = artifacts.parse_finite_json(paths["audit"].read_text(encoding="utf-8"))
    adapter.verify_result(result, adapter.read_reference())
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert len(request["source_sha256"]) == 34
    assert request["manifest"] == c9b_adapter.manifest_payload()
    assert request["manifest_sha256"] == artifacts.digest_json(request["manifest"])
    assert request["adapter_sha256"] == sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    assert request["reference_file_sha256"] == _fixture_hashes(reference_dir)
    assert request["reference_file_sha256"] == audit["reference_file_sha256"]
    assert request["planned_wake_optimizer_updates"] == 576
    assert request["maximum_executed_optimizer_updates"] == 576
    assert request["planned_guarded_attempts"] == audit["work"]["guarded_attempts"] == 54
    assert request["planned_guard_evaluations"] == audit["work"]["guard_evaluations"] == 108
    assert request["planned_outer_evaluations"] == audit["outer_evaluations"] == 72
    assert request["planned_outer_examples"] == audit["outer_examples"] == 1440
    assert request["paired_contrasts"] == [list(pair) for pair in adapter.CONTRASTS]
    assert audit["contrast_count"] == 60
    assert audit["cell_count"] == 24 and audit["decision_count"] == 216
    assert audit["work"] == c9b_adapter._work_summary(result["train_facts"])
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert result == first["result"]
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    with pytest.raises(FileExistsError, match="already exists"):
        adapter.run_bounded_development(tmp_path)
    assert len(launched) == 1
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


@pytest.mark.parametrize(
    "corruption",
    ["metric", "retention", "contrast", "train", "cell", "seed", "pair", "extra", "final"],
)
def test_should_reject_any_changed_metric_cell_pair_fact_or_seal(
    monkeypatch: pytest.MonkeyPatch,
    reference_dir: Path,
    worker_pair: tuple[subprocess.CompletedProcess[str], ...],
    corruption: str,
) -> None:
    _select_fixture_reference(monkeypatch, reference_dir)
    broken = deepcopy(artifacts.parse_finite_json(worker_pair[0].stdout)["result"])
    last = broken["scored_seeds"][-1]
    if corruption == "metric":
        last["arms"][-1]["final_mean_task_accuracy"] = 0.123
    elif corruption == "retention":
        last["arms"][-1]["retention_ratio_a"] = 0.123
    elif corruption == "contrast":
        last["contrasts"][-1]["signed_forgetting_a"] = 0.123
    elif corruption == "train":
        broken["train_facts"]["seed_results"][-1]["executed_optimizer_updates"] += 1
    elif corruption == "cell":
        last["arms"].pop()
    elif corruption == "seed":
        broken["scored_seeds"].pop()
    elif corruption == "pair":
        last["contrasts"].pop()
    elif corruption == "extra":
        last["arms"][-1]["undeclared_score"] = 0.5
    else:
        broken["final_released"] = True
    with pytest.raises(ValueError, match="differs|differ"):
        adapter.verify_result(broken, adapter.read_reference())


def test_should_reject_source_or_manifest_drift_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_dir: Path
) -> None:
    _select_fixture_reference(monkeypatch, reference_dir)
    original = adapter.ADDITIONAL_SOURCE_SHA256.copy()
    monkeypatch.setitem(
        adapter.ADDITIONAL_SOURCE_SHA256, "src/app/continual_parent_factor_development.py", "0" * 64
    )
    with pytest.raises(ValueError, match="frozen source changed"):
        adapter.run_bounded_development(tmp_path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(adapter, "ADDITIONAL_SOURCE_SHA256", original)
    from dataclasses import replace

    monkeypatch.setattr(
        adapter,
        "fixed_parent_manifest",
        lambda: replace(c9b_adapter.fixed_parent_manifest(), memory_examples=9),
    )
    with pytest.raises(ValueError, match="frozen manifest"):
        adapter.run_bounded_development(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["timeout", "nonfinite", "worker_exit", "incomplete"])
def test_should_keep_failed_run_without_success_result_or_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_dir: Path, kind: str
) -> None:
    _select_fixture_reference(monkeypatch, reference_dir)

    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        if kind == "timeout":
            raise subprocess.TimeoutExpired(cmd="worker", timeout=120)
        if kind == "worker_exit":
            return SimpleNamespace(returncode=2, stdout="", stderr="injected worker failure")
        return SimpleNamespace(
            returncode=0, stdout='{"result": NaN}' if kind == "nonfinite" else "{}", stderr=""
        )

    monkeypatch.setattr(adapter.subprocess, "run", failed)
    with pytest.raises((subprocess.TimeoutExpired, ValueError, RuntimeError, KeyError)):
        adapter.run_bounded_development(tmp_path)
    paths = adapter.artifact_paths(tmp_path)
    assert paths["request"].exists() and paths["failure"].exists()
    assert not paths["result"].exists() and not paths["audit"].exists()
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    assert failure["reason"] == ("wall_limit" if kind == "timeout" else "worker_or_audit")
