"""Public replay factor writer has bounded, exclusive, audited artifacts."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_p63_replay_factor_pilot as adapter


def test_should_run_public_replay_factor_cli_and_read_complete_artifacts(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.run_p63_replay_factor_pilot",
        "--output-dir",
        str(tmp_path),
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert first.returncode == 0, first.stderr
    paths = adapter.artifact_paths(tmp_path)
    assert paths["failure"].exists() is False
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    result = adapter.parse_finite_json(paths["result"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    adapter.verify_result(result)
    assert request["manifest_sha256"] == adapter.digest_json(request["manifest"])
    assert request["source_sha256"] == adapter.check_source_hashes()
    assert request["planned_optimizer_updates"] == 684
    assert request["wall_limit_seconds"] == 120
    assert audit["cell_count"] == 24
    assert audit["result_sha256"] == sha256(paths["result"].read_bytes()).hexdigest()
    assert audit["request_sha256"] == sha256(paths["request"].read_bytes()).hexdigest()

    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    occupied = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert occupied.returncode != 0
    assert "already exists" in occupied.stderr
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


def test_should_reject_changed_source_and_result_facts_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "src/app/continual_replay_factor_pilot.py"
    with monkeypatch.context() as patch:
        patch.setitem(adapter.SOURCE_SHA256, name, "0" * 64)
        with pytest.raises(ValueError, match="frozen source changed"):
            adapter.run_bounded_pilot(tmp_path)
    assert list(tmp_path.iterdir()) == []

    from dataclasses import asdict
    from src.app.continual_replay_factor_pilot import fixed_replay_pilot_manifest, run_replay_pilot

    result = asdict(run_replay_pilot(fixed_replay_pilot_manifest()))
    adapter.verify_result(json.loads(json.dumps(result)))
    broken = deepcopy(result)
    broken["seed_results"][0]["boundaries"][0]["applied_ids_by_method"]["backprop_on"] = []
    with pytest.raises(ValueError, match="applied IDs differ"):
        adapter.verify_result(json.loads(json.dumps(broken)))
    broken = deepcopy(result)
    broken["seed_results"][0]["methods"][0]["development"]["final_mean_task_accuracy"] = 0.0
    with pytest.raises(ValueError, match="derived metrics differ"):
        adapter.verify_result(json.loads(json.dumps(broken)))
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.parse_finite_json('{"x": NaN}')
