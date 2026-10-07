"""The frozen representative selection must remain test-free and auditable."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scripts import run_cifar_representative_selection as selection


def test_should_reject_changed_request_before_source_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text('{"changed":true}\n', encoding="utf-8")
    monkeypatch.setattr(selection, "REQUEST_SHA256", "0" * 64)

    def unexpected(*args: Any) -> Any:
        raise AssertionError("source read before frozen request check")

    monkeypatch.setattr(selection.study, "_read_verified_probe", unexpected)
    monkeypatch.setattr(selection.probe, "_verify_sources", unexpected)

    with pytest.raises(ValueError, match="digest changed"):
        selection._verify_saved_request(request_path)


def test_should_bind_all_selection_rows_and_confirmation_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "selection_seeds": [179],
        "confirmation_seeds": [181, 191, 193],
        "candidate_grid": {
            head: [
                {"candidate_id": candidate, "config": {"rate": candidate}}
                for candidate in ("a", "b")
            ]
            for head in selection.HEAD_NAMES
        },
        "source": {"archive_sha256": "source"},
        "scopes": {
            "wall_time": {"per_head_seconds": 5.0, "epoch_cap": 1000},
            "fixed_data": {"limit_seconds": 240},
        },
        "confirmation_total_limit_seconds": 1080,
    }
    trials = [
        {
            "head_name": head,
            "candidate_id": candidate,
            "seed": 179,
            "config": {"rate": candidate},
            "validation_accuracy": 0.6 if candidate == "a" else 0.5,
            "split_hashes": {"train": "train", "guard": "guard", "validation": "outer"},
            "feature_hashes": {"train": "features"},
            "backbone_hash": "backbone",
            "initial_head_hash": "initial",
        }
        for head in selection.HEAD_NAMES
        for candidate in ("a", "b")
    ]
    payload: dict[str, Any] = {
        "status": "complete",
        "final_test_source_constructions": 0,
        "selection": {
            "attempts": [
                {
                    "head_name": row["head_name"],
                    "candidate_id": row["candidate_id"],
                    "seed": row["seed"],
                    "config": row["config"],
                    "status": "complete",
                }
                for row in trials
            ],
            "trials": trials,
            "selections": [
                {"head_name": head, "candidate_id": "a"} for head in selection.HEAD_NAMES
            ],
            "confirmations": [],
        },
        "confirmation_manifest": {"manifest_digest": "typed"},
    }
    monkeypatch.setattr(
        selection,
        "restore_confirmation_manifest",
        lambda value: SimpleNamespace(
            selection_seeds=(179,),
            confirmation_seeds=(181, 191, 193),
            wall_time_budget_seconds=5.0,
            wall_time_epoch_cap=1000,
            source_selection_digest=selection._digest(payload["selection"]),
        ),
    )

    selection._validate_worker_payload(payload, request)
    frozen = selection._freeze_manifest(request, payload, payload["confirmation_manifest"])

    assert frozen["selection_sha256"] == selection._digest(payload["selection"])
    assert frozen["freeze_digest"] == selection._digest(
        {key: value for key, value in frozen.items() if key != "freeze_digest"}
    )
    assert frozen["confirmation_seeds"] == [181, 191, 193]
    assert frozen["scope_limits"] == request["scopes"]


def test_should_save_failed_worker_trials_without_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {
        "quiet_window": {"max_utilization_percent": 10, "min_free_mib": 5120},
        "selection_limit_seconds": 180,
    }
    monkeypatch.setattr(selection, "_verify_saved_request", lambda path: (request, {}))
    monkeypatch.setattr(selection, "_verify_cuda_runtime", lambda: None)
    monkeypatch.setattr(
        selection.probe,
        "_quiet_window",
        lambda: [{"utilization_percent": 1, "free_mib": 8000}],
    )
    failed = {
        "status": "failed",
        "error": "candidate training failed",
        "attempts": [{"head_name": "backprop_mlp", "status": "complete"}],
        "trials": [{"head_name": "backprop_mlp", "validation_accuracy": 0.2}],
    }
    monkeypatch.setattr(
        selection.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout=json.dumps(failed), stderr="training error"
        ),
    )
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    failure = tmp_path / "failure.json"

    with pytest.raises(RuntimeError, match="candidate training failed"):
        selection.run_selection(
            tmp_path / "request.json", result, manifest, failure, tmp_path / "journal.jsonl"
        )

    saved = json.loads(failure.read_text(encoding="utf-8"))
    assert saved["worker"]["attempts"] == failed["attempts"]
    assert saved["worker"]["trials"] == failed["trials"]
    assert not result.exists() and not manifest.exists()
    assert saved["request_sha256"] == selection.REQUEST_SHA256


def test_should_keep_completed_and_inflight_attempts_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {
        "quiet_window": {"max_utilization_percent": 10, "min_free_mib": 5120},
        "selection_limit_seconds": 180,
    }
    monkeypatch.setattr(selection, "_verify_saved_request", lambda path: (request, {}))
    monkeypatch.setattr(selection, "_verify_cuda_runtime", lambda: None)
    monkeypatch.setattr(
        selection.probe,
        "_quiet_window",
        lambda: [{"utilization_percent": 1, "free_mib": 8000}],
    )
    journal = tmp_path / "journal.jsonl"

    def time_out(*args: Any, **kwargs: Any) -> Any:
        journal.write_text(
            json.dumps(
                {
                    "attempt": {"candidate_id": "a", "status": "complete"},
                    "trial": {"validation_accuracy": 0.2},
                }
            )
            + "\n"
            + json.dumps({"attempt": {"candidate_id": "b", "status": "started"}, "trial": None})
            + "\n",
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired("selection worker", 180)

    monkeypatch.setattr(selection.subprocess, "run", time_out)
    failure = tmp_path / "failure.json"

    with pytest.raises(subprocess.TimeoutExpired):
        selection.run_selection(
            tmp_path / "request.json",
            tmp_path / "result.json",
            tmp_path / "manifest.json",
            failure,
            journal,
        )

    saved = json.loads(failure.read_text(encoding="utf-8"))
    assert [row["attempt"]["status"] for row in saved["attempt_journal"]] == ["complete", "started"]
    assert saved["attempt_journal"][0]["trial"]["validation_accuracy"] == 0.2
    assert not (tmp_path / "manifest.json").exists()
