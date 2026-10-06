"""Representative confirmation must honor frozen gates and scope limits."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scripts import run_cifar_representative_confirmation as confirmation
from scripts import audit_cifar_representative_confirmation as audit


def test_should_reject_failed_restore_before_quiet_gate_or_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject() -> Any:
        raise ValueError("frozen selection changed")

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("quiet gate or worker ran before restore")

    monkeypatch.setattr(confirmation, "read_saved_selection", reject)
    monkeypatch.setattr(confirmation.probe, "_quiet_window", unexpected)
    monkeypatch.setattr(confirmation.subprocess, "run", unexpected)

    with pytest.raises(ValueError, match="frozen selection changed"):
        confirmation.run_confirmation(tmp_path)


def test_should_defer_busy_gpu_without_final_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {"quiet_window": {"readings": 3, "max_utilization_percent": 10, "min_free_mib": 5120}}
    restored = SimpleNamespace(
        request=request,
        freeze={"freeze_digest": "frozen"},
        confirmation_manifest=SimpleNamespace(confirmation_seeds=(181, 191, 193)),
    )
    monkeypatch.setattr(confirmation, "read_saved_selection", lambda: restored)
    monkeypatch.setattr(confirmation.selection, "_verify_cuda_runtime", lambda: None)
    monkeypatch.setattr(
        confirmation.probe,
        "_quiet_window",
        lambda: [{"utilization_percent": value, "free_mib": 8000} for value in (2, 11, 3)],
    )

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("final worker ran under busy GPU")

    monkeypatch.setattr(confirmation.subprocess, "run", unexpected)

    with pytest.raises(RuntimeError, match="quiet CUDA gate"):
        confirmation.run_confirmation(tmp_path)

    assert not (tmp_path / confirmation.RESULT_NAME).exists()
    assert not (tmp_path / confirmation.GATE_NAME).exists()
    assert len(list(tmp_path.glob("*deferred*.json"))) == 1


def test_should_bound_each_scope_by_itself_and_total() -> None:
    request = {
        "scopes": {
            "fixed_data": {"limit_seconds": 240},
            "wall_time": {"limit_seconds": 240},
            "capacity_memory": {"limit_seconds": 600},
        },
        "confirmation_total_limit_seconds": 1080,
    }

    assert confirmation._scope_timeout(request, "fixed_data", 100.0, 100.0) == 240
    assert confirmation._scope_timeout(request, "wall_time", 100.0, 400.0) == 240
    assert confirmation._scope_timeout(request, "capacity_memory", 100.0, 1000.0) == 180
    with pytest.raises(TimeoutError, match="total confirmation budget"):
        confirmation._scope_timeout(request, "capacity_memory", 100.0, 1180.0)


def test_should_keep_partial_attempts_when_fixed_data_child_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {
        "quiet_window": {"readings": 3, "max_utilization_percent": 10, "min_free_mib": 5120},
        "confirmation_total_limit_seconds": 1080,
        "scopes": {
            scope: {"limit_seconds": cap}
            for scope, cap in zip(confirmation.SCOPES, (240, 240, 600), strict=True)
        },
    }
    restored = SimpleNamespace(
        request=request,
        freeze={"freeze_digest": "frozen"},
        confirmation_manifest=SimpleNamespace(confirmation_seeds=(181, 191, 193)),
    )
    monkeypatch.setattr(confirmation, "read_saved_selection", lambda: restored)
    monkeypatch.setattr(confirmation.selection, "_verify_cuda_runtime", lambda: None)
    monkeypatch.setattr(
        confirmation.probe,
        "_quiet_window",
        lambda: [{"utilization_percent": 2, "free_mib": 8000}] * 3,
    )

    def timeout(scope: str, data_dir: Path, limit: float) -> Any:
        assert scope == "fixed_data" and limit <= 240
        (data_dir / confirmation.ATTEMPTS_NAME).write_text(
            '{"attempt":{"status":"started","seed":181},"trial":null}\n', encoding="utf-8"
        )
        raise subprocess.TimeoutExpired("fixed-data worker", limit)

    monkeypatch.setattr(confirmation, "_launch_scope", timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        confirmation.run_confirmation(tmp_path)

    failure = tmp_path / confirmation.FAILURE_NAME
    assert failure.is_file()
    assert confirmation.ATTEMPTS_NAME in failure.read_text(encoding="utf-8")
    assert not (tmp_path / confirmation.RESULT_NAME).exists()


def test_should_reject_worker_before_dataset_when_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject() -> Any:
        raise ValueError("missing frozen journal")

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("worker constructed final source before restore")

    monkeypatch.setattr(confirmation, "read_saved_selection", reject)
    monkeypatch.setattr(confirmation.tuning, "run_matched_head_tuning", unexpected)

    with pytest.raises(ValueError, match="missing frozen journal"):
        confirmation._run_worker("fixed_data", tmp_path)


def test_should_reject_worker_without_quiet_gate_before_final_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored = SimpleNamespace(
        request={
            "quiet_window": {"readings": 3, "max_utilization_percent": 10, "min_free_mib": 5120}
        },
        freeze={"freeze_digest": "frozen"},
    )
    monkeypatch.setattr(confirmation, "read_saved_selection", lambda: restored)
    monkeypatch.setattr(confirmation.selection, "_verify_cuda_runtime", lambda: None)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fixed-data worker opened final before quiet gate")

    monkeypatch.setattr(confirmation, "_run_fixed_data", unexpected)
    with pytest.raises(FileNotFoundError, match="quiet gate is missing"):
        confirmation._run_worker("fixed_data", tmp_path)


def test_should_stop_before_wall_time_when_fixed_result_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored = SimpleNamespace(
        request={
            "quiet_window": {"readings": 3, "max_utilization_percent": 10, "min_free_mib": 5120},
            "confirmation_total_limit_seconds": 1080,
            "scopes": {
                scope: {"limit_seconds": cap}
                for scope, cap in zip(confirmation.SCOPES, (240, 240, 600), strict=True)
            },
        },
        freeze={"freeze_digest": "frozen"},
        confirmation_manifest=SimpleNamespace(confirmation_seeds=(181, 191, 193)),
    )
    monkeypatch.setattr(confirmation, "read_saved_selection", lambda: restored)
    monkeypatch.setattr(confirmation.selection, "_verify_cuda_runtime", lambda: None)
    monkeypatch.setattr(
        confirmation.probe,
        "_quiet_window",
        lambda: [{"utilization_percent": 2, "free_mib": 8000}] * 3,
    )
    launched: list[str] = []

    def launch(scope: str, data_dir: Path, timeout: float) -> str:
        launched.append(scope)
        return "complete"

    monkeypatch.setattr(confirmation, "_launch_scope", launch)

    with pytest.raises(FileNotFoundError, match="fixed_data result missing"):
        confirmation.run_confirmation(tmp_path)

    assert launched == ["fixed_data"]
    assert (tmp_path / confirmation.FAILURE_NAME).is_file()


def test_should_reject_scope_artifact_with_changed_seed(
    tmp_path: Path,
) -> None:
    restored: Any = SimpleNamespace(
        freeze={"freeze_digest": "frozen"},
        confirmation_manifest=SimpleNamespace(manifest_digest="typed"),
    )
    path = tmp_path / "wall.json"
    confirmation._save_exclusive(
        path,
        {
            "schema": "cifar_representative_confirmation_scope_v1",
            "scope": "wall_time",
            "seed": 999,
            "request_sha256": confirmation.selection.REQUEST_SHA256,
            "freeze_digest": "frozen",
            "manifest_digest": "typed",
            "report": {},
        },
    )

    with pytest.raises(ValueError, match="scope or provenance"):
        audit._read_scope(path, "wall_time", 181, restored)
