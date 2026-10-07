"""P5.3a keeps interrupted writes outside completed artifact paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_versioned_v14_bundle import run_versioned_v14_bundle
from src.infra import atomic_artifact_directory as atomic
from src.infra.measured_observation_files import (
    verify_measured_observation_projection,
    verify_wake_diagnostic_sidecar,
    write_measured_observation_projection,
    write_wake_diagnostic_sidecar,
)
from src.infra.observation_projection_files import (
    verify_observation_projection,
    write_observation_projection,
)
from src.infra.versioned_run_files import verify_run_bundle, write_run_bundle


@pytest.fixture(scope="module")
def source_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p53-source")
    return run_versioned_v14_bundle(root, "p53-source", Path.cwd(), capture_wake_diagnostics=True)


def _failed_stage(parent: Path, name: str) -> Path:
    candidates = list(parent.glob(f".{name}.pending.*"))
    assert len(candidates) == 1
    return candidates[0]


def test_should_leave_failed_and_canceled_publications_unlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result"
    original = atomic._write_file

    def fail_second(path: Path, payload: bytes) -> None:
        if path.name == "manifest.json":
            state = json.loads((path.parent / atomic.PUBLICATION_STATE).read_bytes())
            assert state["status"] == "incomplete"
            assert state["completed_files"] == ["data.json"]
            raise RuntimeError("injected write failure")
        original(path, payload)

    monkeypatch.setattr(atomic, "_write_file", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        atomic.publish_artifact_directory(target, {"data.json": b"{}", "manifest.json": b"{}"})
    assert not target.exists()
    assert (
        json.loads((_failed_stage(tmp_path, "result") / atomic.PUBLICATION_STATE).read_bytes())[
            "status"
        ]
        == "failed"
    )

    canceled = tmp_path / "canceled"

    def interrupt(_path: Path, _payload: bytes) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(atomic, "_write_file", interrupt)
    with pytest.raises(KeyboardInterrupt):
        atomic.publish_artifact_directory(canceled, {"data.json": b"{}"})
    assert not canceled.exists()
    assert (
        json.loads((_failed_stage(tmp_path, "canceled") / atomic.PUBLICATION_STATE).read_bytes())[
            "status"
        ]
        == "canceled"
    )


def test_should_refuse_another_writers_existing_publication_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".locked.publish.lock"
    lock.write_bytes(b"other writer\n")
    with pytest.raises(FileExistsError):
        atomic.publish_artifact_directory(tmp_path / "locked", {"data.json": b"{}"})
    assert lock.read_bytes() == b"other writer\n"
    assert not (tmp_path / "locked").exists()


def test_should_not_expose_partial_v14_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_run: Path
) -> None:
    manifest = json.loads((source_run / "manifest.json").read_bytes())
    manifest["run_id"] = "p53-atomic-bundle"
    training = (source_run / "training.json").read_bytes()
    outcomes = (source_run / "outcomes.json").read_bytes()
    original = atomic._write_file

    def fail_manifest(path: Path, payload: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("injected bundle write failure")
        original(path, payload)

    monkeypatch.setattr(atomic, "_write_file", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        write_run_bundle(tmp_path, manifest, training, outcomes)
    assert not (tmp_path / "p53-atomic-bundle").exists()
    with pytest.raises(ValueError, match="manifest"):
        verify_run_bundle(tmp_path / "p53-atomic-bundle")
    state = json.loads(
        (_failed_stage(tmp_path, "p53-atomic-bundle") / atomic.PUBLICATION_STATE).read_bytes()
    )
    assert state["status"] == "failed"
    assert state["completed_files"] == ["training.json", "outcomes.json"]


def test_should_reject_fully_staged_bundle_when_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_run: Path
) -> None:
    manifest = json.loads((source_run / "manifest.json").read_bytes())
    manifest["run_id"] = "p53-rename-failure"

    def fail_rename(_source: Path, _target: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(atomic.os, "rename", fail_rename)
    with pytest.raises(OSError, match="injected rename"):
        write_run_bundle(
            tmp_path,
            manifest,
            (source_run / "training.json").read_bytes(),
            (source_run / "outcomes.json").read_bytes(),
        )
    assert not (tmp_path / "p53-rename-failure").exists()
    stage = _failed_stage(tmp_path, "p53-rename-failure")
    assert (stage / "manifest.json").exists()
    assert json.loads((stage / atomic.PUBLICATION_STATE).read_bytes())["status"] == "failed"
    with pytest.raises(ValueError, match="directory"):
        verify_run_bundle(stage)


def test_should_publish_only_complete_measured_sidecar_and_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_run: Path
) -> None:
    manifest = json.loads((source_run / "manifest.json").read_bytes())
    manifest["run_id"] = "p53-measured-target"
    run = write_run_bundle(
        tmp_path,
        manifest,
        (source_run / "training.json").read_bytes(),
        (source_run / "outcomes.json").read_bytes(),
    )
    payload = (source_run / "measurements-v1/wake-diagnostics.jsonl").read_bytes()
    original = atomic._write_file

    def fail_manifest(path: Path, data: bytes) -> None:
        if path.name in {"measurement-manifest.json", "projection-manifest.json"}:
            raise OSError("injected sidecar/projection failure")
        original(path, data)

    monkeypatch.setattr(atomic, "_write_file", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        write_wake_diagnostic_sidecar(run, payload)
    assert not (run / "measurements-v1").exists()
    assert (
        json.loads((_failed_stage(run, "measurements-v1") / atomic.PUBLICATION_STATE).read_bytes())[
            "status"
        ]
        == "failed"
    )
    monkeypatch.setattr(atomic, "_write_file", original)
    write_wake_diagnostic_sidecar(run, payload)
    verify_wake_diagnostic_sidecar(run)

    monkeypatch.setattr(atomic, "_write_file", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        write_measured_observation_projection(run)
    assert not (run / "observations-measured-v1").exists()
    assert (
        json.loads(
            (_failed_stage(run, "observations-measured-v1") / atomic.PUBLICATION_STATE).read_bytes()
        )["status"]
        == "failed"
    )
    monkeypatch.setattr(atomic, "_write_file", original)
    write_measured_observation_projection(run)
    verify_measured_observation_projection(run)

    monkeypatch.setattr(atomic, "_write_file", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        write_observation_projection(run)
    assert not (run / "observations-v1").exists()
    assert (
        json.loads((_failed_stage(run, "observations-v1") / atomic.PUBLICATION_STATE).read_bytes())[
            "status"
        ]
        == "failed"
    )
    monkeypatch.setattr(atomic, "_write_file", original)
    write_observation_projection(run)
    verify_observation_projection(run)
