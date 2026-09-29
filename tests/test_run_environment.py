"""Git and dependency provenance must report what the process actually saw."""

from __future__ import annotations

from pathlib import Path
import subprocess

from src.infra.run_environment import capture_run_environment


def test_should_report_explicit_unavailable_source_outside_git(tmp_path: Path) -> None:
    environment = capture_run_environment(tmp_path)
    assert environment.source["commit_sha"] is None
    assert environment.source["dirty"] is None
    assert environment.source["unavailable_reason"]
    assert environment.dependency_versions["python"]
    assert environment.dependency_versions["numpy"]
    assert environment.hardware["compute_device"] == "cpu"


def test_should_hash_dirty_untracked_source_content(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("fixed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    clean = capture_run_environment(tmp_path).source
    assert clean["dirty"] is False
    assert clean["untracked_file_count"] == 0

    untracked = tmp_path / "new.py"
    untracked.write_text("one\n", encoding="utf-8")
    first = capture_run_environment(tmp_path).source
    untracked.write_text("two\n", encoding="utf-8")
    second = capture_run_environment(tmp_path).source
    assert first["dirty"] is True
    assert first["untracked_file_count"] == 1
    assert first["status_sha256"] == second["status_sha256"]
    assert first["workspace_sha256"] != second["workspace_sha256"]
