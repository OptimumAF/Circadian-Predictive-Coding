"""Metadata-only adapter publication and conservative local source closure."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import inspect_p67_scoring_training_references as adapter
from scripts.run_p67_confirmation_training import read_completed_bundle
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.infra.continual_confirmation_io import encoded_digest, read_json
from src.infra.continual_confirmation_training_references import stream_file_identity


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("inspection unexpectedly entered a reader or new work")


@pytest.fixture
def metadata_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # Only publication metadata; no complete scientific readback authority.
    report = {
        "fixture_only": True,
        "bundles": [{"fixture_only": True}, {"fixture_only": True}],
        "final_released": False,
    }

    def read(root: Path, manifest: Any, reader: Any) -> dict[str, Any]:
        assert root == adapter.REPO_ROOT
        assert manifest == fixed_scoring_manifest()
        assert reader is read_completed_bundle
        return deepcopy(report)

    monkeypatch.setattr(adapter, "read_training_references", read)
    return report


def test_should_publish_exclusive_metadata_with_the_intended_byte_identity(
    metadata_spy: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "new" / "metadata.json"
    report = adapter.inspect_training_references(path)
    assert report == read_json(path) == metadata_spy
    assert stream_file_identity(path)["sha256"] == encoded_digest(metadata_spy)
    assert not (path.parent / "confirmation-train.failure.json").exists()


@pytest.mark.parametrize("directory", [False, True])
def test_should_refuse_an_occupied_output_before_reading_either_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: bool
) -> None:
    path = tmp_path / "occupied.json"
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b"user metadata\n")
    monkeypatch.setattr(adapter, "read_training_references", _forbid)
    with pytest.raises(FileExistsError, match="already exists"):
        adapter.inspect_training_references(path)
    if not directory:
        assert path.read_bytes() == b"user metadata\n"


def test_should_preserve_a_foreign_writer_collision_without_a_failure_marker(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"

    def read(*args: Any) -> Any:
        path.write_bytes(b"foreign metadata\n")
        return deepcopy(metadata_spy)

    monkeypatch.setattr(adapter, "read_training_references", read)
    with pytest.raises(FileExistsError):
        adapter.inspect_training_references(path)
    assert path.read_bytes() == b"foreign metadata\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metadata.json"]


def test_should_reject_changed_published_metadata_bytes(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    original = adapter.write_exclusive

    def changed(target: Path, value: Any) -> None:
        original(target, value)
        with target.open("a", encoding="utf-8") as output:
            output.write(" ")

    monkeypatch.setattr(adapter, "write_exclusive", changed)
    with pytest.raises(ValueError, match="published"):
        adapter.inspect_training_references(path)


def test_should_propagate_reader_failure_without_publishing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"

    def failed(*args: Any) -> Any:
        raise ValueError("fabricated complete-reader failure")

    monkeypatch.setattr(adapter, "read_training_references", failed)
    with pytest.raises(ValueError, match="complete-reader"):
        adapter.inspect_training_references(path)
    assert not path.exists()


def test_should_reject_nonfinite_metadata_before_claiming_the_output(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.json"
    monkeypatch.setattr(
        adapter, "read_training_references", lambda *args: {"invalid": float("nan")}
    )
    with pytest.raises(ValueError):
        adapter.inspect_training_references(path)
    assert not path.exists()


def test_should_expose_the_read_only_cli_without_reading_training_or_final_values() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.inspect_p67_scoring_training_references", "--help"],
        cwd=adapter.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0 and "--result-file" in result.stdout


def test_should_extend_the_entire_available_static_local_closure() -> None:
    root = adapter.REPO_ROOT
    pending = [
        "scripts/inspect_p67_scoring_training_references.py",
        "src/app/continual_confirmation_scoring_validation.py",
        "src/infra/continual_confirmation_final.py",
    ]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8-sig"))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                modules.extend(node.module + "." + alias.name for alias in node.names)
            for module in modules:
                if not module.startswith(("src.", "scripts.")):
                    continue
                parts = module.split(".")
                for index in range(1, len(parts) + 1):
                    package = "/".join(parts[:index])
                    for candidate in (package + ".py", package + "/__init__.py"):
                        if (root / candidate).is_file():
                            pending.append(candidate)
    assert len(visited) == 90
    assert {
        "scripts/run_p67_confirmation_training.py",
        "src/app/continual_confirmation_scoring.py",
        "src/app/continual_confirmation_scoring_manifest.py",
        "src/app/continual_confirmation_scoring_state.py",
        "src/app/continual_confirmation_analysis.py",
        "src/app/continual_confirmation_analysis_contract.py",
        "src/core/seed_statistics.py",
        "src/core/confirmation_final_roles.py",
        "src/infra/continual_confirmation_training_references.py",
        "src/app/continual_confirmation_scoring_validation.py",
    } <= visited
