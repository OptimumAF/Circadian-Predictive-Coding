"""Exclusive metadata publication/readback; spies are not scientific evidence."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import inspect_p611_confirmation_costs as adapter
from src.infra.continual_confirmation_io import read_json


@pytest.fixture
def metadata_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    body = {
        "fixture_only": True,
        "projection": {"cells": []},
        "statistical_seed_report_complete": False,
    }
    monkeypatch.setattr(adapter, "current_cost_sources", lambda: {"fixture.py": "a" * 64})

    def read(root: Path, reader: Any) -> dict[str, Any]:
        assert root == adapter.REPO_ROOT
        assert reader is adapter.read_completed_bundle
        return deepcopy(body)

    monkeypatch.setattr(adapter, "read_confirmation_cost_references", read)
    return body


def test_should_publish_exclusive_metadata_and_rederive_every_byte_on_readback(
    metadata_spy: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "new" / "costs.json"
    body = adapter.inspect_confirmation_costs(path)
    assert body == read_json(path)
    assert body["cost_references"] == metadata_spy
    before = path.read_bytes()
    assert adapter.inspect_confirmation_costs(path, read_only=True) == body
    assert path.read_bytes() == before


@pytest.mark.parametrize("directory", [False, True])
def test_should_preserve_occupied_outputs_before_any_complete_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: bool
) -> None:
    path = tmp_path / "costs.json"
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b"user data\n")

    def forbid(*args: Any) -> Any:
        raise AssertionError("occupied output entered readback")

    monkeypatch.setattr(adapter, "read_confirmation_cost_references", forbid)
    with pytest.raises(FileExistsError):
        adapter.inspect_confirmation_costs(path)
    if not directory:
        assert path.read_bytes() == b"user data\n"


def test_should_reject_missing_readback_before_reading_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbid(*args: Any) -> Any:
        raise AssertionError("missing readback entered inputs")

    monkeypatch.setattr(adapter, "read_confirmation_cost_references", forbid)
    with pytest.raises(ValueError, match="missing"):
        adapter.inspect_confirmation_costs(tmp_path / "missing.json", read_only=True)


@pytest.mark.parametrize("change", ["source", "cost", "extra", "noncanonical"])
def test_should_reject_stale_resealed_or_incomplete_readback(
    metadata_spy: dict[str, Any], tmp_path: Path, change: str
) -> None:
    path = tmp_path / "costs.json"
    body = adapter.inspect_confirmation_costs(path)
    if change == "source":
        body["source_sha256"]["fixture.py"] = "b" * 64
    elif change == "cost":
        body["cost_references"]["projection"]["cells"].append({"invented": True})
    elif change == "extra":
        body["undeclared"] = True
    if change == "noncanonical":
        with path.open("a", encoding="utf-8") as output:
            output.write(" ")
    else:
        path.unlink()
        adapter.write_exclusive(path, body)
    with pytest.raises(ValueError):
        adapter.inspect_confirmation_costs(path, read_only=True)


def test_should_reject_late_source_drift_before_publication(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def drift() -> Any:
        calls.append(1)
        return {"fixture.py": ("a" if len(calls) == 1 else "b") * 64}

    monkeypatch.setattr(adapter, "current_cost_sources", drift)
    path = tmp_path / "costs.json"
    with pytest.raises(ValueError, match="source"):
        adapter.inspect_confirmation_costs(path)
    assert not path.exists()


def test_should_preserve_foreign_collision_during_complete_readback(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "costs.json"

    def collide(*args: Any) -> Any:
        path.write_bytes(b"foreign metadata\n")
        return deepcopy(metadata_spy)

    monkeypatch.setattr(adapter, "read_confirmation_cost_references", collide)
    with pytest.raises(FileExistsError):
        adapter.inspect_confirmation_costs(path)
    assert path.read_bytes() == b"foreign metadata\n"


def test_should_reject_changed_output_after_publication(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = adapter.write_exclusive

    def changed(path: Path, value: Any) -> None:
        original(path, value)
        with path.open("a", encoding="utf-8") as output:
            output.write(" ")

    monkeypatch.setattr(adapter, "write_exclusive", changed)
    with pytest.raises(ValueError, match="published"):
        adapter.inspect_confirmation_costs(tmp_path / "costs.json")


def test_should_reject_nonfinite_metadata_before_creating_output(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adapter, "read_confirmation_cost_references", lambda *a: {"bad": float("nan")}
    )
    path = tmp_path / "costs.json"
    with pytest.raises(ValueError):
        adapter.inspect_confirmation_costs(path)
    assert not path.exists()


def test_should_propagate_complete_reader_failure_without_output(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> Any:
        raise ValueError("complete reader failure")

    monkeypatch.setattr(adapter, "read_confirmation_cost_references", fail)
    path = tmp_path / "costs.json"
    with pytest.raises(ValueError, match="complete reader"):
        adapter.inspect_confirmation_costs(path)
    assert not path.exists()


@pytest.mark.parametrize("options,code", [(["--help"], 0), ([], 2), (["--seed", "41"], 2)])
def test_should_keep_cli_dry_or_reject_scientific_overrides(options: list[str], code: int) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.inspect_p611_confirmation_costs", *options],
        cwd=adapter.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == code


def test_should_extend_the_entire_frozen_scored_source_closure_with_exactly_three_modules() -> None:
    sources = adapter.current_cost_sources()
    assert len(sources) == 100
    original = adapter.current_sources(adapter.REPO_ROOT)
    assert all(sources[name] == digest for name, digest in original.items())
    pending = list(original) + ["scripts/inspect_p611_confirmation_costs.py"]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(ast.parse((adapter.REPO_ROOT / name).read_text(encoding="utf-8-sig"))):
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
                        if (adapter.REPO_ROOT / candidate).is_file():
                            pending.append(candidate)
    assert set(sources) == visited


@pytest.mark.parametrize(
    "name",
    [
        "src/app/continual_confirmation_report_costs.py",
        "src/infra/continual_confirmation_report_cost_references.py",
        "scripts/run_p67_confirmation_scoring.py",
    ],
)
def test_should_reject_any_changed_pinned_component_before_reading_costs(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = adapter.current_cost_sources()
    for relative in sources:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((adapter.REPO_ROOT / relative).read_bytes())
    path = tmp_path / name
    with path.open("a", encoding="utf-8") as output:
        output.write("\n# changed fixture\n")
    monkeypatch.setattr(adapter, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError):
        adapter.current_cost_sources()


def test_should_refuse_output_drift_during_independent_readback(
    metadata_spy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "costs.json"
    adapter.inspect_confirmation_costs(path)

    def drift(*args: Any) -> Any:
        with path.open("a", encoding="utf-8") as output:
            output.write(" ")
        return deepcopy(metadata_spy)

    monkeypatch.setattr(adapter, "read_confirmation_cost_references", drift)
    with pytest.raises(ValueError, match="changed during readback"):
        adapter.inspect_confirmation_costs(path, read_only=True)


def test_should_refuse_duplicate_json_fields_on_readback(
    metadata_spy: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "costs.json"
    adapter.inspect_confirmation_costs(path)
    encoded = path.read_text(encoding="utf-8")
    path.write_text(encoded.replace("{", '{"schema_id": "duplicate",', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        adapter.inspect_confirmation_costs(path, read_only=True)
