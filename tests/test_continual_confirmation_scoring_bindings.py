"""Real source/request files with explicit reference-byte spies only.

The scientific manifest and saved reference report remain fully validated.
No ignored training body, source construction, model or final value is needed.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any

import pytest

from test_continual_confirmation_scoring_execution import (
    reference_report as reference_report,
)
from test_continual_confirmation_scoring import _seal_scoring, _forbid
from src.app.continual_confirmation_scoring_execution import (
    encoded_identity,
    scoring_execution_request,
)
from src.infra import continual_confirmation_scoring_bindings as bindings
from src.infra.continual_confirmation_io import read_json, write_exclusive


REPO_ROOT = Path(__file__).resolve().parents[1]
_READ_BYTES = Path.read_bytes
_READ_TEXT = Path.read_text


@pytest.fixture(autouse=True)
def seal_file_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    _seal_scoring(monkeypatch)
    # This boundary must read real files; restore only filesystem entry points.
    monkeypatch.setattr(Path, "read_bytes", _READ_BYTES)
    monkeypatch.setattr(Path, "read_text", _READ_TEXT)
    from src.infra import continual_confirmation_final as final
    from src.infra import continual_confirmation_scoring_worker as worker

    monkeypatch.setattr(final, "release_confirmation_final", _forbid)
    monkeypatch.setattr(final, "evaluate_confirmation_final", _forbid)
    monkeypatch.setattr(worker, "train_confirmation", _forbid)


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    names = (
        set(bindings.PRIOR_SOURCE_SHA256)
        | set(bindings.ADDITIONAL_SOURCE_SHA256)
        | set(bindings.OWN_SOURCE_PATHS)
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, path)
    return tmp_path


@pytest.fixture
def saved_request(
    source_root: Path,
    reference_report: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any]]:
    # Delegation is limited to six historical file bytes. The actual two
    # complete readers and all 97 current files have separate preflight evidence.
    monkeypatch.setattr(bindings, "_check_reference_files", lambda *args: None)
    path, scope = source_root / "request.json", source_root / "scope.json"
    body = scoring_execution_request(
        started_utc="2026-10-01T00:00:00+00:00",
        **bindings.scoring_bindings(source_root, path, scope, reference_report),
    )
    write_exclusive(path, body)
    return path, scope, body


def _closure(root: Path) -> set[str]:
    pending = [
        "scripts/run_p67_confirmation_scoring.py",
        "scripts/inspect_p67_scoring_training_references.py",
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
                pieces = module.split(".")
                for index in range(1, len(pieces) + 1):
                    package = "/".join(pieces[:index])
                    for candidate in (package + ".py", package + "/__init__.py"):
                        if (root / candidate).is_file():
                            pending.append(candidate)
    return visited


def test_should_bind_the_conservative_full_adapter_closure_and_every_old_pin(
    source_root: Path,
) -> None:
    sources = bindings.current_sources(source_root)
    assert len(sources) == 97 and set(sources) == _closure(source_root)
    assert all(sources[name] == digest for name, digest in bindings.PRIOR_SOURCE_SHA256.items())
    assert sources == bindings.current_sources(REPO_ROOT)


@pytest.mark.parametrize(
    "name",
    [
        "src/core/backprop_mlp.py",
        "src/app/continual_confirmation_scoring.py",
        "src/app/continual_confirmation_scoring_execution.py",
        "src/infra/continual_confirmation_scoring_worker.py",
        "src/infra/continual_confirmation_scoring_artifacts.py",
        "src/infra/continual_confirmation_scoring_bindings.py",
        "scripts/run_p67_confirmation_scoring.py",
    ],
)
@pytest.mark.parametrize("kind", ["missing", "changed"])
def test_should_reject_changed_or_missing_old_new_and_own_sources_against_saved_request(
    source_root: Path,
    saved_request: tuple[Path, Path, dict[str, Any]],
    name: str,
    kind: str,
) -> None:
    path, scope, body = saved_request
    source = source_root / name
    if kind == "missing":
        source.unlink()
    else:
        source.write_bytes(source.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError, match="source|request"):
        bindings.checked_scoring_request(source_root, path, scope, encoded_identity(body))


def test_should_check_canonical_request_bytes_environment_command_and_late_identity(
    source_root: Path, saved_request: tuple[Path, Path, dict[str, Any]]
) -> None:
    path, scope, body = saved_request
    assert (
        bindings.checked_scoring_request(source_root, path, scope, encoded_identity(body)) == body
    )
    assert body["command"] == bindings.worker_command(path, scope)
    assert body["command"][0] == bindings.sys.executable


@pytest.mark.parametrize("kind", ["whitespace", "nonfinite", "duplicate", "unknown", "identity"])
def test_should_reject_noncanonical_ambiguous_changed_or_wrong_request_bytes(
    source_root: Path,
    saved_request: tuple[Path, Path, dict[str, Any]],
    kind: str,
) -> None:
    path, scope, body = saved_request
    identity = encoded_identity(body)
    if kind == "whitespace":
        path.write_bytes(path.read_bytes() + b" ")
    elif kind == "nonfinite":
        path.write_text('{"x":NaN}', encoding="utf-8")
    elif kind == "duplicate":
        path.write_text('{"x":1,"x":2}', encoding="utf-8")
    elif kind == "unknown":
        body["unknown"] = True
        path.unlink()
        write_exclusive(path, body)
    else:
        identity["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        bindings.checked_scoring_request(
            source_root, path, scope, identity if kind == "identity" else None
        )


@pytest.mark.parametrize("kind", ["environment", "command", "request_during_checks", "reference"])
def test_should_reject_current_or_late_bindings_before_any_scientific_access(
    source_root: Path,
    saved_request: tuple[Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    path, scope, _ = saved_request
    if kind == "environment":
        monkeypatch.setattr(bindings.platform, "processor", lambda: "changed")
    elif kind == "command":
        monkeypatch.setattr(bindings, "worker_command", lambda *args: ["changed"])
    elif kind == "reference":

        def fail(*args: Any) -> None:
            raise ValueError("fabricated current reference byte drift")

        monkeypatch.setattr(bindings, "_check_reference_files", fail)
    else:
        original = bindings.current_sources

        def changed(root: Path) -> dict[str, str]:
            value = original(root)
            path.write_bytes(path.read_bytes() + b" ")
            return value

        monkeypatch.setattr(bindings, "current_sources", changed)
    with pytest.raises(ValueError):
        bindings.checked_scoring_request(source_root, path, scope)


def test_should_validate_report_before_any_reference_or_source_io(
    reference_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid(*args: Any) -> None:
        raise AssertionError("malformed report reached file bindings")

    monkeypatch.setattr(bindings, "_check_reference_files", forbid)
    monkeypatch.setattr(bindings, "current_sources", forbid)
    bad = deepcopy(reference_report)
    bad["unexpected"] = True
    with pytest.raises(ValueError):
        bindings.scoring_bindings(Path("unopened"), Path("request"), Path("scope"), bad)


@pytest.mark.parametrize("kind", ["scope", "files"])
def test_should_check_actual_scope_and_all_six_reference_identities(
    reference_report: dict[str, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest

    scope_sha = fixed_scoring_manifest().scope_record_sha256
    files = {
        bundle["reference"]["name"]: deepcopy(bundle["files"])
        for bundle in reference_report["bundles"]
    }
    monkeypatch.setattr(bindings, "stream_file_identity", lambda path: {"sha256": scope_sha})
    monkeypatch.setattr(bindings, "verify_training_reference_bytes", lambda *args: files)
    bindings._check_reference_files(Path("root"), Path("scope"), reference_report)
    if kind == "scope":
        monkeypatch.setattr(bindings, "stream_file_identity", lambda path: {"sha256": "0" * 64})
    else:
        next(iter(files.values()))["result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        bindings._check_reference_files(Path("root"), Path("scope"), reference_report)


def test_should_preserve_every_file_when_reading_a_valid_request(
    source_root: Path, saved_request: tuple[Path, Path, dict[str, Any]]
) -> None:
    path, scope, _ = saved_request
    before = path.read_bytes()
    assert bindings.checked_scoring_request(source_root, path, scope) == read_json(path)
    assert path.read_bytes() == before
