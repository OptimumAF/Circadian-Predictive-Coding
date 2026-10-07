"""Local binding fixtures exercise provenance without ignored saved runs."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_p67_confirmation_training as adapter
from src.app import continual_arrived_benchmark as arrived
from src.app.continual_confirmation_execution import execution_request, json_value
from src.app import continual_confirmation_training as training
from src.app.continual_confirmation_validation import _verify_seed_envelope
from src.app.continual_confirmation_work_validation import SeedWork, _verify_seed_work
from src.app.continual_confirmation_manifest import (
    fixed_confirmation_manifest,
    validate_confirmation_manifest,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra import continual_confirmation_io as files


@pytest.fixture
def bound_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    source = tmp_path / "src/app/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source fixture\n", encoding="utf-8")
    source_map = {"src/app/fixture.py": files.file_digest(source)}
    manifest = fixed_confirmation_manifest()
    report = json_value(
        {
            "schema_id": "p67_confirmation_scope_inspection_v1",
            "protocol_id": manifest.protocol_id,
            "manifest": asdict(manifest),
            "summary": validate_confirmation_manifest(manifest),
            "development_references": [
                {
                    "family": family.name,
                    "bundles": [{"source_sha256": source_map}, {"source_sha256": source_map}],
                }
                for family in manifest.families
            ],
            "inspection_source_sha256": source_map,
            "reservation_usage": {"result_files_checked": [], "observed_source_seeds": []},
            "confirmation_source_constructed": False,
            "confirmation_scored": False,
            "final_released": False,
        }
    )
    path = tmp_path / "scope.json"
    files.write_exclusive(path, report)
    monkeypatch.setattr(adapter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "SCOPE_FILE_SHA256", files.file_digest(path))
    monkeypatch.setattr(adapter, "EXTRA_SOURCE_SHA256", source_map.copy())
    monkeypatch.setattr(adapter.inventory, "inspect_confirmation_scope", lambda: deepcopy(report))
    return path, report


def _forbid(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("binding entered a reserved builder/model/update/score")


def test_should_bind_exact_scope_sources_command_and_environment_before_data(
    bound_scope: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "__init__", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", _forbid)
    scope_path, _ = bound_scope
    before = scope_path.read_bytes()
    result = adapter.execution_bindings(tmp_path / "request.json", scope_path)
    assert result["manifest"] == fixed_confirmation_manifest()
    assert result["source_sha256"] == adapter.EXTRA_SOURCE_SHA256
    assert result["scope_sha256"] == files.file_digest(scope_path)
    assert result["adapter_sha256"] == files.file_digest(Path(adapter.__file__))
    assert result["command"][-4:] == [
        "--request-file",
        str(tmp_path / "request.json"),
        "--scope-file",
        str(scope_path),
    ]
    assert scope_path.read_bytes() == before
    assert not (tmp_path / "request.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "changed_bytes",
        "revalidation",
        "source_missing",
        "source_changed",
        "source_conflict",
        "unfrozen",
    ],
)
def test_should_reject_scope_or_source_drift_before_any_data(
    bound_scope: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    scope_path, report = bound_scope
    if kind == "missing":
        scope_path.unlink()
    elif kind == "changed_bytes":
        scope_path.write_bytes(scope_path.read_bytes() + b" ")
    elif kind == "revalidation":
        report["summary"]["cell_count"] -= 1
    elif kind == "source_missing":
        (tmp_path / "src/app/fixture.py").unlink()
    elif kind == "source_changed":
        (tmp_path / "src/app/fixture.py").write_text("# changed\n", encoding="utf-8")
    elif kind == "source_conflict":
        adapter.EXTRA_SOURCE_SHA256["src/app/fixture.py"] = "0" * 64
    else:
        monkeypatch.setattr(adapter, "EXTRA_SOURCE_SHA256", {})
    with pytest.raises(ValueError, match="scope|source|pins"):
        adapter.execution_bindings(tmp_path / "request.json", scope_path)
    assert not (tmp_path / "request.json").exists()


@pytest.mark.parametrize(
    "payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', "[]", "{"]
)
def test_should_reject_ambiguous_nonfinite_or_malformed_json(payload: str) -> None:
    with pytest.raises(ValueError, match="confirmation JSON"):
        files.parse_json(payload)


def test_should_preserve_existing_bytes_and_avoid_claiming_nonfinite_output(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        files.write_exclusive(target, {"result": 1})
    assert target.read_bytes() == b"existing"
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        files.write_exclusive(invalid, {"result": float("nan")})
    assert not invalid.exists()


@pytest.fixture
def stub_child(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Metadata-only IO fixture: the scientific verifier is delegated to a spy.
    # This never represents a reserved trained body or a scientific success.
    work = (SeedWork("fixture", 41, 1, 1, 0, 0, 0, 0, 0, 8, 0),)
    result = {
        "fixture_only": True,
        "seed_results": [
            {
                "family": "gating",
                "after_b": {"fixture": {"model_type": "src.core.backprop_mlp.BackpropMLP"}},
                "legacy_train_facts": {
                    "methods": [{"method": "fixture", "wake_updates": 1, "replay_updates": 0}]
                },
            }
        ],
    }

    def verified(payload: dict[str, Any], manifest: Any) -> Any:
        if payload != result:
            raise ValueError("fixture result forgery")
        return work

    monkeypatch.setattr(adapter, "verify_confirmation_payload", verified)

    def completed(command: list[str], **kwargs: Any) -> Any:
        request_file = Path(command[command.index("--request-file") + 1])
        payload = {
            "result": deepcopy(result),
            "request_sha256": files.file_digest(request_file),
            "observed_updates": {
                "attempted_updates": 1,
                "executed_updates": 1,
                "by_model_kind": {"backprop": 1, "pc": 0, "circadian": 0},
            },
            "process_rss": {
                "pid": 123,
                "start_bytes": 100,
                "peak_bytes": 200,
                "sample_count": 2,
                "interval_seconds": 0.005,
            },
            "worker_elapsed_seconds": 1.0,
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    return completed


@pytest.mark.parametrize("name", ["request", "result", "audit", "failure", "claim"])
def test_should_preserve_any_occupied_artifact_before_binding_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    paths = adapter.artifact_paths(output)
    paths[name].write_bytes(b"existing bytes")
    monkeypatch.setattr(adapter, "execution_bindings", _forbid)
    monkeypatch.setattr(adapter.subprocess, "run", _forbid)
    with pytest.raises(FileExistsError, match="already exists"):
        adapter.run_bounded_training(output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {
        paths[name].name: b"existing bytes"
    }


def test_should_publish_and_read_back_exclusive_bound_bundle(
    bound_scope: tuple[Path, dict[str, Any]],
    stub_child: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_file, _ = bound_scope
    launches = []

    def completed(command: list[str], **kwargs: Any) -> Any:
        launches.append((command, kwargs))
        assert adapter.artifact_paths(tmp_path / "run")["request"].is_file()
        return stub_child(command, **kwargs)

    monkeypatch.setattr(adapter.subprocess, "run", completed)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    audit = adapter.run_bounded_training(tmp_path / "run", scope_file)
    assert len(launches) == 1 and launches[0][1]["timeout"] == 600
    assert audit["status"] == "completed"
    assert audit["work"]["totals"]["executed_optimizer_updates"] == 1
    assert audit["scope_record_sha256"] == adapter.SCOPE_FILE_SHA256
    paths = adapter.artifact_paths(tmp_path / "run")
    assert not paths["failure"].exists()
    request, result, reread = adapter.read_completed_bundle(tmp_path / "run", scope_file)
    assert result["fixture_only"] is True and reread == audit
    assert audit["request_sha256"] == files.file_digest(paths["request"])
    assert audit["result_sha256"] == files.file_digest(paths["result"])
    assert request["source_sha256"] == adapter.EXTRA_SOURCE_SHA256
    before = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    with pytest.raises(FileExistsError):
        adapter.run_bounded_training(tmp_path / "run", scope_file)
    assert len(launches) == 1
    assert before == {name: path.read_bytes() for name, path in paths.items() if path.exists()}


@pytest.mark.parametrize(
    "kind",
    [
        "timeout",
        "canceled",
        "worker_exit",
        "missing",
        "malformed",
        "nonfinite",
        "duplicate",
        "result",
        "observed",
        "attempted",
        "model_kind",
        "rss",
        "worker_wall",
        "worker_bool_time",
        "request_echo",
        "late_request",
        "late_source",
    ],
)
def test_should_record_failure_without_a_success_bundle(
    bound_scope: tuple[Path, dict[str, Any]],
    stub_child: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    scope_file, _ = bound_scope

    def failed(command: list[str], **kwargs: Any) -> Any:
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        if kind == "canceled":
            raise KeyboardInterrupt("injected cancel")
        response = stub_child(command, **kwargs)
        if kind == "worker_exit":
            response.returncode = 2
            response.stderr = "injected worker failure"
        elif kind in {"missing", "malformed", "nonfinite", "duplicate"}:
            response.stdout = {
                "missing": "{}",
                "malformed": "{",
                "nonfinite": '{"result":NaN}',
                "duplicate": '{"result":1,"result":2}',
            }[kind]
        else:
            payload = json.loads(response.stdout)
            if kind == "result":
                payload["result"]["fixture_only"] = False
            elif kind == "observed":
                payload["observed_updates"]["executed_updates"] += 1
            elif kind == "attempted":
                payload["observed_updates"]["attempted_updates"] += 1
            elif kind == "model_kind":
                payload["observed_updates"]["by_model_kind"] = {
                    "backprop": 0,
                    "pc": 1,
                    "circadian": 0,
                }
            elif kind == "rss":
                payload["process_rss"]["peak_bytes"] = 512 * 1024 * 1024 + 1
            elif kind in {"worker_wall", "worker_bool_time"}:
                payload["worker_elapsed_seconds"] = 600 if kind == "worker_wall" else True
            elif kind == "request_echo":
                payload["request_sha256"] = "0" * 64
            elif kind == "late_request":
                request_file = Path(command[command.index("--request-file") + 1])
                request_file.write_bytes(request_file.read_bytes() + b" ")
            else:
                (tmp_path / "src/app/fixture.py").write_text("changed", encoding="utf-8")
            response.stdout = json.dumps(payload)
        return response

    monkeypatch.setattr(adapter.subprocess, "run", failed)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt)):
        adapter.run_bounded_training(tmp_path / "run", scope_file)
    paths = adapter.artifact_paths(tmp_path / "run")
    assert paths["request"].exists() and paths["failure"].exists()
    assert not paths["result"].exists() and not paths["audit"].exists()
    failure = files.read_json(paths["failure"])
    assert failure["reason"] == (
        "wall_limit"
        if kind == "timeout"
        else "canceled"
        if kind == "canceled"
        else "worker_or_audit"
    )
    with pytest.raises(ValueError, match="complete successful"):
        adapter.read_completed_bundle(tmp_path / "run", scope_file)


@pytest.mark.parametrize("name", ["result", "audit"])
def test_should_keep_publication_failure_as_an_incomplete_bundle(
    bound_scope: tuple[Path, dict[str, Any]],
    stub_child: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    scope_file, _ = bound_scope
    original = adapter.write_exclusive

    def failed(path: Path, value: Any) -> None:
        if path.name.endswith(f".{name}.json"):
            raise OSError("injected publication failure")
        original(path, value)

    monkeypatch.setattr(adapter, "write_exclusive", failed)
    monkeypatch.setattr(adapter.subprocess, "run", stub_child)
    with pytest.raises(OSError, match="publication failure"):
        adapter.run_bounded_training(tmp_path / "run", scope_file)
    paths = adapter.artifact_paths(tmp_path / "run")
    assert paths["failure"].exists() and not paths["audit"].exists()
    assert paths["result"].exists() == (name == "audit")
    with pytest.raises(ValueError, match="complete successful"):
        adapter.read_completed_bundle(tmp_path / "run", scope_file)


@pytest.mark.parametrize(
    "corruption",
    [
        "audit_work",
        "audit_extra",
        "audit_elapsed",
        "result",
        "request",
        "missing_audit",
        "failure_marker",
    ],
)
def test_should_refuse_a_tampered_or_partial_saved_bundle(
    bound_scope: tuple[Path, dict[str, Any]],
    stub_child: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    scope_file, _ = bound_scope
    monkeypatch.setattr(adapter.subprocess, "run", stub_child)
    output = tmp_path / "run"
    adapter.run_bounded_training(output, scope_file)
    paths = adapter.artifact_paths(output)
    if corruption == "missing_audit":
        paths["audit"].unlink()
    elif corruption == "failure_marker":
        paths["failure"].write_bytes(b"{}")
    elif corruption == "result":
        paths["result"].write_text('{"fixture_only":false}', encoding="utf-8")
    elif corruption == "request":
        request = files.read_json(paths["request"])
        request["source_sha256"]["src/app/fixture.py"] = "0" * 64
        paths["request"].write_text(json.dumps(request), encoding="utf-8")
    else:
        audit = files.read_json(paths["audit"])
        if corruption == "audit_work":
            audit["work"]["totals"]["executed_optimizer_updates"] += 1
        elif corruption == "audit_elapsed":
            audit["elapsed_seconds"] = True
        else:
            audit["extra"] = 1
        paths["audit"].write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError):
        adapter.read_completed_bundle(output, scope_file)


def test_should_preserve_a_competing_request_created_during_binding(
    bound_scope: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope_file, _ = bound_scope
    original = adapter.execution_bindings
    output = tmp_path / "run"

    def racing(request_file: Path, scope_file: Path) -> Any:
        result = original(request_file, scope_file)
        output.mkdir()
        request_file.write_bytes(b"other writer's request")
        return result

    monkeypatch.setattr(adapter, "execution_bindings", racing)
    monkeypatch.setattr(adapter.subprocess, "run", _forbid)
    with pytest.raises(FileExistsError, match="already exists"):
        adapter.run_bounded_training(output, scope_file)
    paths = adapter.artifact_paths(output)
    assert paths["request"].read_bytes() == b"other writer's request"
    assert not paths["claim"].exists() and not paths["failure"].exists()


def _save_request(tmp_path: Path, scope_file: Path) -> Path:
    request_file = tmp_path / "request.json"
    request = execution_request(
        started_utc="2026-09-30T12:00:00+00:00",
        **adapter.execution_bindings(request_file, scope_file),
    )
    files.write_exclusive(request_file, request)
    return request_file


@pytest.mark.parametrize(
    "kind", ["missing", "malformed", "manifest", "source", "command", "final", "extra", "time"]
)
def test_should_refuse_a_bad_saved_child_request_before_training(
    bound_scope: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    scope_file, _ = bound_scope
    request_file = _save_request(tmp_path, scope_file)
    request = files.read_json(request_file)
    if kind == "missing":
        request_file.unlink()
    elif kind == "malformed":
        request_file.write_bytes(b'{"source_sha256":1,"source_sha256":2}')
    else:
        if kind == "manifest":
            request["manifest"]["families"].pop()
        elif kind == "source":
            request["source_sha256"]["src/app/fixture.py"] = "0" * 64
        elif kind == "command":
            request["command"].append("--seed=101")
        elif kind == "final":
            request["final_released"] = True
        elif kind == "extra":
            request["extra"] = 1
        else:
            request["started_utc"] = None
        request_file.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(adapter, "train_confirmation", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter._worker_parts(request_file, scope_file)


def test_should_emit_structured_failure_for_an_unbound_private_worker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(adapter.sys, "argv", ["run", "--worker"])
    monkeypatch.setattr(adapter, "train_confirmation", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    with pytest.raises(SystemExit) as stopped:
        adapter.main()
    assert stopped.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert files.parse_json(output.err)["error"] == "confirmation worker requires its saved request"


@pytest.mark.parametrize("corruption", ["late_a", "late_b_selector", "late_role"])
def test_should_reject_last_development_held_state_corruption_after_independent_validation(
    bound_scope: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    scope_file, _ = bound_scope
    request_file = _save_request(tmp_path, scope_file)
    families = tuple(
        replace(item, seeds=(item.development_seeds[0],))
        for item in fixed_confirmation_manifest().families
    )
    held_fixture: dict[str, Any] = {}

    def development_only(manifest: Any) -> Any:
        held = training._train_families(families)
        held_fixture["held"] = held
        return training.TrainedConfirmation(
            training.ConfirmationTrainingFacts(manifest, tuple(item.facts for item in held)), held
        )

    def verify_development(payload: dict[str, Any], manifest: Any) -> Any:
        # The production gate still requires 560 cells. This private fixture
        # tests live comparison on the six fixed development bodies only.
        work = []
        for row, family in zip(payload["seed_results"], families, strict=True):
            _verify_seed_envelope(row, family)
            work.append(_verify_seed_work(row, family))
        last = held_fixture["held"][-1]
        if corruption == "late_a":
            last.models_after_a["random_growth"].weight_input_hidden[0, 0] += 0.01
        elif corruption == "late_b_selector":
            last.models_after_b["random_growth"]._parent_selection_rng.random()
        else:
            last.roles_b.train.target[0, 0] = 1 - last.roles_b.train.target[0, 0]
        return tuple(work)

    reserved = {seed for family in fixed_confirmation_manifest().families for seed in family.seeds}
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles

    def source_a(config: Any, seed: int) -> Any:
        assert seed not in reserved, "fixture opened a reserved A builder"
        return original_a(config, seed)

    def source_b(config: Any, seed: int) -> Any:
        assert seed not in reserved, "fixture opened a reserved B builder"
        return original_b(config, seed)

    monkeypatch.setattr(arrived, "_build_phase_a_roles", source_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", source_b)
    monkeypatch.setattr(adapter, "train_confirmation", development_only)
    monkeypatch.setattr(adapter, "verify_confirmation_payload", verify_development)
    with pytest.raises(ValueError, match="checkpoint|role metadata|content differs from role hash"):
        adapter._worker_parts(request_file, scope_file)
    assert not adapter.artifact_paths(tmp_path)["result"].exists()


def test_should_keep_request_publication_error_as_failure_without_launch(
    bound_scope: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope_file, _ = bound_scope
    original = adapter.write_exclusive

    def failed(path: Path, value: Any) -> None:
        if path.name.endswith(".request.json"):
            raise OSError("injected request publication failure")
        original(path, value)

    monkeypatch.setattr(adapter, "write_exclusive", failed)
    monkeypatch.setattr(adapter.subprocess, "run", _forbid)
    with pytest.raises(OSError, match="request publication failure"):
        adapter.run_bounded_training(tmp_path / "run", scope_file)
    paths = adapter.artifact_paths(tmp_path / "run")
    assert paths["failure"].exists() and not paths["claim"].exists()
    assert (
        not paths["request"].exists()
        and not paths["result"].exists()
        and not paths["audit"].exists()
    )


@pytest.mark.parametrize("kind", ["rss", "unavailable", "wall"])
def test_should_fail_worker_resources_before_any_builder_or_training(
    bound_scope: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from src.infra.continual_confirmation_runtime import ExecutionObserver
    from src.shared.process_memory import ProcessRssSampler

    scope_file, _ = bound_scope
    request_file = _save_request(tmp_path, scope_file)
    monkeypatch.setattr(adapter, "train_confirmation", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    if kind in {"rss", "unavailable"}:
        value = 512 * 1024 * 1024 + 1 if kind == "rss" else None
        monkeypatch.setattr(
            adapter,
            "ProcessRssSampler",
            lambda **kwargs: ProcessRssSampler(read_rss_bytes=lambda: value, **kwargs),
        )
    else:
        ticks = iter((0.0, 601.0))
        monkeypatch.setattr(
            adapter,
            "ExecutionObserver",
            lambda manifest, sampler: ExecutionObserver(
                manifest, sampler, clock=lambda: next(ticks)
            ),
        )
    with pytest.raises(RuntimeError, match="rss_limit|RSS unavailable|wall_limit"):
        adapter._worker_parts(request_file, scope_file)


def test_should_pin_the_entire_static_local_dependency_closure_without_saved_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(adapter.__file__).resolve().parents[1]
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "__init__", _forbid)
    scope = {
        "inspection_source_sha256": {
            name: files.file_digest(root / name)
            for name in (
                "src/app/continual_confirmation_manifest.py",
                "scripts/inspect_p67_confirmation_scope.py",
            )
        },
        "development_references": [
            {"bundles": [{"source_sha256": old.check_source_hashes()}]}
            for old, _ in adapter.inventory.ADAPTERS.values()
        ],
    }
    sources = adapter.check_source_hashes(scope)
    pending = ["scripts/run_p67_confirmation_training.py"]
    visited = set()
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
    assert set(sources) | {"scripts/run_p67_confirmation_training.py"} == visited
    assert len(sources) == 78 and len(visited) == 79


@pytest.mark.parametrize("name", ["request", "result"])
def test_should_refuse_changed_published_bytes_before_success_audit(
    bound_scope: tuple[Path, dict[str, Any]],
    stub_child: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    scope_file, _ = bound_scope
    original = adapter.write_exclusive

    def changed(path: Path, value: Any) -> None:
        original(path, value)
        if path.name.endswith(f".{name}.json"):
            path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(adapter, "write_exclusive", changed)
    monkeypatch.setattr(adapter.subprocess, "run", stub_child if name == "result" else _forbid)
    with pytest.raises(ValueError, match="published .* bytes"):
        adapter.run_bounded_training(tmp_path / "run", scope_file)
    paths = adapter.artifact_paths(tmp_path / "run")
    assert paths["failure"].exists() and not paths["audit"].exists()
    assert not paths["claim"].exists()


def test_should_not_mark_a_noncooperating_writer_request_failed(
    bound_scope: tuple[Path, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope_file, _ = bound_scope
    original = adapter.write_exclusive

    def racing(path: Path, value: Any) -> None:
        if path.name.endswith(".request.json"):
            path.write_bytes(b"foreign request")
            raise FileExistsError("injected foreign request")
        original(path, value)

    monkeypatch.setattr(adapter, "write_exclusive", racing)
    monkeypatch.setattr(adapter.subprocess, "run", _forbid)
    with pytest.raises(FileExistsError):
        adapter.run_bounded_training(tmp_path / "run", scope_file)
    paths = adapter.artifact_paths(tmp_path / "run")
    assert paths["request"].read_bytes() == b"foreign request"
    assert not paths["failure"].exists() and not paths["claim"].exists()
