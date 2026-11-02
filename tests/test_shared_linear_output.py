"""Actual finally/encoder/CLI AST with JSON records only; no learner imports."""

import argparse
import ast
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/recurring_ticket_workflow_example.py"


def extract(path, name, env):
    tree = ast.parse(path.read_text(encoding="utf8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


def encoder():
    return extract(ROOT / "scripts/run_ticket_workflow.py", "_encode_report", dict(json=json))


def finalizer():
    # Exercise the production finally block, without executing training setup.
    # JSON-only records never enter the unchanged native-array comparator port.
    env: dict[str, Any] = dict(
        Any=Any,
        fields=fields,
        is_dataclass=is_dataclass,
        np=SimpleNamespace(ndarray=type("NoNativeArrays", (), {})),
    )
    extract(ROOT / "src/app/coupled_attribution.py", "_same_value", env)
    tree = ast.parse(RUNNER.read_text(encoding="utf8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_example")
    block = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "publication is not None"
        and any(isinstance(c, ast.Attribute) and c.attr == "close" for c in ast.walk(n))
    )
    wrapper = ast.parse("def finalize(report, publication):\n    pass\n").body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    wrapper.body = [block]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, "actual-publication-finally", "exec"), env)
    return env["finalize"]


class PublicationRecord:
    """Metadata double for the existing close/preserved_work boundary only."""

    def __init__(self, row, close_error=False):
        self.row = row
        self.trusted = deepcopy(row)
        self.close_error = close_error
        self.closes = 0
        self.reads = 0

    def close(self):
        self.closes += 1
        if self.close_error:
            raise ValueError("close failed")

    def preserved_work(self):
        self.reads += 1
        return deepcopy(self.trusted)


def record(status="published_and_rolled_back", large=False):
    # Forty fixed synthetic metadata slots, never executed tasks/native evidence.
    row = dict(
        complete=True,
        status=status,
        fit_returns=16,
        episodes=[
            dict(slot=i, success=False, evidence="y" * (15000 if large else 0)) for i in range(16)
        ],
    )
    report = dict(
        complete=True,
        source_unchanged=True,
        actor_unchanged=True,
        episodes=[
            dict(slot=i, success=False, evidence="x" * (8000 if large else 0)) for i in range(24)
        ],
        shared_serving=row,
    )
    return report, PublicationRecord(row)


@pytest.mark.parametrize("status", ["published_and_rolled_back", "rejected"])
def test_complete_exact_record_is_retained_once(status):
    report, owner = record(status)
    original = deepcopy(report)
    finalizer()(report, owner)
    assert report == original and "shared_serving_preserved_work" not in report
    assert owner.closes == owner.reads == 1


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "incomplete_row",
        "corrupt",
        "missing",
        "counter_type",
        "complete_type",
        "close",
    ],
)
def test_failed_or_changed_public_record_keeps_detached_trusted_work(fault):
    report, owner = record()
    if fault == "incomplete":
        report["complete"] = False
    elif fault == "incomplete_row":
        owner.trusted["complete"] = owner.row["complete"] = False
    elif fault == "corrupt":
        report["shared_serving"]["episodes"][0]["success"] = True
    elif fault == "missing":
        del report["shared_serving"]
    elif fault == "counter_type":
        report["shared_serving"]["fit_returns"] = True
    elif fault == "complete_type":
        report["complete"] = 1
    elif fault == "close":
        owner.close_error = True
    finalizer()(report, owner)
    paid = report["shared_serving_preserved_work"]
    assert paid == owner.trusted and paid is not owner.trusted and paid is not owner.row
    assert owner.closes == owner.reads == 1
    if fault == "close":
        assert report["complete"] is False and report["source_unchanged"] is None
        assert report["actor_unchanged"] is None and "close failed" in report["error"]


def test_no_publication_preserves_default_report():
    report = dict(complete=True, episodes=[])
    before = deepcopy(report)
    finalizer()(report, None)
    assert report == before


def test_complete_forty_slot_report_fits_original_bound_without_losing_unique_evidence():
    report, owner = record(large=True)
    duplicate = {**deepcopy(report), "shared_serving_preserved_work": deepcopy(owner.trusted)}
    with pytest.raises(ValueError, match="exceeds512KiB"):
        encoder()(duplicate)
    original = deepcopy(report)
    finalizer()(report, owner)
    encoded = encoder()(report)
    assert len(encoded) <= 512 * 1024 and json.loads(encoded) == original
    assert len(report["episodes"]) + len(report["shared_serving"]["episodes"]) == 40


@pytest.mark.parametrize("over", [0, 1])
def test_existing_exact_encoder_limit_remains_binding(over):
    overhead = len((json.dumps(dict(padding=""), separators=(",", ":")) + "\n").encode())
    report = dict(padding="x" * (512 * 1024 - overhead + over))
    if over:
        with pytest.raises(ValueError, match="exceeds512KiB"):
            encoder()(report)
    else:
        data = encoder()(report)
        assert len(data) == 512 * 1024 and json.loads(data) == report


@pytest.mark.parametrize("case", ["shared", "default", "incomplete", "existing", "oversize"])
def test_actual_cli_saves_complete_record_and_preserves_output_guards(case, tmp_path, monkeypatch):
    # Only pure mode validators occupy these names; no project module is loaded.
    config = ModuleType("scripts.shared_linear_publication")
    validators: dict[str, Any] = {}
    setattr(
        config,
        "require_mode",
        extract(ROOT / "scripts/shared_linear_publication.py", "require_mode", validators),
    )
    package = ModuleType("scripts")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "scripts", package)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    calls = []

    def run_example(**options):
        calls.append(options)
        if case == "default":
            return dict(complete=True, episodes=[])
        report, owner = record(large=case == "shared")
        if case == "incomplete":
            report["complete"] = False
        if case == "oversize":
            report["extra"] = "z" * (512 * 1024)
        finalizer()(report, owner)
        return report

    env: dict[str, Any] = dict(
        argparse=argparse,
        Path=Path,
        __doc__="metadata CLI",
        _encode_report=encoder(),
        run_example=run_example,
    )
    for name in (
        "_require_shared_replay_mode",
        "_require_ce_probe_update_mode",
        "_require_train_readout_mode",
    ):
        extract(RUNNER, name, env)
    main = extract(RUNNER, "main", env)
    output = tmp_path / "report.json"
    options = (
        []
        if case == "default"
        else [
            "--method",
            "linear_fit",
            "--context-return",
            "--train-update",
            "minimum_change",
            "--replay",
            "coverage",
            "--replay-joint",
            "joint",
            "--shared-serving",
        ]
    )
    argv = options + ["--output", str(output)]
    if case == "existing":
        output.write_bytes(b"original")
        with pytest.raises(FileExistsError):
            main(argv)
        assert output.read_bytes() == b"original" and calls == []
    elif case == "oversize":
        with pytest.raises(ValueError, match="exceeds512KiB"):
            main(argv)
        assert output.read_bytes() == b"{}\n" and len(calls) == 1
    else:
        assert main(argv) == (1 if case == "incomplete" else 0)
        saved = json.loads(output.read_bytes())
        assert len(calls) == 1
        if case == "shared":
            assert calls[0]["shared_serving"] is True and len(output.read_bytes()) <= 512 * 1024
            assert len(saved["episodes"]) == 24 and len(saved["shared_serving"]["episodes"]) == 16
            assert "shared_serving_preserved_work" not in saved
        elif case == "incomplete":
            assert "shared_serving_preserved_work" in saved and saved["complete"] is False
