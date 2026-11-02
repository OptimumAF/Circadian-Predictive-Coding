"""Actual observer meter/audit extracted without data/model/native imports."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def observer(tmp_path):
    source = Path("artifacts/runs/complete-b22-neural-study-20261101/study.py")
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("Meter", "wrap", "audit")
        or isinstance(node, ast.Assign)
        and any(isinstance(n, ast.Name) and n.id == "LIMITS" for n in node.targets)
    ]
    clock = SimpleNamespace(value=0.0)
    namespace = dict(
        Path=Path,
        os=os,
        time=SimpleNamespace(monotonic=lambda: clock.value),
        ROOT=tmp_path,
        STAGE=tmp_path / "artifacts/new",
        INPUTS=(tmp_path / "artifacts/memory.json", tmp_path / "artifacts/teacher.json"),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    io_work = dict(bytes=0)
    meter = namespace["Meter"](0, SimpleNamespace(peak_bytes=1024), io_work)
    return namespace, meter, clock, io_work


def test_should_preserve_attempt_without_successful_return(observer):
    ns, meter, _, _ = observer

    def failed():
        raise ValueError("lost")

    owner = SimpleNamespace(step=failed)
    ns["wrap"](meter, owner, "step", "controls", returned="control_returns")
    with pytest.raises(ValueError):
        owner.step()
    assert meter.counts["controls"] == 1 and meter.counts["control_returns"] == 0


def test_should_acknowledge_constructor_before_elapsed_boundary_and_allow_cleanup(observer):
    ns, meter, clock, _ = observer
    token = object()

    def created():
        clock.value = 90.0
        return token

    owner = SimpleNamespace(create=created, close=lambda: None)
    ns["wrap"](meter, owner, "create", "creations", returned="creation_returns")
    ns["wrap"](meter, owner, "close", "closes", returned="close_returns", cleanup=True)
    assert owner.create() is token
    with pytest.raises(RuntimeError):
        meter.check()
    owner.close()
    assert meter.counts["creation_returns"] == meter.counts["close_returns"] == 1
    with pytest.raises(RuntimeError):
        meter.charge("controls")


def test_should_refuse_extra_file_open_even_when_total_source_budget_remains(observer):
    ns, meter, _, _ = observer
    for _ in range(4):
        ns["audit"](meter, "open", (str(ns["INPUTS"][0]), "r", 0))
    with pytest.raises(RuntimeError):
        ns["audit"](meter, "open", (str(ns["INPUTS"][0]), "r", 0))
    assert meter.counts["memory_opens"] == 5 and meter.counts["source_opens"] == 5
    with pytest.raises(RuntimeError):
        ns["audit"](meter, "open", (str(ns["INPUTS"][1]), "r", 0))


def test_should_refuse_unrelated_artifact_and_latch_io_overflow(observer):
    ns, meter, _, io_work = observer
    ns["audit"](meter, "open", (str(ns["STAGE"] / "source-pins.json"), "r", 0))
    io_work["bytes"] = 4 * 1024**2 + 1
    with pytest.raises(RuntimeError):
        meter.check()
    io_work["bytes"] = 0
    with pytest.raises(RuntimeError):
        meter.check()
    other = ns["Meter"](0, SimpleNamespace(peak_bytes=1024), dict(bytes=0))
    with pytest.raises(RuntimeError):
        ns["audit"](other, "open", (str(ns["ROOT"] / "artifacts/other.json"), "r", 0))
    assert other.counts["forbidden"] == 1
