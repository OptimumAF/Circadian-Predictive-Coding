"""Literal actual meter/audit code, no producer execution or scientific input."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/continuous-affine-feasibility-20261101/fit_observe.py"
)


def helpers(now=0.0, peak=64):
    module = ast.parse(SOURCE.read_text())
    nodes: list[ast.stmt] = [
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Meter"
    ]
    nodes += [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "wrap"
    ]
    namespace: dict[str, Any] = dict(
        LIMITS=dict(fits=1, rows=198, returns=1, main_opens=2, correction_opens=3, forbidden=0),
        STARTED=0.0,
        time=SimpleNamespace(monotonic=lambda: now),
    )
    exec(compile(ast.Module(nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    meter = namespace["Meter"]()
    meter.memory = SimpleNamespace(peak_bytes=peak)
    main = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    audit = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    source = os.path.normcase(os.path.abspath("literal/run.json"))
    namespace.update(
        meter=meter,
        os=os,
        source_files={source: "main_opens"},
        old_root=os.path.normcase(os.path.abspath("literal")) + os.sep,
        own_root=os.path.normcase(os.path.abspath("literal/own")) + os.sep,
    )
    exec(compile(ast.Module([audit], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return meter, namespace


def test_should_charge_before_real_helper_delegation_and_latch_exhaustion():
    meter, namespace = helpers()
    called = []
    operation = SimpleNamespace(fit=lambda rows: called.append(len(rows)))
    namespace["wrap"](meter, operation, "fit", "fits", rows=("rows", 0), returned="returns")
    operation.fit([0] * 198)
    assert called == [198] and meter.counts["rows"] == 198 and meter.counts["returns"] == 1
    with pytest.raises(RuntimeError):
        operation.fit([0])
    assert called == [198] and meter.counts["fits"] == 2
    with pytest.raises(RuntimeError):
        operation.fit([0])
    assert meter.counts["fits"] == 2


def test_should_count_only_declared_old_source_opens_and_refuse_more():
    meter, namespace = helpers()
    audit = namespace["audit"]
    for _ in range(2):
        audit("open", ("literal/run.json", "r", 0))
    assert meter.counts["main_opens"] == 2
    with pytest.raises(RuntimeError):
        audit("open", ("literal/run.json", "r", 0))
    assert meter.counts["main_opens"] == 3
    fresh, other = helpers()
    other["audit"]("open", ("literal/own/observations.json", "r", 0))
    assert fresh.counts["forbidden"] == 0
    with pytest.raises(RuntimeError):
        other["audit"]("open", ("literal/final.json", "r", 0))
    assert fresh.counts["forbidden"] == 1


def test_should_never_renew_actual_time_or_rss_workstops():
    for now, peak in ((76.0, 64), (0.0, 481 * 1024**2)):
        meter, namespace = helpers(now, peak)
        with pytest.raises(RuntimeError):
            meter.check()
        namespace["time"].monotonic = lambda: 0.0
        meter.memory.peak_bytes = 64
        with pytest.raises(RuntimeError):
            meter.charge("fits")
        assert meter.counts["fits"] == 0
