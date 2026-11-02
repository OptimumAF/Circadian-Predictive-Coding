"""AST-extracted real native observer helpers; no producer execution/import."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/recurring-current-native-20261101/native_observe.py"
)


def helpers(now=0.0, peak=64):
    module = ast.parse(SOURCE.read_text())
    selected: list[ast.stmt] = [
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Meter"
    ]
    selected += [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "wrap"]
    bind = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "bind")
    finish = next(n for n in bind.body if isinstance(n, ast.FunctionDef) and n.name == "finish")
    namespace: dict[str, Any] = dict(
        LIMITS=dict(calls=1, rows=2, returns=1, closes=2, close_returns=2),
        STARTED=0.0,
        time=SimpleNamespace(monotonic=lambda: now),
    )
    exec(compile(ast.Module(selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    meter = namespace["Meter"]()
    meter.memory = SimpleNamespace(peak_bytes=peak)
    return meter, namespace, finish


def test_should_charge_actual_helper_before_delegate_and_permanently_refuse_more_work():
    meter, namespace, _ = helpers()
    calls = []
    owner = SimpleNamespace(run=lambda x: calls.append(len(x)))
    namespace["wrap"](meter, owner, "run", "calls", rows=("rows", 0), returned="returns")
    owner.run([1, 2])
    assert calls == [2] and meter.counts["rows"] == 2 and meter.counts["returns"] == 1
    with pytest.raises(RuntimeError, match="cap exceeded"):
        owner.run([1])
    assert calls == [2] and meter.counts["calls"] == 2 and meter.stopped
    with pytest.raises(RuntimeError):
        owner.run([1])
    assert calls == [2] and meter.counts["calls"] == 2


def test_should_latch_actual_workstop_even_if_time_and_rss_are_later_reset():
    meter, namespace, _ = helpers(now=281)
    with pytest.raises(RuntimeError, match="work deadline"):
        meter.check()
    namespace["time"].monotonic = lambda: 0.0
    meter.memory.peak_bytes = 64
    with pytest.raises(RuntimeError, match="work deadline"):
        meter.charge("calls")
    assert meter.counts["calls"] == 0
    rss_meter, _, _ = helpers(peak=481 * 1024**2)
    with pytest.raises(RuntimeError, match="work RSS"):
        rss_meter.check()
    rss_meter.memory.peak_bytes = 64
    with pytest.raises(RuntimeError, match="work RSS"):
        rss_meter.charge("calls")
    assert rss_meter.counts["calls"] == 0


def test_should_cleanup_retained_partial_owner_after_workstop_without_renewal():
    meter, namespace, finish = helpers(now=281)
    owner = object()
    meter.owners.append(owner)
    with pytest.raises(RuntimeError):
        meter.check()
    closed = []
    namespace.update(meter=meter, close=lambda value: closed.append(value))
    exec(compile(ast.Module([finish], type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace["finish"](owner)
    assert closed == [owner] and meter.closed == {id(owner)}
    assert meter.counts["closes"] == meter.counts["close_returns"] == 1
    assert meter.stopped == "280-second work deadline"
    with pytest.raises(RuntimeError):
        meter.charge("calls")
