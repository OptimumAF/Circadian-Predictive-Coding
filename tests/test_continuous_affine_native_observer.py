"""AST-extracted actual observer only; no producer/model/SDK imports or fits."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def observer(tmp_path):
    source = Path("artifacts/runs/continuous-affine-native-20261101/native_case.py")
    tree = ast.parse(source.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("Meter", "wrap", "audit_read")
        or isinstance(node, ast.Assign)
        and any(isinstance(n, ast.Name) and n.id == "LIMITS" for n in node.targets)
    ]
    clock = SimpleNamespace(value=0.0)
    namespace = dict(
        time=SimpleNamespace(monotonic=lambda: clock.value),
        os=os,
        Path=Path,
        ROOT=tmp_path,
        STAGE=tmp_path / "artifacts/new",
        SOURCE=tmp_path / "artifacts/accepted/FITTED-MODEL.json",
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    memory = SimpleNamespace(peak_bytes=1024)
    meter = namespace["Meter"](0.0, memory)
    return namespace, meter, clock, memory


def test_should_preserve_overcap_charge_and_latch_refusal(observer):
    _, meter, _, _ = observer
    meter.charge("model_reads")
    with pytest.raises(RuntimeError, match="permanent"):
        meter.charge("model_reads")
    assert meter.counts["model_reads"] == 2
    with pytest.raises(RuntimeError):
        meter.charge("predictions")
    assert meter.counts["predictions"] == 0


def test_should_keep_attempt_without_return_when_operation_raises(observer):
    ns, meter, _, _ = observer

    def failed():
        raise ValueError("lost return")

    owner = SimpleNamespace(call=failed)
    ns["wrap"](meter, owner, "call", "controls", "control_returns")
    with pytest.raises(ValueError):
        owner.call()
    assert meter.counts["controls"] == 1 and meter.counts["control_returns"] == 0


def test_should_acknowledge_returned_constructor_despite_elapsed_resource_boundary(observer):
    ns, meter, clock, _ = observer
    token = object()

    def created():
        clock.value = 110.0
        return token

    owner = SimpleNamespace(create=created)
    ns["wrap"](meter, owner, "create", "creations", "creation_returns")
    assert owner.create() is token
    assert meter.counts["creations"] == meter.counts["creation_returns"] == 1
    with pytest.raises(RuntimeError):
        meter.charge("controls")


def test_should_allow_counted_cleanup_without_clearing_permanent_stop(observer):
    ns, meter, _, _ = observer
    meter.stopped = "spent"
    calls = []
    owner = SimpleNamespace(close=lambda: calls.append(True))
    ns["wrap"](meter, owner, "close", "closes", "close_returns", cleanup=True)
    owner.close()
    assert calls == [True]
    assert meter.counts["closes"] == meter.counts["close_returns"] == 1
    assert meter.stopped == "spent"


@pytest.mark.parametrize("fault", ["time", "rss"])
def test_should_keep_original_clock_and_latched_resource_failure(observer, fault):
    _, meter, clock, memory = observer
    if fault == "time":
        clock.value = 110.0
    else:
        memory.peak_bytes = 1792 * 1024**2 + 1
    with pytest.raises(RuntimeError):
        meter.charge("predictions")
    clock.value, memory.peak_bytes = 0.0, 1024
    with pytest.raises(RuntimeError):
        meter.charge("predictions")
    assert meter.counts["predictions"] == 0


def test_should_count_exact_cached_source_read_without_old_artifact_access(observer):
    ns, meter, _, _ = observer
    ns["audit_read"](meter, "open", (str(ns["SOURCE"]), "r", 0))
    assert meter.counts["model_reads"] == 1
    with pytest.raises(RuntimeError):
        ns["audit_read"](meter, "open", (str(ns["SOURCE"]), "r", 0))
    assert meter.counts["model_reads"] == 2


def test_should_refuse_old_artifact_reads_and_allow_own_metadata(observer):
    ns, meter, _, _ = observer
    ns["audit_read"](meter, "open", (str(ns["STAGE"] / "source-pins.json"), "r", 0))
    ns["audit_read"](meter, "open", (str(ns["ROOT"] / "src/module.py"), "r", 0))
    with pytest.raises(RuntimeError):
        ns["audit_read"](meter, "open", (str(ns["ROOT"] / "artifacts/old/actual.json"), "r", 0))
    assert meter.counts["forbidden"] == 1
