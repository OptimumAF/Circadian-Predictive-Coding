"""Actual study observer extracted without imports, models or dataset reads."""

import ast
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def observer(tmp_path):
    path = Path("artifacts/runs/nearest-reserved-probe-study-20261101/study.py")
    tree = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in ("Meter", "CountedReader", "wrap", "audit")
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "LIMITS" for t in n.targets)
    ]
    clock = SimpleNamespace(value=0.0)
    namespace = dict(
        Path=Path,
        os=os,
        time=SimpleNamespace(monotonic=lambda: clock.value),
        ROOT=tmp_path,
        STAGE=tmp_path / "artifacts/new",
        INPUTS=((tmp_path / "artifacts/source.json", 4, "unused", 2),),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    meter = namespace["Meter"](0.0, SimpleNamespace(peak_bytes=1024))
    return namespace, meter, clock


def test_should_preserve_failed_read_attempt_without_return(observer):
    ns, meter, _ = observer

    class Failed:
        def read(self, size):
            raise OSError("lost read")

    with pytest.raises(OSError):
        ns["CountedReader"](Failed(), meter).read(4)
    assert meter.counts["read_attempts"] == 1 and meter.counts["read_returns"] == 0


def test_should_keep_actual_bytes_and_return_when_read_crosses_deadline(observer):
    ns, meter, clock = observer

    class Late(io.BytesIO):
        def read(self, size: int | None = -1, /) -> bytes:
            clock.value = 35.0
            return super().read(size)

    handle = Late(b"paid")
    with pytest.raises(RuntimeError):
        with ns["CountedReader"](handle, meter) as reader:
            reader.read(4)
    assert handle.closed
    assert meter.counts["read_returns"] == 1 and meter.counts["bytes"] == 4
    clock.value = 0.0
    with pytest.raises(RuntimeError):
        meter.charge("appends")


def test_should_refuse_source_open_beyond_specific_cap_permanently(observer):
    ns, meter, _ = observer
    source = ns["INPUTS"][0][0]
    meter.admit_open(source)
    meter.admit_open(source)
    with pytest.raises(RuntimeError):
        meter.admit_open(source)
    assert meter.counts["opens"] == 3
    with pytest.raises(RuntimeError):
        meter.charge("appends")


def test_should_keep_validator_return_and_known_copy_payload_after_late_ack(observer):
    ns, meter, clock = observer

    def copied():
        clock.value = 35.0
        return SimpleNamespace(
            prototype_features=SimpleNamespace(nbytes=16),
            prototype_actions=SimpleNamespace(nbytes=8),
        )

    owner = SimpleNamespace(snapshot=copied)
    ns["wrap"](meter, owner, "snapshot", "validators", "validator_returns", payload=True)
    with pytest.raises(RuntimeError):
        owner.snapshot()
    assert meter.counts["validator_returns"] == 1 and meter.counts["known_copy_bytes"] == 24


def test_should_refuse_unrelated_artifact_before_any_actual_read(observer):
    ns, meter, _ = observer
    with pytest.raises(RuntimeError):
        ns["audit"](meter, "open", (str(ns["ROOT"] / "artifacts/oldmodel.json"), "r", 0))
    assert meter.counts["forbidden"] == 1 and meter.counts["opens"] == 0
