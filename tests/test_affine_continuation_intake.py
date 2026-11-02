"""Actual continuation intake on literal envelope; zero saved-file/native work."""

import ast
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import pytest

from src.core.continuous_affine_fit import ContinuousAffineFit, DEFINITION
from src.core.cube_taskspace_state import cube_taskspace_spec


@pytest.mark.parametrize("corrupt", [False, True])
def test_should_admit_complete_report_diagnostic_and_refuse_changed_provenance(corrupt):
    model = ContinuousAffineFit(
        cube_taskspace_spec(), ((0.0,) * 4,) * 13, 198, 13, (1.0,) * 13, 1.0, 1.0
    )
    values = dict(asdict(model), coefficient_norm=0.0)
    envelope = dict(
        definition=DEFINITION,
        source_actual_sha256="bad"
        if corrupt
        else "4d42e4dafe86713cb81221bc9b6867e692e270e2de8e8ef7418e92b4ad90fa3a",
        qualification="accepted TRAIN fit only; physical/native/deployed use unqualified",
        model=values,
    )
    raw = json.dumps(envelope).encode()
    assert len(raw) < 2066
    raw += b" " * (2066 - len(raw))
    opens = []

    class Source:
        def open(self, mode):
            opens.append(mode)
            return io.BytesIO(raw)

    source = Path("artifacts/runs/continuous-affine-native-20261101-continuation/native_case.py")
    tree = ast.parse(source.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_reader"
    ]
    namespace = dict(
        SOURCE=Source(), DIGEST=hashlib.sha256(raw).hexdigest(), hashlib=hashlib, json=json
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    if corrupt:
        with pytest.raises(ValueError, match="provenance"):
            namespace["load_reader"]()
    else:
        reader = namespace["load_reader"]()
        assert reader.snapshot_state().model == model
    assert opens == ["rb"]
