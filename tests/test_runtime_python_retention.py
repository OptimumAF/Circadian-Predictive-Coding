"""Retention behavior and full real-process proofs; no scientific admission."""

import gc
import json
import marshal
from pathlib import Path
import subprocess
import sys
import types
import weakref
import warnings

import pytest

from test_runtime_marshal_controls import CONSTANT_KINDS, make_constant_code
from src.infra.runtime_python_objects import retain_python_executable_objects


def operation():
    return None


def closure_operation(value):
    def closed():
        return value

    return closed


@pytest.mark.parametrize("kind", CONSTANT_KINDS)
def test_should_retain_code_constants_before_first_return(kind):
    code = make_constant_code(kind)
    function = types.FunctionType(code, {})
    filters_before = list(warnings.filters)
    retained = retain_python_executable_objects()
    assert warnings.filters == filters_before
    before = marshal.dumps(code)

    held = function()
    assert held is code.co_consts[1]
    assert any(value is code for value in retained)
    assert any(value is held for value in retained)
    assert marshal.dumps(code) == before
    del held
    assert marshal.dumps(code) == before


@pytest.mark.parametrize(
    "route", ("namespace", "globals", "defaults", "keyword-defaults", "closure", "attributes")
)
def test_should_retain_untracked_code_from_every_observed_binding_route(route):
    code = make_constant_code("str")
    function = types.FunctionType(operation.__code__, {})
    holder: object
    if route == "namespace":
        holder = types.SimpleNamespace(binding={"nested": ((code,),)})
    elif route == "globals":
        holder = types.FunctionType(operation.__code__, {"binding": {"nested": ((code,),)}})
    elif route == "defaults":
        function.__defaults__ = (code,)
        holder = function
    elif route == "keyword-defaults":
        function.__kwdefaults__ = {"code": code}
        holder = function
    elif route == "closure":
        holder = closure_operation(code)
    else:
        setattr(function, "binding", code)
        holder = function
    gc.collect()
    retained = retain_python_executable_objects()
    before = marshal.dumps(code)

    held = code.co_consts[1]
    assert any(value is code for value in retained)
    assert any(value is held for value in retained)
    assert marshal.dumps(code) == before
    del held
    assert marshal.dumps(code) == before
    assert holder is not None


def test_should_release_function_identity_after_retained_values_are_released():
    function = types.FunctionType(operation.__code__, {})
    reference = weakref.ref(function)
    retained = retain_python_executable_objects()
    del function
    gc.collect()
    assert reference() is not None

    del retained
    gc.collect()
    assert reference() is None


def test_should_collect_native_code_without_running_user_descriptors_or_metaclasses():
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append("metaclass equality")
            raise AssertionError("retention must use identity dispatch")

        def __hash__(cls):
            calls.append("metaclass hashing")
            raise AssertionError("retention must use object IDs")

    class Subject(metaclass=Meta):
        @property
        def __dict__(self):  # type: ignore[override]
            calls.append("dictionary property")
            raise AssertionError("retention must read native namespaces")

    subject = Subject()
    retained = retain_python_executable_objects()
    assert any(value is Subject for value in retained)
    assert subject is not None and calls == []


def test_should_preserve_complete_actual_runtime_records_across_retained_code_use(
    tmp_path, record_property
):
    script = Path(__file__).with_name("runtime_retention_fixtures.py")
    completed = subprocess.run(
        [sys.executable, "-B", str(script), str(tmp_path / "retention")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["passed"] and rows[0]["group"] == "retention"
    assert rows[0]["science_calls"] == {} and rows[0]["actual_complete_observations"] == 3
    assert all(row["passed"] for row in rows[0]["all_control_names_and_outcomes"])
    record_property("complete_runtime_artifacts", str(tmp_path / "retention"))
