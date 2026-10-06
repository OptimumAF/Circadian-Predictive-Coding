"""Diagnostic marshal behavior; these controls grant no runtime/source authority.

Inputs are fresh local code objects, constants, and ordinary reference lifetimes.
Outputs compare full raw serializations with all public native code fields and
recursive constant contents. The observer and its complete graph stay unchanged.
"""

from __future__ import annotations

import marshal
import sys
import types

import pytest

from src.infra.runtime_python_objects import _code_identity

from runtime_marshal_value_fixtures import (
    CONSTANT_KINDS,
    describe_native_code,
    make_constant_code,
    retain_code_values,
)


@pytest.mark.parametrize("kind", CONSTANT_KINDS)
def test_should_change_only_serialization_when_unique_constant_gains_reference(kind):
    code = make_constant_code(kind)
    contents = describe_native_code(code)
    before = marshal.dumps(code)
    identity = _code_identity(code)
    # Why this: assertion rewriting can retain the constant while evaluating it.
    reference_count = sys.getrefcount(code.co_consts[1])
    assert reference_count == 2

    held = code.co_consts[1]
    during = marshal.dumps(code)
    reference_count = sys.getrefcount(code.co_consts[1])
    assert reference_count == 3
    assert before != during and identity != _code_identity(code)
    assert describe_native_code(code) == contents
    assert describe_native_code(marshal.loads(before)) == contents
    assert describe_native_code(marshal.loads(during)) == contents
    del held

    reference_count = sys.getrefcount(code.co_consts[1])
    assert reference_count == 2
    assert marshal.dumps(code) == before and _code_identity(code) == identity


@pytest.mark.parametrize("kind", CONSTANT_KINDS)
def test_should_preserve_raw_code_bytes_when_all_constants_are_retained(kind):
    code = make_constant_code(kind)
    retained = retain_code_values(code)
    contents = describe_native_code(code)
    identity = _code_identity(code)
    before = marshal.dumps(code)
    function = types.FunctionType(code, {})

    held = function()
    assert held is code.co_consts[1]
    assert marshal.dumps(code) == before and _code_identity(code) == identity
    assert describe_native_code(code) == contents
    del held
    assert marshal.dumps(code) == before and _code_identity(code) == identity
    assert any(value is code.co_consts[1] for value in retained)


def test_should_keep_serialization_stable_while_an_exception_retains_a_constant():
    module = compile(
        "def controlled():\n    unused = None\n    raise ValueError('placeholder')\n",
        "<marshal-exception-control>",
        "exec",
    )
    code = module.co_consts[0].replace(co_consts=make_constant_code("str").co_consts)
    function = types.FunctionType(code, {"ValueError": ValueError})
    retained = retain_code_values(code)
    contents = describe_native_code(code)
    before = marshal.dumps(code)
    error = None
    try:
        function()
    except ValueError as caught:
        error = caught
        assert caught.args[0] is code.co_consts[1]
        assert marshal.dumps(code) == before
        assert describe_native_code(code) == contents
    assert error is not None and error.__traceback__ is not None
    assert error.__traceback__.tb_next is not None
    assert error.__traceback__.tb_next.tb_frame.f_code is code
    assert marshal.dumps(code) == before
    assert any(value is code.co_consts[1] for value in retained)


@pytest.mark.parametrize("change", ("constant", "bytecode", "nested-code"))
def test_should_detect_actual_code_changes_even_when_original_constants_are_retained(change):
    code = make_constant_code("code" if change == "nested-code" else "str")
    retained = retain_code_values(code)
    contents = describe_native_code(code)
    before = marshal.dumps(code)
    identity = _code_identity(code)
    if change == "constant":
        replacement = code.replace(co_consts=(None, "changed constant"))
    elif change == "bytecode":
        other = compile("def changed():\n    return None\n", "<marshal-control>", "exec")
        replacement = code.replace(co_code=other.co_consts[0].co_code)
    else:
        inner = code.co_consts[1].replace(co_consts=(None, "changed nested constant"))
        replacement = code.replace(co_consts=(None, inner))

    assert describe_native_code(replacement) != contents
    assert marshal.dumps(replacement) != before
    assert _code_identity(replacement) != identity
    assert marshal.dumps(code) == before
    assert describe_native_code(code) == contents
    assert any(value is code for value in retained)


@pytest.mark.parametrize("retain_fields", (False, True))
def test_should_expose_shared_line_table_reference_unless_native_fields_are_retained(retain_fields):
    code = make_constant_code("str")
    retained = retain_code_values(code) if retain_fields else ()
    contents = describe_native_code(code)
    before = marshal.dumps(code)
    replacement = code.replace(co_consts=(None, "changed constant"))
    during = marshal.dumps(code)

    assert replacement.co_linetable is code.co_linetable
    assert describe_native_code(code) == contents
    assert (during == before) is retain_fields
    del replacement
    assert marshal.dumps(code) == before
    assert bool(retained) is retain_fields
