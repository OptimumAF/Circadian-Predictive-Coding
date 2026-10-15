"""Independent controls for explicit native code paths retained before birth.

Why native code constants: nested callback code exists before factory dispatch.
Synthetic paths are inspected only; the sole integration runs a real factory and
its callback once each. Code continuity does not establish callback birth or source
authority, and no original factories or controls are imported.
"""

from dataclasses import FrozenInstanceError
import json
import sys
import threading
from types import CodeType, FunctionType
from typing import cast

import pytest

from provider_code_path import (
    ProviderCodePath,
    pin_provider_code_path,
    require_provider_callback_code,
    require_provider_code_path,
)
from provider_entry_profile import ProviderEntryProfile


def assert_pin_refused(root, root_code, indexes):
    with pytest.raises((TypeError, ValueError)):
        pin_provider_code_path(root, root_code, indexes)


def assert_path_refused(witness):
    with pytest.raises((TypeError, ValueError)):
        require_provider_code_path(witness)


def assert_callback_refused(witness, callback):
    with pytest.raises((TypeError, ValueError)):
        require_provider_callback_code(witness, callback)


def assert_witness_fields_refused(root, root_code, indexes, token):
    with pytest.raises((TypeError, ValueError)):
        require_provider_code_path(ProviderCodePath(root, root_code, indexes, token))


def make_native_chain(depth: int) -> tuple[FunctionType, CodeType, CodeType]:
    """Create bounded native constants for inspection, without calling the root."""

    def template():
        return None

    leaf = template.__code__
    node = leaf
    for _ in range(depth):
        node = template.__code__.replace(co_consts=(node,))
    return FunctionType(node, {}), node, leaf


def test_should_retain_empty_and_explicit_nested_paths_without_factory_dispatch():
    def root():
        raise AssertionError("unmeasured root must not dispatch")

    def factory():
        def middle():
            def provider():
                return None

            return provider

        return middle

    root_function = cast(FunctionType, root)
    root_code = root_function.__code__
    empty = pin_provider_code_path(root_function, root_code, ())
    require_provider_code_path(empty)
    assert empty.root is root_function and empty.root_code is root_code
    assert empty.indexes == () and empty.token is root_code
    factory_function = cast(FunctionType, factory)
    factory_code = factory_function.__code__
    middle_index, middle_code = next(
        (index, value)
        for index, value in enumerate(factory_code.co_consts)
        if type(value) is CodeType
    )
    provider_index, provider_code = next(
        (index, value)
        for index, value in enumerate(middle_code.co_consts)
        if type(value) is CodeType
    )
    indexes = (middle_index, provider_index)
    witness = pin_provider_code_path(factory_function, factory_code, indexes)
    require_provider_code_path(witness)
    assert witness.token is provider_code
    assert factory_function.__code__ is factory_code
    with pytest.raises(FrozenInstanceError):
        setattr(witness, "indexes", ())


def test_should_accept_depth_sixteen_index_511_and_512_constants_including_leaf():
    root, root_code, leaf = make_native_chain(16)
    witness = pin_provider_code_path(root, root_code, (0,) * 16)
    require_provider_code_path(witness)
    assert witness.token is leaf and len(witness.indexes) == 16
    large_leaf = leaf.replace(co_consts=(None,) * 512)
    large_root_code = root_code.replace(co_consts=(None,) * 511 + (large_leaf,))
    large_root = FunctionType(large_root_code, {})
    boundary = pin_provider_code_path(large_root, large_root_code, (511,))
    require_provider_code_path(boundary)
    assert boundary.token is large_leaf
    assert len(large_root_code.co_consts) == len(large_leaf.co_consts) == 512


def test_should_retain_repeated_literal_code_identity_on_separate_explicit_paths():
    _, code, leaf = make_native_chain(1)
    root_code = code.replace(co_consts=(leaf, leaf))
    root = FunctionType(root_code, {})
    first = pin_provider_code_path(root, root_code, (0,))
    second = pin_provider_code_path(root, root_code, (1,))
    require_provider_code_path(first)
    require_provider_code_path(second)
    assert first.token is second.token is leaf
    assert first.indexes == (0,) and second.indexes == (1,)


def test_should_refuse_malformed_native_inputs_without_foreign_dispatch():
    events: list[str] = []

    class Foreign:
        def __getattribute__(self, name):
            events.append("attribute")
            raise AssertionError("foreign attribute dispatched")

        def __eq__(self, other):
            events.append("equality")
            raise AssertionError("foreign equality dispatched")

        def __hash__(self):
            events.append("hash")
            raise AssertionError("foreign hash dispatched")

        def __repr__(self):
            events.append("repr")
            raise AssertionError("foreign repr dispatched")

    class ForeignTuple(tuple):
        def __len__(self):
            events.append("tuple_length")
            raise AssertionError("foreign tuple length dispatched")

        def __iter__(self):
            events.append("tuple_iteration")
            raise AssertionError("foreign tuple iteration dispatched")

    class ForeignInt(int):
        def __index__(self):
            events.append("integer_index")
            raise AssertionError("foreign integer conversion dispatched")

    root, root_code, leaf = make_native_chain(1)
    foreign = Foreign()
    invalid_roots: list[object] = [None, root_code, len, foreign]
    for invalid_root in invalid_roots:
        assert_pin_refused(invalid_root, root_code, (0,))
        assert_witness_fields_refused(invalid_root, root_code, (0,), leaf)
    invalid_codes: list[object] = [None, root, (), foreign]
    for invalid_code in invalid_codes:
        assert_pin_refused(root, invalid_code, (0,))
        assert_witness_fields_refused(root, invalid_code, (0,), leaf)
    invalid_paths: list[object] = [
        None,
        [0],
        ForeignTuple((0,)),
        (True,),
        (ForeignInt(0),),
        (-1,),
        (512,),
        (1.0,),
        (foreign,),
    ]
    for indexes in invalid_paths:
        assert_pin_refused(root, root_code, indexes)
        assert_witness_fields_refused(root, root_code, indexes, leaf)
    invalid_tokens: list[object] = [None, root, (), foreign]
    for invalid_token in invalid_tokens:
        assert_witness_fields_refused(root, root_code, (0,), invalid_token)
    assert events == []


def test_should_refuse_non_code_missing_indexes_depth_seventeen_and_513_constants():
    root, root_code, leaf = make_native_chain(1)
    assert_pin_refused(root, root_code, (1,))
    non_code_root_code = root_code.replace(co_consts=(None,))
    non_code_root = FunctionType(non_code_root_code, {})
    assert_pin_refused(non_code_root, non_code_root_code, (0,))
    deep_root, deep_code, _ = make_native_chain(17)
    assert_pin_refused(deep_root, deep_code, (0,) * 17)
    wide_root_code = root_code.replace(co_consts=(leaf,) * 513)
    wide_root = FunctionType(wide_root_code, {})
    assert_pin_refused(wide_root, wide_root_code, ())
    assert_pin_refused(wide_root, wide_root_code, (0,))
    assert_path_refused(ProviderCodePath(wide_root, wide_root_code, (), wide_root_code))
    wide_leaf = leaf.replace(co_consts=(None,) * 513)
    leaf_root_code = root_code.replace(co_consts=(wide_leaf,))
    leaf_root = FunctionType(leaf_root_code, {})
    assert_pin_refused(leaf_root, leaf_root_code, (0,))
    assert_path_refused(ProviderCodePath(leaf_root, leaf_root_code, (0,), wide_leaf))
    wide_middle = root_code.replace(co_consts=(leaf,) * 513)
    middle_root_code = root_code.replace(co_consts=(wide_middle,))
    middle_root = FunctionType(middle_root_code, {})
    assert_pin_refused(middle_root, middle_root_code, (0, 0))


def test_should_refuse_changed_root_and_equal_foreign_code_tokens():
    root, root_code, leaf = make_native_chain(1)
    witness = pin_provider_code_path(root, root_code, (0,))
    equal_root = root_code.replace()
    equal_leaf = leaf.replace()
    assert equal_root == root_code and equal_root is not root_code
    assert equal_leaf == leaf and equal_leaf is not leaf
    assert_pin_refused(root, equal_root, (0,))
    assert_path_refused(ProviderCodePath(root, equal_root, (0,), leaf))
    assert_path_refused(ProviderCodePath(root, root_code, (0,), equal_leaf))
    root.__code__ = equal_root
    assert_path_refused(witness)
    assert_pin_refused(root, root_code, (0,))
    original_callback = FunctionType(leaf, {})
    assert_callback_refused(witness, original_callback)
    root.__code__ = root_code
    require_provider_code_path(witness)
    changed_child_root = root_code.replace(co_consts=(equal_leaf,))
    root.__code__ = changed_child_root
    assert_path_refused(witness)
    root.__code__ = root_code


def test_should_refuse_incomplete_foreign_and_inconsistent_witnesses_without_row_getters():
    events: list[str] = []

    class ForeignRow(ProviderCodePath):
        def __getattribute__(self, name):
            events.append("row_attribute")
            raise AssertionError("foreign row attribute dispatched")

        def __repr__(self):
            events.append("row_repr")
            raise AssertionError("foreign row repr dispatched")

    root, root_code, leaf = make_native_chain(1)
    assert_path_refused(object.__new__(ProviderCodePath))
    assert_path_refused(object.__new__(ForeignRow))
    for incomplete in (None, object(), (), {"root": root}):
        assert_path_refused(incomplete)
    malformed_rows = (
        ProviderCodePath(root, root_code, (), leaf),
        ProviderCodePath(root, root_code, (0,), root_code),
    )
    for row in malformed_rows:
        assert_path_refused(row)
    assert events == []


def test_should_require_exact_callback_code_without_claiming_callback_birth():
    events: list[str] = []

    class ForeignCallable:
        def __getattribute__(self, name):
            events.append("attribute")
            raise AssertionError("foreign callable attribute dispatched")

        def __call__(self):
            events.append("body")
            raise AssertionError("foreign callable body dispatched")

        def __eq__(self, other):
            events.append("equality")
            raise AssertionError("foreign callable equality dispatched")

        def __hash__(self):
            events.append("hash")
            raise AssertionError("foreign callable hash dispatched")

        def __repr__(self):
            events.append("repr")
            raise AssertionError("foreign callable repr dispatched")

    class Receiver:
        def callback(self):
            events.append("body")

    root, root_code, leaf = make_native_chain(1)
    witness = pin_provider_code_path(root, root_code, (0,))
    callback = FunctionType(leaf, {})
    same_code_callback = FunctionType(leaf, {})
    require_provider_callback_code(witness, callback)
    # Why accept both: this API binds code identity, not function object birth.
    require_provider_callback_code(witness, same_code_callback)
    assert callback is not same_code_callback and callback.__code__ is same_code_callback.__code__
    equal_code_callback = FunctionType(leaf.replace(), {})
    assert equal_code_callback.__code__ == leaf and equal_code_callback.__code__ is not leaf
    invalid_callbacks: list[object] = [
        equal_code_callback,
        root,
        leaf,
        None,
        len,
        Receiver().callback,
        ForeignCallable(),
    ]
    for invalid_callback in invalid_callbacks:
        assert_callback_refused(witness, invalid_callback)
    callback.__code__ = leaf.replace()
    assert_callback_refused(witness, callback)
    assert_callback_refused(object.__new__(ProviderCodePath), same_code_callback)
    assert events == []


def test_should_ignore_hostile_nonselected_constants_without_any_foreign_dispatch():
    events: list[str] = []

    class HostileConstant:
        def __getattribute__(self, name):
            events.append("attribute")
            raise AssertionError("unused constant attribute dispatched")

        def __eq__(self, other):
            events.append("equality")
            raise AssertionError("unused constant equality dispatched")

        def __hash__(self):
            events.append("hash")
            raise AssertionError("unused constant hash dispatched")

        def __repr__(self):
            events.append("repr")
            raise AssertionError("unused constant repr dispatched")

    _, code, leaf = make_native_chain(1)
    hostile = HostileConstant()
    root_code = code.replace(co_consts=(hostile, leaf, hostile))
    root = FunctionType(root_code, {})
    events.clear()
    witness = pin_provider_code_path(root, root_code, (1,))
    require_provider_code_path(witness)
    require_provider_callback_code(witness, FunctionType(leaf, {}))
    assert witness.token is leaf and events == []
    assert_pin_refused(root, root_code, (0,))
    assert events == []


def test_should_pin_before_factory_and_count_both_original_code_entries_once(tmp_path):
    bodies: list[str] = []

    def factory():
        bodies.append("factory")

        def provider():
            bodies.append("provider")

        return provider

    root = cast(FunctionType, factory)
    root_code = root.__code__
    child_index, child_code = next(
        (index, value) for index, value in enumerate(root_code.co_consts) if type(value) is CodeType
    )
    witness = pin_provider_code_path(root, root_code, (child_index,))
    require_provider_code_path(witness)
    assert witness.token is child_code and bodies == []
    guard = ProviderEntryProfile(
        ((root_code, "factory", 1), (child_code, "provider", 1)), tmp_path / "breach.json"
    )
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    with guard.measure():
        callback = cast(FunctionType, factory())
        assert bodies == ["factory"]
        require_provider_callback_code(witness, callback)
        assert callback.__code__ is child_code and root.__code__ is root_code
        callback()
    assert bodies == ["factory", "provider"]
    assert guard.counts == {"factory": 1, "provider": 1}
    assert callback.__code__ is witness.token and root.__code__ is witness.root_code
    assert sys.getprofile() is before_main and threading.getprofile() is before_thread
    assert not (tmp_path / "breach.json").exists()
    (tmp_path / "work.json").write_text(
        json.dumps(
            dict(
                bodies=2,
                factory_calls=1,
                provider_calls=1,
                counts=guard.counts,
                workers=0,
                joined=True,
                restored=True,
            )
        ),
        encoding="utf8",
    )
