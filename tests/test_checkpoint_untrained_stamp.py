"""Exact stored proof validation; no checkpoint, native work or authority."""

from typing import Any

import pytest

from src.app.managed_replay_checkpoints import (
    _require_inbox_keys,
    _require_partition_maps,
    _require_untrained_stamp,
)
from src.app.untrained_inbox_origins import UntrainedInboxOrigin


def test_should_accept_exact_bounded_scalar_stamp_schema():
    _require_untrained_stamp(((1, 2, 3, 4, "a" * 64, "b" * 64),), 4)


@pytest.mark.parametrize("mode", ["outer", "digest", "tuple_subclass", "boolean"])
def test_should_reject_foreign_stamp_forms_before_their_comparison_can_run(mode):
    calls: list[str] = []

    class ForeignEquality:
        def __eq__(self, other):
            calls.append("equality")
            raise AssertionError("foreign proof equality must not execute")

    class ForeignTuple(tuple):
        def __eq__(self, other):
            calls.append("tuple equality")
            raise AssertionError("foreign proof tuple equality must not execute")

    fields: Any = (1, 2, 3, 4, "a" * 64, "b" * 64)
    value: Any = (fields,)
    if mode == "outer":
        value = ForeignEquality()
    elif mode == "digest":
        value = ((1, 2, 3, 4, ForeignEquality(), "b" * 64),)
    elif mode == "tuple_subclass":
        value = (ForeignTuple(fields),)
    else:
        value = ((True, 2, 3, 4, "a" * 64, "b" * 64),)
    with pytest.raises(ValueError):
        _require_untrained_stamp(value, 4)
    assert calls == []


def test_should_accept_exact_empty_witness_operation_map():
    _require_partition_maps({1: ()}, 4, UntrainedInboxOrigin)


@pytest.mark.parametrize("mode", ["map", "key", "value"])
def test_should_reject_foreign_operation_maps_before_lookup_or_equality(mode):
    calls: list[str] = []

    class ForeignMap(dict):
        def get(self, *args):
            calls.append("get")
            raise AssertionError("foreign map get must not execute")

    class ForeignKey(int):
        __hash__ = int.__hash__

        def __eq__(self, other):
            calls.append("equality")
            raise AssertionError("foreign map key equality must not execute")

    mapping: Any = (
        ForeignMap({1: ()})
        if mode == "map"
        else ({ForeignKey(1): ()} if mode == "key" else {1: [object()]})
    )
    with pytest.raises(ValueError):
        _require_partition_maps(mapping, 4, UntrainedInboxOrigin)
    assert calls == []


def test_should_accept_exact_bounded_inbox_keys():
    _require_inbox_keys({("episode", "sample"): object()}, 4, 4096)


@pytest.mark.parametrize("mode", ["string", "tuple"])
def test_should_reject_foreign_inbox_keys_before_their_comparison_can_run(mode):
    calls: list[str] = []

    class ForeignString(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append("string equality")
            raise AssertionError("foreign key equality must not execute")

    class ForeignTuple(tuple):
        __hash__ = tuple.__hash__

        def __eq__(self, other):
            calls.append("tuple equality")
            raise AssertionError("foreign key equality must not execute")

    key: Any = (
        ("episode", ForeignString("sample"))
        if mode == "string"
        else ForeignTuple(("episode", "sample"))
    )
    mapping = {key: object()}
    with pytest.raises(ValueError):
        _require_inbox_keys(mapping, 4, 4096)
    assert calls == []
