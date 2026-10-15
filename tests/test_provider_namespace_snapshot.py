"""Ordered snapshot equivalence on bounded, inactive namespace fixtures."""

import json
from typing import Any

import pytest

import provider_source_loader as loader


class Probe(dict):
    """Controlled call-order evidence, never installed under a profile."""

    def __init__(self, pairs, hostile_events: list[str], last_allowed=None):
        dict.__init__(self, pairs)
        self.calls: list[str] = []
        self.yielded: int = 0
        self.hostile_events = hostile_events
        self.last_allowed = last_allowed

    def __len__(self):
        self.calls.append("len")
        return dict.__len__(self)

    def __iter__(self):
        self.calls.append("iter")
        for index, key in enumerate(dict.__iter__(self)):
            if self.last_allowed is not None and index > self.last_allowed:
                self.hostile_events.append("late_iteration")
                raise AssertionError
            self.yielded += 1
            yield key

    def items(self):
        self.calls.append("items")
        return dict.items(self)


def reference_snapshot(ns, bound=512):
    """Literal original generator predicate, restricted to owned unit fixtures."""
    if len(ns) > bound or any(type(key) is not str or len(key) > 128 for key in ns):
        raise ValueError("invalid source witness")
    return tuple(ns.items())


def reference_unchanged(ns, saved):
    reference_snapshot(ns)
    if len(ns) != len(saved) or any(key not in ns or ns[key] is not value for key, value in saved):
        raise ValueError("invalid source witness")


def sentinel(events: list[str]):
    class Value:
        def __eq__(self, other):
            events.append("value_equality")
            raise AssertionError

    return object.__new__(Value)


def compare(ns, expected, bound=512, error=None, unchanged=False, saved=None):
    traces: list[dict[str, Any]] = []
    results: list[Any] = []
    failures: list[BaseException] = []
    for current in (True, False):
        if isinstance(ns, Probe):
            ns.calls.clear()
            ns.yielded = 0
        if error is None:
            if unchanged:
                function = loader._unchanged if current else reference_unchanged
                results.append(function(ns, saved))
            else:
                function = loader._snapshot if current else reference_snapshot
                results.append(function(ns, bound))
        else:
            with pytest.raises(error) as raised:
                if unchanged:
                    function = loader._unchanged if current else reference_unchanged
                    function(ns, saved)
                else:
                    function = loader._snapshot if current else reference_snapshot
                    function(ns, bound)
            failures.append(raised.value)
        traces.append(
            dict(calls=list(ns.calls), yielded=ns.yielded)
            if isinstance(ns, Probe)
            else dict(calls=[], yielded=0)
        )
    assert traces[0] == traces[1]
    if error is not None:
        assert type(failures[0]) is type(failures[1]) is error
        assert str(failures[0]) == str(failures[1])
        assert failures[0].__suppress_context__ is failures[1].__suppress_context__
    elif unchanged:
        assert results[0] is None and results[1] is None
    else:
        for result in results:
            assert type(result) is tuple and len(result) == len(expected)
            assert all(
                result[index][0] is key and result[index][1] is value
                for index, (key, value) in enumerate(expected)
            )
    return traces[0]


def save_record(
    tmp_path, name, fixtures, entries, successes, value_errors, type_errors, traces, events
):
    assert 0 <= entries <= 2048 and 0 <= fixtures <= 2048 and events == []
    assert len(traces) == successes + value_errors + type_errors
    (tmp_path / "snapshot-work.json").write_text(
        json.dumps(
            dict(
                test_name=name,
                fixtures=fixtures,
                fixture_entries=entries,
                comparisons=len(traces),
                successes=successes,
                errors=dict(ValueError=value_errors, TypeError=type_errors),
                call_orders=traces,
                hostile_events=events,
                provider_calls=0,
                factory_calls=0,
                constructor_calls=0,
                delegate_calls=0,
                source_compilations=0,
            )
        ),
        encoding="utf8",
    )


def test_snapshot_preserves_empty_and_native_ordered_identity(tmp_path):
    events: list[str] = []
    traces: list[dict[str, Any]] = []
    pairs = tuple((f"k{index}", sentinel(events)) for index in range(3))
    traces.append(compare({}, ()))
    traces.append(compare(dict(pairs), pairs))
    assert traces == [dict(calls=[], yielded=0)] * 2
    save_record(
        tmp_path,
        "test_snapshot_preserves_empty_and_native_ordered_identity",
        2,
        3,
        2,
        0,
        0,
        traces,
        events,
    )


def test_snapshot_preserves_size_boundary_and_early_refusal(tmp_path):
    events: list[str] = []
    value = sentinel(events)
    pairs = tuple((f"k{index}", value) for index in range(513))
    accepted = Probe(pairs[:512], events)
    refused = Probe(pairs, events, last_allowed=-1)
    traces = [compare(accepted, pairs[:512]), compare(refused, (), error=ValueError)]
    assert traces == [
        dict(calls=["len", "iter", "items"], yielded=512),
        dict(calls=["len"], yielded=0),
    ]
    save_record(
        tmp_path,
        "test_snapshot_preserves_size_boundary_and_early_refusal",
        2,
        1025,
        1,
        1,
        0,
        traces,
        events,
    )


def test_snapshot_preserves_native_key_length_boundary(tmp_path):
    events: list[str] = []
    value = sentinel(events)
    short, long = "x" * 128, "x" * 129
    traces = [
        compare(Probe(((short, value),), events), ((short, value),)),
        compare(Probe(((long, value),), events), (), error=ValueError),
    ]
    assert traces == [
        dict(calls=["len", "iter", "items"], yielded=1),
        dict(calls=["len", "iter"], yielded=1),
    ]
    save_record(
        tmp_path,
        "test_snapshot_preserves_native_key_length_boundary",
        2,
        2,
        1,
        1,
        0,
        traces,
        events,
    )


def test_snapshot_refuses_wrong_native_key_types_before_key_length(tmp_path):
    events: list[str] = []
    traces: list[dict[str, Any]] = []
    value = sentinel(events)
    keys: tuple[object, ...] = (0, False, None, object())
    for key in keys:
        traces.append(compare(Probe(((key, value),), events), (), error=ValueError))
    assert traces == [dict(calls=["len", "iter"], yielded=1)] * 4
    save_record(
        tmp_path,
        "test_snapshot_refuses_wrong_native_key_types_before_key_length",
        4,
        4,
        0,
        4,
        0,
        traces,
        events,
    )


def test_snapshot_refuses_hostile_string_subclasses_before_hooks(tmp_path):
    events: list[str] = []
    traces: list[dict[str, Any]] = []
    value = sentinel(events)

    class ForeignName(str):
        __hash__ = str.__hash__

        def __len__(self):
            events.append("key_length")
            raise AssertionError

        def __eq__(self, other):
            events.append("key_equality")
            raise AssertionError

    for text in ("short", "x" * 129):
        key = str.__new__(ForeignName, text)
        traces.append(compare(Probe(((key, value),), events), (), error=ValueError))
    assert traces == [dict(calls=["len", "iter"], yielded=1)] * 2
    save_record(
        tmp_path,
        "test_snapshot_refuses_hostile_string_subclasses_before_hooks",
        2,
        2,
        0,
        2,
        0,
        traces,
        events,
    )


def test_snapshot_stops_at_first_bad_key_without_later_effects(tmp_path):
    events: list[str] = []
    value = sentinel(events)
    first = Probe(((object(), value), ("later", value)), events, last_allowed=0)
    second = Probe((("good", value), ("x" * 129, value), ("later", value)), events, last_allowed=1)
    traces = [compare(first, (), error=ValueError), compare(second, (), error=ValueError)]
    assert traces == [
        dict(calls=["len", "iter"], yielded=1),
        dict(calls=["len", "iter"], yielded=2),
    ]
    save_record(
        tmp_path,
        "test_snapshot_stops_at_first_bad_key_without_later_effects",
        2,
        5,
        0,
        2,
        0,
        traces,
        events,
    )


def test_snapshot_preserves_configured_bounds_and_native_error_semantics(tmp_path):
    events: list[str] = []
    traces: list[dict[str, Any]] = []
    value = sentinel(events)
    variants: tuple[tuple[tuple, Any, type[Exception] | None], ...] = (
        ((), -1, ValueError),
        ((("key", value),), 0, ValueError),
        ((("key", value),), 1, None),
        ((("key", value),), False, ValueError),
        ((), False, None),
        ((("key", value),), True, None),
        ((("key", value),), 0.5, ValueError),
        ((("key", value),), 1.0, None),
        ((("key", value),), "1", TypeError),
        ((("key", value),), None, TypeError),
    )
    for pairs, bound, error in variants:
        traces.append(compare(Probe(pairs, events), pairs, bound=bound, error=error))
    traces.append(compare(object(), (), error=TypeError))
    assert traces == [
        dict(calls=["len"], yielded=0),
        dict(calls=["len"], yielded=0),
        dict(calls=["len", "iter", "items"], yielded=1),
        dict(calls=["len"], yielded=0),
        dict(calls=["len", "iter", "items"], yielded=0),
        dict(calls=["len", "iter", "items"], yielded=1),
        dict(calls=["len"], yielded=0),
        dict(calls=["len", "iter", "items"], yielded=1),
        dict(calls=["len"], yielded=0),
        dict(calls=["len"], yielded=0),
        dict(calls=[], yielded=0),
    ]
    save_record(
        tmp_path,
        "test_snapshot_preserves_configured_bounds_and_native_error_semantics",
        11,
        8,
        4,
        4,
        3,
        traces,
        events,
    )


def test_snapshot_preserves_unchanged_identity_refusal_and_restoration(tmp_path):
    events: list[str] = []
    traces: list[dict[str, Any]] = []
    pairs = tuple((f"k{index}", sentinel(events)) for index in range(3))
    namespace = dict(pairs)
    saved = pairs
    traces.append(compare(namespace, (), unchanged=True, saved=saved))
    key, original = pairs[1]
    namespace[key] = object()
    try:
        traces.append(compare(namespace, (), error=ValueError, unchanged=True, saved=saved))
    finally:
        namespace[key] = original
    traces.append(compare(namespace, (), unchanged=True, saved=saved))
    key, original = pairs[0]
    del namespace[key]
    try:
        traces.append(compare(namespace, (), error=ValueError, unchanged=True, saved=saved))
    finally:
        namespace[key] = original
    # Original unchanged semantics match saved identities despite insertion order.
    traces.append(compare(namespace, (), unchanged=True, saved=saved))
    namespace["extra"] = object()
    try:
        traces.append(compare(namespace, (), error=ValueError, unchanged=True, saved=saved))
    finally:
        del namespace["extra"]
    traces.append(compare(namespace, (), unchanged=True, saved=saved))
    wrong_key = object()
    native_namespace: Any = namespace
    native_namespace[wrong_key] = object()
    try:
        traces.append(compare(namespace, (), error=ValueError, unchanged=True, saved=saved))
    finally:
        del native_namespace[wrong_key]
    traces.append(compare(namespace, (), unchanged=True, saved=saved))
    assert traces == [dict(calls=[], yielded=0)] * 9
    save_record(
        tmp_path,
        "test_snapshot_preserves_unchanged_identity_refusal_and_restoration",
        1,
        4,
        5,
        4,
        0,
        traces,
        events,
    )
