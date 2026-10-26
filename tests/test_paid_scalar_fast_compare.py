"""Behavioral equivalence when an already sealed scalar report changes shape.

The legacy oracle remains value-first and owns only the original container
paths. These checks exercise repeated calls and fallback transitions, without
native learners, predictions, physics, or experiment artifacts.
"""

from copy import deepcopy

import pytest

from src.app.probe_return_phase import paid_report_guard
from test_paid_report_guard import Tensor, original_guard


@pytest.mark.parametrize("side", ["actual", "expected", "both"])
def test_should_preserve_callback_trace_for_late_values_with_custom_metaclass_equality(side):
    traces = []
    for seal in (original_guard, paid_report_guard):
        trace = []

        class Metaclass(type):
            def __eq__(cls, other):
                trace.append("metaclass equality")
                return False

        class Value(metaclass=Metaclass):
            def __eq__(self, other):
                trace.append("value equality")
                return True

        report = dict(value=1)
        guard, expected = seal(report)
        if side in ("actual", "both"):
            report["value"] = Value()
        if side in ("expected", "both"):
            expected["value"] = Value()
        result = outcome(guard)
        traces.append((result, trace))
    assert traces[0] == traces[1]
    if side != "both":
        assert "metaclass equality" not in traces[1][1]


def outcome(guard):
    try:
        guard()
    except Exception as error:
        return type(error), str(error)
    return None


@pytest.mark.parametrize(
    "change",
    [
        "bool",
        "float",
        "none",
        "string",
        "new_container",
        "nan",
        "remove_dict_owner",
        "remove_list_owner",
        "list_to_tuple",
        "reorder",
        "expected_alias_split",
        "current_alias_split",
        "value_and_owner",
        "remove_leaf",
        "add_leaf",
    ],
)
def test_should_preserve_legacy_outcomes_when_report_and_returned_reference_drift_together(change):
    results = []
    for factory in (original_guard, paid_report_guard):
        shared = [dict(calls=1)]
        report = dict(first=shared, second=shared, scalar=1, nested=dict(child=[]))
        guard, expected = factory(report)
        observed = [outcome(guard)]
        if change in ("bool", "float", "none", "string", "new_container", "nan"):
            value = {
                "bool": True,
                "float": 1.0,
                "none": None,
                "string": "1",
                "new_container": dict(new=[1]),
                "nan": float("nan"),
            }[change]
            report["scalar"] = value
            expected["scalar"] = deepcopy(value)
        elif change == "remove_dict_owner":
            del report["nested"]["child"]
            del expected["nested"]["child"]
        elif change == "remove_list_owner":
            report["first"].clear()
            expected["first"].clear()
        elif change == "list_to_tuple":
            report["first"] = tuple(report["first"])
            expected["first"] = tuple(expected["first"])
        elif change == "reorder":
            for value in (report, expected):
                first = value.pop("first")
                value["first"] = first
        elif change == "expected_alias_split":
            expected["second"] = deepcopy(expected["second"])
        elif change == "current_alias_split":
            report["second"] = deepcopy(report["second"])
        elif change == "value_and_owner":
            report["first"] = deepcopy(report["first"])
            report["scalar"] = 2
        elif change == "remove_leaf":
            del report["scalar"]
            del expected["scalar"]
        else:
            report["new"] = dict(values=[None, False, 0.0])
            expected["new"] = dict(values=[None, False, 0.0])
        observed.append(outcome(guard))
        # A failed read is not cached as success, and a newly introduced
        # container does not silently gain original ownership authority.
        observed.append(outcome(guard))
        if change == "new_container":
            report["scalar"] = deepcopy(report["scalar"])
            observed.append(outcome(guard))
        results.append(observed)
    assert results[1] == results[0]
    if change == "nan":
        assert results[0][1][0] is ValueError
    elif change == "remove_dict_owner":
        assert results[0][1][0] is KeyError
    elif change == "remove_list_owner":
        assert results[0][1][0] is IndexError


@pytest.mark.parametrize("target", ["scalar", "earlier_owner", "later_value"])
def test_should_preserve_late_opaque_comparison_callbacks_and_error_precedence(target):
    observations = []
    for factory in (original_guard, paid_report_guard):
        events = []
        report = dict(first=[1], scalar=1, opaque=None, last=[2])
        guard, expected = factory(report)

        def callback():
            events.append("opaque comparison")
            if target == "scalar":
                report["scalar"] = 2
            elif target == "earlier_owner":
                report["first"] = [1]
            else:
                report["first"] = [1]
                report["last"][0] = 3

        report["opaque"] = Tensor(callback=callback)
        expected["opaque"] = Tensor()
        first = outcome(guard)
        second = outcome(guard)
        observations.append((first, second, tuple(events)))
    assert observations[1] == observations[0]
    assert observations[0][2]
    if target == "scalar":
        assert observations[0][0] is None
        assert "report changed" in observations[0][1][1]
    elif target == "earlier_owner":
        assert "report was replaced" in observations[0][0][1]
    else:
        assert "report changed" in observations[0][0][1]


def test_should_add_no_hash_or_equality_callbacks_when_a_custom_key_appears_after_sealing():
    observations = []
    for factory in (original_guard, paid_report_guard):
        events = []

        class Key:
            def __hash__(self):
                events.append("hash")
                return 17

            def __eq__(self, other):
                events.append("equal")
                return self is other

        report = dict(first=[1], nested=dict(value=1))
        guard, expected = factory(report)
        key = Key()
        report["nested"][key] = dict(calls=2)
        expected["nested"][key] = dict(calls=2)
        events.clear()
        first = outcome(guard)
        report["first"] = [1]
        second = outcome(guard)
        observations.append((first, second, tuple(events)))
    assert observations[1] == observations[0]
    assert observations[0][0] is None
    assert "report was replaced" in observations[0][1][1]


def test_should_refuse_late_scalar_subclasses_without_calling_custom_equality():
    for factory in (original_guard, paid_report_guard):
        calls = []

        class Integer(int):
            def __eq__(self, other):
                calls.append(other)
                raise AssertionError("unsupported scalar subtype equality was invoked")

        report = dict(value=1)
        guard, expected = factory(report)
        report["value"] = Integer(1)
        expected["value"] = Integer(1)
        with pytest.raises(ValueError, match="report changed"):
            guard()
        assert calls == []
