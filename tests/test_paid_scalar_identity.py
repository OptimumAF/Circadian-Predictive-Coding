"""Compare repeated immutable-leaf reads with the original paid-report contract.

Builtin leaves cannot mutate in place, but equal replacement leaves and edits
to the returned restoration reference must retain the original value semantics.
Container ownership remains attached to the report's original paths. These
tests exercise public guards, without models, predictions, physics, or mocks.
"""

from copy import deepcopy

import pytest

from src.app.probe_return_phase import paid_report_guard
from test_paid_report_guard import original_guard


@pytest.fixture(params=[original_guard, paid_report_guard], ids=["original", "optimized"])
def seal(request):
    return request.param


@pytest.mark.parametrize("place", ["leaf", "nested"])
def test_should_refuse_a_deepcopy_callback_that_seals_different_builtin_values(seal, place):
    report = {}
    events = []

    class ChangeDuringCopy:
        def __deepcopy__(self, memo):
            events.append("copy")
            # Both trees become entirely builtin before ownership is sealed.
            # Purity alone therefore cannot prove their initial equality.
            report["value"] = 7 if place == "leaf" else {"rows": [7]}
            return 8 if place == "leaf" else {"rows": [8]}

    report["value"] = ChangeDuringCopy()
    guard, expected = seal(report)
    assert events == ["copy"]
    assert report["value"] != expected["value"]
    for _ in range(2):
        with pytest.raises(ValueError, match="report changed"):
            guard()
    if place == "leaf":
        expected["value"] = 7
    else:
        expected["value"]["rows"][0] = 7
    guard()
    assert events == ["copy"]


def distinct_equal_leaf(kind):
    if kind == "integer":
        return int("123456789012345678901234567890")
    if kind == "float":
        return float("1.23456789")
    return b"a paid observation with a deliberately noninterned scalar".decode("ascii")


@pytest.mark.parametrize("kind", ["integer", "float", "string"])
@pytest.mark.parametrize("side", ["actual", "expected", "both"])
def test_should_accept_equal_scalar_replacements_then_recheck_values_and_owners(seal, kind, side):
    original_leaf = distinct_equal_leaf(kind)
    child = {"value": original_leaf}
    report = {"rows": [[child]], "status": "complete"}
    guard, expected = seal(report)
    guard()

    for _ in range(2):
        for tree, name in ((report, "actual"), (expected, "expected")):
            if side in (name, "both"):
                replacement = distinct_equal_leaf(kind)
                assert replacement == original_leaf and replacement is not original_leaf
                assert replacement is not tree["rows"][0][0]["value"]
                tree["rows"][0][0]["value"] = replacement
        guard()

    report["rows"][0][0]["value"] = "different paid value"
    with pytest.raises(ValueError, match="report changed"):
        guard()
    report["rows"][0][0]["value"] = original_leaf
    guard()

    report["rows"][0][0] = dict(child)
    with pytest.raises(ValueError, match="report was replaced"):
        guard()
    report["rows"][0][0] = child
    guard()


@pytest.mark.parametrize("side", ["actual", "expected", "both"])
def test_should_preserve_zero_sign_equality_and_refuse_same_object_nan_after_success(seal, side):
    report = {"work": {"seconds": 0.0}}
    guard, expected = seal(report)
    guard()
    for tree, name in ((report, "actual"), (expected, "expected")):
        if side in (name, "both"):
            tree["work"]["seconds"] = -0.0
    guard()

    nan = float("nan")
    report["work"]["seconds"] = nan
    expected["work"]["seconds"] = nan
    with pytest.raises(ValueError, match="report changed"):
        guard()
    report["work"]["seconds"] = expected["work"]["seconds"] = 0.0
    guard()


def test_should_allow_trusted_subtree_replacement_without_transferring_actual_owner_authority(seal):
    original_child = {"values": [distinct_equal_leaf("integer")]}
    report = {"first": original_child, "second": original_child}
    guard, expected = seal(report)
    report["first"]["values"][0] = distinct_equal_leaf("integer")
    guard()

    expected["second"] = deepcopy(expected["second"])
    assert expected["first"] is not expected["second"]
    for tree in (report, expected):
        first = tree.pop("first")
        tree["first"] = first
    guard()

    report["second"] = deepcopy(original_child)
    with pytest.raises(ValueError, match="report was replaced"):
        guard()
    report["second"] = original_child
    guard()
