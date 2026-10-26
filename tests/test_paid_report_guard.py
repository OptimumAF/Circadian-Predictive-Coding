"""Paid scalar evidence and opaque-leaf compatibility against the original guard."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard


def original_guard(report):
    """Frozen pre-optimization contract: values first, then original owner paths."""
    expected = deepcopy(report)

    def containers(value, path=()):
        if type(value) in (dict, list, tuple):
            yield path, value
            children = value.items() if type(value) is dict else enumerate(value)
            for key, child in children:
                yield from containers(child, (*path, key))

    owners = tuple(containers(report))

    def guard():
        if not _same_value(report, expected):
            raise ValueError("preceding paid probe report changed during A return")
        for path, owner in owners:
            current = report
            for key in path:
                current = current[key]
            if current is not owner:
                raise ValueError("preceding paid probe report was replaced during A return")

    return guard, expected


@pytest.fixture(params=[original_guard, paid_report_guard], ids=["original", "optimized"])
def seal(request):
    return request.param


def scalar_report():
    return dict(
        status="complete",
        work=dict(calls=4, seconds=0.25, admitted=True, error=None),
        scores=[dict(ce=0.5), (1, "A", False)],
    )


def test_should_accept_unchanged_report_and_dict_reordering_without_replacing_owners(seal):
    report = scalar_report()
    guard, expected = seal(report)
    work = report.pop("work")
    report["work"] = work
    guard()
    assert expected == report and expected is not report
    assert expected["work"] is not report["work"]


@pytest.mark.parametrize("value", [True, 4.0, "4", None, float("nan"), float("inf"), -1, 5])
def test_should_refuse_exact_scalar_value_or_type_changes(seal, value):
    report = dict(calls=4)
    guard, _ = seal(report)
    report["calls"] = value
    with pytest.raises(ValueError, match="report changed"):
        guard()


@pytest.mark.parametrize("value", [float("nan"), np.float64(1), object(), {1, 2}])
def test_should_preserve_original_refusal_of_unsupported_or_nan_leaves(seal, value):
    guard, _ = seal(dict(value=value))
    with pytest.raises(ValueError, match="report changed"):
        guard()


class UnsupportedEquality:
    def __eq__(self, other):
        raise AssertionError("unsupported leaves must not gain arbitrary equality calls")


def test_should_refuse_unsupported_objects_without_invoking_their_equality(seal):
    guard, _ = seal(dict(value=UnsupportedEquality()))
    with pytest.raises(ValueError, match="report changed"):
        guard()


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), -0.0, 0.0, None])
def test_should_preserve_supported_scalar_equality_including_infinities(seal, value):
    guard, _ = seal(dict(value=value))
    guard()


@pytest.mark.parametrize("path", [("work",), ("scores",), ("scores", 0), ("scores", 1)])
def test_should_refuse_equal_valued_replacement_at_every_original_container_path(seal, path):
    report = scalar_report()
    guard, _ = seal(report)
    parent: Any = report
    for key in path[:-1]:
        parent = parent[key]
    original = parent[path[-1]]
    replacement = tuple(list(original)) if type(original) is tuple else deepcopy(original)
    assert replacement is not original
    parent[path[-1]] = replacement
    with pytest.raises(ValueError, match="report was replaced"):
        guard()


def test_should_check_both_alias_paths_for_same_value_replacement(seal):
    shared = [dict(calls=4)]
    report = dict(first=shared, second=shared)
    guard, expected = seal(report)
    assert expected["first"] is expected["second"]
    assert expected["first"] is not shared
    report["second"] = deepcopy(shared)
    with pytest.raises(ValueError, match="report was replaced"):
        guard()


def test_should_give_value_error_precedence_over_an_earlier_equal_owner_replacement(seal):
    report: dict[str, Any] = dict(first=[1], last=dict(calls=4))
    guard, _ = seal(report)
    report["first"] = [1]
    report["last"]["calls"] = 5
    with pytest.raises(ValueError, match="report changed"):
        guard()


@pytest.mark.parametrize("mutation", ["expected", "both", "both_replaced"])
def test_should_keep_returned_expected_mutations_and_current_owner_authority(seal, mutation):
    report = dict(work=dict(calls=4))
    guard, expected = seal(report)
    expected["work"]["calls"] = 5
    if mutation != "expected":
        report["work"]["calls"] = 5
    if mutation == "both_replaced":
        report["work"] = dict(calls=5)
    if mutation == "both":
        guard()
    else:
        reason = "report changed" if mutation == "expected" else "report was replaced"
        with pytest.raises(ValueError, match=reason):
            guard()


def test_should_accept_expected_shape_changes_when_both_sides_match_original_owner_paths(seal):
    report: dict[str, Any] = dict(rows=[1])
    guard, expected = seal(report)
    report["rows"].append(dict(calls=4))
    expected["rows"].append(dict(calls=4))
    guard()
    # New expected containers never acquire authority over original-owner paths.
    report["rows"][1] = dict(calls=4)
    guard()


@pytest.mark.parametrize("kind,error", [("dict", KeyError), ("list", IndexError)])
def test_should_keep_original_missing_owner_path_errors_after_both_sides_remove_a_container(
    seal, kind, error
):
    report: dict[str, Any] = dict(rows=dict(child=[]) if kind == "dict" else [[]])
    guard, expected = seal(report)
    key = "child" if kind == "dict" else 0
    del report["rows"][key]
    del expected["rows"][key]
    with pytest.raises(error):
        guard()


def test_should_not_add_custom_key_hash_callbacks_during_construction_or_guarding():
    def observations(factory):
        calls = []

        class Key:
            def __hash__(self):
                calls.append("hash")
                return 19

            def __deepcopy__(self, memo):
                return self

        report = {Key(): dict(calls=4)}
        calls.clear()
        guard, _ = factory(report)
        constructed = tuple(calls)
        calls.clear()
        guard()
        return constructed, tuple(calls)

    original = observations(original_guard)
    optimized = observations(paid_report_guard)
    assert original[0] and original[1]
    assert optimized == original


def test_should_preserve_nonstr_dict_key_and_key_equality_semantics(seal):
    report = {1: [4], ("context", 2): dict(status="complete")}
    guard, expected = seal(report)
    guard()
    expected[True] = expected.pop(1)
    guard()
    report[1] = [4]
    with pytest.raises(ValueError, match="report was replaced"):
        guard()


@dataclass
class OpaqueRecord:
    rows: list[Any]


def test_should_value_check_dataclass_contents_without_owning_its_nested_containers(seal):
    report = dict(record=OpaqueRecord([dict(calls=4)]))
    guard, expected = seal(report)
    assert expected["record"].rows is not report["record"].rows
    report["record"] = OpaqueRecord([dict(calls=4)])
    guard()
    report["record"].rows[0]["calls"] = True
    with pytest.raises(ValueError, match="report changed"):
        guard()


@pytest.mark.parametrize("mutation", ["values", "dtype", "shape", "nan", "equal_copy"])
def test_should_preserve_numpy_value_dtype_shape_and_detachment_semantics(seal, mutation):
    report = dict(values=np.array([[1.0, 2.0]], dtype=np.float64))
    guard, expected = seal(report)
    assert not np.shares_memory(report["values"], expected["values"])
    if mutation == "values":
        report["values"][0, 0] = 3
    elif mutation == "dtype":
        report["values"] = report["values"].astype(np.float32)
    elif mutation == "shape":
        report["values"] = report["values"].reshape(2, 1)
    elif mutation == "nan":
        report["values"][0, 0] = np.nan
    else:
        report["values"] = report["values"].copy()
        guard()
        return
    with pytest.raises(ValueError, match="report changed"):
        guard()


class Tensor:
    """Minimal tensor comparison protocol; importing native Torch is unnecessary."""

    __module__ = "torch"

    def __init__(self, values=(1, 2), *, dtype="float64", device="cpu", callback=None):
        self.values = values
        self.dtype = dtype
        self.device = device
        self.callback = callback

    def equal(self, other):
        if self.callback is not None:
            self.callback()
        return self.values == other.values


@pytest.mark.parametrize("mutation", ["values", "dtype", "device", "equal_copy"])
def test_should_preserve_tensor_value_dtype_device_and_opaque_owner_semantics(seal, mutation):
    report = dict(tensor=Tensor())
    guard, expected = seal(report)
    assert report["tensor"] is not expected["tensor"]
    if mutation == "equal_copy":
        report["tensor"] = Tensor()
        guard()
        return
    setattr(report["tensor"], mutation, (2, 1) if mutation == "values" else "changed")
    with pytest.raises(ValueError, match="report changed"):
        guard()


def test_should_check_earlier_owned_paths_after_an_opaque_comparison_callback(seal):
    report: dict[str, Any] = dict(first=[1])

    def replace_earlier_owner():
        report["first"] = [1]

    report["tensor"] = Tensor(callback=replace_earlier_owner)
    guard, _ = seal(report)
    with pytest.raises(ValueError, match="report was replaced"):
        guard()


def test_should_preserve_original_single_value_pass_when_opaque_callback_changes_earlier_scalar(
    seal,
):
    report: dict[str, Any] = dict(first=1)

    def change_earlier_scalar():
        report["first"] = 2

    report["tensor"] = Tensor(callback=change_earlier_scalar)
    guard, expected = seal(report)
    guard()
    assert report["first"] == 2 and expected["first"] == 1
    with pytest.raises(ValueError, match="report changed"):
        guard()
