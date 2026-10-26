"""Grouped scalar consistency oracles, every pair retained, no model work.

Prospective native kernel totals35 calls,1219 row exposures,4876 conversions,
39718 pair attempts/39717 returns (one arithmetic overflow). Parent alone runs.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import pytest

from src.core import grouped_input_target_consistency as grouped
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)

UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")


def spec(dimension=2):
    return ContinuousActionSpec(dimension, UNITS, (-1.0,) * 4, (1.0,) * 4)


def episode(name, observations, targets, *, selected_spec=None):
    rows = tuple(
        ContinuousTransition(
            index,
            tuple(observation),
            tuple(target),
            0.0,
            tuple(observations[index + 1] if index + 1 < len(observations) else observation),
            False,
            index == len(observations) - 1,
        )
        for index, (observation, target) in enumerate(zip(observations, targets, strict=True))
    )
    return ContinuousEpisode(
        "episode-" + name,
        "TRAIN-" + name,
        "train",
        selected_spec or spec(len(observations[0])),
        rows,
    )


def small_groups():
    return (
        (
            "a",
            episode(
                "a", ((0.0, 0.0), (0.0, 0.0)), ((0.0, 0.0, 0.0, 1.0), (0.5, -0.25, 0.125, -1.0))
            ),
        ),
        (
            "b",
            episode(
                "b",
                ((0.0, 0.0), (1e-300, 0.0)),
                ((0.5 + 1e-12, -0.25, 0.125, -1.0), (0.5, -0.25, 0.125, -1.0)),
            ),
        ),
        ("c", episode("c", ((1.0, 0.0),), ((0.0, 0.0, 0.0, 1.0),))),
    )


def full_groups():
    return tuple(
        (
            name,
            episode(
                name,
                tuple((float(index),) + (0.0,) * 11 for index in range(count)),
                ((0.0, 0.0, 0.0, -1.0),) * count,
            ),
        )
        for name, count in zip(
            ("current_corrections", "corrections", "push", "a", "b", "train20"),
            (50, 50, 18, 15, 50, 15),
            strict=True,
        )
    )


def scalar_pairs(data):
    rows = [row for _, item in data for row in item.transitions]
    pairs = []
    for first, left in enumerate(rows):
        for second in range(first + 1, len(rows)):
            right = rows[second]
            native_left = tuple(
                struct.unpack("!f", struct.pack("!f", value))[0] for value in left.action
            )
            native_right = tuple(
                struct.unpack("!f", struct.pack("!f", value))[0] for value in right.action
            )
            raw = tuple(b - a for a, b in zip(left.action, right.action, strict=True))
            native = tuple(b - a for a, b in zip(native_left, native_right, strict=True))
            duplicate = left.observation == right.observation
            pairs.append(
                dict(
                    first_row=first,
                    second_row=second,
                    input_l2=math.hypot(
                        *(b - a for a, b in zip(left.observation, right.observation, strict=True))
                    ),
                    target_delta=list(raw),
                    native_target_delta=list(native),
                    target_l2=math.hypot(*raw),
                    native_target_l2=math.hypot(*native),
                    duplicate_input=duplicate,
                    raw_target_conflict=duplicate and left.action != right.action,
                    native_target_conflict=duplicate and native_left != native_right,
                    opposite_gripper=left.action[3] != right.action[3],
                )
            )
    return pairs


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        kernel_attempts=0,
        kernel_returns=0,
        rows=0,
        pair_attempts=0,
        pair_returns=0,
        native_conversion_attempts=0,
        native_conversion_returns=0,
    )
    kernel = grouped.continuous_input_target_consistency

    def measured(features, targets, selected_spec, *, _work):
        before = asdict(_work)
        counts["kernel_attempts"] += 1
        counts["rows"] += len(features)
        try:
            result = kernel(features, targets, selected_spec, _work=_work)
            counts["kernel_returns"] += 1
            return result
        finally:
            for key in (
                "pair_attempts",
                "pair_returns",
                "native_conversion_attempts",
                "native_conversion_returns",
            ):
                counts[key] += getattr(_work, key) - before[key]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(grouped, "continuous_input_target_consistency", measured)
        yield counts
    if os.environ.get("GROUPED_CONSISTENCY_FIXTURE_LEDGER"):
        with Path(os.environ["GROUPED_CONSISTENCY_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts == dict(
        kernel_attempts=35,
        kernel_returns=34,
        rows=1219,
        pair_attempts=39718,
        pair_returns=39717,
        native_conversion_attempts=4876,
        native_conversion_returns=4876,
    )


@pytest.fixture(scope="module")
def small_reports(ledger):
    return {size: grouped.diagnose_grouped_input_targets(small_groups()[:size]) for size in (2, 3)}


@pytest.mark.parametrize("size", [2, 3])
def test_should_preserve_every_signed_raw_native_pair_and_global_nearest_tie(small_reports, size):
    data, result = small_groups()[:size], small_reports[size]
    expected = scalar_pairs(data)
    assert result["complete"] is True and result["error"] is None
    assert json.loads(json.dumps(result["pairs"])) == expected
    rows = sum(len(item.transitions) for _, item in data)
    nearest = []
    for row in range(rows):
        choices = [
            (
                value["input_l2"],
                value["second_row"] if value["first_row"] == row else value["first_row"],
            )
            for value in expected
            if row in (value["first_row"], value["second_row"])
        ]
        nearest.append(min(choices)[1])
    assert result["nearest_other_rows"] == nearest
    for field, pair_key in (
        ("duplicate_input_pairs", "duplicate_input"),
        ("raw_conflict_pairs", "raw_target_conflict"),
        ("native_conflict_pairs", "native_target_conflict"),
        ("opposite_gripper_pairs", "opposite_gripper"),
    ):
        assert result[field] == sum(pair[pair_key] for pair in expected)
    rounding = next(
        pair for pair in result["pairs"] if (pair["first_row"], pair["second_row"]) == (1, 2)
    )
    assert rounding["raw_target_conflict"] is True and rounding["native_target_conflict"] is False
    tiny = next(
        pair for pair in result["pairs"] if (pair["first_row"], pair["second_row"]) == (0, 3)
    )
    assert tiny["input_l2"] == 1e-300 and tiny["duplicate_input"] is False


@pytest.fixture(scope="module")
def full_data():
    return full_groups()


@pytest.fixture(scope="module")
def full_report(ledger, full_data):
    before = tuple(asdict(item) for _, item in full_data)
    result = grouped.diagnose_grouped_input_targets(full_data)
    assert tuple(asdict(item) for _, item in full_data) == before
    return result


def test_should_keep_all198_rows_once_and_charge_all21_kernel_calls_with_repeated_work(full_report):
    report = full_report
    assert report["complete"] is True and report["rows"] == 198 and report["input_dimension"] == 12
    assert [(row["first_row"], row["second_row"]) for row in report["pairs"]] == [
        (first, second) for first in range(198) for second in range(first + 1, 198)
    ]
    assert len(report["blocks"]) == 21 and all(row["validated"] is True for row in report["blocks"])
    assert [row["groups"] for row in report["blocks"][:6]] == [[name] for name, _ in full_groups()]
    work = report["work"]
    for key, expected in dict(
        kernel_calls_attempted=21,
        kernel_calls_returned=21,
        kernel_rows=1188,
        pair_attempts=39693,
        pair_returns=39693,
        native_conversion_attempts=4752,
        native_conversion_returns=4752,
        retained_pairs=19503,
        discarded_repeated_pairs=20190,
        global_nearest_comparisons=39006,
        logical_owned_array_bytes=152064,
        native_target_float32_numeric_bytes=19008,
        owned_native_target_numeric_bytes=38016,
    ).items():
        assert work[key] == expected
    bindings = report["row_bindings"]
    flattened = [(name, item, row) for name, item in full_groups() for row in item.transitions]
    assert bindings == [
        dict(
            global_row=index,
            group=name,
            episode_id=item.episode_id,
            context=item.context,
            row=row.step,
            step=row.step,
        )
        for index, (name, item, row) in enumerate(flattened)
    ]
    for index, (_, _, row) in enumerate(flattened):
        duplicate_indices = [
            other
            for other, (_, _, candidate) in enumerate(flattened)
            if other != index and candidate.observation == row.observation
        ]
        assert report["nearest_other_rows"][index] == min(duplicate_indices)
    assert (
        report["raw_conflict_pairs"]
        == report["native_conflict_pairs"]
        == report["opposite_gripper_pairs"]
        == 0
    )


def test_should_detach_group_bindings_pairs_and_counters_from_caller_inputs(full_report, full_data):
    data = full_data
    before = tuple(asdict(item) for _, item in data)
    # Mutating caller records after completion cannot change the cached report.
    object.__setattr__(data[0][1], "context", "changed-after-return")
    assert full_report["groups"][0]["context"] == before[0]["context"]
    assert full_report["row_bindings"][0]["context"] == before[0]["context"]


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "pair_list",
        "one",
        "seven",
        "duplicate_name",
        "duplicate_id",
        "role",
        "spec",
        "dimension",
        "rows",
        "total",
        "gripper",
        "nonfinite",
        "name",
    ],
)
def test_should_refuse_all_bad_groups_before_any_kernel_call(monkeypatch, fault):
    data: Any = small_groups()
    if fault == "list":
        data = list(data)
    elif fault == "pair_list":
        data = (list(data[0]),) + data[1:]
    elif fault == "one":
        data = data[:1]
    elif fault == "seven":
        data = tuple((str(index), replace(data[0][1], episode_id=str(index))) for index in range(7))
    elif fault == "duplicate_name":
        data = (data[0], (data[0][0], data[1][1]))
    elif fault == "duplicate_id":
        data = (data[0], (data[1][0], replace(data[1][1], episode_id=data[0][1].episode_id)))
    elif fault == "role":
        data = data[:-1] + (("c", replace(data[-1][1], role="probe_test")),)
    elif fault == "spec":
        item = data[-1][1]
        data = data[:-1] + (
            (
                "c",
                replace(
                    item,
                    spec=replace(item.spec, action_units=("other",) + item.spec.action_units[1:]),
                ),
            ),
        )
    elif fault == "dimension":
        item = episode("large", ((0.0,) * 65,), ((0.0, 0.0, 0.0, -1.0),))
        data = (("x", item), ("y", replace(item, episode_id="other")))
    elif fault == "rows":
        data = (("x", episode("large", ((0.0, 0.0),) * 51, ((0.0, 0.0, 0.0, -1.0),) * 51)), data[1])
    elif fault == "total":
        data = tuple(
            (
                str(index),
                episode(str(index), ((0.0, 0.0),) * count, ((0.0, 0.0, 0.0, -1.0),) * count),
            )
            for index, count in enumerate((50, 50, 25, 24, 25, 25))
        )
    elif fault == "gripper":
        object.__setattr__(data[-1][1].transitions[0], "action", (0.0, 0.0, 0.0, 0.0))
    elif fault == "nonfinite":
        object.__setattr__(data[-1][1].transitions[0], "observation", (math.nan, 0.0))
    else:
        data = ((True, data[0][1]),) + data[1:]
    monkeypatch.setattr(
        grouped,
        "continuous_input_target_consistency",
        lambda *args, **options: pytest.fail("invalid groups reached kernel"),
    )
    with pytest.raises(ValueError):
        grouped.diagnose_grouped_input_targets(data)


def test_should_keep_known_overflow_work_and_refuse_unobserved_cross_return():
    data = tuple(
        (name, episode(name, ((value, 0.0),), ((0.0, 0.0, 0.0, -1.0),)))
        for name, value in (("negative", -1e308), ("positive", 1e308))
    )
    result = grouped.diagnose_grouped_input_targets(data)
    assert result["complete"] is False and result["unknown_unreturned_kernel_work"] is True
    assert len(result["blocks"]) == 3 and [row["returned"] for row in result["blocks"]] == [
        True,
        True,
        False,
    ]
    assert (
        result["pairs"] == []
        and result["work"]["pair_attempts"] == 1
        and result["work"]["pair_returns"] == 0
    )
    assert result["work"]["native_conversion_returns"] == 16


def test_should_preserve_validated_prefix_and_paid_counters_when_next_kernel_return_is_lost(
    monkeypatch,
):
    kernel = grouped.continuous_input_target_consistency
    calls = 0

    def unavailable(*args, **options):
        nonlocal calls
        calls += 1
        result = kernel(*args, **options)
        if calls == 2:
            raise RuntimeError("paid kernel return unavailable")
        return result

    monkeypatch.setattr(grouped, "continuous_input_target_consistency", unavailable)
    result = grouped.diagnose_grouped_input_targets(small_groups())
    assert (
        calls == 2
        and result["complete"] is False
        and result["unknown_unreturned_kernel_work"] is True
    )
    assert (
        len(result["pairs"]) == 1
        and result["pairs"][0]["first_row"] == 0
        and result["pairs"][0]["second_row"] == 1
    )
    assert [row["validated"] for row in result["blocks"]] == [True, False]
    assert (
        result["work"]["kernel_calls_attempted"] == 2
        and result["work"]["kernel_calls_returned"] == 1
    )
    assert result["work"]["pair_returns"] == 2
