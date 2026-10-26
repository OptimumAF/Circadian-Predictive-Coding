"""TRAIN feature support from twenty prepared rows; no learner fixtures.

Expected summaries use the synthetic source commands and displacements directly,
including the difference between a commanded close and a provisional open target.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pytest

from src.adapters import joint_readiness_support as support
from src.adapters.joint_readiness_data import JointReadinessData, prepare_joint_readiness_data
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from test_joint_readiness_corrections import inputs


STAGES = ("align", "descend", "grasp", "lift", "settle")
FEATURE_NAMES = [
    "is_grasped",
    "object_minus_tcp_x_over_0p1m",
    "object_minus_tcp_y_over_0p1m",
    "object_minus_tcp_z_over_0p1m",
    "goal_minus_object_x_over_0p1m",
    "goal_minus_object_y_over_0p1m",
    "goal_minus_object_z_over_0p1m",
    "previous_executed_dx",
    "previous_executed_dy",
    "previous_executed_dz",
    "previous_executed_gripper",
    "closing_count_over_6",
]


@pytest.fixture(scope="module", autouse=True)
def bounded_diagnostic_work():
    counts = dict(prepared_rows=0, diagnostic_calls=0, diagnostic_row_visit_upper_bound=0)
    original = support.summarize_joint_readiness_support

    def summarize(data):
        counts["diagnostic_calls"] += 1
        if type(data) is JointReadinessData and type(data.rows) is tuple:
            # Conservatively charge all three summary passes even when admission
            # rejects the input before traversing its rows.
            counts["diagnostic_row_visit_upper_bound"] += 3 * len(data.rows)
        return original(data)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(support, "summarize_joint_readiness_support", summarize)
        yield counts
    if os.environ.get("TFS_LEDGER"):
        with Path(os.environ["TFS_LEDGER"]).open("x", encoding="utf8") as handle:
            json.dump(counts, handle, sort_keys=True)
            handle.write("\n")
    assert counts["prepared_rows"] == 20
    assert counts["diagnostic_row_visit_upper_bound"] <= 4096


@pytest.fixture(scope="module")
def prepared(bounded_diagnostic_work):
    reference, corrections, curated = inputs()
    original = (asdict(reference), asdict(corrections), asdict(curated))
    ab = prepare_joint_readiness_data(reference, corrections, None, now=103)
    abc = prepare_joint_readiness_data(reference, corrections, curated, now=103)
    bounded_diagnostic_work["prepared_rows"] += len(ab.rows) + len(abc.rows)
    assert (asdict(reference), asdict(corrections), asdict(curated)) == original
    return ab, abc


def test_should_report_exact_global_marginal_support_and_distinct_preparation_cost(prepared):
    _, data = prepared
    before = deepcopy(asdict(data))
    report = support.summarize_joint_readiness_support(data)
    assert report["protocol"] == "pick_cube_train_feature_support_v1"
    assert report["feature_names"] == FEATURE_NAMES
    summary = report["global"]
    assert summary["rows"] == summary["unique_feature_vectors"] == 6
    assert summary["unique_previous_actions"] == 3
    assert summary["grasp_counts"] == [6, 0]
    assert summary["closing_count_counts"] == [4, 2, 0, 0, 0, 0, 0]
    expected = [
        (0.0, 0.0),
        (-0.02 / 0.1, (0.03 + 0.001) / 0.1),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.25, 0.25),
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (0.0, 1 / 6),
    ]
    assert summary["coordinate_ranges"] == [dict(min=lo, max=hi) for lo, hi in expected]
    assert summary["constant_coordinates"] == [0, 2, 3, 4, 5, 6, 8, 9]
    for key, value in dict(
        rows_validated=6,
        reprojected_rows=6,
        source_values_validated=252,
        emitted_feature_values=72,
        logical_feature_numeric_bytes=576,
        history_advances=6,
        summary_row_visits=18,
        summary_coordinate_visits=216,
    ).items():
        assert report["work"][key] == value
    assert asdict(data) == before
    assert data.work["observation_projections"] == 12


@pytest.mark.parametrize("index,names", [(0, ("a", "b")), (1, ("a", "b", "curated_a"))])
def test_should_group_each_original_source_and_all_five_recorded_stages(prepared, index, names):
    data = prepared[index]
    report = support.summarize_joint_readiness_support(data)
    assert tuple(report["by_source"]) == names
    assert tuple(report["by_source_stage"]) == names
    for name in names:
        source = report["by_source"][name]
        assert source["rows"] == source["unique_feature_vectors"] == 2
        assert source["unique_previous_actions"] == 2
        assert source["grasp_counts"] == [2, 0]
        assert source["closing_count_counts"] == (
            [2, 0, 0, 0, 0, 0, 0] if name == "a" else [1, 1, 0, 0, 0, 0, 0]
        )
        assert tuple(report["by_source_stage"][name]) == STAGES
        assert report["by_source_stage"][name]["align"] == source
        for stage in STAGES[1:]:
            empty = report["by_source_stage"][name][stage]
            assert empty == dict(
                rows=0,
                coordinate_ranges=None,
                constant_coordinates=[],
                unique_feature_vectors=0,
                unique_previous_actions=0,
                grasp_counts=[0, 0],
                closing_count_counts=[0] * 7,
            )
    assert report["global"]["rows"] == (4 if index == 0 else 6)
    assert ("curated_a" in report["by_source"]) is bool(index)


def test_should_use_recorded_stage_and_executed_command_history_without_reading_targets_or_clocks(
    prepared,
):
    changed = deepcopy(prepared[1])
    changed.rows[1]["stage"] = "descend"
    for row in changed.rows:
        for key in (
            "target",
            "observed_at",
            "labels_arrived_at",
            "source_observed_at",
            "queried_at",
            "arrived_at",
            "source_available_at",
        ):
            row.pop(key, None)
    report = support.summarize_joint_readiness_support(changed)
    assert report["global"] == support.summarize_joint_readiness_support(prepared[1])["global"]
    assert report["by_source_stage"]["a"]["align"]["rows"] == 1
    assert report["by_source_stage"]["a"]["descend"]["rows"] == 1
    # B and C labels command an open gripper, while their original executed
    # commands close it. Only the latter supplies previous-action support.
    for source in ("b", "curated_a"):
        assert report["by_source"][source]["coordinate_ranges"][10] == dict(min=-1.0, max=0.0)
        assert report["by_source"][source]["closing_count_counts"] == [1, 1, 0, 0, 0, 0, 0]


def test_should_detach_every_returned_summary_container_and_preserve_input(prepared):
    data = prepared[1]
    before = deepcopy(asdict(data))
    report = support.summarize_joint_readiness_support(data)
    expected = deepcopy(report)
    report["feature_names"][0] = "changed"
    report["global"]["coordinate_ranges"][0]["min"] = -999.0
    assert report["by_source"]["a"]["coordinate_ranges"][0]["min"] == 0.0
    report["by_source"]["a"]["grasp_counts"][0] = 999
    assert report["by_source_stage"]["a"]["align"]["grasp_counts"] == [2, 0]
    report["by_source_stage"]["a"]["align"]["closing_count_counts"][0] = 999
    assert report["by_source"]["a"]["closing_count_counts"] == [2, 0, 0, 0, 0, 0, 0]
    report["work"]["rows_validated"] = 999
    assert asdict(data) == before
    assert support.summarize_joint_readiness_support(data) == expected
    assert expected["by_source"]["a"]["grasp_counts"] == [2, 0]


def test_should_report_all_closing_bins_and_saturation_from_actual_commands(
    bounded_diagnostic_work,
):
    reference, corrections, _ = inputs(b_rows=8)
    data = prepare_joint_readiness_data(reference, corrections, None, now=103)
    bounded_diagnostic_work["prepared_rows"] += len(data.rows)
    report = support.summarize_joint_readiness_support(data)
    assert report["by_source"]["b"]["closing_count_counts"] == [1, 1, 1, 1, 1, 1, 2]
    assert report["by_source"]["b"]["coordinate_ranges"][11] == dict(min=0.0, max=1.0)
    assert report["by_source"]["a"]["closing_count_counts"] == [2, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "context",
        "episode_id",
        "source_kind",
        "step",
        "row_index",
        "unknown_stage",
        "raw_nan",
        "feature_nan",
        "geometry",
        "history",
        "integer_action",
        "bool_action",
        "nonfinite_action",
        "out_of_bounds_action",
        "unquantized_action",
    ],
)
def test_should_refuse_altered_source_identity_geometry_history_or_commands(prepared, fault):
    data = deepcopy(prepared[1])
    row = data.rows[3]
    if fault in ("role", "context", "episode_id", "source_kind", "unknown_stage"):
        field = "stage" if fault == "unknown_stage" else fault
        row[field] = {
            "role": "inner_guard",
            "context": "pick-cube-inner-guard-31",
            "episode_id": "pick-cube-guard-smoke-31",
            "source_kind": "qualified_demonstration",
            "stage": "unknown",
        }[field]
    elif fault == "step":
        row["step"] = 0
    elif fault == "row_index":
        row["row_index"] = 3.0
    elif fault == "raw_nan":
        row["source_observation"][36] = float("nan")
    elif fault == "feature_nan":
        row["features"][1] = float("nan")
    elif fault == "geometry":
        row["features"][1] += 0.125
    elif fault == "history":
        row["features"][10] = 1.0  # provisional target cannot replace executed history
    else:
        row["actual_command"][0] = {
            "integer_action": 0,
            "bool_action": False,
            "nonfinite_action": float("inf"),
            "out_of_bounds_action": 1.25,
            "unquantized_action": 0.1,
        }[fault]
    with pytest.raises(ValueError):
        support.summarize_joint_readiness_support(data)


@pytest.mark.parametrize(
    "fault",
    [
        "foreign",
        "foreign_spec",
        "spec",
        "bounds",
        "units",
        "zero_recipe",
        "history47",
        "reordered_groups",
        "gap",
        "bool_boundary",
        "empty_source",
        "oversize",
    ],
)
def test_should_refuse_unsupported_prepared_spec_recipe_or_source_partition(prepared, fault):
    data = deepcopy(prepared[1])
    if fault == "foreign":
        data = asdict(data)
    elif fault == "foreign_spec":

        class EqualSpec:
            def __eq__(self, other):
                return True

        data = replace(data, spec=EqualSpec())
    elif fault == "spec":
        data = replace(data, spec=replace(data.spec, observation_dim=11))
    elif fault == "bounds":
        data = replace(data, spec=replace(data.spec, lower_bounds=(-2.0,) * 4))
    elif fault == "units":
        data = replace(
            data, spec=replace(data.spec, action_units=tuple(reversed(data.spec.action_units)))
        )
    elif fault == "zero_recipe":
        data = replace(data, recipe=PickCubeTaskSpaceHistoryRecipe(mode="zero"))
    elif fault == "history47":
        data = replace(data, recipe=PickCubeActionHistoryRecipe())
    elif fault == "reordered_groups":
        data = replace(data, groups=(("b", 0, 2), ("a", 2, 4), ("curated_a", 4, 6)))
    elif fault == "gap":
        data = replace(data, groups=(("a", 0, 2), ("b", 3, 4), ("curated_a", 4, 6)))
    elif fault == "bool_boundary":
        data = replace(data, groups=(("a", False, 2), ("b", 2, 4), ("curated_a", 4, 6)))
    elif fault == "empty_source":
        data = replace(data, groups=(("a", 0, 2), ("b", 2, 4), ("curated_a", 4, 4)))
    else:
        # Copies are invalid admission data, not additional prepared/projected rows.
        data = replace(
            data,
            rows=tuple(deepcopy(data.rows[i % 6]) for i in range(129)),
            groups=(("a", 0, 50), ("b", 50, 100), ("curated_a", 100, 129)),
        )
    with pytest.raises(ValueError):
        support.summarize_joint_readiness_support(data)
