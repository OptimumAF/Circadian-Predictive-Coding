"""Exact-input label conflicts and native command quantization, without learning.

Prospective:7 core calls reach arithmetic (two are partial overflow refusals),
19 complete pairs+2 attempted failed pairs;13 other core entries refuse
before conversion/pairs. Two adapter paths project2 teacher+2 query rows each.
Zero model/RNG/fit/SGD/forward/gradient/scene/native work. Root alone executes.
"""

from dataclasses import FrozenInstanceError, asdict
import math
from typing import Any

import numpy as np
import pytest

from src.adapters import readiness_label_separation as adapter
from src.core import continuous_bc_diagnostics as core
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_readiness_corrections import (
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
    GoalConsistentObservedGraspCorrectionBatch,
    ObservedGraspCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")
RECIPE = PickCubeTaskSpaceHistoryRecipe()


def spec(dimension=2):
    return ContinuousActionSpec(dimension, UNITS, (-1.0,) * 4, (1.0,) * 4)


def arrays():
    return (
        np.asarray([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1e-300, 0.0)], dtype=np.float64),
        np.asarray(
            [
                (0.0, 0.0, 0.0, 1.0),
                (0.5, -0.25, 0.125, -1.0),
                (0.5 + 1e-12, -0.25, 0.125, -1.0),
                (0.5, -0.25, 0.125, -1.0),
            ],
            dtype=np.float64,
        ),
    )


@pytest.fixture(scope="module")
def measured():
    features, targets = arrays()
    before = features.copy(), targets.copy()
    work = core._InputTargetConsistencyWork()
    result = core.continuous_input_target_consistency(features, targets, spec(), _work=work)
    np.testing.assert_array_equal(features, before[0])
    np.testing.assert_array_equal(targets, before[1])
    return result, work


def test_should_keep_all_original_pairs_and_count_conflicts_only_at_identical_inputs(measured):
    result, _ = measured
    assert result.definition == "continuous_input_target_exact_consistency_v1"
    assert (result.rows, result.input_dimension, result.action_dimension) == (4, 2, 4)
    assert [(pair.first_row, pair.second_row) for pair in result.pairs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert result.duplicate_input_pairs == result.raw_conflict_pairs == 3
    assert result.native_conflict_pairs == 2 and result.opposite_gripper_pairs == 3
    first = result.pairs[0]
    assert first.duplicate_input and first.raw_target_conflict and first.native_target_conflict
    assert first.opposite_gripper is True
    assert first.target_delta == first.native_target_delta == (0.5, -0.25, 0.125, -2.0)
    assert (
        first.target_l2
        == first.native_target_l2
        == pytest.approx(math.sqrt(4 + 0.25 + 0.0625 + 0.015625))
    )
    assert result.pairs[2].duplicate_input is False
    assert result.pairs[2].raw_target_conflict is result.pairs[2].native_target_conflict is False
    assert result.pairs[2].opposite_gripper is True


def test_should_preserve_raw_xyz_disagreement_that_native_float32_cannot_distinguish(measured):
    result, _ = measured
    pair = result.pairs[3]
    delta = (0.5 + 1e-12) - 0.5
    assert pair.first_row == 1 and pair.second_row == 2 and pair.duplicate_input
    assert pair.target_delta == (delta, 0.0, 0.0, 0.0)
    assert pair.native_target_delta == (0.0,) * 4
    assert pair.target_l2 == delta and pair.native_target_l2 == 0.0
    assert pair.raw_target_conflict is True and pair.native_target_conflict is False
    assert pair.opposite_gripper is False


def test_should_keep_tiny_input_distance_nonzero_and_use_lowest_neighbor_on_ties(measured):
    result, _ = measured
    assert result.nearest_other_rows == (1, 0, 0, 0)
    for index in (2, 4, 5):
        pair = result.pairs[index]
        assert pair.input_l2 == 1e-300 and pair.duplicate_input is False
    assert result.pairs[0].input_l2 == result.pairs[1].input_l2 == result.pairs[3].input_l2 == 0.0


def test_should_own_native_scalar_records_and_detach_returned_work_from_mutable_attempt_counters(
    measured,
):
    result, work = measured
    expected = dict(
        native_conversion_attempts=16,
        native_conversion_returns=16,
        pair_attempts=6,
        pair_returns=6,
        input_coordinate_differences=12,
        target_coordinate_differences=24,
        native_target_coordinate_differences=24,
        nearest_comparisons=12,
    )
    assert asdict(work) == expected
    assert {name: result.work[name] for name in expected} == expected
    for pair in result.pairs:
        assert type(pair.target_delta) is tuple and all(
            type(value) is float for value in pair.target_delta
        )
        assert type(pair.native_target_delta) is tuple and all(
            type(value) is float for value in pair.native_target_delta
        )
    with pytest.raises(FrozenInstanceError):
        setattr(result.pairs[0], "input_l2", 9.0)
    try:
        work.pair_attempts += 1
        assert result.work["pair_attempts"] == 6
    finally:
        work.pair_attempts -= 1


def test_should_return_unknown_nearest_for_a_single_row_without_fabricating_pairs():
    result = core.continuous_input_target_consistency(
        np.zeros((1, 2), dtype=np.float64),
        np.asarray([(0.0, 0.0, 0.0, -1.0)], dtype=np.float64),
        spec(),
    )
    assert result.pairs == () and result.nearest_other_rows == (None,)
    assert (
        result.duplicate_input_pairs
        == result.raw_conflict_pairs
        == result.native_conflict_pairs
        == 0
    )
    assert result.work["native_conversion_returns"] == 4
    assert result.work["pair_attempts"] == result.work["nearest_comparisons"] == 0


def test_should_measure_large_finite_distances_without_squaring_overflow():
    result = core.continuous_input_target_consistency(
        np.asarray([(0.0, 0.0), (1e200, 1e200)], dtype=np.float64),
        np.asarray([(0.0, 0.0, 0.0, -1.0), (0.25, 0.0, 0.0, 1.0)], dtype=np.float64),
        spec(),
    )
    assert result.pairs[0].input_l2 == pytest.approx(math.sqrt(2) * 1e200)
    assert result.pairs[0].opposite_gripper is True
    assert result.nearest_other_rows == (1, 0)


def test_should_refuse_unrepresentable_subtraction_and_keep_accumulated_partial_work():
    features = np.asarray([(-1e308, 0.0), (1e308, 0.0)], dtype=np.float64)
    targets = np.asarray([(0.0, 0.0, 0.0, -1.0)] * 2, dtype=np.float64)
    work = core._InputTargetConsistencyWork(
        native_conversion_attempts=2, native_conversion_returns=2
    )
    before = features.copy(), targets.copy()
    with pytest.raises(FloatingPointError):
        core.continuous_input_target_consistency(features, targets, spec(), _work=work)
    assert asdict(work) == dict(
        native_conversion_attempts=10,
        native_conversion_returns=10,
        pair_attempts=1,
        pair_returns=0,
        input_coordinate_differences=1,
        target_coordinate_differences=0,
        native_target_coordinate_differences=0,
        nearest_comparisons=0,
    )
    np.testing.assert_array_equal(features, before[0])
    np.testing.assert_array_equal(targets, before[1])


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "shape",
        "nan",
        "inf",
        "bounds",
        "target_bounds",
        "gripper",
        "rows",
        "dimension",
        "actions",
    ],
)
def test_should_refuse_invalid_numeric_inputs_before_native_conversion_or_pair_work(fault):
    features, targets = arrays()
    selected_spec = spec()
    if fault == "dtype":
        features = features.astype(np.float32)
    elif fault == "shape":
        features = features.ravel()
    elif fault in ("nan", "inf"):
        features[0, 0] = float(fault)
    elif fault == "bounds":
        selected_spec = ContinuousActionSpec(
            2, UNITS, (-2.0, -1.0, -1.0, -1.0), (2.0, 1.0, 1.0, 1.0)
        )
    elif fault == "target_bounds":
        targets[0, 0] = 1.01
    elif fault == "gripper":
        targets[0, 3] = 0.0
    elif fault == "rows":
        features = np.zeros((101, 2), dtype=np.float64)
        targets = np.asarray([(0.0, 0.0, 0.0, -1.0)] * 101, dtype=np.float64)
    elif fault == "dimension":
        selected_spec = spec(3)
    else:
        selected_spec = ContinuousActionSpec(2, UNITS[:3], (-1.0,) * 3, (1.0,) * 3)
    work = core._InputTargetConsistencyWork()
    with pytest.raises(ValueError):
        core.continuous_input_target_consistency(features, targets, selected_spec, _work=work)
    assert all(value == 0 for value in asdict(work).values())


@pytest.mark.parametrize("fault", ["mapping", "bool", "negative"])
def test_should_refuse_foreign_or_corrupt_work_records_without_numeric_work(fault):
    features, targets = arrays()
    work: Any = core._InputTargetConsistencyWork()
    if fault == "mapping":
        work = asdict(work)
    elif fault == "bool":
        work.pair_returns = True
    else:
        work.nearest_comparisons = -1
    with pytest.raises(ValueError):
        core.continuous_input_target_consistency(features, targets, spec(), _work=work)


def raw(index):
    values = [0.0] * 42
    values[0], values[7], values[8], values[18] = index / 8, 0.02, 0.02, 1.0
    values[22], values[32] = 1.0, 1.0
    values[36:42] = (0.1 - index * 0.025, 0.0, 0.0, 0.018, 0.018, 0.0)
    return tuple(values)


def teacher(*, seed=17, overflow=False):
    def observation(index):
        values = list(raw(index))
        if overflow:
            values[0] = -1e308 if index == 0 else 1e308
        return tuple(values)

    episode = ContinuousEpisode(
        f"pick-cube-readiness-teacher-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        spec(42),
        tuple(
            ContinuousTransition(
                index,
                observation(index),
                action,
                float(index == 1),
                observation(index + 1),
                index == 1,
                False,
            )
            for index, action in enumerate(((0.25, 0.0, 0.0, -1.0), (0.0, 0.0, 0.0, -1.0)))
        ),
    )
    return ReadinessStageBinding(episode, ("lift", "settle"))


def corrections():
    episode = ContinuousEpisode(
        "pick-cube-smoke-17",
        "pick-cube",
        "train",
        spec(42),
        tuple(
            ContinuousTransition(
                index, raw(index), action, float(index == 1), raw(index + 1), index == 1, False
            )
            for index, action in enumerate(((0.5, 0.0, 0.0, 0.5), (0.0, -0.25, 0.0, -0.5)))
        ),
    )
    return GoalConsistentObservedGraspCorrectionBatch(
        17,
        episode,
        (
            ReadinessCorrectionLabel(0, raw(0), (0.18, 0.18, 0.0, -1.0), "lift"),
            ReadinessCorrectionLabel(1, raw(1), (0.0, 0.0, 0.0, -1.0), "settle"),
        ),
        (True, True),
    )


@pytest.fixture(scope="module")
def separated():
    source, query = teacher(), corrections()
    before = asdict(source), asdict(query)
    captures = []
    original = adapter.continuous_input_target_consistency

    def capture(features, targets, specification, **options):
        captures.append((features.copy(), targets.copy(), specification))
        return original(features, targets, specification, **options)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter, "continuous_input_target_consistency", capture)
        report = adapter.run_readiness_label_separation(source, query, now=3, recipe=RECIPE)
    assert report["complete"] is True and report["error"] is None
    assert (asdict(source), asdict(query)) == before
    return dict(report=report, source=source, query=query, captures=captures)


def test_should_keep_teacher_provenance_distinct_from_provisional_query_stages_and_original_arrivals(
    separated,
):
    report, source, query = separated["report"], separated["source"], separated["query"]
    assert report["definition"] == "readiness_raw42_taskspace12_label_separation_v1"
    assert len(report["rows"]) == 4 and report["recipe"] == asdict(RECIPE)
    for index, row in enumerate(report["rows"]):
        assert row["index"] == index and row["step"] == index % 2
        assert row["context"] == "pick-cube" and row["role"] == "train"
        assert row["stage"] == ("lift" if index % 2 == 0 else "settle")
        if index < 2:
            assert row["source_kind"] == "qualified_teacher"
            assert (
                row["episode_id"] == source.episode.episode_id
                and row["definition"] == READINESS_DEFINITION
            )
            assert "observed_at" not in row and "labels_arrived_at" not in row
        else:
            assert row["source_kind"] == "provisional_correction"
            assert row["episode_id"] == query.episode.episode_id
            assert row["definition"] == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
            assert row["provenance"] == GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
            assert (row["observed_at"], row["labels_arrived_at"]) == (
                (0, 1) if index == 2 else (2, 3)
            )
    expected_targets = tuple(row.action for row in source.episode.transitions) + tuple(
        label.target for label in query.labels
    )
    expected_actual = tuple(
        row.action for row in source.episode.transitions + query.episode.transitions
    )
    assert report["targets"] == expected_targets and report["executed_actions"] == expected_actual
    assert report["targets"][2:] != report["executed_actions"][2:]
    for _, targets, _ in separated["captures"]:
        assert targets.tolist() == [list(value) for value in expected_targets]


def test_should_encode_the_original_executed_history_before_comparing_inputs_without_changing_targets(
    separated,
):
    report, source, query, captures = (
        separated[name] for name in ("report", "source", "query", "captures")
    )
    assert [capture[2].observation_dim for capture in captures] == [42, 12]
    raw_values = [row.observation for row in source.episode.transitions] + [
        label.observation for label in query.labels
    ]
    assert captures[0][0].tolist() == [list(value) for value in raw_values]
    tails = [(0.0,) * 5, (0.25, 0.0, 0.0, -1.0, 1 / 6), (0.0,) * 5, (0.5, 0.0, 0.0, 0.5, 0.0)]
    expected = [
        list((value[18],) + tuple(component / 0.1 for component in value[36:42]) + tail)
        for value, tail in zip(raw_values, tails, strict=True)
    ]
    assert captures[1][0].tolist() == expected
    assert captures[1][0][3, -5:-1].tolist() != list(query.labels[0].target)
    raw_result, model_result = (
        report["views"][name]["result"] for name in ("raw42", "taskspace12")
    )
    assert raw_result["duplicate_input_pairs"] == 2
    assert model_result["duplicate_input_pairs"] == 1
    assert raw_result["raw_conflict_pairs"] == model_result["raw_conflict_pairs"] == 1
    assert raw_result["native_conflict_pairs"] == model_result["native_conflict_pairs"] == 1
    assert raw_result["opposite_gripper_pairs"] == model_result["opposite_gripper_pairs"] == 0


def test_should_report_actual_pair_and_projection_work_without_inventing_learner_or_acquisition_work(
    separated,
):
    report = separated["report"]
    work = report["work"]
    assert work["native_conversion_attempts"] == work["native_conversion_returns"] == 32
    assert work["pair_attempts"] == work["pair_returns"] == 12
    assert work["input_coordinate_differences"] == 324
    assert (
        work["target_coordinate_differences"] == work["native_target_coordinate_differences"] == 48
    )
    assert work["nearest_comparisons"] == 24
    for name in (
        "model_creation_calls",
        "rng_draws",
        "fit_calls",
        "optimizer_updates",
        "predictions",
        "gradients",
        "native_controls",
    ):
        assert work[name] == 0
    projections = report["projection_work"]
    assert projections["teacher_attempts"] == projections["teacher_returns"] == 1
    assert projections["correction_attempts"] == projections["correction_returns"] == 1
    assert projections["attempted_observation_projections_upper_bound"] == 8
    assert projections["teacher"]["observation_projections"] == 4
    assert projections["teacher"]["projected_values"] == 48
    assert projections["teacher"]["physical_scale_divisions"] == 24
    assert projections["corrections"]["observation_projections"] == 4
    assert projections["corrections"]["projected_values"] == 48
    assert projections["corrections"]["physical_scale_divisions"] == 24


def test_should_keep_raw_failure_work_and_stop_the_second_view_after_finite_subtraction_overflow(
    monkeypatch,
):
    calls = []
    original = adapter.continuous_input_target_consistency

    def capture(features, targets, specification, **options):
        calls.append(specification.observation_dim)
        return original(features, targets, specification, **options)

    monkeypatch.setattr(adapter, "continuous_input_target_consistency", capture)
    report = adapter.run_readiness_label_separation(
        teacher(overflow=True), corrections(), now=3, recipe=RECIPE
    )
    assert report["complete"] is False and "unrepresentable" in report["error"]
    assert calls == [42]
    raw_view, model_view = report["views"]["raw42"], report["views"]["taskspace12"]
    assert raw_view["complete"] is False and raw_view["result"] is None
    assert (
        raw_view["work"]["native_conversion_attempts"]
        == raw_view["work"]["native_conversion_returns"]
        == 16
    )
    assert raw_view["work"]["pair_attempts"] == 1 and raw_view["work"]["pair_returns"] == 0
    assert raw_view["work"]["input_coordinate_differences"] == 1
    assert (
        raw_view["work"]["target_coordinate_differences"]
        == raw_view["work"]["nearest_comparisons"]
        == 0
    )
    assert model_view == dict(
        complete=False, result=None, work=asdict(core._InputTargetConsistencyWork())
    )
    assert (
        report["projection_work"]["teacher_returns"]
        == report["projection_work"]["correction_returns"]
        == 1
    )


def forbidden(*args, **options):
    pytest.fail("invalid/unreleased evidence admitted projection or pair arithmetic")


@pytest.mark.parametrize(
    "fault",
    [
        "unarrived",
        "teacher_role",
        "query_role",
        "context",
        "model_input",
        "zero_recipe",
        "v1_batch",
        "mixed_teacher_id",
        "raw_flag",
        "target",
    ],
)
def test_should_refuse_unreleased_foreign_or_relabelled_evidence_before_projection_or_numeric_work(
    monkeypatch, fault
):
    source, query, now, recipe = teacher(), corrections(), 3, RECIPE
    if fault == "unarrived":
        now = 2
    elif fault == "teacher_role":
        object.__setattr__(source.episode, "role", "final_test")
    elif fault == "query_role":
        object.__setattr__(query.episode, "role", "probe_train")
    elif fault == "context":
        source = teacher(seed=19)
    elif fault == "model_input":
        source = ReadinessStageBinding(
            history.action_history_training_episode(source.episode, RECIPE), source.stages
        )
    elif fault == "zero_recipe":
        recipe = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    elif fault == "v1_batch":
        query = ObservedGraspCorrectionBatch(
            query.train_seed, query.episode, query.labels, query.observed_grasp
        )
    elif fault == "mixed_teacher_id":
        object.__setattr__(source.episode, "episode_id", "pick-cube-smoke-17")
    elif fault == "raw_flag":
        values = list(query.episode.transitions[0].observation)
        values[18] = True
        object.__setattr__(query.episode.transitions[0], "observation", tuple(values))
    else:
        object.__setattr__(query.labels[0], "target", (0.18, 0.18, 0.0, 0.0))
    monkeypatch.setattr(adapter, "action_history_training_episode", forbidden)
    monkeypatch.setattr(adapter, "action_history_correction_view", forbidden)
    monkeypatch.setattr(adapter, "continuous_input_target_consistency", forbidden)
    with pytest.raises(ValueError):
        adapter.run_readiness_label_separation(source, query, now=now, recipe=recipe)
