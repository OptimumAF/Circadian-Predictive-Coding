"""Describe released TRAIN features without learning or changing data authority.

Input is already-prepared JointReadinessData; qualification and label release
remain the caller/preparer's responsibility. Output owns marginal feature ranges
and recorded source/stage history counts. No labels, clocks, stage inference,
learner, decision threshold, simulator, file IO or promotion belongs here.
"""

import math
from typing import Any

from src.adapters.joint_readiness_data import JointReadinessData
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.pick_cube_action_history import (
    ActionHistoryState,
    action_history_observation,
    advance_action_history,
)
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_taskspace_history import validate_taskspace_history_recipe
from src.core.pick_cube_train_context import pick_cube_training_context


FEATURE_NAMES = (
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
)
STAGES = ("align", "descend", "grasp", "lift", "settle")
_SOURCES = {
    "a": ("qualified_demonstration", "pick-cube-observed-grasp-teacher-17", 17),
    "b": ("provisional_correction", "pick-cube-smoke-19", 19),
    "curated_a": ("curated_provisional_correction", "pick-cube-smoke-17", 17),
}


def _vector(value: Any, size: int) -> tuple[float, ...]:
    if (
        type(value) not in (list, tuple)
        or len(value) != size
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"TRAIN support requires {size} finite native numeric values")
    return tuple(float(v) for v in value)


def _validate_groups(data: JointReadinessData) -> None:
    if type(data.rows) is not tuple or not 1 <= len(data.rows) <= 128:
        raise ValueError("TRAIN support requires one to128 prepared rows")
    if type(data.groups) is not tuple or len(data.groups) not in (2, 3):
        raise ValueError("TRAIN support requires original a,b[,curated_a] groups")
    start = 0
    for group, name in zip(data.groups, _SOURCES, strict=False):
        if (
            type(group) is not tuple
            or len(group) != 3
            or group[0] != name
            or any(type(i) is not int for i in group[1:])
            or group[1] != start
            or not 1 <= group[2] - start <= 50
        ):
            raise ValueError("TRAIN support requires ordered contiguous original source groups")
        start = group[2]
    if start != len(data.rows):
        raise ValueError("TRAIN support groups must cover every prepared row exactly once")


def _source_features(data, name, start, stop, recipe):
    kind, identity, seed = _SOURCES[name]
    state = ActionHistoryState()
    rows = []
    for index in range(start, stop):
        row = data.rows[index]
        if (
            type(row) is not dict
            or row.get("role") != "train"
            or row.get("context") != pick_cube_training_context(seed)
            or row.get("episode_id") != identity
            or row.get("source_kind") != kind
            or type(row.get("row_index")) is not int
            or row["row_index"] != index
            or type(row.get("step")) is not int
            or row["step"] != index - start
            or row.get("stage") not in STAGES
        ):
            raise ValueError("TRAIN support row source, recorded stage or ordered identity differs")
        features = _vector(row.get("features"), 12)
        raw = _vector(row.get("source_observation"), 42)
        command_values = row.get("actual_command")
        if (
            command_values is None
            or type(command_values) not in (list, tuple)
            or any(type(value) is not float for value in command_values)
        ):
            raise ValueError("TRAIN support requires original nativefloat32 executed commands")
        command = _vector(command_values, 4)
        expected = action_history_observation(raw, state, recipe)
        if features != expected:
            raise ValueError("TRAIN support geometry or causal executed history differs")
        state = advance_action_history(state, command)
        rows.append((row["stage"], features))
    return rows


def _summary(features: list[tuple[float, ...]]) -> dict[str, Any]:
    grasp = [0, 0]
    closing = [0] * 7
    lower = list(features[0]) if features else []
    upper = lower.copy()
    for row in features:
        grasp[int(row[0])] += 1
        closing[round(row[11] * 6)] += 1
        for i, value in enumerate(row):
            lower[i] = min(lower[i], value)
            upper[i] = max(upper[i], value)
    return dict(
        rows=len(features),
        coordinate_ranges=(
            [dict(min=a, max=b) for a, b in zip(lower, upper, strict=True)] if features else None
        ),
        constant_coordinates=[i for i, (a, b) in enumerate(zip(lower, upper)) if a == b],
        unique_feature_vectors=len(set(features)),
        unique_previous_actions=len({row[7:11] for row in features}),
        grasp_counts=grasp,
        closing_count_counts=closing,
    )


def summarize_joint_readiness_support(data: JointReadinessData) -> dict[str, Any]:
    """Summarize original TRAIN geometry/history; never qualify or select data.

    Why explicit: diagnostics incur validation/projection work and should not
    silently alter the existing preparation or learning protocol. All five
    recorded stages appear per source, including absent stages. Coordinate
    envelopes describe marginal observations, not joint coverage/reachability.
    """
    if type(data) is not JointReadinessData:
        raise ValueError("TRAIN support requires exact already-prepared JointReadinessData")
    recipe = validate_taskspace_history_recipe(data.recipe)
    expected_spec = ContinuousActionSpec(12, _UNITS, (-1.0,) * 4, (1.0,) * 4)
    if (
        recipe.mode != "executed"
        or type(data.spec) is not ContinuousActionSpec
        or data.spec != expected_spec
    ):
        raise ValueError("TRAIN support requires the original executed TS12 normalized spec")
    _validate_groups(data)
    # Validate every row before producing summaries. Targets/clocks are never read.
    sources = {
        name: _source_features(data, name, start, stop, recipe) for name, start, stop in data.groups
    }
    all_features = [feature for rows in sources.values() for _, feature in rows]
    count = len(all_features)
    return dict(
        protocol="pick_cube_train_feature_support_v1",
        feature_names=list(FEATURE_NAMES),
        **{"global": _summary(all_features)},
        by_source={name: _summary([f for _, f in rows]) for name, rows in sources.items()},
        by_source_stage={
            name: {stage: _summary([f for s, f in rows if s == stage]) for stage in STAGES}
            for name, rows in sources.items()
        },
        work=dict(
            rows_validated=count,
            reprojected_rows=count,
            source_values_validated=42 * count,
            emitted_feature_values=12 * count,
            logical_feature_numeric_bytes=96 * count,
            history_advances=count,
            summary_row_visits=3 * count,
            summary_coordinate_visits=36 * count,
            scope="source values count distinct coordinates, not repeated validation passes; "
            "logical feature storage only; validation/selection/sets/Python containers/RSS additional; "
            "zero learner/RNG/label-query/clock/IO/native work",
        ),
        interpretation="marginal TRAIN feature support only; recorded stages are not inferred; "
        "closing count is an executed-command proxy, not contact; no joint coverage, "
        "reachability, qualification, release authority or reliability claim",
    )
