"""Diagnose saved TRAIN robot control without fitting or running a policy.

Accepted teacher and learned-rollout JSON are inputs, not newly qualified data.
Errors use only exactly identical observations; time-aligned teacher commands
are not ground truth on a diverged student trajectory. Frozen normalization and
support distances describe departure, not its cause or a corrective oracle.
No model, optimizer, RNG, simulator, privileged geometry or final/probe IO.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from scripts.push_cube_correction_inputs import load_push_correction_inputs
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    continuous_training_pairs,
    copy_continuous_episode,
)
from src.core.cube_taskspace_state import CUBE_TASKSPACE_DEFINITION, cube_taskspace_spec
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.observation_standardization import (
    ObservationStandardization,
    ScaleFlooredObservationStandardization,
    standardize_observations,
)
from src.infra.maniskill_continuous import ACTION_UNITS, CONTROL_MODE
from src.infra.maniskill_push_cube import PushCubeWork

PUSH_STUDY_SHA256 = "1f4d1a7018e522c737641ea371c1ccc98200dc57f03cfe1e84e8062027d5a6b4"


def _episode(value: dict[str, Any]) -> ContinuousEpisode:
    spec = value["spec"]
    return ContinuousEpisode(
        value["episode_id"],
        value["context"],
        value["role"],
        ContinuousActionSpec(
            spec["observation_dim"],
            tuple(spec["action_units"]),
            tuple(spec["lower_bounds"]),
            tuple(spec["upper_bounds"]),
        ),
        tuple(
            ContinuousTransition(
                row["step"],
                tuple(row["observation"]),
                tuple(row["action"]),
                row["reward"],
                tuple(row["next_observation"]),
                row["terminated"],
                row["truncated"],
            )
            for row in value["transitions"]
        ),
    )


def _admit(
    teacher: dict[str, Any], learned: dict[str, Any], *, state_observation: bool = False
) -> tuple[ContinuousEpisode, ContinuousEpisode, ObservationStandardization]:
    if type(state_observation) is not bool:
        raise ValueError("state_observation must be an exact bool")
    if type(teacher) is not dict or type(learned) is not dict:
        raise ValueError("diagnosis requires accepted saved JSON objects")
    if (
        teacher["qualified"] is not True
        or teacher["role"] != "train"
        or type(teacher["seed"]) is not int
        or teacher["seed"] != 17
    ):
        raise ValueError("diagnosis requires the accepted qualified TRAIN17 teacher")
    if learned["complete"] is not True or learned["candidate_unchanged"] is not True:
        raise ValueError("diagnosis requires a complete unchanged learned-rollout result")
    training, rollout = learned["training"], learned["rollout"]
    if state_observation:
        if (
            learned.get("observation_projection") is not None
            or rollout.get("observation_projection") is not None
            or learned.get("canonical_state_quaternions", False) is not False
            or "input_projection_work" in learned
        ):
            raise ValueError("raw state42 diagnosis cannot reinterpret projected learner features")
        for name, record in (("teacher", teacher), ("rollout", rollout)):
            for field in ("observation_recipe", "visual_work"):
                if field not in record or record[field] is not None:
                    raise ValueError(f"state42 diagnosis requires explicit {name} {field}=None")
        policy, source = training["policy"], training["source_episodes"]["warm"]
        if (
            teacher.get("task") != "PickCube-v1"
            or rollout.get("task") != "PickCube-v1"
            or teacher.get("controller") != CONTROL_MODE
            or teacher.get("context") != "pick-cube"
            or rollout.get("context") != "pick-cube"
            or policy.get("training_role") != "train"
            or type(policy.get("seed")) is not int
            or policy["seed"] != 17
            or source.get("context") != "pick-cube"
            or source.get("role") != "train"
        ):
            raise ValueError("state42 diagnosis requires the bound default TRAIN17 metadata")
    counters = (
        rollout["seed"],
        rollout["optimizer_updates"],
        rollout["source_updates"],
        rollout["source_training_rows"],
        training["work"]["optimizer_updates"],
        training["work"]["training_rows"],
    )
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("saved diagnosis seeds and costs require native nonnegative integers")
    if (
        training["complete"] is not True
        or rollout["complete"] is not True
        or rollout["role"] != "train"
        or rollout["seed"] != 17
        or rollout["controller"] != CONTROL_MODE
        or rollout["optimizer_updates"] != 0
        or rollout["frozen_state_exact"] is not True
        or rollout["original_owner_exact"] is not True
        or rollout["learner_policy"] != training["policy"]
        or rollout["source_updates"] != training["work"]["optimizer_updates"]
        or rollout["source_training_rows"] != training["work"]["training_rows"]
    ):
        raise ValueError("saved TRAIN policy, controller, frozen ownership or costs differ")
    warm, student = _episode(teacher["candidate_episode"]), _episode(rollout["episode"])
    if (
        warm.role != "train"
        or student.role != "train"
        or warm.context != student.context
        or warm.spec != student.spec
        or warm.spec.observation_dim != (42 if state_observation else 789)
        or (state_observation and warm.context != "pick-cube")
        or warm.spec.action_units != ACTION_UNITS
        or warm.spec.lower_bounds != (-1.0,) * 4
        or warm.spec.upper_bounds != (1.0,) * 4
        or len(warm.transitions) > 50
        or len(student.transitions) > 50
        or json.dumps(asdict(warm.spec), sort_keys=True)
        != json.dumps(training["policy"]["spec"], sort_keys=True)
        or training["source_episodes"]["warm"]["episode_id"] != warm.episode_id
    ):
        raise ValueError(
            "saved diagnosis requires matching bounded TRAIN state42 episodes"
            if state_observation
            else "saved diagnosis requires matching bounded TRAIN RGB789 episodes"
        )
    raw = training["policy"]["observation_standardization"]
    base_fields = {"means", "scales", "fitted_rows"}
    selected_fields = base_fields | {"scale_floors", "effective_scales"}
    transform: ObservationStandardization
    if type(raw) is not dict or set(raw) not in (base_fields, selected_fields):
        raise ValueError("saved normalization metadata has unsupported or missing fields")
    for name in ("means", "scales", *(["scale_floors"] if set(raw) == selected_fields else [])):
        if type(raw[name]) not in (list, tuple):
            raise ValueError(f"saved normalization {name} must be a native sequence")
    if set(raw) == selected_fields:
        if not state_observation:
            raise ValueError("saved scale-floor diagnosis requires the explicit state42 route")
        selected = ScaleFlooredObservationStandardization(
            tuple(raw["means"]),
            tuple(raw["scales"]),
            raw["fitted_rows"],
            tuple(raw["scale_floors"]),
        )
        effective = raw["effective_scales"]
        if (
            type(effective) not in (list, tuple)
            or len(effective) != len(selected.effective_scales)
            or any(type(value) is not float for value in effective)
            or tuple(effective) != selected.effective_scales
        ):
            raise ValueError("saved scale-floor effective scales differ from their declaration")
        transform = selected
    else:
        transform = ObservationStandardization(
            tuple(raw["means"]), tuple(raw["scales"]), raw["fitted_rows"]
        )
    if (
        transform.fitted_rows != len(warm.transitions)
        or len(transform.means) != warm.spec.observation_dim
    ):
        raise ValueError("diagnosis requires original warm-only fitted statistics")
    return warm, student, transform


def diagnose_control(
    teacher: dict[str, Any], learned: dict[str, Any], *, state_observation: bool = False
) -> dict[str, Any]:
    """Describe exact-input fit and support; explicit state mode uses all42 axes."""
    if type(state_observation) is not bool:
        raise ValueError("state_observation must be an exact bool")
    warm, student, transform = (
        _admit(teacher, learned, state_observation=True)
        if state_observation
        else _admit(teacher, learned)
    )
    return _describe_control(
        warm, student, transform, learned["rollout"], state_observation=state_observation
    )


def _describe_control(
    warm: ContinuousEpisode,
    student: ContinuousEpisode,
    transform: ObservationStandardization | None,
    rollout: dict[str, Any],
    *,
    state_observation: bool = False,
    push_taskspace: bool = False,
) -> dict[str, Any]:
    """Share exact-input/support arithmetic across explicitly admitted formats."""
    warm_pairs = continuous_training_pairs(warm)
    student_pairs = continuous_training_pairs(student)
    tx, ty = (np.asarray(value, dtype=np.float64) for value in warm_pairs)
    sx, sy = (np.asarray(value, dtype=np.float64) for value in student_pairs)
    if not np.array_equal(tx[0], sx[0]):
        raise ValueError("saved diagnosis reset observation differs from teacher")
    if push_taskspace:
        # D12 declares displacement/.1 and closing_count/6. A fitted
        # transform would invent a different distance from the serving inputs.
        tz, nz = tx.copy(), sx.copy()
    else:
        if transform is None:
            raise ValueError("saved diagnosis requires its frozen normalization")
        tz, nz = (standardize_observations(value, transform, warm.spec) for value in (tx, sx))
    # State mode is an explicit full-input diagnosis, never an inferred or
    # truncated RGB/public-prefix view of another observation contract.
    support_dimensions = 12 if push_taskspace else (42 if state_observation else 21)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = nz[:, None, :support_dimensions] - tz[None, :, :support_dimensions]
            distances = np.mean(delta**2, axis=2)
        if not np.isfinite(distances).all():
            raise FloatingPointError("nonfinite support distances")
    except FloatingPointError as overflow:
        raise ValueError("support distance arithmetic must remain finite") from overflow
    nearest = np.argmin(distances, axis=1)
    rms = np.sqrt(distances[np.arange(len(sx)), nearest])
    # Exact raw observation equality, not nearest/time labels, admits an error.
    matches = np.all(sx[:, None, :] == tx[None, :, :], axis=2)
    known = []
    for index in range(len(sx)):
        selected = np.flatnonzero(matches[index])
        if not len(selected):
            continue
        targets = ty[selected]
        consistent = bool(np.all(targets == targets[0]))
        row: dict[str, Any] = dict(
            student_step=index, teacher_steps=selected.tolist(), labels_consistent=consistent
        )
        if consistent:
            error = sy[index] - targets[0]
            row.update(
                action=sy[index].tolist(),
                teacher_action=targets[0].tolist(),
                action_error=error.tolist(),
                max_abs_action_error=float(np.max(np.abs(error))),
                desired_translation_delta_error_m=(0.1 * error[:3]).tolist(),
            )
        known.append(row)
    blocks = {}
    sections = (
        (("physical7", slice(0, 7)), ("history5", slice(7, 12)))
        if push_taskspace
        else (
            (("state42", slice(0, 42)),)
            if state_observation
            else (("public21", slice(0, 21)), ("rgb768", slice(21, None)))
        )
    )
    for name, section in sections:
        old, new = tx[:, section], sx[:, section]
        outside = np.sum((new < old.min(axis=0)) | (new > old.max(axis=0)), axis=1)
        blocks[name] = dict(
            outside_training_range_coordinates_by_step=outside.tolist(),
            max_abs_frozen_normalized_by_step=np.max(np.abs(nz[:, section]), axis=1).tolist(),
        )
    named_bytes = sum(
        a.nbytes for a in (tx, ty, sx, sy, tz, nz, delta, distances, nearest, rms, matches)
    )
    exact_comparison_bytes = len(sx) * len(tx) * warm.spec.observation_dim
    # Why: exact matching and range/absolute-value summaries allocate temporary
    # arrays too. This bounds selected array payload, not validation/JSON/RSS.
    temporary_bytes = exact_comparison_bytes + 2 * max(tx.nbytes, sx.nbytes) + delta.nbytes
    if named_bytes + temporary_bytes > 4 * 1024**2:
        raise ValueError("diagnostic selected numeric/boolean payload exceeds4MiB")
    return dict(
        complete=True,
        definition=(
            "saved_train_push_taskspace12_exact_input_fit_and_support_v1"
            if push_taskspace
            else (
                "saved_train_state42_exact_input_fit_and_support_v1"
                if state_observation
                else "saved_train_exact_input_fit_and_support_v1"
            )
        ),
        **(
            dict(reference_episode=warm.episode_id)
            if push_taskspace
            else dict(teacher_episode=warm.episode_id)
        ),
        student_episode=student.episode_id,
        rows=dict(teacher=len(tx), student=len(sx)),
        reset_observation_exact=True,
        exact_observation_matches=known,
        **(
            dict(
                nearest_taskspace12_reference_steps=nearest.tolist(),
                nearest_taskspace12_normalized_rms=rms.tolist(),
                first_nonzero_taskspace12_support_distance=next(
                    (i for i, x in enumerate(rms) if x > 0), None
                ),
            )
            if push_taskspace
            else dict(
                nearest_state42_teacher_steps=nearest.tolist(),
                nearest_state42_normalized_rms=rms.tolist(),
                first_nonzero_state42_support_distance=next(
                    (i for i, x in enumerate(rms) if x > 0), None
                ),
            )
            if state_observation
            else dict(
                nearest_public21_teacher_steps=nearest.tolist(),
                nearest_public21_normalized_rms=rms.tolist(),
                first_nonzero_prefix_support_distance=next(
                    (i for i, x in enumerate(rms) if x > 0), None
                ),
            )
        ),
        observation_blocks=blocks,
        recorded_outcome=dict(
            sparse_return=rollout["sparse_return"],
            any_success=rollout["any_success"],
            final_success=rollout["final_success"],
        ),
        work=dict(
            normalization_calls=0 if push_taskspace else 2,
            normalization_rows=0 if push_taskspace else len(tx) + len(sx),
            normalization_coordinates=(
                0 if push_taskspace else (len(tx) + len(sx)) * warm.spec.observation_dim
            ),
            **(
                dict(
                    identity_copy_calls=2,
                    identity_copy_rows=len(tx) + len(sx),
                    identity_copy_coordinates=(len(tx) + len(sx)) * 12,
                )
                if push_taskspace
                else {}
            ),
            **{
                (
                    "taskspace12_distance_coordinates"
                    if push_taskspace
                    else (
                        "state42_distance_coordinates"
                        if state_observation
                        else "prefix_distance_coordinates"
                    )
                ): len(tx) * len(sx) * support_dimensions
            },
            exact_match_coordinate_comparisons=exact_comparison_bytes,
            selected_numeric_boolean_bytes=named_bytes,
            selected_temporary_allowance_bytes=temporary_bytes,
            selected_working_payload_bound_bytes=named_bytes + temporary_bytes,
            model_initializations=0,
            fits=0,
            optimizer_updates=0,
            forwards=0,
            gradients=0,
            rng_draws=0,
            simulator_controls=0,
        ),
        qualifications=[
            "accepted saved TRAIN inputs; not third-party teacher/model authentication",
            "only exact raw matched observations have supervised action errors",
            "nearest/time correspondence is not counterfactual ground truth",
            "desired controller delta error is not achieved physical displacement",
            (
                "identity physical input scaling, no fitted statistics or clipping; descriptive not causal"
                if push_taskspace
                else "frozen normalization, no refit or clipping; distances are descriptive not causal"
            ),
            "first nonzero distance is departure from stored points, not a calibrated OOD threshold",
            "selected payload excludes validation copies, Python/JSON containers and process RSS",
        ],
    )


def _same_json(value: Any, expected: Any) -> bool:
    """Keep native bool/int distinctions when binding saved scalar receipts."""
    return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False
    )


def _push_episode(
    episode: ContinuousEpisode, identity: str, *, rows: int | None = None, executed: bool = True
) -> ContinuousEpisode:
    owned = copy_continuous_episode(episode)
    if (
        owned.episode_id != identity
        or owned.context != "push-cube-train-17"
        or owned.role != "train"
        or owned.spec != cube_taskspace_spec()
        or not 1 <= len(owned.transitions) <= 50
        or (rows is not None and len(owned.transitions) != rows)
    ):
        raise ValueError("Push diagnosis requires its bounded original TRAIN17 D12 episodes")
    history = ActionHistoryState()
    for row in owned.transitions:
        for observation in (row.observation, row.next_observation):
            if observation[0] not in (0.0, 1.0) or observation[11] not in tuple(
                n / 6 for n in range(7)
            ):
                raise ValueError("Push observed grasp/closing history differs from D12")
            count = next(n for n in range(7) if observation[11] == n / 6)
            ActionHistoryState(tuple(observation[7:11]), count)
        if executed:
            if row.observation[7:] != (*history.previous_action, history.closing_count / 6):
                raise ValueError("Push observation history must contain prior executed commands")
            history = advance_action_history(history, row.action)
            if row.next_observation[7:] != (*history.previous_action, history.closing_count / 6):
                raise ValueError("Push next history must contain the actual executed command")
    # Correction targets are supervision on the original student's observations;
    # advancing those histories using targets would fabricate an expert rollout.
    return owned


def _push_rollout(study: dict[str, Any], student: ContinuousEpisode) -> dict[str, Any]:
    rollout = study["native_push"]
    count = len(student.transitions)
    expected_metadata = dict(
        task_id="PushCube-v1",
        role="train",
        seed=17,
        context=student.context,
        episode_id=student.episode_id,
        horizon=50,
        controller=dict(
            frame="root_translation", mode=CONTROL_MODE, no_target=True, normalized_delta=True
        ),
        recipe=dict(
            definition=CUBE_TASKSPACE_DEFINITION,
            displacement_frame="world",
            displacement_scale=0.1,
            feature_dim=12,
            history="executed",
            closing_limit=6,
        ),
        reward=dict(
            source="info.success", mode="sparse", rule="xy_distance_lt_0.1_and_object_z_lt_0.025"
        ),
    )
    metadata = rollout.get("metadata", {})
    if type(metadata) is not dict or not _same_json(
        {name: metadata.get(name) for name in expected_metadata}, expected_metadata
    ):
        raise ValueError("Push task/controller/observation/reset metadata differs")
    if (
        any(rollout.get(name) is not True for name in ("complete", "source_unchanged"))
        or any(
            rollout.get(name) is not False
            for name in ("unreturned_native_work_unknown", "unreturned_prediction_possible")
        )
        or rollout.get("error") is not None
    ):
        raise ValueError("Push diagnosis requires complete frozen known native evidence")
    expected_counts = dict(
        task_id="PushCube-v1",
        role="train",
        seed=17,
        control_attempts=count,
        control_returns=count,
        prediction_attempts=count,
        prediction_returns=count,
    )
    if not _same_json({name: rollout.get(name) for name in expected_counts}, expected_counts):
        raise ValueError("Push control/prediction receipts differ from its episode")
    expected_work = asdict(
        PushCubeWork(
            1,
            1,
            1,
            1,
            count,
            count,
            5 * count,
            0,
            count + 1,
            count + 1,
            count + 1,
            count + 1,
            2 * (count + 1),
            2 * (count + 1),
            1,
            1,
        )
    )
    if not _same_json(rollout.get("work"), expected_work) or not _same_json(
        rollout.get("transitions"), [asdict(row) for row in student.transitions]
    ):
        raise ValueError("Push paid work or copied transition evidence differs")
    outputs = rollout.get("prediction_outputs")
    if type(outputs) is not list or len(outputs) != count:
        raise ValueError("Push prediction captures are incomplete")
    for row, output in zip(student.transitions, outputs, strict=True):
        if (
            type(output) is not dict
            or output.get("dtype") != "float64"
            or not _same_json(output.get("shape"), [1, 4])
            or type(output.get("values_omitted")) is not int
            or output["values_omitted"] != 0
            or type(output.get("values")) is not list
            or len(output["values"]) != 4
            or any(type(value) is not str for value in output["values"])
        ):
            raise ValueError("Push prediction shape/dtype/capture differs")
        values = tuple(float(value) for value in output["values"])
        if any(not math.isfinite(value) or not -1 <= value <= 1 for value in values):
            raise ValueError("Push predictions require finite normalized axes")
        executed = tuple(struct.unpack("=f", struct.pack("=f", value))[0] for value in values)
        if executed != row.action:
            raise ValueError("Push prediction capture differs from executed float32 command")
    rewards = [row.reward for row in student.transitions]
    last = student.transitions[-1]
    scores = dict(
        sparse_return=float(sum(rewards)), any_success=any(rewards), final_success=bool(rewards[-1])
    )
    if (
        any(reward not in (0.0, 1.0) for reward in rewards)
        or not _same_json({name: rollout.get(name) for name in scores}, scores)
        or last.terminated != bool(rewards[-1])
        or last.truncated == last.terminated
    ):
        raise ValueError("Push sparse scores/end flags differ from recorded rewards")
    return rollout


def diagnose_push_control(
    study: dict[str, Any], correction_packet: ContinuousEpisode
) -> dict[str, Any]:
    """Describe accepted saved Push traces; never label or evaluate a new state."""
    if (
        type(study) is not dict
        or study.get("protocol") != "push_cube_prefix_group_balanced_correction_train_v1"
    ):
        raise ValueError("Push diagnosis requires the accepted balanced study format")
    if (
        study.get("complete") is not True
        or study.get("source_unchanged") is not True
        or study.get("unreturned_stage_work_unknown") is not False
        or study.get("error") is not None
    ):
        raise ValueError("Push diagnosis requires complete unchanged preceding study evidence")
    continuation = study.get("correction_continuation", {})
    if (
        type(continuation) is not dict
        or any(
            continuation.get(name) is not True
            for name in ("complete", "source_exact", "actor_exact")
        )
        or continuation.get("unreturned_training_work_unknown") is not False
        or not _same_json(continuation.get("initial_counts"), dict(updates=112, training_rows=9832))
        or not _same_json(continuation.get("final_counts"), dict(updates=120, training_rows=11016))
    ):
        raise ValueError(
            "Push diagnosis requires the original validated weighted continuation counts"
        )
    fit = study.get("correction_fit_protocol", {})
    expected_fit = dict(initial_counts=[112, 9832], final_counts=[120, 11016], rows=148, updates=8)
    if type(fit) is not dict or not _same_json(
        {name: fit.get(name) for name in expected_fit}, expected_fit
    ):
        raise ValueError("Push diagnosis requires the original fixed corrective dose")
    expert = _push_episode(_episode(study["training"]), "push-cube-teacher-17", rows=18)
    packet = _push_episode(
        correction_packet, "push-cube-correction-targets-17", rows=50, executed=False
    )
    student = _push_episode(_episode(study["native_push"]["episode"]), "push-frozen-mixed120")
    rollout = _push_rollout(study, student)
    if any(
        reference.transitions[0].observation != student.transitions[0].observation
        for reference in (expert, packet)
    ):
        raise ValueError("Push diagnosis requires the same original full D12 reset")
    comparisons = {
        name: _describe_control(reference, student, None, rollout, push_taskspace=True)
        for name, reference in (("expert", expert), ("corrections", packet))
    }
    additive = (
        "normalization_calls",
        "normalization_rows",
        "normalization_coordinates",
        "identity_copy_calls",
        "identity_copy_rows",
        "identity_copy_coordinates",
        "taskspace12_distance_coordinates",
        "exact_match_coordinate_comparisons",
        "model_initializations",
        "fits",
        "optimizer_updates",
        "forwards",
        "gradients",
        "rng_draws",
        "simulator_controls",
    )
    work = {name: sum(item["work"][name] for item in comparisons.values()) for name in additive}
    work["comparison_calls"] = 2
    work["selected_peak_payload_bound_bytes"] = max(
        item["work"]["selected_working_payload_bound_bytes"] for item in comparisons.values()
    )
    return dict(
        complete=True,
        definition="saved_train_push_taskspace12_control_gap_v1",
        source_counts=dict(continuation["final_counts"]),
        comparisons=comparisons,
        work=work,
        input_scaling="identity physical D12; no fitted statistics",
        qualifications=[
            "accepted saved TRAIN inputs; CLI binds original report bytes and released packet",
            "correction targets remain provisional labels on original112 student observations",
            "nearest/time-aligned reference actions at different inputs are not counterfactual labels",
            "coverage and action discrepancies are descriptive, not a causal diagnosis or native success",
        ],
    )


def _read_push_study(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2 or hashlib.sha256(raw).hexdigest() != PUSH_STUDY_SHA256:
        raise ValueError("accepted balanced Push study binding differs")
    return json.loads(raw)


def _same_push_receipt(receipt: dict[str, Any], original: dict[str, Any]) -> bool:
    # The same original file has different path separators on Windows and WSL.
    # Resolve only that representation; every other receipt value stays exact.
    if type(receipt.get("path")) is not str or type(original.get("path")) is not str:
        return False
    left, right = dict(receipt), dict(original)
    left["path"], right["path"] = (
        str(Path(left["path"]).resolve()),
        str(Path(right["path"]).resolve()),
    )
    return _same_json(left, right)


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        raise ValueError("saved diagnostic input exceeds8MiB")
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--state-observation",
        action="store_true",
        help="diagnose explicit native state42 inputs using all42 support coordinates",
    )
    modes.add_argument(
        "--push-taskspace",
        action="store_true",
        help="compare accepted saved Push D12 study with its released correction/expert records",
    )
    args = parser.parse_args()
    options: dict[str, Any] = {}
    if args.state_observation:
        options["state_observation"] = True
    if args.push_taskspace:
        study = _read_push_study(args.learned)
        packet, receipt = load_push_correction_inputs(args.teacher)
        if not _same_push_receipt(receipt, study["correction_fit_protocol"]["input_receipt"]):
            raise ValueError("Push diagnosis correction receipt differs from the original study")
        result = diagnose_push_control(study, packet)
    else:
        result = diagnose_control(_read(args.teacher), _read(args.learned), **options)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if len(encoded.encode("utf8")) > 65536:
        raise ValueError("saved diagnostic output exceeds64KiB")
    print(encoded)


if __name__ == "__main__":
    main()
