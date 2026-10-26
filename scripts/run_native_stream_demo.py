"""Run with python -m scripts.run_native_stream_demo; write one JSON report."""

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.adapters.native_stream_demo import run_demo, run_coupled_demo, run_matched_demo
from src.adapters.promotion_rejection_demo import run_rejection_demo
from src.adapters.continuous_stream_demo import run_continuous_demo
from src.adapters.continuous_learnability_demo import (
    run_continuous_learnability_demo,
    validate_phase_updates,
)
from src.adapters.continuous_learnability_modes import FAMILIES
from src.adapters.continuous_probe import ContinuousProbePlan, copy_probe_plan
from src.infra.continuous_probe_json import MAX_PROBE_FILE_BYTES, read_continuous_probe_episode
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.pick_cube_camera_context import (
    PickCubeCameraContext,
    camera_context_recipe,
    pick_cube_camera_context,
)
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_public_vectors import (
    public_vector_projection,
    public_vector_training_episode,
)
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    pick_cube_mass_context,
    validate_pick_cube_mass_context,
)
from src.core.pick_cube_lighting_context import (
    PickCubeLightingContext,
    pick_cube_lighting_context,
    validate_pick_cube_lighting_context,
)
from src.infra.maniskill_continuous import (
    ACTION_UNITS,
    CONTROL_MODE,
    HORIZON,
    TASK_ID,
    ManiSkillLightingReadback,
    ManiSkillLightingWork,
    validate_lighting_visual_work,
    ManiSkillMassReadback,
    ManiSkillMassWork,
    ManiSkillWork,
    ManiSkillVisualWork,
)
from src.infra.maniskill_task_teacher import TaskTeacherResult, ObservedGraspTaskTeacherResult


def _check_visual_recipe(
    value: object, *, camera_context: PickCubeCameraContext | None = None
) -> None:
    """Require the complete JSON form; never backfill missing recipe fields."""
    expected = asdict(
        pick_cube_rgb_recipe() if camera_context is None else camera_context_recipe(camera_context)
    )
    if type(value) is not dict or value.keys() != expected.keys():
        raise ValueError("visual teacher requires the complete frozen RGB recipe")
    for name, target in expected.items():
        actual = value[name]
        if type(target) is tuple:
            valid = (
                type(actual) is list
                and len(actual) == len(target)
                and all(type(a) is type(b) and a == b for a, b in zip(actual, target, strict=True))
            )
        else:
            valid = type(actual) is type(target) and actual == target
        if not valid:
            raise ValueError(f"visual teacher frozen RGB recipe differs at {name}")


def _unique_visual_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse ambiguous visual declarations before a duplicate can overwrite one."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"visual teacher contains duplicate field {name}")
        result[name] = value
    return result


def _teacher_episode(value, context, identity, *, visual_observation):
    """Decode the shared bounded episode view; callers bind receipt provenance."""
    if value["role"] != "train" or value["context"] != context or value["episode_id"] != identity:
        raise ValueError("teacher episode role/context/identity differs from declared TRAIN seed")
    spec = value["spec"]
    dimension = pick_cube_rgb_recipe().feature_dim if visual_observation else 42
    expected = ContinuousActionSpec(dimension, ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    actual = ContinuousActionSpec(
        spec["observation_dim"],
        tuple(spec["action_units"]),
        tuple(spec["lower_bounds"]),
        tuple(spec["upper_bounds"]),
    )
    if actual != expected or not 1 <= len(value["transitions"]) <= HORIZON:
        raise ValueError(
            f"teacher requires declared {dimension}D PickCube units/bounds and at most50 steps"
        )
    transitions = tuple(
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
    )
    # The first21 public vector coordinates retain physical units; only
    # the frozen pooled uint8 RGB suffix has the declared unit interval.
    if visual_observation and any(
        not 0.0 <= pixel <= 1.0
        for row in transitions
        for observation in (row.observation, row.next_observation)
        for pixel in observation[21:]
    ):
        raise ValueError("visual teacher RGB features must lie in range [0,1]")
    if transitions[-1].reward != 1 or any(row.reward not in (0, 1) for row in transitions):
        raise ValueError("qualified PickCube teacher requires sparse0/1 rewards and final success")
    return ContinuousEpisode(value["episode_id"], context, "train", actual, transitions)


def _teacher_payload(path, *, visual_observation, strict=False, payload=None):
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    limit = (4 if visual_observation else 1) * 1024**2
    if payload is None:
        with path.open("rb") as handle:
            payload = handle.read(limit + 1)
    elif type(payload) is not bytes:
        raise ValueError("teacher receipt payload must be exact bytes")
    if len(payload) > limit:
        raise ValueError(f"teacher receipt exceeds{limit // 1024**2}MiB")
    raw = (
        json.loads(payload, object_pairs_hook=_unique_visual_fields)
        if visual_observation or strict
        else json.loads(payload)
    )
    if type(raw) is not dict:
        raise ValueError("teacher receipt must be a JSON object")
    if strict:
        json.dumps(raw, allow_nan=False)
    return raw, len(payload)


def _read_teacher(path: Path, seed: int, *, visual_observation: bool = False) -> ContinuousEpisode:
    """Read a default TRAIN receipt; context interventions require explicit routes."""
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    context = pick_cube_training_context(seed)
    try:
        raw, _ = _teacher_payload(path, visual_observation=visual_observation)
        if "observed_grasp" in raw:
            raise ValueError("observed-grasp teacher requires explicit readiness selection")
        if raw.get("teacher_definition") is not None:
            raise ValueError("readiness teacher requires explicit --readiness-teachers selection")
        _refuse_lighting_markers(raw)
        if any(raw.get(key) is not None for key in ("mass_context", "mass_readback", "mass_work")):
            raise ValueError("mass teacher requires explicit --mass-drift selection")
        if any(
            raw.get(key) is not None
            for key in ("guard_seed", "labels_arrived_at", "camera_context")
        ):
            raise ValueError("default TRAIN teacher cannot carry guard or camera metadata")
        if (
            raw["qualified"] is not True
            or raw["final_success"] is not True
            or raw["error"] is not None
            or raw["role"] != "train"
            or type(raw["seed"]) is not int
            or raw["seed"] != seed
            or raw["task"] != TASK_ID
            or raw["controller"] != CONTROL_MODE
            or raw.get("context", context if seed == 17 and not visual_observation else None)
            != context
        ):
            raise ValueError(
                "expected qualified successful TRAIN teacher with declared reset binding"
            )
        if visual_observation:
            _check_visual_recipe(raw.get("observation_recipe"))
        elif raw.get("observation_recipe") is not None:
            raise ValueError("state teacher cannot declare a visual recipe")
        return _teacher_episode(
            raw["candidate_episode"],
            context,
            f"pick-cube-task-teacher-{seed}",
            visual_observation=visual_observation,
        )
    except (KeyError, TypeError, AttributeError, UnicodeError) as error:
        raise ValueError("malformed qualified TRAIN teacher receipt") from error


def _observed_teacher_json_feedback(raw) -> tuple[bool, ...]:
    """Check original serialized feedback/state flags before episode ownership."""
    if raw["observation_recipe"] is not None or raw["visual_work"] is not None:
        raise ValueError("observed-grasp teacher requires explicit nonvisual receipt fields")
    value = raw["candidate_episode"]
    if type(value) is not dict or type(value["spec"]) is not dict:
        raise ValueError("observed-grasp teacher requires its original episode/spec object")
    dimension = value["spec"]["observation_dim"]
    rows, feedback = value["transitions"], raw["observed_grasp"]
    if (
        type(dimension) is not int
        or dimension != 42
        or type(rows) is not list
        or not 1 <= len(rows) <= HORIZON
        or type(feedback) is not list
        or len(feedback) != len(rows)
        or any(type(flag) is not bool for flag in feedback)
    ):
        raise ValueError("observed-grasp teacher requires raw42 and one original bool per row")
    for flag, row in zip(feedback, rows, strict=True):
        if type(row) is not dict:
            raise ValueError("observed-grasp teacher requires original transition objects")
        for key in ("observation", "next_observation"):
            observation = row[key]
            if (
                type(observation) is not list
                or len(observation) != 42
                or type(observation[18]) is not float
                or observation[18] not in (0.0, 1.0)
            ):
                raise ValueError("serialized official current/next grasp flag must be float0/1")
        if flag is not (row["observation"][18] == 1.0):
            raise ValueError("serialized feedback differs from the current official grasp flag")
    return tuple(feedback)


def _read_readiness_teacher(
    path: Path,
    seed: int,
    *,
    visual_observation: bool = False,
    observed_grasp: bool = False,
    _payload: bytes | None = None,
) -> tuple[ContinuousEpisode, dict[str, Any]]:
    """Admit a distinct readiness TRAIN receipt through existing qualification.

    Why: successful readiness demonstrations have different identity and phase
    semantics. Preserve them rather than relabeling them as fixed-time teachers.
    Retained acquisition work describes the saved collection, not new simulator IO.
    """
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    if type(observed_grasp) is not bool:
        raise ValueError("observed_grasp must be an exact bool")
    if observed_grasp and visual_observation:
        raise ValueError("observed-grasp readiness teacher requires raw42 state observations")
    context = pick_cube_training_context(seed)
    definition = OBSERVED_GRASP_DEFINITION if observed_grasp else READINESS_DEFINITION
    identity = (
        f"pick-cube-observed-grasp-teacher-{seed}"
        if observed_grasp
        else f"pick-cube-readiness-teacher-{seed}"
    )
    try:
        # Why: receipt intake can qualify the exact bytes it already owns and
        # records, without a second path read observing a different file.
        payload_options = {} if _payload is None else dict(payload=_payload)
        raw, size = _teacher_payload(
            path, visual_observation=visual_observation, strict=True, **payload_options
        )
        _refuse_lighting_markers(raw)
        if not observed_grasp and "observed_grasp" in raw:
            raise ValueError("observed-grasp teacher requires explicit selected admission")
        if (
            raw["qualified"] is not True
            or raw["observed_success"] is not True
            or raw["final_success"] is not True
            or raw["error"] is not None
            or raw["traceback"] is not None
            or type(raw["seed"]) is not int
            or raw["seed"] != seed
            or raw["role"] != "train"
            or raw["task"] != TASK_ID
            or raw["controller"] != CONTROL_MODE
            or raw["context"] != context
            or raw["teacher_definition"] != definition
            or type(raw["stages"]) is not list
            or any(type(stage) is not str for stage in raw["stages"])
        ):
            raise ValueError("expected qualified geometry-readiness TRAIN teacher")
        if any(
            raw.get(key) is not None
            for key in (
                "guard_seed",
                "labels_arrived_at",
                "camera_context",
                "mass_context",
                "mass_readback",
                "mass_work",
                "readiness_request_error",
                "correction_provenance",
                "provenance",
            )
        ):
            raise ValueError("readiness teacher cannot mix guard, context or correction metadata")
        if "optimizer_updates" in raw and (
            type(raw["optimizer_updates"]) is not int or raw["optimizer_updates"] != 0
        ):
            raise ValueError("readiness teacher optimizer_updates must be exact zero when present")
        if visual_observation:
            _check_visual_recipe(raw["observation_recipe"])
        elif raw.get("observation_recipe") is not None:
            raise ValueError("readiness state teacher cannot declare a visual recipe")
        feedback_options: dict[str, Any] = {}
        if observed_grasp:
            feedback = _observed_teacher_json_feedback(raw)
            feedback_options["observed_grasp"] = feedback
        episode = _teacher_episode(
            raw["candidate_episode"],
            context,
            identity,
            visual_observation=visual_observation,
        )
        result_type = ObservedGraspTaskTeacherResult if observed_grasp else TaskTeacherResult
        result = result_type(
            episode,
            raw["observed_success"],
            raw["final_success"],
            raw["creation_attempts"],
            raw["geometry_attempts"],
            raw["geometry_returns"],
            raw["teacher_actions"],
            tuple(raw["stages"]),
            ManiSkillWork(**raw["work"]),
            raw["error"],
            raw["traceback"],
            observation_recipe=pick_cube_rgb_recipe() if visual_observation else None,
            visual_work=ManiSkillVisualWork(**raw["visual_work"])
            if raw.get("visual_work") is not None
            else None,
            teacher_definition=raw["teacher_definition"],
            **feedback_options,
        )
        released = result.training_episode()
        assert result.work is not None
        metadata = dict(
            mode="observed_grasp_readiness" if observed_grasp else "geometry_readiness",
            teacher_definition=result.teacher_definition,
            seed=seed,
            context=context,
            episode_id=released.episode_id,
            stages=list(result.stages),
            teacher_work=asdict(result.work),
            teacher_visual_work=asdict(result.visual_work) if result.visual_work else None,
            teacher_creation_attempts=result.creation_attempts,
            teacher_geometry_attempts=result.geometry_attempts,
            teacher_geometry_returns=result.geometry_returns,
            teacher_actions=result.teacher_actions,
            source_file_bytes=size,
            per_file_byte_limit=(4 if visual_observation else 1) * 1024**2,
            acquisition_scope="retained_teacher_work_not_new_simulator_work",
        )
        if observed_grasp:
            metadata["observed_grasp"] = list(feedback_options["observed_grasp"])
        return released, metadata
    except (KeyError, TypeError, AttributeError, UnicodeError, RecursionError) as error:
        raise ValueError("malformed qualified readiness TRAIN teacher receipt") from error


def _mass_visual_work(value, rows, *, visual_observation):
    """Validate retained teacher encoding costs; this route never exports frames."""
    if not visual_observation:
        if value is not None:
            raise ValueError("state mass teacher cannot declare visual work")
        return None
    work = ManiSkillVisualWork(**value)
    encodings = rows + 1
    expected = dict(
        observation_attempts=encodings,
        observation_returns=encodings,
        input_frame_bytes_upper_bound=encodings * 49152,
        feature_values_upper_bound=encodings * 789,
        returned_frame_bytes=encodings * 49152,
        returned_feature_values=encodings * 789,
        returned_feature_numeric_bytes=encodings * 6312,
        current_owned_frame_bytes=49152,
        frame_export_copies=0,
        frame_export_bytes=0,
    )
    actual = asdict(work)
    if any(type(actual[key]) is not int or actual[key] != count for key, count in expected.items()):
        raise ValueError("mass visual work differs from the complete teacher lifecycle")
    return work


def _read_mass_teacher(path: Path, *, visual_observation: bool = False):
    """Release owned fixed-mass TRAIN pairs and original acquisition metadata.

    Why: reusing TaskTeacherResult qualification keeps serialized input subject
    to the same mass identity, lifecycle and label-release rules as live teachers.
    Nothing here creates a simulator, learner, fitted transform or training call.
    """
    try:
        raw, size = _teacher_payload(path, visual_observation=visual_observation, strict=True)
        _refuse_lighting_markers(raw)
        mass = _mass_context(raw["mass_context"])
        if (
            raw["qualified"] is not True
            or raw["observed_success"] is not True
            or raw["final_success"] is not True
            or raw["error"] is not None
            or raw["traceback"] is not None
            or type(raw["seed"]) is not int
            or raw["seed"] != mass.seed
            or raw["role"] != "train"
            or raw["task"] != TASK_ID
            or raw["controller"] != CONTROL_MODE
            or raw["context"] != mass.context
        ):
            raise ValueError("expected qualified fixed mass TRAIN17 teacher")
        if visual_observation:
            _check_visual_recipe(raw["observation_recipe"])
        elif raw.get("observation_recipe") is not None:
            raise ValueError("state mass teacher cannot declare a visual recipe")
        episode = _teacher_episode(
            raw["candidate_episode"],
            mass.context,
            f"{mass.context}-teacher",
            visual_observation=visual_observation,
        )
        readback = raw["mass_readback"]
        if type(readback) is not dict or set(readback) != {
            "context",
            "source_mass_kg",
            "applied_mass_kg",
        }:
            raise ValueError("mass readback must retain its complete original fields")
        original = ManiSkillMassReadback(
            _mass_context(readback["context"]),
            readback["source_mass_kg"],
            readback["applied_mass_kg"],
        )
        mass_work = ManiSkillMassWork(**raw["mass_work"])
        work = ManiSkillWork(**raw["work"])
        visual_work = _mass_visual_work(
            raw.get("visual_work"), len(episode.transitions), visual_observation=visual_observation
        )
        if type(raw["stages"]) is not list or any(
            type(stage) is not str for stage in raw["stages"]
        ):
            raise ValueError("mass teacher stages must retain their JSON string-list shape")
        result = TaskTeacherResult(
            episode,
            raw["observed_success"],
            raw["final_success"],
            raw["creation_attempts"],
            raw["geometry_attempts"],
            raw["geometry_returns"],
            raw["teacher_actions"],
            tuple(raw["stages"]),
            work,
            raw["error"],
            raw["traceback"],
            observation_recipe=pick_cube_rgb_recipe() if visual_observation else None,
            visual_work=visual_work,
            guard_seed=raw.get("guard_seed"),
            labels_arrived_at=raw.get("labels_arrived_at"),
            camera_context=raw.get("camera_context"),
            mass_context=mass,
            mass_readback=original,
            mass_work=mass_work,
        )
        released = result.training_episode()
        metadata = dict(
            mode="mass_drift",
            warm_seed=17,
            new_seed=17,
            warm_context=pick_cube_training_context(17),
            new_context=mass.context,
            mass_context=asdict(mass),
            mass_readback=asdict(original),
            mass_work=asdict(mass_work),
            teacher_work=asdict(work),
            teacher_creation_attempts=result.creation_attempts,
            teacher_geometry_attempts=result.geometry_attempts,
            teacher_geometry_returns=result.geometry_returns,
            teacher_actions=result.teacher_actions,
            source_file_bytes=size,
            per_file_byte_limit=(4 if visual_observation else 1) * 1024**2,
            acquisition_scope="retained_teacher_work_not_new_simulator_work",
        )
        if visual_work is not None:
            metadata["teacher_visual_work"] = asdict(visual_work)
        return released, metadata
    except (KeyError, TypeError, AttributeError, UnicodeError, RecursionError) as error:
        raise ValueError("malformed qualified mass TRAIN teacher receipt") from error


def _refuse_lighting_markers(raw):
    """Keep default and mass receipts from silently discarding light metadata."""
    if any(
        raw.get(key) is not None
        for key in ("lighting_context", "lighting_readback", "lighting_work")
    ):
        raise ValueError("lighting teacher requires explicit --lighting-drift selection")


def _require_fixed_context_fields(value, expected, name):
    # Dataclass defaults serve Python callers, not missing serialized evidence.
    if (
        type(value) is not dict
        or value.keys() != expected.keys()
        or any(
            type(value[key]) is not type(target) or value[key] != target
            for key, target in expected.items()
        )
    ):
        raise ValueError(f"{name} context requires its complete original typed fields")


def _lighting_context(value):
    _require_fixed_context_fields(value, asdict(pick_cube_lighting_context()), "lighting")
    return validate_pick_cube_lighting_context(PickCubeLightingContext(**value))


def _mass_context(value):
    _require_fixed_context_fields(value, asdict(pick_cube_mass_context()), "mass")
    return validate_pick_cube_mass_context(PickCubeMassContext(**value))


def _lighting_readback(value):
    if type(value) is not dict or set(value) != {
        "context",
        "source_ambient_rgb",
        "applied_ambient_rgb",
    }:
        raise ValueError("lighting readback requires its complete original fields")
    for key in ("source_ambient_rgb", "applied_ambient_rgb"):
        if type(value[key]) is not list or len(value[key]) != 3:
            raise ValueError("lighting readback requires original JSON RGB lists")
    return ManiSkillLightingReadback(
        _lighting_context(value["context"]),
        tuple(value["source_ambient_rgb"]),
        tuple(value["applied_ambient_rgb"]),
    )


def _check_lighting_teacher_binding(raw, context):
    if (
        raw["qualified"] is not True
        or raw["observed_success"] is not True
        or raw["final_success"] is not True
        or raw["error"] is not None
        or raw["traceback"] is not None
        or type(raw["seed"]) is not int
        or raw["seed"] != context.seed
        or raw["role"] != "train"
        or raw["task"] != TASK_ID
        or raw["controller"] != CONTROL_MODE
        or raw["context"] != context.context
    ):
        raise ValueError("expected qualified fixed lighting TRAIN17 teacher")
    if any(
        raw.get(key) is not None
        for key in (
            "guard_seed",
            "labels_arrived_at",
            "camera_context",
            "mass_context",
            "mass_readback",
            "mass_work",
        )
    ):
        raise ValueError("lighting TRAIN teacher cannot mix guard, camera or mass metadata")
    # Public collector results have no optimizer field; never invent one in
    # historical receipts. Explicit claims must still describe zero SGD work.
    if "optimizer_updates" in raw and (
        type(raw["optimizer_updates"]) is not int or raw["optimizer_updates"] != 0
    ):
        raise ValueError("lighting teacher optimizer_updates must be exact zero when present")
    _check_visual_recipe(raw["observation_recipe"])
    if type(raw["stages"]) is not list or any(type(x) is not str for x in raw["stages"]):
        raise ValueError("lighting stages must retain their JSON string-list shape")


def _lighting_teacher_result(raw):
    context = _lighting_context(raw["lighting_context"])
    _check_lighting_teacher_binding(raw, context)
    episode = _teacher_episode(
        raw["candidate_episode"],
        context.context,
        f"{context.context}-teacher",
        visual_observation=True,
    )
    visual = validate_lighting_visual_work(
        ManiSkillVisualWork(**raw["visual_work"]), steps=len(episode.transitions)
    )
    return TaskTeacherResult(
        episode,
        raw["observed_success"],
        raw["final_success"],
        raw["creation_attempts"],
        raw["geometry_attempts"],
        raw["geometry_returns"],
        raw["teacher_actions"],
        tuple(raw["stages"]),
        ManiSkillWork(**raw["work"]),
        raw["error"],
        raw["traceback"],
        observation_recipe=pick_cube_rgb_recipe(),
        visual_work=visual,
        lighting_context=context,
        lighting_readback=_lighting_readback(raw["lighting_readback"]),
        lighting_work=ManiSkillLightingWork(**raw["lighting_work"]),
    )


def _lighting_teacher_metadata(result, size):
    context, original, work = (
        result.lighting_context,
        result.lighting_readback,
        result.lighting_work,
    )
    if context is None or original is None or work is None or result.visual_work is None:
        raise ValueError("qualified lighting teacher requires complete retained acquisition work")
    return dict(
        mode="lighting_drift",
        warm_seed=17,
        new_seed=17,
        warm_context=pick_cube_training_context(17),
        new_context=context.context,
        lighting_context=asdict(context),
        lighting_readback=asdict(original),
        lighting_work=asdict(work),
        teacher_work=asdict(result.work),
        teacher_visual_work=asdict(result.visual_work),
        teacher_creation_attempts=result.creation_attempts,
        teacher_geometry_attempts=result.geometry_attempts,
        teacher_geometry_returns=result.geometry_returns,
        teacher_actions=result.teacher_actions,
        source_file_bytes=size,
        per_file_byte_limit=4 * 1024**2,
        acquisition_scope="retained_teacher_work_not_new_simulator_work",
    )


def _read_lighting_teacher(path: Path, *, visual_observation: bool = False):
    """Release fixed lighting TRAIN pairs and paid work from one bounded payload.

    Why: reuse live teacher qualification and the existing stream, without
    refitting, opening a simulator or relabelling this TRAIN17 context as seed19.
    """
    if visual_observation is not True:
        raise ValueError("lighting teacher requires --visual-observation")
    try:
        raw, size = _teacher_payload(path, visual_observation=True, strict=True)
        result = _lighting_teacher_result(raw)
        episode = result.training_episode()
        return episode, _lighting_teacher_metadata(result, size)
    except (KeyError, TypeError, AttributeError, UnicodeError, RecursionError) as error:
        raise ValueError("malformed qualified lighting TRAIN teacher receipt") from error


def _check_camera_teacher_binding(raw, camera):
    _require_fixed_context_fields(raw["camera_context"], asdict(camera), "camera")
    if (
        raw["qualified"] is not True
        or raw["observed_success"] is not True
        or raw["final_success"] is not True
        or raw["error"] is not None
        or raw["traceback"] is not None
        or type(raw["seed"]) is not int
        or raw["seed"] != camera.seed
        or raw["role"] != "train"
        or raw["context"] != camera.context
        or raw["task"] != TASK_ID
        or raw["controller"] != CONTROL_MODE
    ):
        raise ValueError("expected qualified fixed visual camera TRAIN17 teacher")
    if any(
        raw.get(key) is not None
        for key in (
            "guard_seed",
            "labels_arrived_at",
            "mass_context",
            "mass_readback",
            "mass_work",
            "lighting_context",
            "lighting_readback",
            "lighting_work",
        )
    ):
        raise ValueError("camera TRAIN teacher cannot mix guard, mass or lighting metadata")
    if "optimizer_updates" in raw and (
        type(raw["optimizer_updates"]) is not int or raw["optimizer_updates"] != 0
    ):
        raise ValueError("camera teacher optimizer_updates must be exact zero when present")
    _check_visual_recipe(raw["observation_recipe"], camera_context=camera)


def _camera_teacher_result(raw):
    camera = pick_cube_camera_context()
    _check_camera_teacher_binding(raw, camera)
    episode = _teacher_episode(
        raw["candidate_episode"],
        camera.context,
        f"{camera.context}-teacher",
        visual_observation=True,
    )
    _check_camera_acquisition(raw, len(episode.transitions))
    # All fixed RGB recipes share encoding geometry and the complete paid
    # lifecycle; reuse its validator without broadening default recipe admission.
    visual = validate_lighting_visual_work(
        ManiSkillVisualWork(**raw["visual_work"]), steps=len(episode.transitions)
    )
    return TaskTeacherResult(
        episode,
        raw["observed_success"],
        raw["final_success"],
        raw["creation_attempts"],
        raw["geometry_attempts"],
        raw["geometry_returns"],
        raw["teacher_actions"],
        tuple(raw["stages"]),
        ManiSkillWork(**raw["work"]),
        raw["error"],
        raw["traceback"],
        observation_recipe=camera_context_recipe(camera),
        visual_work=visual,
        camera_context=camera,
    )


def _check_camera_acquisition(raw, rows):
    """Camera qualification alone does not validate retained acquisition costs."""
    expected = dict(
        creation_attempts=1, geometry_attempts=rows, geometry_returns=rows, teacher_actions=rows
    )
    if any(type(raw[key]) is not int or raw[key] != count for key, count in expected.items()):
        raise ValueError("camera teacher counters differ from its complete acquisition")
    work = ManiSkillWork(**raw["work"])
    if any(type(value) is not int for value in asdict(work).values()) or work != ManiSkillWork(
        rows, rows, rows * 5, 0
    ):
        raise ValueError("camera teacher controller work differs from its complete acquisition")
    stages = (
        ("align",) * 8 + ("descend",) * 8 + ("grasp",) * 6 + ("lift",) * 18 + ("settle",) * 10
    )[:rows]
    if type(raw["stages"]) is not list or tuple(raw["stages"]) != stages:
        raise ValueError("camera teacher stages differ from the fixed controller")


def _read_camera_teacher(path: Path, *, visual_observation: bool = False):
    """Read one bounded fixed-camera TRAIN receipt; no fit, simulator or refit."""
    if visual_observation is not True:
        raise ValueError("camera teacher requires --visual-observation")
    try:
        raw, size = _teacher_payload(path, visual_observation=True, strict=True)
        result = _camera_teacher_result(raw)
        episode = result.training_episode()
        return episode, dict(
            mode="camera_drift",
            warm_seed=17,
            new_seed=17,
            warm_context=pick_cube_training_context(17),
            new_context=episode.context,
            camera_context=asdict(pick_cube_camera_context()),
            teacher_work=asdict(result.work),
            teacher_visual_work=asdict(result.visual_work),
            teacher_creation_attempts=result.creation_attempts,
            teacher_geometry_attempts=result.geometry_attempts,
            teacher_geometry_returns=result.geometry_returns,
            teacher_actions=result.teacher_actions,
            source_file_bytes=size,
            per_file_byte_limit=4 * 1024**2,
            acquisition_scope="retained_teacher_work_not_new_simulator_work",
        )
    except (KeyError, TypeError, AttributeError, UnicodeError, RecursionError) as error:
        raise ValueError("malformed qualified camera TRAIN teacher receipt") from error


def _project_public_inputs(warm, new, *, camera_context: PickCubeCameraContext | None = None):
    """Own known TRAIN views and expose projection work separately from SGD."""
    projection, recipe = public_vector_projection(), pick_cube_rgb_recipe()
    if camera_context is None:
        projected = tuple(
            public_vector_training_episode(episode, projection, source_recipe=recipe)
            for episode in (warm, new)
        )
    else:
        projected = (
            public_vector_training_episode(warm, projection, source_recipe=recipe),
            public_vector_training_episode(
                new,
                projection,
                source_recipe=camera_context_recipe(camera_context),
                camera_context=camera_context,
            ),
        )
    observations = 2 * sum(len(episode.transitions) for episode in (warm, new))
    work = dict(
        projection=asdict(projection),
        source_recipe=asdict(recipe),
        observation_projections=observations,
        source_values_validated=observations * projection.source_dim,
        projected_values=observations * projection.feature_dim,
        projected_numeric_bytes=observations * projection.feature_dim * 8,
    )
    if camera_context is not None:
        work["new_source_recipe"] = asdict(camera_context_recipe(camera_context))
    return projected[0], projected[1], work


def _read_probe_plan(args, warm, new):
    """Separate declared probe inputs; never adapt TRAIN teacher roles or shapes."""
    training, training_bytes = read_continuous_probe_episode(args.probe_train, role="probe_train")
    evaluation, evaluation_bytes = read_continuous_probe_episode(args.probe_test, role="probe_test")
    plan = copy_probe_plan(
        ContinuousProbePlan(
            training,
            evaluation,
            1 if args.probe_updates is None else args.probe_updates,
            args.probe_diagnostics,
            args.probe_zero_velocity_control,
        ),
        (warm, new),
    )
    if (
        args.visual_observation
        and not args.public_vector_only
        and any(
            not 0.0 <= pixel <= 1.0
            for episode in (training, evaluation)
            for row in episode.transitions
            for observation in (row.observation, row.next_observation)
            for pixel in observation[21:]
        )
    ):
        raise ValueError("visual probe RGB features must lie in range [0,1]")
    metadata = dict(
        training=dict(
            episode_id=training.episode_id, role=training.role, source_file_bytes=training_bytes
        ),
        evaluation=dict(
            episode_id=evaluation.episode_id,
            role=evaluation.role,
            source_file_bytes=evaluation_bytes,
        ),
        per_file_byte_limit=MAX_PROBE_FILE_BYTES,
        reservation="caller_declared_reserved_episodes; no collection qualification or runtime release evidence",
    )
    return plan, metadata


def main() -> None:
    scenarios = {
        "stream": run_demo,
        "promotion-rejection": run_rejection_demo,
        "coupled-neutral": run_coupled_demo,
        "matched-baselines": run_matched_demo,
    }
    parser = argparse.ArgumentParser(description="Local delayed-label native adapter stream")
    parser.add_argument("--seed", type=int)
    continuous_scenarios: dict[
        str, Callable[[ContinuousEpisode, ContinuousEpisode], dict[str, Any]]
    ] = {
        "continuous-nearest": run_continuous_demo,
        "continuous-mlp-learnability": run_continuous_learnability_demo,
    }
    parser.add_argument("--scenario", choices=(*scenarios, *continuous_scenarios), default="stream")
    parser.add_argument("--warm-teacher", type=Path)
    parser.add_argument("--new-teacher", type=Path)
    parser.add_argument(
        "--readiness-teachers",
        action="store_true",
        help="Admit distinct qualified geometry-readiness TRAIN17/19 demonstrations.",
    )
    parser.add_argument(
        "--probe-train", type=Path, help="Reserved probe_train episode JSON; MLP only."
    )
    parser.add_argument(
        "--probe-test", type=Path, help="Reserved probe_test score-only episode JSON."
    )
    parser.add_argument(
        "--probe-updates", type=int, help="Fixed updates per probe scope (1..16; default1)."
    )
    parser.add_argument(
        "--probe-diagnostics", action="store_true", help="Add probe_train fit/direction readouts."
    )
    parser.add_argument(
        "--probe-zero-velocity-control",
        action="store_true",
        help="Add independently budgeted disposable velocity-only control.",
    )
    parser.add_argument(
        "--learner-family",
        choices=FAMILIES,
        help="Matched learner in continuous-mlp-learnability only (default bp).",
    )
    parser.add_argument(
        "--a-only-control",
        action="store_true",
        help="Train only warm A in all MLP phases; keep diagnostics on both A and B.",
    )
    parser.add_argument(
        "--camera-drift",
        action="store_true",
        help="Read fixed visual TRAIN17 displaced-camera B; warm remains default TRAIN17.",
    )
    parser.add_argument(
        "--mass-drift",
        action="store_true",
        help="Read new teacher as fixed TRAIN17 mass drift; warm remains default TRAIN17.",
    )
    parser.add_argument(
        "--lighting-drift",
        action="store_true",
        help="Read fixed visual TRAIN17 half-ambient B; warm remains default TRAIN17.",
    )
    parser.add_argument(
        "--phase-updates",
        type=int,
        help="Fixed updates per A/B/A phase in continuous-mlp-learnability (1..128; default8).",
    )
    parser.add_argument(
        "--binary-gripper-axis",
        type=int,
        help="Use existing mixed binary-gripper loss on this axis in continuous-mlp-learnability only.",
    )
    parser.add_argument(
        "--rgb-range-normalization",
        action="store_true",
        help="Fit TRAIN vector prefix and use fixed[0,1] RGB scaling in visual MLP recipe only.",
    )
    parser.add_argument(
        "--visual-observation",
        action="store_true",
        help="Read both continuous teachers with the fixed public RGB recipe.",
    )
    parser.add_argument(
        "--public-vector-only",
        action="store_true",
        help="Use the fixed public21 model input in visual continuous-mlp-learnability.",
    )
    parser.add_argument(
        "--diagnose-units",
        action="store_true",
        help="Observe native importance/age on the legacy circadian stream only.",
    )
    args = parser.parse_args()
    probe_options = (
        args.probe_train is not None
        or args.probe_test is not None
        or args.probe_updates is not None
        or args.probe_diagnostics
        or args.probe_zero_velocity_control
    )
    if probe_options:
        if args.scenario != "continuous-mlp-learnability":
            parser.error("probe options require --scenario continuous-mlp-learnability")
        if args.probe_train is None or args.probe_test is None:
            parser.error("probe options require both --probe-train and --probe-test")
        if args.probe_updates is not None and not 1 <= args.probe_updates <= 16:
            parser.error("--probe-updates must be an integer in [1,16]")
    if args.learner_family is not None and args.scenario != "continuous-mlp-learnability":
        parser.error("--learner-family requires --scenario continuous-mlp-learnability")
    if args.learner_family not in (None, "bp") and args.binary_gripper_axis is not None:
        parser.error("continuous PC/CPC supports only the default continuous likelihood")
    if args.a_only_control and args.scenario != "continuous-mlp-learnability":
        parser.error("--a-only-control requires --scenario continuous-mlp-learnability")
    if args.phase_updates is not None:
        if args.scenario != "continuous-mlp-learnability":
            parser.error("--phase-updates requires --scenario continuous-mlp-learnability")
        try:
            validate_phase_updates(args.phase_updates)
        except ValueError as error:
            parser.error(str(error))
    if args.public_vector_only and (
        args.scenario != "continuous-mlp-learnability" or not args.visual_observation
    ):
        parser.error("--public-vector-only requires visual continuous-mlp-learnability")
    if args.public_vector_only and args.rgb_range_normalization:
        parser.error("--public-vector-only excludes --rgb-range-normalization")
    if args.rgb_range_normalization and (
        args.scenario != "continuous-mlp-learnability" or not args.visual_observation
    ):
        parser.error("--rgb-range-normalization requires visual continuous-mlp-learnability")
    if args.binary_gripper_axis is not None and args.scenario != "continuous-mlp-learnability":
        parser.error("--binary-gripper-axis requires --scenario continuous-mlp-learnability")
    if args.diagnose_units and args.scenario != "stream":
        parser.error("--diagnose-units requires --scenario stream")
    continuous = args.scenario in continuous_scenarios
    if args.readiness_teachers and (
        not continuous or args.camera_drift or args.mass_drift or args.lighting_drift
    ):
        parser.error(
            "--readiness-teachers requires a continuous scenario without context interventions"
        )
    if args.camera_drift:
        if not continuous or not args.visual_observation:
            parser.error("--camera-drift requires a visual continuous stream scenario")
        if args.mass_drift or args.lighting_drift:
            parser.error("--camera-drift excludes --mass-drift and --lighting-drift")
    if args.lighting_drift:
        if not continuous or not args.visual_observation:
            parser.error("--lighting-drift requires a visual continuous stream scenario")
        if args.mass_drift or args.public_vector_only:
            parser.error("--lighting-drift excludes --mass-drift and --public-vector-only")
    if args.mass_drift and not continuous:
        parser.error("--mass-drift requires a continuous stream scenario")
    if args.mass_drift and args.public_vector_only:
        # The existing projection binds only its declared default/camera contexts.
        parser.error("--mass-drift does not support --public-vector-only")
    if args.visual_observation and not continuous:
        parser.error("--visual-observation requires --scenario continuous-nearest")
    if continuous and (args.warm_teacher is None or args.new_teacher is None):
        parser.error("continuous-nearest requires --warm-teacher and --new-teacher")
    if continuous and args.seed is not None:
        parser.error("--seed does not apply to fixed TRAIN17/19 teacher receipts")
    if not continuous and (args.warm_teacher is not None or args.new_teacher is not None):
        parser.error("teacher paths require --scenario continuous-nearest")
    try:
        if continuous:
            options: dict[str, Any] = (
                {"visual_observation": True} if args.visual_observation else {}
            )
            if args.readiness_teachers:
                warm, warm_metadata = _read_readiness_teacher(args.warm_teacher, 17, **options)
                new, new_metadata = _read_readiness_teacher(args.new_teacher, 19, **options)
            else:
                warm = _read_teacher(args.warm_teacher, 17, **options)
                if args.camera_drift:
                    new, context_metadata = _read_camera_teacher(args.new_teacher, **options)
                elif args.lighting_drift:
                    new, context_metadata = _read_lighting_teacher(args.new_teacher, **options)
                elif args.mass_drift:
                    new, context_metadata = _read_mass_teacher(args.new_teacher, **options)
                else:
                    new = _read_teacher(args.new_teacher, 19, **options)
            if args.public_vector_only:
                warm, new, projection_work = (
                    _project_public_inputs(warm, new, camera_context=pick_cube_camera_context())
                    if args.camera_drift
                    else _project_public_inputs(warm, new)
                )
            run_options: dict = {}
            if probe_options:
                probe_plan, probe_metadata = _read_probe_plan(args, warm, new)
                run_options["probe_plan"] = probe_plan
            if args.learner_family not in (None, "bp"):
                run_options["learner_family"] = args.learner_family
            if args.a_only_control:
                run_options["a_only_control"] = True
            if args.phase_updates is not None:
                run_options["phase_updates"] = args.phase_updates
            if args.binary_gripper_axis is not None:
                run_options["binary_gripper_axis"] = args.binary_gripper_axis
            if args.rgb_range_normalization:
                run_options["rgb_range_recipe"] = pick_cube_rgb_recipe()
            report = (
                run_continuous_learnability_demo(warm, new, **run_options)
                if run_options
                else continuous_scenarios[args.scenario](warm, new)
            )
            if probe_options:
                report["input_probes"] = probe_metadata
            if args.readiness_teachers:
                report["input_teachers"] = dict(warm=warm_metadata, new=new_metadata)
            if args.mass_drift or args.lighting_drift or args.camera_drift:
                report["input_context"] = context_metadata
            if args.visual_observation:
                report["input_observation"] = dict(
                    mode="public_rgb",
                    recipe=asdict(pick_cube_rgb_recipe()),
                    per_file_byte_limit=4 * 1024**2,
                )
                if args.camera_drift:
                    report["input_observation"] = dict(
                        mode="public_rgb_camera_drift",
                        warm_recipe=asdict(pick_cube_rgb_recipe()),
                        new_recipe=asdict(camera_context_recipe(pick_cube_camera_context())),
                        per_file_byte_limit=4 * 1024**2,
                    )
                if args.public_vector_only:
                    report["input_observation"].update(
                        mode="public_vector21",
                        source_mode="public_rgb_camera_drift"
                        if args.camera_drift
                        else "public_rgb",
                    )
                    report["input_projection"] = projection_work
        else:
            seed = 23 if args.seed is None else args.seed
            report = (
                run_demo(seed, include_unit_diagnostics=True)
                if args.diagnose_units
                else scenarios[args.scenario](seed)
            )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False, indent=2))
    if continuous and report.get("complete") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
