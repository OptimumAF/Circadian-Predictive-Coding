"""Run one real CPU PickCube episode; export an owned trace, never train a model.

Usage: python -m scripts.maniskill_continuous_smoke --output new-trace.json
Uses TRAIN seed17 and zero normalized actions by default; seed19 is explicit. Python callers may
pass main(learner=owned_train_bc_model) or main(actor=runtime.actor) for a
frozen-policy TRAIN rollout. Exact PC/CPC learners use a frozen feedforward
policy with full native/wake state checks. The actor route uses actual nearest-BC serving.
Requires the optional pinned simulator environment. No training or selection.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback
from typing import cast, Any

import numpy as np

from src.app.readiness_roll_in import ReadinessRollInLedger
from src.core.pick_cube_readiness_roll_in import (
    DEFINITION as READINESS_ROLL_IN_DEFINITION,
    ReadinessBehaviorPlan,
    ReadinessBehaviorDecision,
    ReadinessRollInPlan,
    ReadinessTeacherCompletionPlan,
    readiness_roll_in_decision,
    readiness_roll_in_metadata,
    validate_readiness_roll_in_plan,
)
from src.adapters.numpy_continuous_bc import (
    ContinuousBCSnapshot,
    NumpyContinuousBCLearner,
    _snapshot as validate_bc_snapshot,
)
from src.adapters.continuous_correction_demo import run_continuous_correction_update
from src.adapters.continuous_head_learnability import head_protocol
from src.adapters.numpy_continuous_pc import (
    ContinuousPCSnapshot,
    NumpyContinuousPCLearner,
    _snapshot as validate_pc_snapshot,
)
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.adapters.numpy_nearest_continuous_bc import (
    OBJECTIVE as NEAREST_UPDATE,
    FixedInputNearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
    _snapshot as validate_nearest_snapshot,
)
from src.app.actor_shadow import StableActor
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan, prepare_replay_bank
from src.app.serving_promotion import PromotableActor
from src.adapters.promotable_robot_binding import PromotableRobotBinding, bind_promotable_robot
from src.core.actor_ports import VersionedPrediction, VersionedState
from src.core.experience import require_identifier
from src.core import continuous_bc
from src.core import continuous_episode_clock as clocks
from src.core import pick_cube_public_vectors as public_vectors
from src.core import pick_cube_state_quaternions as state_quaternions
from src.core import pick_cube_action_history as action_history
from src.app.action_history_encoding import ActionHistoryEncoding
from src.app.action_decoding import ActionDecodingLedger
from src.core.pick_cube_action_decoding import (
    PickCubeBinaryGripperRecipe,
    validate_gripper_recipe,
)
from src.core.pick_cube_action_history_corrections import action_history_correction_metadata
from src.core.continuous_demonstration import ContinuousEpisode
from src.core.nearest_continuous_bc import NearestPrediction
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.pick_cube_guard_context import pick_cube_guard_context
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.joint_position_limits import (
    JointPositionLimits,
    copy_joint_position_summary,
    check_joint_position_samples,
)
from src.core.pick_cube_mass_context import (
    PickCubeMassContext,
    pick_cube_mass_context,
    require_mass_binding,
    validate_pick_cube_mass_context,
)
from src.core.pick_cube_lighting_context import (
    PickCubeLightingContext,
    pick_cube_lighting_context,
    require_lighting_binding,
    validate_pick_cube_lighting_context,
)
from src.infra.maniskill_lighting import (
    ManiSkillLightingReadback,
    ManiSkillLightingWork,
    validate_lighting_readback,
    validate_lighting_work,
)
from src.core.pick_cube_corrections import (
    PROVENANCE as FIXED_CORRECTION_PROVENANCE,
    PickCubeCorrectionBatch,
)
from src.core.pick_cube_readiness_corrections import (
    PROVENANCE as READINESS_CORRECTION_PROVENANCE,
    ReadinessCorrectionBatch,
    ObservedGraspCorrectionBatch,
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    readiness_correction_provenance,
    readiness_correction_definition,
)
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_TEACHER_DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_readiness_corrections import (
    OBSERVED_GRASP_PROVENANCE,
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
)
from src.core.pick_cube_state_corrections import StateQuaternionCorrectionWork
from src.core.continuous_episode_clock import EpisodeClock
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.pick_cube_camera_context import (
    PickCubeCameraContext,
    pick_cube_camera_context,
    require_camera_binding,
    camera_context_recipe,
    validate_camera_recipe,
    validate_pick_cube_camera_context,
)
from src.infra.maniskill_continuous import (
    ACTION_UNITS,
    CONTROL_MODE,
    HORIZON,
    TASK_ID,
    ManiSkillPickCubeEpisode,
    ManiSkillMassReadback,
    ManiSkillMassWork,
    validate_mass_readback,
    validate_mass_work,
    ManiSkillVisualWork,
    validate_lighting_visual_work,
    validate_joint_position_work,
    PANDA_JOINT_NAMES,
)
from src.infra.maniskill_corrections import PickCubeCorrectionQuery


Learner = NumpyContinuousBCLearner | NumpyNearestContinuousBCLearner | NumpyContinuousPCLearner
Snapshot = ContinuousBCSnapshot | NearestContinuousBCSnapshot | ContinuousPCSnapshot
ObservationProjection = (
    public_vectors.PickCubePublicVectorProjection
    | state_quaternions.PickCubeStateQuaternionProjection
)
NearestActor = StableActor[
    continuous_bc.Array, continuous_bc.Array, continuous_bc.Array, NearestContinuousBCSnapshot
]


@dataclass(frozen=True)
class _StableNearestBinding:
    actor: NearestActor
    backend: NumpyNearestContinuousBCLearner
    version: str
    state: NearestContinuousBCSnapshot


def _base_policy(state: Snapshot):
    """Only controller/spec reads unwrap PC; identity keeps its full native policy."""
    return state.policy.base if type(state) is ContinuousPCSnapshot else state.policy


def _same_state(before: Snapshot, after: Snapshot) -> bool:
    if type(before) is ContinuousPCSnapshot or type(after) is ContinuousPCSnapshot:
        if isinstance(before, ContinuousPCSnapshot) and isinstance(after, ContinuousPCSnapshot):
            return same_continuous_state(before, after)
        return False
    if type(before) not in (ContinuousBCSnapshot, NearestContinuousBCSnapshot) or type(
        before
    ) is not type(after):
        return False
    if isinstance(before, NearestContinuousBCSnapshot) and isinstance(
        after, NearestContinuousBCSnapshot
    ):
        return (
            before.policy == after.policy
            and before.completed_updates == after.completed_updates
            and before.training_examples == after.training_examples
            and np.array_equal(before.prototype_features, after.prototype_features)
            and np.array_equal(before.prototype_actions, after.prototype_actions)
        )
    if not isinstance(before, ContinuousBCSnapshot) or not isinstance(after, ContinuousBCSnapshot):
        return False
    return (
        before.policy == after.policy
        and before.completed_updates == after.completed_updates
        and before.training_examples == after.training_examples
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                (*before.parameters, *before.velocity),
                (*after.parameters, *after.velocity),
                strict=True,
            )
        )
    )


def _require_controller_policy(policy) -> None:
    if policy.training_role != "train":
        raise ValueError("this rollout requires a TRAIN policy")
    if (
        policy.spec.action_units != ACTION_UNITS
        or policy.spec.lower_bounds != (-1.0,) * 4
        or policy.spec.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("BC policy must match the normalized four-axis robot controller")


def _prepare_learner(learner: Learner) -> Snapshot:
    if type(learner) is NumpyContinuousPCLearner:
        pc_state = validate_pc_snapshot(learner.snapshot_state(), learner.policy)
        _require_controller_policy(pc_state.policy.base)
        return pc_state
    if type(learner) not in (NumpyContinuousBCLearner, NumpyNearestContinuousBCLearner):
        raise ValueError("rollout learner must be an exact supported continuous BC learner")
    _require_controller_policy(learner.policy)
    state = learner.snapshot_state()
    if type(learner) is NumpyNearestContinuousBCLearner and state.training_examples == 0:
        raise ValueError("nearest rollout requires released prototypes before native creation")
    return state


def _prepare_actor(actor: NearestActor) -> _StableNearestBinding | PromotableRobotBinding:
    if type(actor) is PromotableActor:
        binding = bind_promotable_robot(actor)
        if type(binding.state.policy) is FixedInputNearestContinuousBCPolicy:
            raise ValueError("fixed-input nearest serving requires an explicit taskspace learner")
        _require_controller_policy(binding.state.policy)
        return binding
    if (
        type(actor) is not StableActor
        or type(actor._learner) is not NumpyNearestContinuousBCLearner
    ):
        raise ValueError(
            "actor rollout requires an exact StableActor with an exact nearest BC backend"
        )
    # Type/identity inspection is admission only. All serving and state reads
    # use the actor's public locked methods; no restored learner substitutes.
    backend, version_id = actor._learner, actor.version
    require_identifier(version_id, "actor_version")
    tagged = actor.snapshot_state()
    if type(tagged) is not VersionedState or type(tagged.state) is not NearestContinuousBCSnapshot:
        raise ValueError("actor requires an exact versioned nearest BC snapshot")
    if (
        tagged.actor_version != version_id
        or actor.version != version_id
        or actor._learner is not backend
    ):
        raise ValueError("actor version or backend changed during admission")
    state = validate_nearest_snapshot(tagged.state, tagged.state.policy)
    if type(state.policy) is FixedInputNearestContinuousBCPolicy:
        raise ValueError("fixed-input nearest serving requires an explicit taskspace learner")
    _require_controller_policy(state.policy)
    if state.training_examples == 0:
        raise ValueError("actor rollout requires released prototypes before native creation")
    return _StableNearestBinding(actor, backend, version_id, state)


def _check_actor_frozen(binding: _StableNearestBinding | PromotableRobotBinding, result) -> None:
    result["frozen_state_exact"] = result["original_owner_exact"] = None
    result["actor_version_exact"] = result["actor_backend_exact"] = None
    keys = (
        "actor_generation_exact",
        "actor_configuration_exact",
        "actor_cache_exact",
        "actor_metadata_exact",
        "actor_last_cache_tick_exact",
        "actor_serving_bundle_exact",
    )
    for key in keys:
        result[key] = None
    if type(binding) is PromotableRobotBinding:
        binding.check()
        for key in (
            *keys,
            "frozen_state_exact",
            "original_owner_exact",
            "actor_version_exact",
            "actor_backend_exact",
        ):
            result[key] = True
        return
    actor = binding.actor
    result["actor_backend_exact"] = (
        type(actor) is StableActor
        and actor._learner is binding.backend
        and type(actor._learner) is NumpyNearestContinuousBCLearner
    )
    if not result["actor_backend_exact"]:
        raise ValueError("admitted stable actor backend was replaced")
    version_before = actor.version
    tagged = actor.snapshot_state()
    result["actor_backend_exact"] = actor._learner is binding.backend
    result["actor_version_exact"] = (
        type(tagged) is VersionedState
        and type(version_before) is str
        and type(actor.version) is str
        and type(tagged.actor_version) is str
        and version_before == actor.version == tagged.actor_version == binding.version
    )
    if not result["actor_backend_exact"] or not result["actor_version_exact"]:
        raise ValueError("admitted stable actor version or backend changed")
    observed = validate_nearest_snapshot(tagged.state, binding.state.policy)
    result["frozen_state_exact"] = _same_state(binding.state, observed)
    # Both fields concern the actual serving owner; this route has no private fork.
    result["original_owner_exact"] = result["frozen_state_exact"]
    if not result["frozen_state_exact"]:
        raise ValueError("frozen serving actor state changed")


def _check_frozen(learner, model, original, result) -> None:
    if type(model) in (_StableNearestBinding, PromotableRobotBinding):
        _check_actor_frozen(model, result)
        return
    # Each observation replaces the previous flags. A corrupt snapshot can
    # refuse observation; an earlier True must not survive as current evidence.
    result["frozen_state_exact"] = result["original_owner_exact"] = None
    if type(original) is ContinuousPCSnapshot:
        private_exact, source_exact = _check_pc_owners(learner, model, original)
        result["frozen_state_exact"], result["original_owner_exact"] = private_exact, source_exact
    else:
        result["frozen_state_exact"] = _same_state(original, model.snapshot_state())
        result["original_owner_exact"] = _same_state(original, learner.snapshot_state())
    if not result["frozen_state_exact"] or not result["original_owner_exact"]:
        raise ValueError("frozen BC model or original owner state changed")


def _check_pc_owners(learner, model, original, private_original=None):
    """Keep flags unknown on failure, including late public-observer callbacks."""
    if private_original is None:
        private_original = original
    if type(learner) is not NumpyContinuousPCLearner or type(model) is not NumpyContinuousPCLearner:
        raise ValueError("admitted continuous PC owners changed type")
    if learner is model:
        raise ValueError("continuous PC source and inference fork became aliased")
    private_observed = _same_state(private_original, model.snapshot_state())
    source_observed = _same_state(original, learner.snapshot_state())
    if type(learner) is not NumpyContinuousPCLearner or type(model) is not NumpyContinuousPCLearner:
        raise ValueError("continuous PC owner changed during observation")
    # Public callbacks may change the other owner. Finish from exact native-owned
    # fields, without another callback loop or legitimizing a bad public return.
    private = validate_pc_snapshot(model._state, model._policy)
    source = validate_pc_snapshot(learner._state, learner._policy)
    return (
        private_observed and _same_state(private_original, private),
        source_observed and _same_state(original, source),
    )


def _check_corrective_owners(source, original, candidate=None, candidate_state=None) -> bool:
    """Finish PC observations from native fields after all public getters.

    Correction changes the private candidate, so its expected state differs from
    the frozen source. No snapshot is restored to make a failed check pass.
    """
    if type(original) is ContinuousPCSnapshot:
        if type(source) is not NumpyContinuousPCLearner:
            raise ValueError("admitted continuous PC correction source changed type")
        if candidate is not None:
            if type(candidate_state) is not ContinuousPCSnapshot:
                raise ValueError("continuous PC correction lacks an admitted candidate state")
            private_exact, source_exact = _check_pc_owners(
                source, candidate, original, candidate_state
            )
            return private_exact and source_exact
        observed = _same_state(original, source.snapshot_state())
        if type(source) is not NumpyContinuousPCLearner:
            raise ValueError("continuous PC correction source changed during observation")
        native = validate_pc_snapshot(source._state, source._policy)
        return observed and _same_state(original, native)
    return _same_state(original, source.snapshot_state()) and (
        candidate is None or _same_state(candidate_state, candidate.snapshot_state())
    )


def _check_canonical_corrective_owners(source, original, candidate=None, candidate_state=None):
    """Close public observer boundaries using the admitted native owners."""
    expected_type: type
    validator: Any
    if type(original) is ContinuousPCSnapshot:
        expected_type, validator = NumpyContinuousPCLearner, validate_pc_snapshot
    elif type(original) is NearestContinuousBCSnapshot:
        expected_type, validator = NumpyNearestContinuousBCLearner, validate_nearest_snapshot
    else:
        expected_type, validator = NumpyContinuousBCLearner, validate_bc_snapshot
    if type(source) is not expected_type:
        raise ValueError("canonical correction source owner changed")
    if candidate is not None and (type(candidate) is not expected_type or candidate is source):
        raise ValueError("canonical correction requires a separate exact candidate")
    observed = _check_corrective_owners(source, original, candidate, candidate_state)
    exact = observed and _same_state(original, validator(source._state, source._policy))
    if candidate is not None:
        exact = exact and _same_state(
            candidate_state, validator(candidate._state, candidate._policy)
        )
    return exact


def _predict_nearest(model, original: NearestContinuousBCSnapshot, features, result):
    rows, dimensions = original.training_examples, _base_policy(original).spec.observation_dim
    result["nearest_prediction_attempts"] += 1
    result["nearest_distance_candidate_upper_bound"] += rows
    result["nearest_distance_feature_upper_bound"] += rows * dimensions
    report = model.predict_with_work(features)
    result["nearest_prediction_returns"] += 1
    result["prediction_returns"] += 1
    if (
        type(report) is not NearestPrediction
        or type(report.distance_candidates) is not int
        or report.distance_candidates != rows
        or type(report.distance_feature_values) is not int
        or report.distance_feature_values != rows * dimensions
        or type(report.selected_indices) is not tuple
        or len(report.selected_indices) != 1
        or type(report.selected_indices[0]) is not int
        or not 0 <= report.selected_indices[0] < rows
    ):
        raise ValueError("nearest prediction work or selected row differs from admitted prototypes")
    result["nearest_distance_candidates_completed"] += report.distance_candidates
    result["nearest_distance_feature_values_completed"] += report.distance_feature_values
    result["nearest_selected_indices"].extend(report.selected_indices)
    return report.actions


def _predict_stable(binding: _StableNearestBinding | PromotableRobotBinding, features, result):
    result["actor_prediction_version_exact"] = None
    _check_actor_frozen(binding, result)
    rows, dimensions = binding.state.training_examples, binding.state.policy.spec.observation_dim
    result["actor_prediction_attempts"] += 1
    result["actor_distance_candidate_upper_bound"] += rows
    result["actor_distance_feature_upper_bound"] += rows * dimensions
    tagged = binding.actor.predict(features)
    result["actor_prediction_returns"] += 1
    result["prediction_returns"] += 1
    result["actor_prediction_version_exact"] = (
        type(tagged) is VersionedPrediction
        and type(tagged.actor_version) is str
        and tagged.actor_version == binding.version
    )
    if not result["actor_prediction_version_exact"]:
        raise ValueError("actor prediction version differs from admitted serving version")
    result["actor_prediction_versions"].append(tagged.actor_version)
    return tagged.prediction


def _predict_action(
    learner, model, original, observation, result, *, projection_port=None, projection=None
) -> tuple[float, ...]:
    if type(original) is ContinuousPCSnapshot:
        _check_frozen(learner, model, original, result)
    if projection_port is not None:
        _check_projection(projection, result)
        _check_state_projection_port(projection_port, projection, result)
    result["prediction_attempts"] += 1
    features = np.array([observation], dtype=np.float64)
    if type(model) in (_StableNearestBinding, PromotableRobotBinding):
        prediction = _predict_stable(model, features, result)
    elif isinstance(original, NearestContinuousBCSnapshot):
        prediction = _predict_nearest(model, original, features, result)
    else:
        prediction = model.predict(features)
        result["prediction_returns"] += 1
    # Why: refuse unexpected state changes before sending any control action;
    # do not roll them back or obscure the failure with a restored success flag.
    _check_frozen(learner, model, original, result)
    action = continuous_bc._targets(prediction, 1, _base_policy(original).spec)
    result["validated_prediction_rows"] += 1
    result["validated_prediction_bytes"] += action.nbytes
    if type(model) in (_StableNearestBinding, PromotableRobotBinding):
        # StableActor returns actions, not NearestPrediction. These counts are
        # derived only after a valid return/full-state/action observation.
        result["actor_distance_candidates_completed"] += original.training_examples
        result["actor_distance_feature_values_completed"] += (
            original.training_examples * _base_policy(original).spec.observation_dim
        )
    return tuple(float(value) for value in action[0])


def _check_clock(clock, signature, result) -> None:
    result["clock_configuration_exact"] = None
    observed = clocks._clock(clock)
    result["clock_configuration_exact"] = (observed.horizon, observed.elapsed) == signature
    if not result["clock_configuration_exact"]:
        raise ValueError("admitted clock configuration changed")


def _check_visual_recipe(port, result) -> None:
    if result["observation_recipe"] is None:
        return
    result["visual_recipe_exact"] = None
    camera = None
    if "camera_context" in result:
        metadata = result["camera_context"]
        if type(metadata) is not dict or set(metadata) != {"context", "seed"}:
            raise ValueError("declared rollout camera metadata changed")
        camera = validate_pick_cube_camera_context(PickCubeCameraContext(**metadata))
        require_camera_binding(
            camera,
            context=result["context"],
            seed=result["seed"],
            role="train",
            visual_observation=True,
        )
        if validate_pick_cube_camera_context(port.camera_context) != camera:
            raise ValueError("declared rollout camera context changed")
    observed = validate_camera_recipe(port.observation_recipe, camera)
    result["visual_recipe_exact"] = asdict(observed) == result["observation_recipe"]
    if not result["visual_recipe_exact"]:
        raise ValueError("admitted visual observation recipe changed")


def _check_mass_port(port, context, result, original=None) -> ManiSkillMassReadback | None:
    if context is None:
        return None
    result["mass_context_exact"] = None
    expected = validate_pick_cube_mass_context(context)
    metadata = result["mass_context"]
    if type(metadata) is not dict or set(metadata) != set(asdict(expected)):
        raise ValueError("declared rollout mass metadata changed")
    if validate_pick_cube_mass_context(PickCubeMassContext(**metadata)) != expected:
        raise ValueError("declared rollout mass metadata changed")
    require_mass_binding(expected, context=result["context"], seed=result["seed"], role="train")
    observed = validate_mass_readback(port.mass_readback, expected)
    work = validate_mass_work(port.mass_work)
    # Retain returned work before later callbacks or comparison failures.
    result["mass_work"] = asdict(work)
    if original is None:
        original = observed
        result["mass_readback"] = asdict(original)
    if observed != original:
        raise ValueError("original rollout mass readback changed")
    if json.dumps(result["mass_readback"], sort_keys=True, allow_nan=False) != json.dumps(
        asdict(original), sort_keys=True, allow_nan=False
    ):
        raise ValueError("retained rollout mass readback changed")
    completed = port.work.completed_steps
    if validate_pick_cube_mass_context(port.mass_context) != expected:
        raise ValueError("declared rollout mass context changed")
    if validate_mass_readback(port.mass_readback, expected) != original:
        raise ValueError("rollout mass readback changed after metadata getters")
    reads = 3 + 2 * completed + int("episode" in result)
    if (
        work.read_attempts != reads
        or work.read_returns != reads
        or (work.write_attempts != 1 or work.write_returns != 1)
    ):
        raise ValueError("mass work differs from the frozen rollout lifecycle")
    ManiSkillPickCubeEpisode.check_mass_metadata(port, expected, original, work)
    result["mass_context_exact"] = True
    return original


def _lighting_report_context(context, result) -> PickCubeLightingContext:
    expected = validate_pick_cube_lighting_context(context)
    metadata = result["lighting_context"]
    if type(metadata) is not dict or set(metadata) != set(asdict(expected)):
        raise ValueError("declared rollout lighting metadata changed")
    if validate_pick_cube_lighting_context(PickCubeLightingContext(**metadata)) != expected:
        raise ValueError("declared rollout lighting metadata changed")
    require_lighting_binding(
        expected,
        context=result["context"],
        seed=result["seed"],
        role=result["role"],
        visual_observation=True,
    )
    return expected


def _check_lighting_snapshot(port, context, result, original) -> None:
    """Check retained metadata after owner getters, without another virtual read."""
    expected = _lighting_report_context(context, result)
    owned = validate_lighting_readback(original, expected)
    if json.dumps(result["lighting_readback"], sort_keys=True, allow_nan=False) != json.dumps(
        asdict(owned), sort_keys=True, allow_nan=False
    ):
        raise ValueError("retained original rollout lighting readback changed")
    work = validate_lighting_work(ManiSkillLightingWork(**result["lighting_work"]))
    visual_work = validate_lighting_visual_work(ManiSkillVisualWork(**result["visual_work"]))
    ManiSkillPickCubeEpisode.check_lighting_metadata(
        port, expected, owned, work, observed_visual_work=visual_work
    )


def _check_lighting_port(port, context, result, original=None) -> ManiSkillLightingReadback | None:
    if context is None:
        return None
    result["lighting_context_exact"] = None
    expected = _lighting_report_context(context, result)
    previous = result["lighting_work"]
    previous = (
        validate_lighting_work(ManiSkillLightingWork(**previous)) if previous is not None else None
    )
    previous_visual = result.get("visual_work")
    previous_visual = (
        validate_lighting_visual_work(ManiSkillVisualWork(**previous_visual))
        if previous_visual is not None
        else None
    )
    if original is not None and json.dumps(
        result["lighting_readback"], sort_keys=True, allow_nan=False
    ) != json.dumps(asdict(original), sort_keys=True, allow_nan=False):
        raise ValueError("retained original rollout lighting readback changed")
    work = validate_lighting_work(port.lighting_work)
    if previous is not None and any(
        getattr(work, name) < getattr(previous, name) for name in asdict(work)
    ):
        raise ValueError("paid rollout lighting counters decreased")
    # Paid returns are retained before later getters/lifecycle checks may fail.
    result["lighting_work"] = asdict(work)
    visual_work = validate_lighting_visual_work(port.visual_work)
    if previous_visual is not None and any(
        getattr(visual_work, name) < getattr(previous_visual, name)
        for name in asdict(visual_work)
        if name != "current_owned_frame_bytes"
    ):
        raise ValueError("paid rollout visual counters decreased")
    result["visual_work"] = asdict(visual_work)
    if validate_pick_cube_lighting_context(port.lighting_context) != expected:
        raise ValueError("declared rollout lighting context changed")
    observed = validate_lighting_readback(port.lighting_readback, expected)
    if original is None:
        original = observed
        result["lighting_readback"] = asdict(original)
    elif observed != original:
        raise ValueError("original rollout lighting readback changed")
    completed = port.work.completed_steps
    validate_lighting_visual_work(visual_work, steps=completed)
    reads = 3 + 2 * completed + int("episode" in result)
    if (
        work.read_attempts != reads
        or work.read_returns != reads
        or work.write_attempts != 1
        or work.write_returns != 1
        or work.refresh_attempts != 1
        or work.refresh_returns != 1
        or work.discarded_reset_observations != 1
    ):
        raise ValueError("lighting work differs from the frozen rollout lifecycle")
    _check_lighting_snapshot(port, expected, result, original)
    result["lighting_context_exact"] = True
    return original


def _validate_projection(projection):
    if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
        return state_quaternions.validate_state_quaternion_projection(projection)
    return public_vectors.validate_public_vector_projection(projection)


def _projection_spec(spec, projection):
    if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
        return state_quaternions.state_quaternion_spec(spec, projection)
    return public_vectors.public_vector_spec(spec, projection)


def _check_state_projection_port(port, projection, result) -> None:
    if type(projection) is not state_quaternions.PickCubeStateQuaternionProjection:
        return
    # Finish after opaque getters from native-owned fields, without another
    # sensor read or treating a late visual/context switch as a new admission.
    if (
        type(port) is not ManiSkillPickCubeEpisode
        or port._visual_observation is not False
        or port._camera_context is not None
        or port._mass_context is not None
        or port._mass_context_active is not False
        or port._lighting_context is not None
        or port._lighting_context_active is not False
        or type(port._seed) is not int
        or port._seed not in (17, 19)
        or port._context != pick_cube_training_context(port._seed)
        or port._role != "train"
        or port._episode_id != f"pick-cube-smoke-{port._seed}"
        or result.get("observation_recipe") is not None
        or result.get("visual_work") is not None
        or result.get("camera_context") is not None
        or result.get("mass_context") is not None
        or result.get("lighting_context") is not None
        or type(result.get("seed")) is not int
        or result["seed"] != port._seed
        or result.get("context") != port._context
        or result.get("role") != "train"
    ):
        raise ValueError("canonical state42 projection lost its default TRAIN state binding")
    spec = state_quaternions.state_quaternion_spec(port._spec, projection)
    if (
        spec.action_units != ACTION_UNITS
        or spec.lower_bounds != (-1.0,) * 4
        or spec.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("canonical state42 native controller spec changed")


def _check_projection(projection, result) -> None:
    if projection is None:
        return
    result["projection_exact"] = None
    expected = _validate_projection(projection)
    metadata = result["observation_projection"]
    if type(metadata) is not dict or set(metadata) != set(asdict(expected)):
        raise ValueError("admitted observation projection metadata changed")
    observed = _validate_projection(type(expected)(**metadata))
    if observed != expected:
        raise ValueError("admitted observation projection changed")
    _check_projection_work(expected, result)
    result["projection_exact"] = True


def _check_state_projection_frozen(learner, model, original, result) -> None:
    """Finish this recipe's public ownership observers from native-owned state."""
    _check_frozen(learner, model, original, result)
    expected_type: type
    validator: Any
    if type(original) is ContinuousPCSnapshot:
        expected_type, validator = NumpyContinuousPCLearner, validate_pc_snapshot
    elif type(original) is NearestContinuousBCSnapshot:
        expected_type, validator = NumpyNearestContinuousBCLearner, validate_nearest_snapshot
    else:
        expected_type, validator = NumpyContinuousBCLearner, validate_bc_snapshot
    result["frozen_state_exact"] = result["original_owner_exact"] = None
    if type(learner) is not expected_type or type(model) is not expected_type or learner is model:
        raise ValueError("canonical state42 source or inference owner was replaced")
    private, source = (
        validator(model._state, model._policy),
        validator(learner._state, learner._policy),
    )
    result["frozen_state_exact"] = _same_state(original, private)
    result["original_owner_exact"] = _same_state(original, source)
    if not result["frozen_state_exact"] or not result["original_owner_exact"]:
        raise ValueError("canonical state42 frozen native owner changed")


def _check_projection_work(expected, result) -> None:
    work = result["projection_work"]
    keys = {
        "observation_attempts",
        "observation_returns",
        "source_values_upper_bound",
        "source_values_validated",
        "projected_values",
        "projected_numeric_bytes",
    }
    if (
        type(work) is not dict
        or set(work) != keys
        or any(type(value) is not int or value < 0 for value in work.values())
        or not 0 <= work["observation_returns"] <= work["observation_attempts"] <= HORIZON
        or type(result["prediction_attempts"]) is not int
        or not result["prediction_attempts"]
        <= work["observation_returns"]
        <= result["prediction_attempts"] + 1
        or work["observation_attempts"] > work["observation_returns"] + 1
        or (
            result["complete"] is True
            and not work["observation_attempts"]
            == work["observation_returns"]
            == result["prediction_attempts"]
        )
        or work["source_values_upper_bound"] != work["observation_attempts"] * expected.source_dim
        or work["source_values_validated"] != work["observation_returns"] * expected.source_dim
        or work["projected_values"] != work["observation_returns"] * expected.feature_dim
        or work["projected_numeric_bytes"] != work["projected_values"] * 8
    ):
        raise ValueError("observation projection work changed")


def _project_observation(port, projection, result, *, raw_observation=None):
    _check_projection(projection, result)
    _check_state_projection_port(port, projection, result)
    work = result["projection_work"]
    work["observation_attempts"] += 1
    work["source_values_upper_bound"] += projection.source_dim
    raw = port.observation if raw_observation is None else raw_observation
    if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
        _check_state_projection_port(port, projection, result)
        observation = state_quaternions.state_quaternion_observation(raw, projection)
    else:
        observation = public_vectors.public_vector_observation(
            raw,
            projection,
            source_recipe=port.observation_recipe,
            camera_context=port.camera_context,
        )
    work["observation_returns"] += 1
    work["source_values_validated"] += projection.source_dim
    work["projected_values"] += projection.feature_dim
    work["projected_numeric_bytes"] += projection.feature_dim * 8
    _check_projection(projection, result)
    _check_state_projection_port(port, projection, result)
    return observation


def _history_policy_matches(base, recipe):
    config = action_history.validate_action_history_recipe(recipe)
    if base.spec.observation_dim != config.feature_dim:
        return False
    stats = base.observation_standardization
    if action_history.is_taskspace_recipe(config):
        return stats is None
    return (
        stats is not None
        and stats.means[-5:] == (0.0,) * 5
        and stats.scales[-5:] == (1.0,) * 5
        and getattr(stats, "effective_scales", stats.scales)[-5:] == (1.0,) * 5
    )


def _probe_test_reset(seed):
    """Bind scoring reset24 without borrowing the reserved teacher episode ID."""
    binding = pick_cube_probe_binding("probe_test")
    if type(seed) is not int or seed != binding.seed:
        raise ValueError("probe-test scoring requires the exact native seed24")
    return binding.seed, binding.context, "pick-cube-probe-test-smoke-24"


def _check_action_history_port(
    port, encoding, result, *, returned_record=None, guard_seed=None, probe_test_seed=None
):
    if guard_seed is not None and probe_test_seed is not None:
        raise ValueError("guard and probe-test history bindings are exclusive")
    encoding.check(result.get("action_history"))
    binding, role = encoding.native_binding, encoding.native_role
    if binding is None:
        raise ValueError("causal action history lost its admitted native reset")
    seed, context, episode_id = binding

    def check_binding():
        expected_role = (
            "probe_test"
            if probe_test_seed is not None
            else "inner_guard"
            if guard_seed is not None
            else "train"
        )
        if probe_test_seed is not None and (
            binding != _probe_test_reset(probe_test_seed)
            or type(result.get("probe_test_seed")) is not int
            or result["probe_test_seed"] != probe_test_seed
            or result.get("scoring_only") is not True
            or "guard_seed" in result
        ):
            raise ValueError(
                "causal action history lost its original scoring-only probe-test request"
            )
        if guard_seed is not None and (
            binding
            != (
                guard_seed,
                pick_cube_guard_context(guard_seed),
                f"pick-cube-guard-smoke-{guard_seed}",
            )
            or type(result.get("guard_seed")) is not int
            or result["guard_seed"] != guard_seed
            or result.get("scoring_only") is not True
        ):
            raise ValueError("causal action history lost its original scoring-only guard request")
        if (
            encoding.native_binding != binding
            or encoding.native_role != role
            or role != expected_role
            or type(port) is not ManiSkillPickCubeEpisode
            or port._visual_observation is not False
            or type(port._role) is not str
            or port._role != role
            or type(result.get("role")) is not str
            or result["role"] != role
            or type(port._seed) is not int
            or port._seed != seed
            or type(result["seed"]) is not int
            or result["seed"] != seed
            or port._context != context
            or result["context"] != context
            or port._episode_id != episode_id
            or port._spec.observation_dim != 42
            or (
                (guard_seed is not None or probe_test_seed is not None)
                and port._monitor_joint_positions is not False
            )
        ):
            raise ValueError("causal action history lost its original native role/reset binding")

    check_binding()
    if action_history.is_taskspace_recipe(encoding.recipe) and (
        result.get("task") != "PickCube-v1"
        or result.get("controller") != "pd_ee_delta_pos"
        or port._camera_context is not None
        or (port._mass_context is not None and encoding.mass_context is None)
        or port._lighting_context is not None
    ):
        raise ValueError("taskspace WORLD axes require the admitted default Panda table controller")
    if encoding.mass_context is not None:
        # Mass changes dynamics, not the admitted WORLD axes or command history.
        mass = validate_pick_cube_mass_context(encoding.mass_context)
        require_mass_binding(port._mass_context, context=context, seed=seed, role=role)
        if (
            port._mass_context != mass
            or port._mass_context_active is not True
            or result.get("mass_context") != asdict(mass)
            or encoding.recipe.mode != "executed"
        ):
            raise ValueError("causal history lost its explicit fixed mass binding")
    action_history.action_history_spec(port._spec, encoding.recipe)
    records = tuple(port._records)
    expected_raw = port._observation
    if returned_record is not None:
        if len(records) != encoding.committed + 1 or records[-1] != returned_record:
            raise ValueError("causal action history lost its single returned pending control")
        expected_raw = records[-1].observation
        records = records[:-1]
    encoding.validate_executions(records, expected_raw)
    # Encoding/report callbacks must not relabel a returned control or the next
    # native action after the first role check has already succeeded.
    encoding.check(result.get("action_history"))
    check_binding()


def _check_guard_native_frozen(learner, model, original):
    """Final guard closure after history callbacks, using only owned native fields."""
    validator: Any
    kind, validator = (
        (NumpyContinuousPCLearner, validate_pc_snapshot)
        if type(original) is ContinuousPCSnapshot
        else (NumpyContinuousBCLearner, validate_bc_snapshot)
    )
    if type(learner) is not kind or type(model) is not kind or learner is model:
        raise ValueError("guard source or inference owner was replaced")
    if not _same_state(original, validator(learner._state, learner._policy)) or not _same_state(
        original, validator(model._state, model._policy)
    ):
        raise ValueError("guard frozen native owner changed")


def _check_mass_history_frozen(port, encoding, learner, model, original, mass, readback, result):
    """Close composed callbacks with owned proofs, without another native read."""
    if encoding is None or encoding.mass_context is None:
        return
    if mass is None or readback is None:
        raise ValueError("mass history lost its admitted native readback")
    _check_guard_native_frozen(learner, model, original)
    ManiSkillPickCubeEpisode.check_mass_metadata(
        port, mass, readback, validate_mass_work(ManiSkillMassWork(**result["mass_work"]))
    )


def _rollout(
    port,
    learner,
    model,
    original,
    result,
    clock=None,
    signature=None,
    corrections=None,
    projection=None,
    mass_context=None,
    mass_readback=None,
    lighting_context=None,
    lighting_readback=None,
    roll_in=None,
    history_encoding=None,
    observed_grasp_corrections=False,
    completion_request=None,
    action_decoding=None,
    guard_seed=None,
    probe_test_seed=None,
) -> None:
    history_binding_options = {
        name: seed
        for name, seed in (("guard_seed", guard_seed), ("probe_test_seed", probe_test_seed))
        if seed is not None
    }
    _check_mass_port(port, mass_context, result, mass_readback)
    _check_lighting_port(port, lighting_context, result, lighting_readback)
    _check_visual_recipe(port, result)
    _check_projection(projection, result)
    expected_spec = clocks.clock_spec(port.spec, clock) if clock is not None else port.spec
    if projection is not None:
        expected_spec = _projection_spec(port.spec, projection)
        _check_state_projection_port(port, projection, result)
    if history_encoding is not None:
        expected_spec = action_history.action_history_spec(port.spec, history_encoding.recipe)
        _check_action_history_port(port, history_encoding, result, **history_binding_options)
    if action_decoding is not None:
        action_decoding.check(result["action_decoding"], tuple(port._records))
    if model is not None and expected_spec != _base_policy(original).spec:
        raise ValueError("native observation/action spec differs from BC policy")
    for step in range(HORIZON):
        if action_decoding is not None:
            action_decoding.check(result["action_decoding"], tuple(port._records))
        _check_completion_request(completion_request)
        if roll_in is not None:
            _check_roll_in_result(roll_in, result)
        _check_mass_port(port, mass_context, result, mass_readback)
        _check_lighting_port(port, lighting_context, result, lighting_readback)
        if (mass_context is not None or lighting_context is not None) and model is not None:
            _check_frozen(learner, model, original, result)
        action: tuple[float, ...] = (0.0,) * 4
        if model is not None:
            if history_encoding is not None or (projection is not None and corrections is not None):
                # The label must bind the same full raw input whose view the
                # student used, rather than a second unobserved sensor value.
                raw_observation = port.observation
                observation = (
                    raw_observation
                    if history_encoding is not None
                    else _project_observation(
                        port, projection, result, raw_observation=raw_observation
                    )
                )
            else:
                observation = (
                    port.observation
                    if projection is None
                    else _project_observation(port, projection, result)
                )
            if history_encoding is not None:
                _check_action_history_port(
                    port, history_encoding, result, **history_binding_options
                )
                try:
                    observation = history_encoding.encode(observation)
                finally:
                    result["action_history"] = history_encoding.report()
                _check_action_history_port(
                    port, history_encoding, result, **history_binding_options
                )
            if clock is not None:
                _check_clock(clock, signature, result)
                result["clock_encoding_attempts"] += 1
                observation = clocks.clock_observation(observation, step, clock)
                result["clock_encoding_returns"] += 1
                result["clock_encoded_values"] += len(observation)
                result["clock_values"].append(observation[-1])
            if history_encoding is not None:
                _check_state_projection_frozen(learner, model, original, result)
                _check_action_history_port(
                    port, history_encoding, result, **history_binding_options
                )
            elif observed_grasp_corrections:
                _check_state_projection_frozen(learner, model, original, result)
            if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
                action = _predict_action(
                    learner,
                    model,
                    original,
                    observation,
                    result,
                    projection_port=port,
                    projection=projection,
                )
            else:
                if action_decoding is not None:
                    action_decoding.check(result["action_decoding"], tuple(port._records))
                action = _predict_action(learner, model, original, observation, result)
            if action_decoding is not None:
                try:
                    action = action_decoding.select(step, action, result["action_decoding"])
                finally:
                    result["action_decoding"] = action_decoding.report()
            if history_encoding is not None:
                _check_state_projection_frozen(learner, model, original, result)
                _check_action_history_port(
                    port, history_encoding, result, **history_binding_options
                )
            elif observed_grasp_corrections:
                _check_state_projection_frozen(learner, model, original, result)
            _check_projection(projection, result)
            _check_state_projection_port(port, projection, result)
            if clock is not None:
                _check_clock(clock, signature, result)
        if corrections is not None:
            corrections.query(
                port,
                step,
                raw_observation
                if (projection is not None or history_encoding is not None)
                else observation,
            )
            # Why: privileged callbacks must not mutate either frozen owner
            # between the student's prediction and sending its actual command.
            _check_frozen(learner, model, original, result)
        decision = None
        if roll_in is not None:
            label = corrections.latest_readiness_label(step, raw_observation)
            proposed_student, queried_target = action, label.target
            decision = readiness_roll_in_decision(
                roll_in.plan, step, proposed_student, queried_target
            )
            if type(decision) is not ReadinessBehaviorDecision:
                raise ValueError("readiness behavior returned a foreign decision")
            action = decision.executed_action
        _check_visual_recipe(port, result)
        _check_projection(projection, result)
        _check_mass_port(port, mass_context, result, mass_readback)
        _check_lighting_port(port, lighting_context, result, lighting_readback)
        # Recipe/clock access may invoke trusted callbacks. Observe the actor
        # after those reads so an intervening commit cannot send a stale action.
        if type(model) in (_StableNearestBinding, PromotableRobotBinding):
            _check_actor_frozen(model, result)
        elif (
            corrections is not None
            or projection is not None
            or ((mass_context is not None or lighting_context is not None) and model is not None)
        ):
            _check_frozen(learner, model, original, result)
        if mass_context is not None:
            ManiSkillPickCubeEpisode.check_mass_metadata(
                port,
                mass_context,
                mass_readback,
                validate_mass_work(ManiSkillMassWork(**result["mass_work"])),
            )
        if lighting_context is not None:
            _check_lighting_snapshot(port, lighting_context, result, lighting_readback)
        if type(original) is ContinuousPCSnapshot:
            # Observe after the last recipe/clock/context callback, before control.
            _check_frozen(learner, model, original, result)
        if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
            _check_projection(projection, result)
        _check_state_projection_port(port, projection, result)
        if roll_in is not None:
            _check_state_projection_frozen(learner, model, original, result)
            _check_projection(projection, result)
            _check_state_projection_port(port, projection, result)
            _check_roll_in_result(roll_in, result)
            roll_in.attempt(
                decision,
                raw_observation,
                proposed_student,
                queried_target,
                result["readiness_roll_in"],
            )
            result["readiness_roll_in"] = roll_in.report()
        if history_encoding is not None:
            _check_state_projection_frozen(learner, model, original, result)
            _check_action_history_port(port, history_encoding, result, **history_binding_options)
        elif observed_grasp_corrections:
            _check_state_projection_frozen(learner, model, original, result)
        _check_completion_request(completion_request)
        if roll_in is not None:
            _check_roll_in_result(roll_in, result)
        if action_decoding is not None:
            action_decoding.check(result["action_decoding"], tuple(port._records))
            try:
                action_decoding.attempt(result["action_decoding"])
            finally:
                result["action_decoding"] = action_decoding.report()
        # A final history callback may change a different mass/source field.
        _check_mass_history_frozen(
            port, history_encoding, learner, model, original, mass_context, mass_readback, result
        )
        record = port.step(action)
        if action_decoding is not None:
            # A returned control is paid before later history/owner checks.
            try:
                action_decoding.returned(record, result["action_decoding"])
            finally:
                result["action_decoding"] = action_decoding.report()
        if roll_in is not None:
            # A returned native control stays paid even if history validation fails.
            try:
                roll_in.returned(record, result["readiness_roll_in"])
            finally:
                result["readiness_roll_in"] = roll_in.report()
        if history_encoding is not None:
            _check_action_history_port(
                port, history_encoding, result, returned_record=record, **history_binding_options
            )
            history_encoding.commit(record, action)
            result["action_history"] = history_encoding.report()
            _check_state_projection_frozen(learner, model, original, result)
            _check_action_history_port(port, history_encoding, result, **history_binding_options)
        if action_decoding is not None:
            action_decoding.check(result["action_decoding"], tuple(port._records))

        _check_mass_port(port, mass_context, result, mass_readback)
        _check_lighting_port(port, lighting_context, result, lighting_readback)
        _check_visual_recipe(port, result)
        _check_projection(projection, result)
        if model is not None:
            _check_frozen(learner, model, original, result)
        _check_state_projection_port(port, projection, result)
        if clock is not None:
            _check_clock(clock, signature, result)
        if record.terminated or record.truncated:
            break
    if history_encoding is not None:
        _check_action_history_port(port, history_encoding, result, **history_binding_options)
    episode = port.episode()
    if history_encoding is not None:
        _check_action_history_port(port, history_encoding, result, **history_binding_options)
        history_encoding.verify_episode(episode)
    if action_decoding is not None:
        action_decoding.verify_episode(episode, result["action_decoding"])
    result.update(
        episode=asdict(episode),
        sparse_return=sum(record.reward for record in episode.transitions),
        any_success=any(record.reward == 1 for record in episode.transitions),
        final_success=episode.transitions[-1].reward == 1,
        complete=True,
    )

    _check_mass_port(port, mass_context, result, mass_readback)
    _check_lighting_port(port, lighting_context, result, lighting_readback)
    if (mass_context is not None or lighting_context is not None) and model is not None:
        _check_frozen(learner, model, original, result)
    if lighting_context is not None:
        _check_lighting_snapshot(port, lighting_context, result, lighting_readback)


@dataclass(frozen=True)
class FrozenRolloutResult:
    """Owned report and usable paired corrections only after all final checks."""

    report: dict[str, Any]
    corrections: PickCubeCorrectionBatch | ReadinessCorrectionBatch | None


def _check_roll_in_result(ledger, result):
    ledger.check(result.get("readiness_roll_in"))
    if (
        result.get("behavior_scope") != _readiness_behavior_scope(ledger.plan)
        or result.get("student_only_success_available") is not False
        or result.get("policy") != "mixed_readiness_teacher_and_frozen_student"
    ):
        raise ValueError("mixed readiness behavior lost its explicit success scope")


def _readiness_behavior_scope(plan: ReadinessBehaviorPlan) -> str:
    return (
        "frozen_student_prefix_then_teacher_completion"
        if type(plan) is ReadinessTeacherCompletionPlan
        else "teacher_prefix_then_frozen_student"
    )


def _check_completion_request(request) -> None:
    if request is not None:
        plan, expected = request
        if (
            type(plan) is not ReadinessTeacherCompletionPlan
            or json.dumps(readiness_roll_in_metadata(plan), sort_keys=True, allow_nan=False)
            != expected
        ):
            raise ValueError("original teacher completion request changed")


def _require_teacher_completion_inputs(plan, learner, recipe, readiness, observed, goal):
    """Admit the specific TRAIN coverage path; keep older roll-in rules intact."""
    if type(plan) is not ReadinessTeacherCompletionPlan:
        return False
    validate_readiness_roll_in_plan(plan)
    if (
        type(learner) is not NumpyContinuousBCLearner
        or readiness is not True
        or observed is not True
        or goal is not True
        or recipe is None
    ):
        raise ValueError(
            "teacher completion requires BP, observed-v2 readiness and executed taskspace12"
        )
    owned = action_history.validate_action_history_recipe(recipe)
    if not action_history.is_taskspace_recipe(owned) or owned.mode != "executed":
        raise ValueError("teacher completion requires the explicit executed taskspace12 recipe")
    return True


def _check_teacher_completion_collection(plan, report, batch):
    """Bind mixed commands to the received raw TRAIN trajectory before updates."""
    observed = report.get("readiness_roll_in")
    if type(observed) is not dict or type(observed.get("controls")) is not list:
        raise ValueError("teacher completion requires complete behavior receipts")
    controls = observed["controls"]
    if len(controls) != len(batch.episode.transitions) or not 1 <= len(controls) <= HORIZON:
        raise ValueError("teacher completion receipt count differs from the raw episode")
    ledger = ReadinessRollInLedger(plan)
    for control, row, label in zip(controls, batch.episode.transitions, batch.labels, strict=True):
        if type(control) is not dict:
            raise ValueError("teacher completion control must be an exact record")
        student = tuple(control.get("student_action", ()))
        decision = readiness_roll_in_decision(plan, row.step, student, label.target)
        ledger.attempt(decision, row.observation, student, label.target, ledger.report())
        ledger.returned(row, ledger.report())
    ledger.verify_episode(batch.episode, batch.labels)
    ledger.check(observed)
    if (
        report.get("behavior_scope") != _readiness_behavior_scope(plan)
        or report.get("student_only_success_available") is not False
        or report.get("policy") != "mixed_readiness_teacher_and_frozen_student"
    ):
        raise ValueError("teacher completion collection lost its mixed-behavior scope")


def _check_correction_mode(
    report, batch, readiness: bool, observed_grasp: bool = False, goal_consistent: bool = False
) -> None:
    """Bind the requested label source to exact returned provenance and family."""
    expected = (
        GoalConsistentObservedGraspCorrectionBatch
        if goal_consistent
        else ObservedGraspCorrectionBatch
        if observed_grasp
        else ReadinessCorrectionBatch
        if readiness
        else PickCubeCorrectionBatch
    )
    if goal_consistent and not observed_grasp:
        raise ValueError("goal consistent requires observed grasp provenance")
    if observed_grasp and not readiness:
        raise ValueError("observed grasp requires readiness correction provenance")
    if type(batch) is not expected or type(report) is not dict:
        raise ValueError("released correction family differs from the requested mode")
    if readiness:
        owned = copy_readiness_correction_batch(cast(ReadinessCorrectionBatch, batch))
        if (
            report.get("provenance") != readiness_correction_provenance(owned)
            or report.get("teacher_definition") != readiness_correction_definition(owned)
            or "teacher_schedule_steps" in report
        ):
            raise ValueError("readiness correction report lost its distinct provenance")
        if observed_grasp:
            feedback = report.get("observed_grasp")
            if (
                type(feedback) is not list
                or any(type(value) is not bool for value in feedback)
                or tuple(feedback) != cast(ObservedGraspCorrectionBatch, batch).observed_grasp
            ):
                raise ValueError("released observed grasp feedback differs from current raw labels")
        elif "observed_grasp" in report:
            raise ValueError("unrequested observed grasp supervision")
        count = report.get("teacher_decision_steps")
    else:
        if (
            report.get("provenance") != FIXED_CORRECTION_PROVENANCE
            or "teacher_definition" in report
            or "teacher_decision_steps" in report
        ):
            raise ValueError("fixed correction report differs from the requested provenance")
        count = report.get("teacher_schedule_steps")
    if type(count) is not int or count != len(batch.labels):
        raise ValueError("correction teacher progress differs from the released labels")
    labels = [
        dict(
            asdict(label), observed_at=label.observed_at, labels_arrived_at=label.labels_arrived_at
        )
        for label in batch.labels
    ]
    if labels != report.get("labels"):
        raise ValueError("released correction targets differ from the query receipts")


def _read_joint_position_report(
    port, result, *, initial_limits: JointPositionLimits | None = None, verify: bool = False
) -> JointPositionLimits:
    """Own native-free reports; complete traces verify every sampled qpos prefix."""
    result["joint_position_exact"] = None
    # Paid work must survive a subsequent malformed or throwing summary getter.
    work = validate_joint_position_work(port.joint_position_work)
    previous = result.get("joint_position_work")
    if previous is not None and any(count < previous[name] for name, count in asdict(work).items()):
        raise ValueError("joint-position paid work moved backwards")
    result["joint_position_work"] = asdict(work)
    control_work = result.get("work")
    if control_work is not None:
        completed = control_work["completed_steps"]
        if type(completed) is not int or not 0 <= completed <= HORIZON:
            raise ValueError("joint-position report requires bounded returned control work")
        if work.samples > completed + 1:
            raise ValueError("joint-position samples exceed reset plus returned controls")
    summary = copy_joint_position_summary(port.joint_position_summary)
    if (
        summary.limits.names != PANDA_JOINT_NAMES
        or summary.limits.units != ("rad",) * 7 + ("m",) * 2
    ):
        raise ValueError("joint-position report requires the declared Panda order and rad/m units")
    result["joint_position_summary"] = asdict(summary)
    if summary.samples != work.samples:
        raise ValueError("joint-position summary and work have different sample counts")
    if initial_limits is None:
        if summary.samples != 1:
            raise ValueError("initial joint-position report must contain exactly the reset sample")
    elif summary.limits != initial_limits:
        raise ValueError("joint-position bounds changed after the reset-time snapshot")
    if verify:
        episode = result["episode"]
        rows = episode["transitions"]
        if not 1 <= len(rows) <= HORIZON or episode["spec"]["observation_dim"] not in (42, 789):
            raise ValueError("joint-position verification requires the full bounded raw trace")
        samples = (tuple(rows[0]["observation"][:9]),) + tuple(
            tuple(row["next_observation"][:9]) for row in rows
        )
        result["joint_position_validation_samples"] = len(samples)
        result["joint_position_validation_coordinates"] = len(samples) * 9
        check_joint_position_samples(summary, samples)
        result["joint_position_exact"] = True
    return summary.limits


def main(
    *,
    learner: Learner | None = None,
    observation_clock: EpisodeClock | None = None,
    train_seed: int = 17,
    actor: NearestActor | None = None,
    visual_observation: bool = False,
    corrective_queries: bool = False,
    camera_context: PickCubeCameraContext | None = None,
    mass_context: PickCubeMassContext | None = None,
    lighting_context: PickCubeLightingContext | None = None,
    observation_projection: ObservationProjection | None = None,
    monitor_joint_positions: bool = False,
    readiness_corrections: bool = False,
    readiness_roll_in: ReadinessBehaviorPlan | None = None,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    observed_grasp_corrections: bool = False,
    goal_consistent_corrections: bool = False,
    action_decoding_recipe: PickCubeBinaryGripperRecipe | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, choices=(17, 19), default=train_seed)
    parser.add_argument("--visual-observation", action="store_true", default=visual_observation)
    parser.add_argument("--corrective-queries", action="store_true", default=corrective_queries)
    parser.add_argument(
        "--readiness-corrections", action="store_true", default=readiness_corrections
    )
    parser.add_argument(
        "--observed-grasp-corrections", action="store_true", default=observed_grasp_corrections
    )
    parser.add_argument(
        "--goal-consistent-corrections", action="store_true", default=goal_consistent_corrections
    )
    behavior_defaults = (
        {} if readiness_roll_in is None else readiness_roll_in_metadata(readiness_roll_in)
    )
    behavior_options = parser.add_mutually_exclusive_group()
    behavior_options.add_argument(
        "--readiness-roll-in-steps",
        type=int,
        default=behavior_defaults.get("teacher_steps"),
    )
    behavior_options.add_argument(
        "--readiness-teacher-completion-student-steps",
        type=int,
        default=behavior_defaults.get("student_steps"),
    )
    parser.add_argument("--camera-drift", action="store_true", default=camera_context is not None)
    parser.add_argument("--mass-drift", action="store_true", default=mass_context is not None)
    parser.add_argument(
        "--lighting-drift", action="store_true", default=lighting_context is not None
    )
    parser.add_argument(
        "--public-vector-only",
        action="store_true",
        default=type(observation_projection) is public_vectors.PickCubePublicVectorProjection,
    )
    parser.add_argument(
        "--monitor-joint-positions", action="store_true", default=monitor_joint_positions
    )
    args = parser.parse_args()
    if (
        args.readiness_roll_in_steps is not None
        and args.readiness_teacher_completion_student_steps is not None
    ):
        parser.error("choose exactly one readiness behavior plan")
    readiness_options: dict[str, Any] = {}
    if action_decoding_recipe is not None:
        readiness_options["action_decoding_recipe"] = action_decoding_recipe
    if action_history_recipe is not None:
        readiness_options["action_history_recipe"] = action_history_recipe
    if observed_grasp_corrections is not False:
        readiness_options["observed_grasp_corrections"] = observed_grasp_corrections
    if goal_consistent_corrections is not False:
        readiness_options["goal_consistent_corrections"] = goal_consistent_corrections
    if readiness_corrections is not False:
        readiness_options["readiness_corrections"] = readiness_corrections
    if args.readiness_roll_in_steps is not None:
        readiness_options["readiness_roll_in"] = ReadinessRollInPlan(args.readiness_roll_in_steps)
    if args.readiness_teacher_completion_student_steps is not None:
        readiness_options["readiness_roll_in"] = ReadinessTeacherCompletionPlan(
            args.readiness_teacher_completion_student_steps
        )
    outcome = _execute_frozen_rollout(
        args,
        learner=learner,
        observation_clock=observation_clock,
        train_seed=train_seed,
        actor=actor,
        visual_observation=visual_observation,
        corrective_queries=corrective_queries,
        camera_context=camera_context,
        mass_context=mass_context,
        lighting_context=lighting_context,
        observation_projection=observation_projection,
        monitor_joint_positions=monitor_joint_positions,
        **readiness_options,
    )
    print(json.dumps({key: value for key, value in outcome.report.items() if key != "episode"}))
    return 0 if outcome.report["complete"] else 1


def run_frozen_rollout(
    output: Path,
    *,
    learner: Learner | None = None,
    observation_clock: EpisodeClock | None = None,
    train_seed: int = 17,
    actor: NearestActor | None = None,
    visual_observation: bool = False,
    corrective_queries: bool = False,
    camera_context: PickCubeCameraContext | None = None,
    mass_context: PickCubeMassContext | None = None,
    lighting_context: PickCubeLightingContext | None = None,
    observation_projection: ObservationProjection | None = None,
    monitor_joint_positions: bool = False,
    readiness_corrections: bool = False,
    readiness_roll_in: ReadinessBehaviorPlan | None = None,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    observed_grasp_corrections: bool = False,
    goal_consistent_corrections: bool = False,
    action_decoding_recipe: PickCubeBinaryGripperRecipe | None = None,
    guard_seed: int | None = None,
    probe_test_seed: int | None = None,
) -> FrozenRolloutResult:
    """Programmatic entry to the same exclusive-output frozen rollout as the CLI."""
    args = argparse.Namespace(
        output=output,
        train_seed=train_seed,
        visual_observation=visual_observation,
        corrective_queries=corrective_queries,
        camera_drift=camera_context is not None,
        mass_drift=mass_context is not None,
        lighting_drift=lighting_context is not None,
        public_vector_only=type(observation_projection)
        is public_vectors.PickCubePublicVectorProjection,
        monitor_joint_positions=monitor_joint_positions,
    )
    readiness_options: dict[str, Any] = {}
    if probe_test_seed is not None:
        readiness_options["probe_test_seed"] = probe_test_seed
    if guard_seed is not None:
        readiness_options["guard_seed"] = guard_seed
    if action_decoding_recipe is not None:
        readiness_options["action_decoding_recipe"] = action_decoding_recipe
    if action_history_recipe is not None:
        readiness_options["action_history_recipe"] = action_history_recipe
    if observed_grasp_corrections is not False:
        readiness_options["observed_grasp_corrections"] = observed_grasp_corrections
    if goal_consistent_corrections is not False:
        readiness_options["goal_consistent_corrections"] = goal_consistent_corrections
    if readiness_corrections is not False:
        args.readiness_corrections = readiness_corrections
        readiness_options["readiness_corrections"] = readiness_corrections
    if observed_grasp_corrections is not False:
        args.observed_grasp_corrections = observed_grasp_corrections
    if goal_consistent_corrections is not False:
        args.goal_consistent_corrections = goal_consistent_corrections
    if readiness_roll_in is not None:
        readiness_options["readiness_roll_in"] = readiness_roll_in
    return _execute_frozen_rollout(
        args,
        learner=learner,
        observation_clock=observation_clock,
        train_seed=train_seed,
        actor=actor,
        visual_observation=visual_observation,
        corrective_queries=corrective_queries,
        camera_context=camera_context,
        mass_context=mass_context,
        lighting_context=lighting_context,
        observation_projection=observation_projection,
        monitor_joint_positions=monitor_joint_positions,
        **readiness_options,
    )


def _execute_frozen_rollout(
    args,
    *,
    learner,
    observation_clock,
    train_seed,
    actor,
    visual_observation,
    corrective_queries,
    camera_context,
    mass_context,
    lighting_context,
    observation_projection,
    monitor_joint_positions,
    readiness_corrections=False,
    readiness_roll_in=None,
    action_history_recipe=None,
    observed_grasp_corrections=False,
    goal_consistent_corrections=False,
    action_decoding_recipe=None,
    guard_seed=None,
    probe_test_seed=None,
) -> FrozenRolloutResult:
    # Reserve the destination before admitting simulator work. No existing
    # trace can be overwritten, including a failed attempt's receipt.
    completion_request = None
    if type(readiness_roll_in) is ReadinessTeacherCompletionPlan:
        completion_request = (
            readiness_roll_in,
            json.dumps(
                readiness_roll_in_metadata(readiness_roll_in), sort_keys=True, allow_nan=False
            ),
        )
    with args.output.open("x", encoding="utf8") as handle:
        result: dict[str, Any] = dict(
            task=TASK_ID,
            controller=CONTROL_MODE,
            seed=None,
            context=None,
            role="train",
            policy=(
                "frozen_stable_nearest_continuous_bc"
                if actor is not None
                else "constant_zero_normalized_4axis"
                if learner is None
                else "frozen_continuous_pc"
                if type(learner) is NumpyContinuousPCLearner
                else "frozen_continuous_bc"
            ),
            complete=False,
            creation_attempts=0,
            creation_returns=0,
            constructor_reset="official_seed2022_reconfigure",
            explicit_reset=None,
            reset_physics_steps=None,
            optimizer_updates=0,
            prediction_attempts=0,
            prediction_returns=0,
            validated_prediction_rows=0,
            validated_prediction_bytes=0,
            frozen_state_exact=None,
            original_owner_exact=None,
            observation_clock=None,
            clock_configuration_exact=None,
            clock_encoding_attempts=0,
            clock_encoding_returns=0,
            clock_encoded_values=0,
            clock_values=[],
            source_update_definition=None,
            nearest_prototype_rows=0,
            nearest_prototype_numeric_bytes=0,
            nearest_prediction_attempts=0,
            nearest_prediction_returns=0,
            nearest_distance_candidate_upper_bound=0,
            nearest_distance_feature_upper_bound=0,
            nearest_distance_candidates_completed=0,
            nearest_distance_feature_values_completed=0,
            nearest_selected_indices=[],
            state_guard_scope=None,
            actor_version=None,
            actor_generation=None,
            actor_generation_exact=None,
            actor_configuration_exact=None,
            actor_cache_exact=None,
            actor_metadata_exact=None,
            actor_last_cache_tick_exact=None,
            actor_serving_bundle_exact=None,
            actor_version_exact=None,
            actor_backend_exact=None,
            actor_prediction_version_exact=None,
            actor_prediction_attempts=0,
            actor_prediction_returns=0,
            actor_prediction_versions=[],
            actor_distance_count_source=None,
            actor_distance_candidate_upper_bound=0,
            actor_distance_feature_upper_bound=0,
            actor_distance_candidates_completed=0,
            actor_distance_feature_values_completed=0,
            work=None,
            observation_recipe=None,
            visual_recipe_exact=None,
            visual_work=None,
        )
        start = time.monotonic()
        port = original = clock = signature = projection = None
        history_encoding = None
        action_decoding = None
        history_episode_evidence = None
        history_finished_batch = None
        history_finished_batch_evidence = None
        mass = mass_readback = None
        light = lighting_readback = None
        joint_limits = None
        joint_monitoring = False
        pc_dense_per_prediction = None
        model: Learner | _StableNearestBinding | PromotableRobotBinding | None = None
        corrections: PickCubeCorrectionQuery | None = None
        released_batch: PickCubeCorrectionBatch | ReadinessCorrectionBatch | None = None
        released_batch_evidence = None
        roll_in = None
        observed_grasp_mode = False
        goal_consistent_mode = False
        history_binding_options = {
            name: seed
            for name, seed in (("guard_seed", guard_seed), ("probe_test_seed", probe_test_seed))
            if seed is not None
        }

        scoring_binding = None
        scoring_role = None

        def check_scoring_request():
            if guard_seed is not None and probe_test_seed is not None:
                raise ValueError("guard and probe-test scoring resets are exclusive")
            context = (
                pick_cube_guard_context(guard_seed)
                if guard_seed is not None
                else _probe_test_reset(probe_test_seed)[1]
            )
            if (
                type(train_seed) is not int
                or train_seed != 17
                or type(args.train_seed) is not int
                or args.train_seed != 17
                or type(learner) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner)
                or actor is not None
                or observation_clock is not None
                or visual_observation is not False
                or args.visual_observation is not False
                or corrective_queries is not False
                or args.corrective_queries is not False
                or readiness_corrections is not False
                or getattr(args, "readiness_corrections", False) is not False
                or observed_grasp_corrections is not False
                or getattr(args, "observed_grasp_corrections", False) is not False
                or goal_consistent_corrections is not False
                or getattr(args, "goal_consistent_corrections", False) is not False
                or readiness_roll_in is not None
                or observation_projection is not None
                or action_decoding_recipe is not None
                or monitor_joint_positions is not False
                or args.monitor_joint_positions is not False
                or args.public_vector_only is not False
                or args.camera_drift is not False
                or args.mass_drift is not False
                or args.lighting_drift is not False
                or camera_context is not None
                or mass_context is not None
                or lighting_context is not None
                or not action_history.is_taskspace_recipe(action_history_recipe)
            ):
                mode = "guard" if guard_seed is not None else "probe-test"
                raise ValueError(
                    mode
                    + " scoring requires exact BC/PC and executed TS12 without TRAIN seed19 or optional modes"
                )
            config = action_history.validate_action_history_recipe(action_history_recipe)
            if config.mode != "executed":
                mode = "guard" if guard_seed is not None else "probe-test"
                raise ValueError(mode + " scoring requires the fixed executed taskspace12 recipe")
            return context

        try:
            _check_completion_request(completion_request)
            # Validate the native API value before argparse can coerce string defaults.
            pick_cube_training_context(train_seed)
            if guard_seed is not None or probe_test_seed is not None:
                scoring_context = check_scoring_request()
                if guard_seed is not None:
                    scoring_binding = (
                        guard_seed,
                        scoring_context,
                        f"pick-cube-guard-smoke-{guard_seed}",
                    )
                    scoring_role = "inner_guard"
                    result.update(guard_seed=guard_seed, scoring_only=True, role=scoring_role)
                else:
                    scoring_binding = _probe_test_reset(probe_test_seed)
                    scoring_role = "probe_test"
                    result.update(
                        probe_test_seed=probe_test_seed, scoring_only=True, role=scoring_role
                    )
            if type(visual_observation) is not bool:
                raise ValueError("visual_observation must be an exact bool")
            if type(corrective_queries) is not bool:
                raise ValueError("corrective_queries must be an exact bool")
            if type(readiness_corrections) is not bool:
                raise ValueError("readiness_corrections must be an exact bool")
            if type(observed_grasp_corrections) is not bool:
                raise ValueError("observed_grasp_corrections must be an exact bool")
            observed_grasp_mode = getattr(args, "observed_grasp_corrections", False)
            if type(observed_grasp_mode) is not bool:
                raise ValueError("observed grasp correction argument must be an exact bool")
            if observed_grasp_corrections and observed_grasp_mode is not True:
                raise ValueError("observed grasp request changed before admission")
            if type(goal_consistent_corrections) is not bool:
                raise ValueError("goal_consistent_corrections must be an exact bool")
            goal_consistent_mode = getattr(args, "goal_consistent_corrections", False)
            if type(goal_consistent_mode) is not bool:
                raise ValueError("goal consistent correction argument must be an exact bool")
            if goal_consistent_corrections and goal_consistent_mode is not True:
                raise ValueError("goal consistent request changed before admission")
            if goal_consistent_mode and not observed_grasp_mode:
                raise ValueError("goal consistent corrections require observed grasp readiness")
            readiness_mode = getattr(args, "readiness_corrections", False)
            if type(readiness_mode) is not bool:
                raise ValueError("readiness_corrections argument must be an exact bool")
            if readiness_corrections and readiness_mode is not True:
                raise ValueError("readiness correction request was changed before admission")
            if readiness_mode and (
                args.corrective_queries is not True
                or learner is None
                or actor is not None
                or observation_clock is not None
                or args.camera_drift
                or args.mass_drift
                or args.lighting_drift
            ):
                raise ValueError(
                    "readiness corrections require corrective queries and a default TRAIN learner "
                    "without actor, clock or interventions"
                )
            if action_decoding_recipe is not None:
                validate_gripper_recipe(action_decoding_recipe)
                if (
                    type(learner) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner)
                    or actor is not None
                    or observation_clock is not None
                    or observation_projection is not None
                    or args.visual_observation is not False
                    or args.public_vector_only
                    or args.corrective_queries
                    or readiness_mode
                    or readiness_roll_in is not None
                    or observed_grasp_mode
                    or goal_consistent_mode
                    or args.camera_drift
                    or args.mass_drift
                    or args.lighting_drift
                    or camera_context is not None
                    or mass_context is not None
                    or lighting_context is not None
                    or not action_history.is_taskspace_recipe(action_history_recipe)
                    or action_history_recipe.mode != "executed"
                ):
                    raise ValueError(
                        "binary gripper execution requires pure common-MSE neural TRAIN "
                        "with executed taskspace12 and no interventions"
                    )
                action_decoding = ActionDecodingLedger(action_decoding_recipe)
                result.update(action_decoding=action_decoding.report(), action_decoding_exact=None)
            teacher_completion = _require_teacher_completion_inputs(
                readiness_roll_in,
                learner,
                action_history_recipe,
                readiness_mode,
                observed_grasp_mode,
                goal_consistent_mode,
            )
            if observed_grasp_mode and (
                not readiness_mode
                or args.visual_observation is not False
                or args.public_vector_only
                or (readiness_roll_in is not None and not teacher_completion)
            ):
                raise ValueError(
                    "observed grasp corrections require raw nonvisual readiness without roll-in"
                )
            if observed_grasp_mode:
                result["observed_grasp_corrections"] = True
            if goal_consistent_mode:
                result["goal_consistent_corrections"] = True
            if type(monitor_joint_positions) is not bool:
                raise ValueError("monitor_joint_positions must be an exact bool")
            joint_monitoring = args.monitor_joint_positions
            if type(joint_monitoring) is not bool:
                raise ValueError("monitor_joint_positions argument must be an exact bool")
            if joint_monitoring:
                result.update(
                    joint_position_summary=None,
                    joint_position_work=None,
                    joint_position_exact=None,
                    joint_position_validation_samples=0,
                    joint_position_validation_coordinates=0,
                )
            if action_history_recipe is not None:
                if type(learner) is NumpyNearestContinuousBCLearner and (
                    type(learner._policy) is not FixedInputNearestContinuousBCPolicy
                    or not action_history.is_taskspace_recipe(action_history_recipe)
                    or action_history_recipe.mode != "executed"
                    or args.corrective_queries
                    or readiness_roll_in is not None
                ):
                    raise ValueError(
                        "nearest action history requires fixed-input executed taskspace12 without queries"
                    )
                if mass_context is not None:
                    if (
                        type(learner) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner)
                        or args.corrective_queries
                        or readiness_mode
                        or observed_grasp_mode
                        or readiness_roll_in is not None
                        or action_decoding_recipe is not None
                        or joint_monitoring
                    ):
                        raise ValueError(
                            "mass history requires frozen BC/PC without optional interventions"
                        )
                    # Validate before fork/native IO; the encoding pins this original request.
                    validate_pick_cube_mass_context(mass_context)
                if (
                    args.visual_observation is not False
                    or args.public_vector_only
                    or type(learner)
                    not in (
                        NumpyContinuousBCLearner,
                        NumpyContinuousPCLearner,
                        NumpyNearestContinuousBCLearner,
                    )
                    or actor is not None
                    or observation_clock is not None
                    or observation_projection is not None
                    or (args.corrective_queries and readiness_mode is not True)
                    or args.camera_drift
                    or (args.mass_drift and mass_context is None)
                    or args.lighting_drift
                    or camera_context is not None
                    or lighting_context is not None
                ):
                    raise ValueError(
                        "action history requires the frozen nonvisual TRAIN state learner without interventions"
                    )
                history_encoding = (
                    ActionHistoryEncoding(action_history_recipe, probe_test_seed=probe_test_seed)
                    if probe_test_seed is not None
                    else ActionHistoryEncoding(action_history_recipe, guard_seed=guard_seed)
                    if guard_seed is not None
                    else ActionHistoryEncoding(
                        action_history_recipe, train_seed=args.train_seed, mass_context=mass_context
                    )
                )
                result["action_history"] = history_encoding.report()
            elif (
                type(learner) is NumpyNearestContinuousBCLearner
                and type(learner._policy) is FixedInputNearestContinuousBCPolicy
            ):
                raise ValueError(
                    "fixed-input nearest serving requires explicit executed taskspace12"
                )
            if observation_projection is not None:
                projection = _validate_projection(observation_projection)
            elif args.public_vector_only:
                projection = public_vectors.public_vector_projection()
            if projection is not None:
                if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
                    if (
                        args.visual_observation is not False
                        or args.public_vector_only
                        or type(learner)
                        not in (
                            NumpyContinuousBCLearner,
                            NumpyNearestContinuousBCLearner,
                            NumpyContinuousPCLearner,
                        )
                        or actor is not None
                        or observation_clock is not None
                        or (args.corrective_queries and readiness_mode is not True)
                        or args.camera_drift
                        or args.mass_drift
                        or args.lighting_drift
                        or camera_context is not None
                        or mass_context is not None
                        or lighting_context is not None
                    ):
                        raise ValueError(
                            "canonical state42 projection requires an exact state learner "
                            "without actor, clock, fixed corrections or context interventions"
                        )
                elif (
                    args.visual_observation is not True
                    or (learner is None and actor is None)
                    or observation_clock is not None
                ):
                    raise ValueError(
                        "public-vector21 requires an explicit visual learner/actor without clock"
                    )
                result["observation_projection"] = asdict(projection)
                result["projection_work"] = dict(
                    observation_attempts=0,
                    observation_returns=0,
                    source_values_upper_bound=0,
                    source_values_validated=0,
                    projected_values=0,
                    projected_numeric_bytes=0,
                )
                _check_projection(projection, result)
            if readiness_roll_in is not None:
                plan = validate_readiness_roll_in_plan(readiness_roll_in)
                if (
                    (
                        type(projection) is not state_quaternions.PickCubeStateQuaternionProjection
                        and history_encoding is None
                    )
                    or readiness_mode is not True
                    or args.corrective_queries is not True
                    or args.visual_observation is not False
                    or learner is None
                    or actor is not None
                    or observation_clock is not None
                ):
                    raise ValueError(
                        "readiness roll-in requires canonical state42 readiness queries"
                    )
                roll_in = ReadinessRollInLedger(plan)
                result.update(
                    readiness_roll_in=roll_in.report(),
                    behavior_scope=_readiness_behavior_scope(plan),
                    student_only_success_available=False,
                )
            context = (
                scoring_binding[1]
                if scoring_binding is not None
                else pick_cube_training_context(args.train_seed)
            )
            camera = None
            if camera_context is not None:
                camera = validate_pick_cube_camera_context(camera_context)
            elif args.camera_drift:
                camera = pick_cube_camera_context()
            if camera is not None:
                context = camera.context
                require_camera_binding(
                    camera,
                    context=context,
                    seed=args.train_seed,
                    role="train",
                    visual_observation=args.visual_observation,
                )
                if args.corrective_queries:
                    raise ValueError("camera drift does not admit corrective queries")
                result["camera_context"] = asdict(camera)
            if mass_context is not None:
                mass = validate_pick_cube_mass_context(mass_context)
            elif args.mass_drift:
                mass = pick_cube_mass_context()
            if mass is not None:
                if camera is not None or args.corrective_queries:
                    raise ValueError(
                        "mass drift does not admit camera mixing or corrective queries"
                    )
                context = mass.context
                require_mass_binding(mass, context=context, seed=args.train_seed, role="train")
                result.update(
                    mass_context=asdict(mass),
                    mass_context_exact=None,
                    mass_readback=None,
                    mass_work=None,
                )
            if lighting_context is not None:
                light = validate_pick_cube_lighting_context(lighting_context)
            elif args.lighting_drift:
                light = pick_cube_lighting_context()
            observed_rollout_options: dict[str, Any] = (
                {"observed_grasp_corrections": True} if observed_grasp_mode else {}
            )
            if probe_test_seed is not None:
                observed_rollout_options["probe_test_seed"] = probe_test_seed
            if guard_seed is not None:
                observed_rollout_options["guard_seed"] = guard_seed
            if completion_request is not None:
                observed_rollout_options["completion_request"] = completion_request
            if action_decoding is not None:
                observed_rollout_options["action_decoding"] = action_decoding
            if light is not None:
                if camera is not None or mass is not None or args.corrective_queries:
                    raise ValueError(
                        "lighting requires separate TRAIN collection without camera/mass/corrections"
                    )
                if projection is not None:
                    raise ValueError("public-vector projection has no declared lighting binding")
                context = light.context
                require_lighting_binding(
                    light,
                    context=context,
                    seed=args.train_seed,
                    role="train",
                    visual_observation=args.visual_observation,
                )
                result.update(
                    lighting_context=asdict(light),
                    lighting_context_exact=None,
                    lighting_readback=None,
                    lighting_work=None,
                )
            result["seed"] = scoring_binding[0] if scoring_binding is not None else args.train_seed
            result["context"] = context
            result["explicit_reset"] = (
                f"seed{scoring_binding[0] if scoring_binding is not None else args.train_seed}_if_construction_reaches_port_reset"
            )
            if args.visual_observation:
                result["observation_recipe"] = asdict(
                    pick_cube_rgb_recipe() if camera is None else camera_context_recipe(camera)
                )
            if actor is not None and learner is not None:
                raise ValueError("actor and learner rollout sources are mutually exclusive")
            if args.corrective_queries:
                if learner is None or actor is not None or observation_clock is not None:
                    raise ValueError(
                        "corrective queries require an explicit learner without actor/clock"
                    )
                corrections = (
                    PickCubeCorrectionQuery(
                        args.train_seed,
                        readiness_corrections=True,
                        observed_grasp_corrections=True,
                        **({"goal_consistent_corrections": True} if goal_consistent_mode else {}),
                    )
                    if observed_grasp_mode
                    else PickCubeCorrectionQuery(args.train_seed, readiness_corrections=True)
                    if readiness_mode
                    else PickCubeCorrectionQuery(args.train_seed)
                )
            if observation_clock is not None:
                clock = clocks._clock(observation_clock)
                signature = (clock.horizon, clock.elapsed)
                result["observation_clock"] = asdict(clock)
                if (learner is None and actor is None) or clock.horizon != HORIZON:
                    raise ValueError(
                        "clock rollout requires a TRAIN learner and existing horizon50"
                    )
            if learner is not None:
                pinned_history_owner = None
                if history_encoding is not None or observed_grasp_mode:
                    pinned_history_owner = (
                        validate_pc_snapshot(learner._state, learner._policy)
                        if type(learner) is NumpyContinuousPCLearner
                        else validate_nearest_snapshot(learner._state, learner._policy)
                        if type(learner) is NumpyNearestContinuousBCLearner
                        else validate_bc_snapshot(learner._state, learner._policy)
                    )
                original = _prepare_learner(learner)
                if pinned_history_owner is not None and not _same_state(
                    pinned_history_owner, original
                ):
                    raise ValueError("action history owner changed during admission")

                if history_encoding is not None:
                    base = _base_policy(original)
                    if action_decoding is not None and base.binary_gripper_axis is not None:
                        raise ValueError("binary gripper execution currently requires common MSE")
                    if not _history_policy_matches(base, history_encoding.recipe):
                        raise ValueError(
                            "action history serving requires fixed zero-mean unit-scale47D policy"
                        )

                if (
                    projection is not None
                    and _base_policy(original).spec.observation_dim != projection.feature_dim
                ):
                    raise ValueError(
                        "projected policy dimensions differ from the observation recipe"
                    )
                forked = learner.fork()
                if type(original) is ContinuousPCSnapshot:
                    if forked is learner or type(forked) is not NumpyContinuousPCLearner:
                        raise ValueError("continuous PC rollout requires a separate exact PC fork")
                    _check_frozen(learner, forked, original, result)
                model = forked
                if history_encoding is not None or observed_grasp_mode:
                    _check_state_projection_frozen(learner, model, pinned_history_owner, result)
                result["state_guard_scope"] = "private_fork_and_original_learner"
            elif actor is not None:
                model = _prepare_actor(actor)
                original = model.state
                if (
                    projection is not None
                    and _base_policy(original).spec.observation_dim != projection.feature_dim
                ):
                    raise ValueError(
                        "public-vector21 actor must declare exactly21 input coordinates"
                    )
                result["policy"] = (
                    "frozen_promotable_nearest_continuous_bc"
                    if type(model) is PromotableRobotBinding
                    else "frozen_stable_nearest_continuous_bc"
                )
                result["state_guard_scope"] = (
                    "actual_promotable_actor_no_private_fork"
                    if type(model) is PromotableRobotBinding
                    else "actual_serving_actor_no_private_fork"
                )
                if type(model) is PromotableRobotBinding:
                    result["actor_generation"] = model.snapshot.generation
                result["actor_version"] = model.version
                result["actor_distance_count_source"] = (
                    "derived_from_exact_unchanged_typed_actor_and_query_shapes"
                )
                _check_actor_frozen(model, result)
            if original is not None:
                result["learner_policy"] = asdict(original.policy)
                result["source_updates"] = original.completed_updates
                result["source_training_rows"] = original.training_examples
                if type(original) is ContinuousPCSnapshot:
                    result["policy"] = "frozen_continuous_pc"
                    result["source_update_definition"] = CONTINUOUS_PC_ENERGY_ID
                    result["source_wake_updates"] = (
                        None if original.wake_state is None else original.wake_state.wake_updates
                    )
                    result["continuous_parameter_momentum_numeric_bytes_per_owner"] = sum(
                        value.nbytes for value in original.parameters + original.velocity
                    )
                    result["continuous_wake_numeric_bytes_per_owner"] = (
                        0
                        if original.wake_state is None
                        else original.wake_state.chemical_state.nbytes
                        + original.wake_state.plasticity_state.nbytes
                    )
                    base = original.policy.base
                    pc_dense_per_prediction = (
                        base.spec.observation_dim * base.hidden_dim
                        + base.hidden_dim * len(base.spec.action_units)
                    )
                    result["continuous_prediction_math"] = (
                        "feedforward_without_latent_inference_or_updates"
                    )
                if isinstance(original, NearestContinuousBCSnapshot):
                    if actor is None:
                        result["policy"] = "frozen_nearest_continuous_bc"
                    result["source_update_definition"] = NEAREST_UPDATE
                    result["nearest_prototype_rows"] = original.training_examples
                    result["nearest_prototype_numeric_bytes"] = (
                        original.prototype_features.nbytes + original.prototype_actions.nbytes
                    )
            result["versions"] = {
                name: version(name)
                for name in ("mani_skill", "sapien", "gymnasium", "torch", "numpy")
            }
            if roll_in is not None:
                result["policy"] = "mixed_readiness_teacher_and_frozen_student"
            if action_decoding is not None:
                action_decoding.check(result["action_decoding"], ())
                _check_state_projection_frozen(learner, model, original, result)
            if scoring_binding is None:
                result["creation_attempts"] += 1
            visual_options: dict[str, Any] = {}
            if args.visual_observation:
                visual_options["visual_observation"] = True
            if camera is not None:
                visual_options["camera_context"] = camera
            if mass is not None:
                visual_options["mass_context"] = mass
            if light is not None:
                visual_options["lighting_context"] = light
            if joint_monitoring:
                visual_options["monitor_joint_positions"] = True
            _check_completion_request(completion_request)
            if scoring_binding is not None:
                # SDK creation follows every opaque model/package observer.
                check_scoring_request()
                assert history_encoding is not None
                history_encoding.check(result.get("action_history"))
                _check_guard_native_frozen(learner, model, original)
                if (
                    type(result.get("guard_seed" if guard_seed is not None else "probe_test_seed"))
                    is not int
                    or result.get("guard_seed" if guard_seed is not None else "probe_test_seed")
                    != scoring_binding[0]
                    or result.get("scoring_only") is not True
                    or result.get("role") != scoring_role
                    or type(result.get("seed")) is not int
                    or result.get("seed") != scoring_binding[0]
                    or result.get("context") != scoring_binding[1]
                ):
                    raise ValueError("original scoring-only guard request metadata changed")
                result["creation_attempts"] += 1
            port = ManiSkillPickCubeEpisode(
                scoring_binding[2]
                if scoring_binding is not None
                else f"pick-cube-smoke-{args.train_seed}",
                context,
                role=cast(str, scoring_role) if scoring_binding is not None else "train",
                seed=scoring_binding[0] if scoring_binding is not None else args.train_seed,
                **visual_options,
            )
            result["creation_returns"] += 1
            if joint_monitoring:
                joint_limits = _read_joint_position_report(port, result)
                if model is not None:
                    _check_frozen(learner, model, original, result)
            if light is not None:
                lighting_readback = _check_lighting_port(port, light, result)
                if model is not None:
                    _check_frozen(learner, model, original, result)
                _check_lighting_snapshot(port, light, result, lighting_readback)
                _rollout(
                    port,
                    learner,
                    model,
                    original,
                    result,
                    clock,
                    signature,
                    lighting_context=light,
                    lighting_readback=lighting_readback,
                )
            elif mass is not None:
                mass_readback = _check_mass_port(port, mass, result)
                if model is not None:
                    _check_frozen(learner, model, original, result)
                _rollout(
                    port,
                    learner,
                    model,
                    original,
                    result,
                    clock,
                    signature,
                    projection=projection,
                    mass_context=mass,
                    mass_readback=mass_readback,
                    history_encoding=history_encoding,
                )
            elif history_encoding is not None:
                _rollout(
                    port,
                    learner,
                    model,
                    original,
                    result,
                    history_encoding=history_encoding,
                    corrections=corrections,
                    roll_in=roll_in,
                    **observed_rollout_options,
                )
            elif projection is not None:
                if corrections is None:
                    _rollout(port, learner, model, original, result, projection=projection)
                else:
                    _rollout(
                        port,
                        learner,
                        model,
                        original,
                        result,
                        corrections=corrections,
                        projection=projection,
                        roll_in=roll_in,
                        **observed_rollout_options,
                    )
            elif corrections is None:
                _rollout(port, learner, model, original, result, clock, signature)
            else:
                _rollout(
                    port,
                    learner,
                    model,
                    original,
                    result,
                    clock,
                    signature,
                    corrections=corrections,
                    **observed_rollout_options,
                )
            if (history_encoding is not None or observed_grasp_mode) and result["complete"]:
                history_episode_evidence = json.dumps(
                    result["episode"], sort_keys=True, allow_nan=False
                )
                if corrections is not None:
                    # finish validates the queried port while the SDK is live.
                    raw_episode = ContinuousEpisode(
                        port._episode_id,
                        port._context,
                        port._role,
                        port._spec,
                        tuple(port._records),
                    )
                    history_finished_batch = corrections.finish(raw_episode)
                    history_finished_batch_evidence = json.dumps(
                        asdict(history_finished_batch), sort_keys=True, allow_nan=False
                    )
        except Exception as error:
            result["complete"] = False
            result["error"] = f"{type(error).__name__}: {error}"
            result["traceback"] = traceback.format_exc()
            if joint_monitoring and hasattr(error, "joint_position_work"):
                try:
                    result["joint_position_work"] = asdict(
                        validate_joint_position_work(error.joint_position_work)
                    )
                except Exception as metadata_error:
                    result["joint_position_error"] = str(metadata_error)
            if light is not None and hasattr(error, "lighting_work"):
                try:
                    result["lighting_work"] = asdict(validate_lighting_work(error.lighting_work))
                    if getattr(error, "lighting_readback", None) is not None:
                        result["lighting_readback"] = asdict(
                            validate_lighting_readback(getattr(error, "lighting_readback"), light)
                        )
                    result["lighting_failed_source_rgb"] = getattr(
                        error, "lighting_source_ambient_rgb", None
                    )
                    result["lighting_failed_requested_rgb"] = getattr(
                        error, "lighting_requested_ambient_rgb", None
                    )
                except Exception as metadata_error:
                    result["lighting_error"] = str(metadata_error)
        finally:
            if port is not None:
                try:
                    port.close()
                except Exception as error:
                    result.update(complete=False, cleanup_error=str(error))
                result["work"] = asdict(port.work)
                if args.visual_observation:
                    try:
                        visual_work = port.visual_work
                        result["visual_work"] = (
                            asdict(visual_work) if visual_work is not None else None
                        )
                        _check_visual_recipe(port, result)
                    except Exception as error:
                        result.update(complete=False, visual_recipe_error=str(error))
            if model is not None:
                try:
                    _check_frozen(learner, model, original, result)
                except Exception as error:
                    result.update(complete=False, state_error=str(error))
            if projection is not None and "observation_projection" in result:
                try:
                    _check_projection(projection, result)
                    if model is not None:
                        _check_frozen(learner, model, original, result)
                    if port is not None:
                        _check_state_projection_port(port, projection, result)
                except Exception as error:
                    result.update(complete=False, projection_error=str(error))
            if clock is not None:
                try:
                    _check_clock(clock, signature, result)
                except Exception as error:
                    result.update(complete=False, clock_error=str(error))
            if corrections is not None:
                result["corrections"] = corrections.report()
                if result["complete"]:
                    try:
                        if port is None:
                            raise ValueError("completed corrections require an executed port")
                        if history_encoding is not None or observed_grasp_mode:
                            if (
                                history_finished_batch is None
                                or json.dumps(
                                    asdict(history_finished_batch), sort_keys=True, allow_nan=False
                                )
                                != history_finished_batch_evidence
                            ):
                                raise ValueError(
                                    "original history query batch changed across close"
                                )
                            batch = history_finished_batch
                        else:
                            batch = corrections.finish(port.episode())
                        _check_correction_mode(
                            result["corrections"],
                            batch,
                            readiness_mode,
                            observed_grasp_mode,
                            goal_consistent_mode,
                        )
                        if asdict(batch.episode) != result["episode"]:
                            raise ValueError(
                                "released corrections differ from retained executed trace"
                            )
                        labels = [
                            dict(
                                asdict(label),
                                observed_at=label.observed_at,
                                labels_arrived_at=label.labels_arrived_at,
                            )
                            for label in batch.labels
                        ]
                        if labels != result["corrections"]["labels"]:
                            raise ValueError(
                                "released correction labels differ from query receipts"
                            )
                        batch_metadata = dict(
                            train_seed=batch.train_seed,
                            episode_id=batch.episode.episode_id,
                            rows=len(batch.labels),
                            labels_arrived_at=batch.labels[-1].labels_arrived_at,
                        )
                        # finish/episode/metadata reads may run trusted callbacks;
                        # revalidate after them before releasing usable labels.
                        _check_visual_recipe(port, result)
                        _check_projection(projection, result)
                        _check_frozen(learner, model, original, result)
                        _check_correction_mode(
                            result["corrections"],
                            batch,
                            readiness_mode,
                            observed_grasp_mode,
                            goal_consistent_mode,
                        )
                        result["corrections"]["batch"] = batch_metadata
                        released_batch = batch
                        if (
                            type(projection) is state_quaternions.PickCubeStateQuaternionProjection
                            or history_encoding is not None
                            or observed_grasp_mode
                        ):
                            released_batch_evidence = json.dumps(asdict(batch), sort_keys=True)
                    except Exception as error:
                        result.update(complete=False, correction_error=str(error))
            if mass is not None and port is not None:
                try:
                    if mass_readback is None:
                        raise ValueError("mass rollout never admitted an original readback")
                    _check_mass_port(port, mass, result, mass_readback)
                    _check_visual_recipe(port, result)
                    _check_projection(projection, result)
                    _check_mass_port(port, mass, result, mass_readback)
                    if model is not None:
                        _check_frozen(learner, model, original, result)
                    ManiSkillPickCubeEpisode.check_mass_metadata(
                        port,
                        mass,
                        mass_readback,
                        validate_mass_work(ManiSkillMassWork(**result["mass_work"])),
                    )
                except Exception as error:
                    result.update(complete=False, mass_context_exact=None, mass_error=str(error))
            if light is not None and port is not None:
                try:
                    if lighting_readback is None:
                        raise ValueError("lighting rollout never admitted an original readback")
                    _check_lighting_port(port, light, result, lighting_readback)
                    _check_visual_recipe(port, result)
                    _check_projection(projection, result)
                    _check_lighting_port(port, light, result, lighting_readback)
                    if model is not None:
                        _check_frozen(learner, model, original, result)
                    _check_lighting_snapshot(port, light, result, lighting_readback)
                except Exception as error:
                    result.update(
                        complete=False, lighting_context_exact=None, lighting_error=str(error)
                    )
            if joint_monitoring and port is not None:
                try:
                    if joint_limits is None:
                        raise ValueError("joint rollout never admitted an initial limit snapshot")
                    _read_joint_position_report(
                        port, result, initial_limits=joint_limits, verify=result["complete"]
                    )
                except Exception as error:
                    result.update(
                        complete=False, joint_position_exact=None, joint_position_error=str(error)
                    )
                # Even a failing telemetry property may run a callback. Observe
                # frozen owners after it before releasing any paired corrections.
                if model is not None:
                    try:
                        _check_frozen(learner, model, original, result)
                    except Exception as error:
                        result.update(
                            complete=False, joint_position_exact=None, state_error=str(error)
                        )
                # Reporting is the last external callback boundary. Reuse the
                # existing pure context checks; do not start another native read.
                if mass is not None:
                    try:
                        if mass_readback is None:
                            raise ValueError("mass rollout never admitted an original readback")
                        ManiSkillPickCubeEpisode.check_mass_metadata(
                            port,
                            mass,
                            mass_readback,
                            validate_mass_work(ManiSkillMassWork(**result["mass_work"])),
                        )
                    except Exception as error:
                        result.update(
                            complete=False,
                            joint_position_exact=None,
                            mass_context_exact=None,
                            mass_error=str(error),
                        )
                if light is not None:
                    try:
                        _check_lighting_snapshot(port, light, result, lighting_readback)
                    except Exception as error:
                        result.update(
                            complete=False,
                            joint_position_exact=None,
                            lighting_context_exact=None,
                            lighting_error=str(error),
                        )
            if pc_dense_per_prediction is not None:
                # Use admitted dimensions, not a possibly corrupted late snapshot.
                # This estimate excludes snapshot/gate validation, copies and RSS.
                result["derived_returned_prediction_dense_multiply_accumulates"] = (
                    result["prediction_returns"] * pc_dense_per_prediction
                )
            if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
                result["elapsed_seconds"] = None
                result["projection_exact"] = None
                try:
                    result["elapsed_seconds"] = time.monotonic() - start
                    if roll_in is not None:
                        try:
                            _check_roll_in_result(roll_in, result)
                            if result["complete"]:
                                if released_batch is None:
                                    raise ValueError(
                                        "completed roll-in requires released raw labels"
                                    )
                                roll_in.verify_episode(
                                    released_batch.episode, released_batch.labels
                                )
                        finally:
                            result["readiness_roll_in"] = roll_in.report()
                    if released_batch is not None:
                        if corrections is None:
                            raise ValueError("released readiness batch lost its query owner")
                        if (
                            json.dumps(asdict(released_batch), sort_keys=True)
                            != released_batch_evidence
                        ):
                            raise ValueError("released raw readiness correction evidence changed")
                        _check_correction_mode(
                            result["corrections"],
                            released_batch,
                            True,
                            observed_grasp_mode,
                            goal_consistent_mode,
                        )
                        if asdict(released_batch.episode) != result["episode"]:
                            raise ValueError("canonical query raw trace changed before release")
                        corrections._check_readiness_binding()
                        if corrections.report()["labels"] != result["corrections"]["labels"]:
                            raise ValueError("canonical query receipts changed before release")
                    if port is not None and model is not None:
                        _check_state_projection_frozen(learner, model, original, result)
                        # These final checks read no virtual properties or native
                        # sensors. All timing/telemetry/owner callbacks precede them.
                        _check_projection(projection, result)
                        _check_state_projection_port(port, projection, result)
                        if roll_in is not None:
                            _check_roll_in_result(roll_in, result)
                except Exception as error:
                    result.update(
                        complete=False, projection_exact=None, projection_error=str(error)
                    )
            elif history_encoding is not None:
                try:
                    result["elapsed_seconds"] = time.monotonic() - start
                except Exception as error:
                    result.update(
                        complete=False, elapsed_seconds=None, action_history_error=str(error)
                    )
            else:
                result["elapsed_seconds"] = time.monotonic() - start
            if history_encoding is not None:
                result["action_history_exact"] = None
                try:
                    if roll_in is not None:
                        _check_roll_in_result(roll_in, result)
                        if result["complete"]:
                            if released_batch is None:
                                raise ValueError(
                                    "completed history roll-in requires original raw labels"
                                )
                            roll_in.verify_episode(released_batch.episode, released_batch.labels)
                    if released_batch is not None:
                        if (
                            corrections is None
                            or json.dumps(asdict(released_batch), sort_keys=True)
                            != released_batch_evidence
                        ):
                            raise ValueError("released history raw readiness evidence changed")
                        _check_correction_mode(
                            result["corrections"],
                            released_batch,
                            True,
                            observed_grasp_mode,
                            goal_consistent_mode,
                        )
                        if asdict(released_batch.episode) != result.get("episode"):
                            raise ValueError("history correction raw trace changed before release")
                        corrections._check_readiness_binding()
                        if corrections.report()["labels"] != result["corrections"]["labels"]:
                            raise ValueError(
                                "history correction query receipts changed before release"
                            )

                    if port is not None and model is not None:
                        _check_state_projection_frozen(learner, model, original, result)
                        _check_action_history_port(
                            port, history_encoding, result, **history_binding_options
                        )
                    else:
                        history_encoding.check(result.get("action_history"))
                    if result["complete"]:
                        if port is None:
                            raise ValueError("completed action history requires a native episode")
                        # The SDK has already closed. Reuse the owned raw records;
                        # public episode() may read a robot disposed by close().
                        expected_episode = ContinuousEpisode(
                            port._episode_id,
                            port._context,
                            port._role,
                            port._spec,
                            tuple(port._records),
                        )
                        history_encoding.verify_episode(expected_episode)
                        _check_state_projection_frozen(learner, model, original, result)
                        _check_action_history_port(
                            port, history_encoding, result, **history_binding_options
                        )
                        if (
                            history_episode_evidence is None
                            or json.dumps(result.get("episode"), sort_keys=True, allow_nan=False)
                            != history_episode_evidence
                            or json.dumps(asdict(expected_episode), sort_keys=True, allow_nan=False)
                            != history_episode_evidence
                        ):
                            raise ValueError(
                                "released raw trace differs from executed action history"
                            )
                    if port is not None:
                        # Final pure composition after history/clock callbacks, with no live IO.
                        _check_mass_history_frozen(
                            port,
                            history_encoding,
                            learner,
                            model,
                            original,
                            mass,
                            mass_readback,
                            result,
                        )
                    result["action_history_exact"] = True if result["complete"] else None
                except Exception as error:
                    result.update(
                        complete=False, action_history_exact=None, action_history_error=str(error)
                    )
            if observed_grasp_mode:
                try:
                    if model is not None and original is not None:
                        _check_state_projection_frozen(learner, model, original, result)
                    if result.get("observed_grasp_corrections") is not True:
                        raise ValueError("observed grasp request metadata changed")
                    if (
                        goal_consistent_mode
                        and result.get("goal_consistent_corrections") is not True
                    ):
                        raise ValueError("goal consistent request metadata changed")
                    if result["complete"]:
                        if released_batch is None or corrections is None or port is None:
                            raise ValueError("completed observed grasp collection lost its batch")
                        if (
                            json.dumps(asdict(released_batch), sort_keys=True)
                            != released_batch_evidence
                        ):
                            raise ValueError("released observed grasp evidence changed")
                        _check_correction_mode(
                            result["corrections"], released_batch, True, True, goal_consistent_mode
                        )
                        corrections._check_readiness_binding()
                        observed_report = corrections.report()
                        observed_report["batch"] = dict(
                            train_seed=released_batch.train_seed,
                            episode_id=released_batch.episode.episode_id,
                            rows=len(released_batch.labels),
                            labels_arrived_at=released_batch.labels[-1].labels_arrived_at,
                        )
                        if observed_report != result["corrections"]:
                            raise ValueError("observed grasp query receipts changed before release")
                        native_episode = ContinuousEpisode(
                            port._episode_id,
                            port._context,
                            port._role,
                            port._spec,
                            tuple(port._records),
                        )
                        if asdict(native_episode) != asdict(released_batch.episode) or asdict(
                            native_episode
                        ) != result.get("episode"):
                            raise ValueError("observed grasp raw executed trajectory changed")
                except Exception as error:
                    result.update(complete=False, observed_grasp_error=str(error))
            if completion_request is not None:
                try:
                    _check_completion_request(completion_request)
                    if roll_in is not None:
                        _check_roll_in_result(roll_in, result)
                except Exception as error:
                    result.update(complete=False, teacher_completion_error=str(error))
            if action_decoding is not None:
                # Timing, telemetry and public owner callbacks have all returned.
                # Reuse owned records without another SDK observation or prediction.
                result["action_decoding_exact"] = None
                try:
                    if port is None:
                        action_decoding.check(result.get("action_decoding"), ())
                    else:
                        action_decoding.check(result.get("action_decoding"), tuple(port._records))
                        if result["complete"]:
                            action_decoding.verify_episode(
                                ContinuousEpisode(
                                    port._episode_id,
                                    port._context,
                                    port._role,
                                    port._spec,
                                    tuple(port._records),
                                ),
                                result["action_decoding"],
                            )
                    result["action_decoding_exact"] = True if result["complete"] else None
                except Exception as error:
                    result.update(complete=False, action_decoding_error=str(error))
                finally:
                    result["action_decoding"] = action_decoding.report()
            if scoring_binding is not None and port is not None and history_encoding is not None:
                try:
                    _check_action_history_port(
                        port, history_encoding, result, **history_binding_options
                    )
                    _check_guard_native_frozen(learner, model, original)
                except Exception as error:
                    result.update(complete=False, action_history_exact=None)
                    result["probe_test_error" if probe_test_seed is not None else "guard_error"] = (
                        str(error)
                    )
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return FrozenRolloutResult(result, released_batch if result["complete"] else None)


def _rollout_summary(result: FrozenRolloutResult, seed: int) -> dict[str, Any]:
    report = result.report
    summary = dict(seed=seed, complete=report["complete"])
    for key in (
        "sparse_return",
        "any_success",
        "final_success",
        "work",
        "visual_work",
        "prediction_attempts",
        "prediction_returns",
        "validated_prediction_bytes",
        "action_history",
        "action_history_exact",
        "observation_projection",
        "projection_work",
        "projection_exact",
        "behavior_scope",
        "student_only_success_available",
        "readiness_roll_in",
        "mass_context",
        "mass_readback",
        "mass_work",
        "mass_context_exact",
        "lighting_context",
        "lighting_readback",
        "lighting_work",
        "lighting_context_exact",
        "joint_position_summary",
        "joint_position_work",
        "joint_position_exact",
        "joint_position_validation_samples",
        "joint_position_validation_coordinates",
    ):
        if key in report:
            summary[key] = report[key]
    if "episode" in report:
        rows = report["episode"]["transitions"]
        summary.update(
            steps=len(rows), terminated=rows[-1]["terminated"], truncated=rows[-1]["truncated"]
        )
    if "corrections" in report:
        summary["queries"] = {
            key: value
            for key, value in report["corrections"].items()
            if key not in ("labels", "batch")
        }
    if report.get("policy") == "frozen_nearest_continuous_bc":
        summary["nearest_work"] = {
            key: report[key]
            for key in (
                "nearest_prototype_rows",
                "nearest_prototype_numeric_bytes",
                "nearest_prediction_attempts",
                "nearest_prediction_returns",
                "nearest_distance_candidate_upper_bound",
                "nearest_distance_feature_upper_bound",
                "nearest_distance_candidates_completed",
                "nearest_distance_feature_values_completed",
                "nearest_selected_indices",
            )
        }
    return summary


def _check_action_history_result(recipe, result, expected_seed, expected_state):
    config = action_history.validate_action_history_recipe(recipe)
    actual = result.get("action_history")
    if type(actual) is not dict or result.get("action_history_exact") is not True:
        raise ValueError("native readiness history is incomplete")
    metadata = actual.get("recipe")
    if type(metadata) is not dict or set(metadata) != set(asdict(config)):
        raise ValueError("native readiness history recipe changed")
    observed = action_history.validate_action_history_recipe(type(config)(**metadata))
    if observed != config:
        raise ValueError("native readiness history differs from requested recipe")
    expected_spec = dict(
        observation_dim=42,
        action_units=ACTION_UNITS,
        lower_bounds=(-1.0,) * 4,
        upper_bounds=(1.0,) * 4,
    )
    if (
        type(result.get("seed")) is not int
        or result["seed"] != expected_seed
        or result.get("context") != pick_cube_training_context(expected_seed)
        or result.get("role") != "train"
        or result.get("complete") is not True
        or any(
            result.get(key) is not True for key in ("frozen_state_exact", "original_owner_exact")
        )
        or result["episode"].get("episode_id") != f"pick-cube-smoke-{expected_seed}"
        or result["episode"].get("context") != pick_cube_training_context(expected_seed)
        or result["episode"].get("role") != "train"
        or json.dumps(result["episode"].get("spec"), sort_keys=True)
        != json.dumps(expected_spec, sort_keys=True)
        or type(result.get("source_updates")) is not int
        or result["source_updates"] != expected_state.completed_updates
        or type(result.get("source_training_rows")) is not int
        or result["source_training_rows"] != expected_state.training_examples
        or json.dumps(result.get("learner_policy"), sort_keys=True)
        != json.dumps(asdict(expected_state.policy), sort_keys=True)
    ):
        raise ValueError("history readiness return differs from requested TRAIN source/reset")
    rows = result["episode"]["transitions"]
    state = action_history.ActionHistoryState()
    for row in rows:
        state = action_history.advance_action_history(state, tuple(row["action"]))
    count = len(rows)
    expected = dict(
        recipe=asdict(config),
        state=asdict(state),
        pending=False,
        work=dict(
            encoding_attempts=count,
            encoding_returns=count,
            committed_controls=count,
            source_values_validated=42 * count,
            model_values_emitted=config.feature_dim * count,
            model_numeric_bytes_emitted=config.feature_dim * 8 * count,
        ),
    )
    if (
        not 1 <= count <= 50
        or type(result["prediction_returns"]) is not int
        or result["prediction_returns"] != count
        or json.dumps(actual, sort_keys=True, allow_nan=False)
        != json.dumps(expected, sort_keys=True, allow_nan=False)
    ):
        raise ValueError("native readiness history differs from actual executed prefix")


def _canonical_correction_metadata(batch, projection):
    """Expected declared work without projecting a second copy of the batch."""
    count = len(batch.labels)
    work = StateQuaternionCorrectionWork(
        2 * count,
        84 * count,
        84 * count,
        672 * count,
        42 * count,
        4 * count,
        336 * count,
        32 * count,
        4 * count,
        16 * count,
    )
    return dict(
        projection=asdict(projection),
        source_episode_id=batch.episode.episode_id,
        source_spec=asdict(batch.episode.spec),
        model_spec=asdict(batch.episode.spec),
        work=asdict(work),
        source_encoding="official_state42_raw",
        model_encoding=projection.definition,
    )


def run_corrective_adaptation(
    source: Learner,
    *,
    output_directory: Path,
    visual_observation: bool = True,
    observation_projection: ObservationProjection | None = None,
    readiness_corrections: bool = False,
    acquisition_only: bool = False,
    readiness_roll_in: ReadinessBehaviorPlan | None = None,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    replay_plan: ContinuousReplayPlan | None = None,
    reference_episodes: tuple[ContinuousEpisode, ...] = (),
    binary_gripper_axis: int | None = None,
    observed_grasp_corrections: bool = False,
    goal_consistent_corrections: bool = False,
    head_gauss_newton: bool = False,
) -> dict[str, Any]:
    """Collect, update privately and return a frozen candidate to TRAIN contexts.

    Native use requires a separate prospective local budget. This composes the
    existing rollout/runtime; no source update, refit, serving commit or retry.
    Explicit canonical acquisition uses17 collection/return; the default keeps
    the original19 collection followed by17 and19 frozen returns. Optional reference
    rehearsal adds separately charged native updates; its bank is already in model
    space and never reaches collection or frozen-rollout callbacks. Explicit BP
    binary axis3 preserves mixed supervision; serving actions remain soft tanh.
    """
    if type(source) not in (
        NumpyContinuousBCLearner,
        NumpyNearestContinuousBCLearner,
        NumpyContinuousPCLearner,
    ):
        raise ValueError("corrective adaptation requires an exact TRAIN MSE, PC or nearest learner")
    if type(head_gauss_newton) is not bool:
        raise ValueError("head_gauss_newton must be an exact bool")
    if head_gauss_newton:
        if type(source) is not NumpyContinuousBCLearner:
            raise ValueError("head corrections require exact BP")
        # Why: reject unsupported compositions before directory or SDK work.
        policy = source._policy
        if (
            policy.training_role != "train"
            or policy.binary_gripper_axis is not None
            or policy.spec.observation_dim > 64
            or policy.hidden_dim > 64
            or len(policy.spec.action_units) > 16
            or binary_gripper_axis is not None
        ):
            raise ValueError("head corrections require bounded TRAIN BP MSE")
    head_request = head_protocol() if head_gauss_newton else None
    head_request_evidence = json.dumps(head_request, sort_keys=True, allow_nan=False)
    mixed = binary_gripper_axis is not None
    if mixed and (
        type(binary_gripper_axis) is not int
        or binary_gripper_axis != 3
        or type(source) is not NumpyContinuousBCLearner
    ):
        raise ValueError("binary gripper corrections require exact BP and axis3")
    if type(observed_grasp_corrections) is not bool:
        raise ValueError("observed_grasp_corrections must be an exact bool")
    if type(goal_consistent_corrections) is not bool:
        raise ValueError("goal_consistent_corrections must be an exact bool")
    if goal_consistent_corrections and observed_grasp_corrections is not True:
        raise ValueError("goal consistent corrections require observed grasp readiness")
    teacher_completion = _require_teacher_completion_inputs(
        readiness_roll_in,
        source,
        action_history_recipe,
        readiness_corrections,
        observed_grasp_corrections,
        goal_consistent_corrections,
    )
    completion_plan_evidence = None
    if teacher_completion:
        assert readiness_roll_in is not None
        completion_plan_evidence = json.dumps(
            readiness_roll_in_metadata(readiness_roll_in), sort_keys=True, allow_nan=False
        )
    if observed_grasp_corrections and (
        readiness_corrections is not True
        or visual_observation is not False
        or (readiness_roll_in is not None and not teacher_completion)
    ):
        raise ValueError("observed grasp requires nonvisual readiness without roll-in")
    if action_history_recipe is not None:
        action_history_recipe = action_history.validate_action_history_recipe(action_history_recipe)
    history_recipe_evidence = (
        None
        if action_history_recipe is None
        else json.dumps(asdict(action_history_recipe), sort_keys=True, allow_nan=False)
    )
    replay_bank: tuple[ContinuousEpisode, ...] = ()
    replay_rows = replay_bytes = 0
    if replay_plan is None:
        if type(reference_episodes) is not tuple or reference_episodes:
            raise ValueError("reference episodes require an explicit replay plan")
    else:
        if type(source) is NumpyNearestContinuousBCLearner:
            raise ValueError("reference rehearsal requires a neural correction learner")
        replay_plan, replay_bank, replay_rows, replay_bytes = prepare_replay_bank(
            replay_plan, reference_episodes
        )
    rehearsal_enabled = replay_plan is not None
    native_original = None
    if (
        type(observation_projection) is state_quaternions.PickCubeStateQuaternionProjection
        or action_history_recipe is not None
        or rehearsal_enabled
        or mixed
        or observed_grasp_corrections
        or head_gauss_newton
    ):
        # Capture authority before public policy/snapshot callbacks can redefine
        # the baseline. The recipe is validated before any native work below.
        validators: dict[type, Any] = {
            NumpyContinuousBCLearner: validate_bc_snapshot,
            NumpyContinuousPCLearner: validate_pc_snapshot,
            NumpyNearestContinuousBCLearner: validate_nearest_snapshot,
        }
        native_original = validators[type(source)](source._state, source._policy)
    original = _prepare_learner(source)
    if native_original is not None:
        if not _same_state(native_original, original):
            raise ValueError("canonical correction source changed during initial observation")
        original = native_original
    base = _base_policy(original)
    if mixed and (type(base.binary_gripper_axis) is not int or base.binary_gripper_axis != 3):
        raise ValueError("requested binary gripper differs from the owned BP policy")
    if any(value.spec != base.spec for value in replay_bank):
        raise ValueError("reference replay bank differs from the frozen model spec")
    if action_history_recipe is not None:
        replay_bank = tuple(
            action_history.validate_action_history_training_episode(value, action_history_recipe)
            for value in replay_bank
        )
    if mixed:
        # Only supervision is binary: actual student commands remain soft tanh.
        for episode in replay_bank:
            continuous_bc._binary_gripper_targets(
                np.asarray([row.action for row in episode.transitions], dtype=np.float64),
                base.spec,
                3,
            )
    replay_birth = (
        json.dumps(
            dict(plan=asdict(replay_plan), bank=[asdict(value) for value in replay_bank]),
            sort_keys=True,
            allow_nan=False,
        )
        if replay_plan is not None
        else None
    )
    if (
        not mixed
        and type(source) is not NumpyNearestContinuousBCLearner
        and base.binary_gripper_axis is not None
    ) or (
        base.observation_standardization is None
        and not action_history.is_taskspace_recipe(action_history_recipe)
    ):
        raise ValueError(
            "corrective adaptation requires default MSE with frozen statistics or fixed taskspace scaling"
        )
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    if type(readiness_corrections) is not bool:
        raise ValueError("readiness_corrections must be an exact bool")
    if type(acquisition_only) is not bool:
        raise ValueError("acquisition_only must be an exact bool")
    history_recipe = None
    if action_history_recipe is not None:
        history_recipe = action_history.validate_action_history_recipe(action_history_recipe)
        if (
            observation_projection is not None
            or visual_observation
            or not readiness_corrections
            or type(source) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner)
            or not _history_policy_matches(base, history_recipe)
        ):
            raise ValueError(
                "history corrections require explicit47D nonvisual readiness with frozen tail scaling"
            )
    projection = None
    if observation_projection is not None:
        projection = _validate_projection(observation_projection)
        if type(projection) is state_quaternions.PickCubeStateQuaternionProjection:
            if visual_observation or not readiness_corrections or base.spec.observation_dim != 42:
                raise ValueError("canonical corrections require nonvisual42 readiness labels")
        elif not visual_observation or base.spec.observation_dim != projection.feature_dim:
            raise ValueError("projected corrective adaptation requires an explicit visual21 policy")
    elif visual_observation and base.spec.observation_dim != pick_cube_rgb_recipe().feature_dim:
        raise ValueError("visual corrective adaptation requires the fixed789-feature policy")
    # Both selected views use the existing native ownership/receipt guards.
    canonical = (
        type(projection) is state_quaternions.PickCubeStateQuaternionProjection
        or history_recipe is not None
    )
    roll_in_plan = None
    if readiness_roll_in is not None:
        roll_in_plan = validate_readiness_roll_in_plan(readiness_roll_in)
        if not canonical:
            raise ValueError("readiness roll-in correction requires canonical readiness inputs")
        if (
            teacher_completion
            and json.dumps(
                readiness_roll_in_metadata(roll_in_plan), sort_keys=True, allow_nan=False
            )
            != completion_plan_evidence
        ):
            raise ValueError("original teacher completion plan changed during admission")
    if acquisition_only and not canonical:
        raise ValueError("acquisition_only requires the explicit canonical readiness recipe")
    collection_seed = 17 if acquisition_only else 19
    return_seeds = (17,) if acquisition_only else (17, 19)
    preserve_evidence = (
        canonical or rehearsal_enabled or mixed or observed_grasp_corrections or head_gauss_newton
    )
    preserve_update = rehearsal_enabled or mixed or observed_grasp_corrections or head_gauss_newton
    check_owners = (
        _check_canonical_corrective_owners if preserve_evidence else _check_corrective_owners
    )
    if (preserve_evidence or type(original) is ContinuousPCSnapshot) and not check_owners(
        source, original
    ):
        raise ValueError("continuous PC correction source changed during admission")
    output_directory.mkdir(parents=False, exist_ok=False)
    report: dict[str, Any] = dict(
        complete=False,
        protocol_id="corrective_robot_train_17_acquisition_v1"
        if acquisition_only
        else "corrective_robot_train_19_17_19_v1",
        source_updates=original.completed_updates,
        source_training_rows=original.training_examples,
        collection=None,
        update=None,
        rollouts=[],
        source_state_exact=None,
        guard_reads=0,
        probe_reads=0,
        final_reads=0,
        promotion_calls=0,
    )
    projection_options: dict[str, Any] = {}
    if projection is not None:
        projection_options["observation_projection"] = projection
        report["observation_projection"] = asdict(projection)
    collection_options: dict[str, Any] = {}
    if readiness_corrections:
        collection_options["readiness_corrections"] = True
        report["teacher_definition"] = (
            GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
            if goal_consistent_corrections
            else OBSERVED_GRASP_DEFINITION
            if observed_grasp_corrections
            else READINESS_TEACHER_DEFINITION
        )
    if observed_grasp_corrections:
        collection_options["observed_grasp_corrections"] = True
        report["observed_grasp_corrections"] = True
    if goal_consistent_corrections:
        collection_options["goal_consistent_corrections"] = True
        report["goal_consistent_corrections"] = True
    if roll_in_plan is not None:
        collection_options["readiness_roll_in"] = roll_in_plan
        report["readiness_roll_in_plan"] = asdict(roll_in_plan)
        report["collection_behavior_scope"] = _readiness_behavior_scope(roll_in_plan)
        report["return_behavior_scope"] = "frozen_student_only"
        report["protocol_id"] = (
            "corrective_robot_mixed_train_17_acquisition_v1"
            if acquisition_only
            else "corrective_robot_mixed_train_19_17_19_v1"
        )
        if teacher_completion:
            report["protocol_id"] = (
                "corrective_robot_student_prefix_teacher_completion_train_17_acquisition_v1"
                if acquisition_only
                else "corrective_robot_student_prefix_teacher_completion_train_19_17_19_v1"
            )
    if acquisition_only:
        report["acquisition_only"] = True
        report["collection_seed"] = collection_seed
        report["return_seeds"] = list(return_seeds)
    completion_protocol_id = report["protocol_id"] if teacher_completion else None
    started = time.monotonic()
    planned_updates = 1 if type(source) is NumpyNearestContinuousBCLearner else 8
    planned_replays = 0 if replay_plan is None else planned_updates // replay_plan.every_updates
    update_options: dict[str, Any] = {}
    if head_gauss_newton:
        update_options["head_gauss_newton"] = True
        report["head_gauss_newton"] = deepcopy(head_request)
    if mixed:
        update_options["binary_gripper_axis"] = 3
        report.update(binary_gripper_axis=3, training_objective=continuous_bc.MIXED_OBJECTIVE)
    if rehearsal_enabled:
        assert replay_plan is not None
        update_options.update(replay_plan=replay_plan, reference_episodes=replay_bank)
        report["reference_rehearsal"] = dict(
            plan=asdict(replay_plan),
            bank_episode_ids=[v.episode_id for v in replay_bank],
            bank_contexts=[v.context for v in replay_bank],
            bank_rows=replay_rows,
            retained_numeric_bytes=replay_bytes,
            planned_events=planned_replays,
            planned_exposures=planned_replays * replay_rows,
            scope="additional optimization and TRAIN exposure; not matched compute",
        )
    rehearsal_metadata = json.dumps(report.get("reference_rehearsal"), sort_keys=True)
    candidate = candidate_state = None
    batch = None
    raw_batch_evidence = None
    # Bounded to this recipe's collection, update and at most two return summaries.
    # Why: a later observer must not silently replace an earlier paid success.
    prior_evidence: list[tuple[str, int | None, str]] = []
    prior_raw_evidence: list[tuple[dict[str, Any], str]] = []

    def evidence(value):
        return json.dumps(value, sort_keys=True, allow_nan=False)

    if history_recipe is not None:
        projection_options["action_history_recipe"] = history_recipe
        report["action_history_recipe"] = asdict(history_recipe)

    def check_prior():
        if head_gauss_newton and (
            evidence(report.get("head_gauss_newton")) != head_request_evidence
            or update_options.get("head_gauss_newton") is not True
        ):
            raise ValueError("original head correction request changed")
        if teacher_completion:
            assert readiness_roll_in is not None and roll_in_plan is not None
            if (
                evidence(readiness_roll_in_metadata(readiness_roll_in)) != completion_plan_evidence
                or evidence(readiness_roll_in_metadata(roll_in_plan)) != completion_plan_evidence
                or evidence(report.get("readiness_roll_in_plan")) != evidence(asdict(roll_in_plan))
                or report.get("collection_behavior_scope")
                != _readiness_behavior_scope(roll_in_plan)
                or report.get("return_behavior_scope") != "frozen_student_only"
                or report.get("protocol_id") != completion_protocol_id
            ):
                raise ValueError("original teacher completion plan or mixed scope changed")
        if goal_consistent_corrections and report.get("goal_consistent_corrections") is not True:
            raise ValueError("original goal consistent correction request changed")
        if observed_grasp_corrections and (
            report.get("observed_grasp_corrections") is not True
            or report.get("teacher_definition")
            != (
                GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
                if goal_consistent_corrections
                else OBSERVED_GRASP_DEFINITION
            )
        ):
            raise ValueError("original observed grasp correction request changed")
        if mixed and (
            type(report.get("binary_gripper_axis")) is not int
            or report["binary_gripper_axis"] != 3
            or report.get("training_objective") != continuous_bc.MIXED_OBJECTIVE
        ):
            raise ValueError("selected binary correction request metadata changed")
        if rehearsal_enabled:
            assert replay_plan is not None
            observed_replay = json.dumps(
                dict(plan=asdict(replay_plan), bank=[asdict(value) for value in replay_bank]),
                sort_keys=True,
                allow_nan=False,
            )
            if (
                observed_replay != replay_birth
                or json.dumps(report.get("reference_rehearsal"), sort_keys=True)
                != rehearsal_metadata
            ):
                raise ValueError("original reference rehearsal request or bank changed")
        if not preserve_evidence:
            return
        if history_recipe is not None:
            observed = report.get("action_history_recipe")
            if (
                evidence(asdict(action_history.validate_action_history_recipe(history_recipe)))
                != history_recipe_evidence
                or evidence(observed) != history_recipe_evidence
            ):
                raise ValueError("history correction original requested recipe changed")
        elif canonical:
            if projection is None:
                raise ValueError("canonical correction projection was removed")
            _validate_projection(projection)
            if evidence(report.get("observation_projection")) != evidence(asdict(projection)):
                raise ValueError("canonical correction recipe report changed")
        if raw_batch_evidence is not None:
            if batch is None or evidence(asdict(batch)) != raw_batch_evidence:
                raise ValueError("original raw correction batch changed")
        for key, index, expected in prior_evidence:
            current = report[key] if index is None else report[key][index]
            if evidence(current) != expected:
                raise ValueError("preceding canonical correction evidence changed")
        for raw_report, expected in prior_raw_evidence:
            if evidence(raw_report) != expected:
                raise ValueError("preceding canonical native report changed")

    try:
        check_prior()
        if (preserve_evidence or type(original) is ContinuousPCSnapshot) and not check_owners(
            source, original
        ):
            raise ValueError("continuous PC correction source changed before collection")
        collected = run_frozen_rollout(
            output_directory / "collection.json",
            learner=source,
            train_seed=collection_seed,
            visual_observation=visual_observation,
            corrective_queries=True,
            **projection_options,
            **collection_options,
        )
        report["collection"] = _rollout_summary(collected, collection_seed)
        if (
            not check_owners(source, original)
            or collected.report["complete"] is not True
            or collected.corrections is None
        ):
            raise ValueError("source collection incomplete or original learner changed")
        batch = collected.corrections
        _check_correction_mode(
            collected.report.get("corrections"),
            batch,
            readiness_corrections,
            observed_grasp_corrections,
            goal_consistent_corrections,
        )
        if (
            batch.train_seed != collection_seed
            or asdict(batch.episode) != collected.report["episode"]
        ):
            raise ValueError("released corrections differ from actual collected TRAIN trace")
        _check_correction_mode(
            collected.report.get("corrections"),
            batch,
            readiness_corrections,
            observed_grasp_corrections,
            goal_consistent_corrections,
        )
        if canonical:
            if history_recipe is not None:
                _check_action_history_result(
                    history_recipe, collected.report, collection_seed, original
                )
            else:
                if collected.report.get("projection_exact") is not True:
                    raise ValueError("canonical collection lost its exact projection")
                _check_projection(projection, collected.report)
            if teacher_completion:
                _check_teacher_completion_collection(roll_in_plan, collected.report, batch)
            elif roll_in_plan is not None:
                assert type(roll_in_plan) is ReadinessRollInPlan
                observed_behavior = collected.report.get("readiness_roll_in")
                if (
                    type(observed_behavior) is not dict
                    or observed_behavior.get("definition") != READINESS_ROLL_IN_DEFINITION
                    or type(observed_behavior.get("teacher_steps")) is not int
                    or observed_behavior["teacher_steps"] != roll_in_plan.teacher_steps
                    or collected.report.get("behavior_scope")
                    != "teacher_prefix_then_frozen_student"
                    or collected.report.get("student_only_success_available") is not False
                    or collected.report.get("policy")
                    != "mixed_readiness_teacher_and_frozen_student"
                ):
                    raise ValueError(
                        "collected behavior differs from the requested readiness roll-in"
                    )
        if preserve_evidence:
            raw_batch_evidence = evidence(asdict(batch))
            prior_evidence.append(("collection", None, evidence(report["collection"])))
            prior_raw_evidence.append((collected.report, evidence(collected.report)))
        if (preserve_evidence or type(original) is ContinuousPCSnapshot) and not check_owners(
            source, original
        ):
            raise ValueError("continuous PC correction source changed before private update")
        check_prior()
        updated = run_continuous_correction_update(
            source,
            batch,
            now=batch.labels[-1].labels_arrived_at,
            **projection_options,
            **update_options,
        )
        # Preserve paid work and native candidate authority before later observers.
        returned_report = updated.report
        report["update"] = deepcopy(returned_report) if preserve_update else returned_report
        if preserve_update:
            prior_raw_evidence.append((returned_report, evidence(report["update"])))
        returned_candidate_state = None
        if (
            preserve_update
            and type(updated.candidate) is type(source)
            and updated.candidate is not source
        ):
            returned_candidate_state = validators[type(source)](
                updated.candidate._state, updated.candidate._policy
            )
        with (output_directory / "update.json").open("x", encoding="utf8") as handle:
            json.dump(report["update"], handle, allow_nan=False)
            handle.write("\n")
        if (
            updated.report["complete"] is not True
            or updated.candidate is None
            or not check_owners(source, original)
        ):
            raise ValueError("private correction update incomplete or original learner changed")
        if head_gauss_newton and (
            updated.report.get("protocol_id")
            != (
                "continuous_head_gauss_newton_reference_correction_8_v1"
                if rehearsal_enabled
                else "continuous_head_gauss_newton_correction_8_v1"
            )
            or evidence(updated.report.get("head_gauss_newton")) != head_request_evidence
            or type(updated.report.get("native_updates")) is not list
            or len(updated.report["native_updates"]) != planned_updates + planned_replays
        ):
            raise ValueError("private correction lost the selected head update evidence")
        if readiness_corrections and (
            updated.report.get("provenance")
            != (
                GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
                if goal_consistent_corrections
                else OBSERVED_GRASP_PROVENANCE
                if observed_grasp_corrections
                else READINESS_CORRECTION_PROVENANCE
            )
            or updated.report.get("teacher_definition")
            != (
                GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
                if goal_consistent_corrections
                else OBSERVED_GRASP_DEFINITION
                if observed_grasp_corrections
                else READINESS_TEACHER_DEFINITION
            )
        ):
            raise ValueError("private readiness update report changed its teacher provenance")
        if mixed and (
            type(updated.report.get("binary_gripper_axis")) is not int
            or updated.report["binary_gripper_axis"] != 3
            or updated.report.get("training_objective") != continuous_bc.MIXED_OBJECTIVE
        ):
            raise ValueError("private correction changed the selected mixed objective")
        if observed_grasp_corrections:
            feedback = updated.report.get("observed_grasp")
            if (
                type(feedback) is not list
                or any(type(value) is not bool for value in feedback)
                or tuple(feedback) != cast(ObservedGraspCorrectionBatch, batch).observed_grasp
            ):
                raise ValueError("private update changed original observed grasp feedback")
        check_prior()
        if canonical:
            expected_projection = (
                action_history_correction_metadata(
                    cast(ReadinessCorrectionBatch, batch), history_recipe
                )
                if history_recipe is not None
                else _canonical_correction_metadata(batch, projection)
            )
            if evidence(updated.report.get("input_projection")) != evidence(expected_projection):
                raise ValueError("canonical correction update lost raw/model input provenance")
        if rehearsal_enabled:
            assert replay_plan is not None
            paid_replay = updated.report.get("replay", {})
            if (
                evidence(paid_replay.get("plan")) != evidence(asdict(replay_plan))
                or paid_replay.get("bank_episode_ids") != [v.episode_id for v in replay_bank]
                or paid_replay.get("bank_contexts") != [v.context for v in replay_bank]
                or type(paid_replay.get("retained_numeric_bytes")) is not int
                or paid_replay.get("retained_numeric_bytes") != replay_bytes
                or any(
                    type(paid_replay.get(key)) is not int
                    for key in (
                        "bank_rows",
                        "planned_events",
                        "committed_updates",
                        "charged_exposures",
                    )
                )
                or paid_replay.get("planned_events") != planned_replays
                or paid_replay.get("bank_rows") != replay_rows
                or paid_replay.get("committed_updates") != planned_replays
                or paid_replay.get("charged_exposures") != planned_replays * replay_rows
                or paid_replay.get("stopped") is not False
            ):
                raise ValueError(
                    "private correction replay differs from requested reference treatment"
                )
        if preserve_evidence:
            prior_evidence.append(("update", None, evidence(report["update"])))
        candidate = updated.candidate
        if preserve_evidence and (type(candidate) is not type(source) or candidate is source):
            raise ValueError("canonical correction requires a separate exact candidate")
        if type(original) is ContinuousPCSnapshot and (
            type(candidate) is not NumpyContinuousPCLearner or candidate is source
        ):
            raise ValueError("continuous PC correction requires a separate exact PC candidate")
        candidate_state = candidate.snapshot_state()
        if preserve_update:
            if returned_candidate_state is None or not _same_state(
                returned_candidate_state, candidate_state
            ):
                raise ValueError("reference correction candidate changed during public observation")
            candidate_state = returned_candidate_state
        if (
            candidate_state.policy != original.policy
            or type(candidate) is not type(source)
            or candidate_state.completed_updates
            != original.completed_updates + planned_updates + planned_replays
            or candidate_state.training_examples
            != original.training_examples
            + planned_updates * len(batch.labels)
            + planned_replays * replay_rows
        ):
            raise ValueError("private correction candidate does not continue original learner")
        if (preserve_evidence or type(original) is ContinuousPCSnapshot) and not check_owners(
            source, original, candidate, candidate_state
        ):
            raise ValueError("continuous PC correction owners changed during candidate admission")
        for seed in return_seeds:
            if not check_owners(source, original, candidate, candidate_state):
                raise ValueError("preceding frozen rollout changed source/candidate")
            check_prior()
            rolled = run_frozen_rollout(
                output_directory / f"updated-{seed}.json",
                learner=candidate,
                train_seed=seed,
                visual_observation=visual_observation,
                **projection_options,
            )
            report["rollouts"].append(_rollout_summary(rolled, seed))
            if roll_in_plan is not None:
                expected_policy = (
                    "frozen_continuous_pc"
                    if type(source) is NumpyContinuousPCLearner
                    else "frozen_nearest_continuous_bc"
                    if type(source) is NumpyNearestContinuousBCLearner
                    else "frozen_continuous_bc"
                )
                if rolled.report.get("policy") != expected_policy or any(
                    key in rolled.report
                    for key in (
                        "readiness_roll_in",
                        "behavior_scope",
                        "student_only_success_available",
                    )
                ):
                    raise ValueError("frozen student return unexpectedly contains mixed behavior")
            check_prior()
            if canonical:
                if history_recipe is not None:
                    _check_action_history_result(
                        history_recipe, rolled.report, seed, candidate_state
                    )
                else:
                    if rolled.report.get("projection_exact") is not True:
                        raise ValueError("canonical return lost its exact projection")
                    _check_projection(projection, rolled.report)
            if preserve_evidence:
                index = len(report["rollouts"]) - 1
                prior_evidence.append(("rollouts", index, evidence(report["rollouts"][index])))
                prior_raw_evidence.append((rolled.report, evidence(rolled.report)))
            owners_exact = False
            if rolled.report["complete"] is True:
                # Preserve the original BC/nearest getter order; PC finishes
                # both observations from native fields to catch late changes.
                owners_exact = (
                    check_owners(source, original, candidate, candidate_state)
                    if preserve_evidence or type(original) is ContinuousPCSnapshot
                    else _same_state(candidate_state, candidate.snapshot_state())
                    and _same_state(original, source.snapshot_state())
                )
            if not owners_exact:
                raise ValueError("updated frozen rollout incomplete or owner changed")
            check_prior()
        report["complete"] = True
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if preserve_evidence or type(original) is ContinuousPCSnapshot:
                # Keep paid results even if the final clock observer fails.
                report["elapsed_seconds"] = time.monotonic() - started
            if preserve_evidence:
                report["source_state_exact"] = report["candidate_state_exact"] = None
                exact = check_owners(source, original, candidate, candidate_state)
                report["source_state_exact"] = exact
                report["candidate_state_exact"] = exact if candidate is not None else None
                check_prior()
                if not exact:
                    raise ValueError("canonical correction owners changed before release")
            elif type(original) is ContinuousPCSnapshot and candidate is not None:
                report["source_state_exact"] = report["candidate_state_exact"] = None
                private_exact, source_exact = _check_pc_owners(
                    source, candidate, original, candidate_state
                )
                report["source_state_exact"] = source_exact
                report["candidate_state_exact"] = private_exact
                if not private_exact or not source_exact:
                    report.update(
                        complete=False,
                        state_error="continuous PC correction owners changed before release",
                    )
            else:
                report["source_state_exact"] = check_owners(source, original)
                if report["source_state_exact"] is not True:
                    report["complete"] = False
        except Exception as error:
            report.update(complete=False, state_error=str(error))
            report["source_state_exact"] = None if type(original) is ContinuousPCSnapshot else False
        if not preserve_evidence and type(original) is not ContinuousPCSnapshot:
            report["elapsed_seconds"] = time.monotonic() - started
        with (output_directory / "summary.json").open("x", encoding="utf8") as handle:
            json.dump(report, handle, allow_nan=False)
            handle.write("\n")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
