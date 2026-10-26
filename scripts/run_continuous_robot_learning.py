"""Run qualified TRAIN learning followed by frozen robot inference.

Inputs are qualified teacher files at the CLI or owned TRAIN episodes at the
Python boundary. Default A/B/A serves TRAIN17; the explicit joint-acquisition
profile uses fixed arrived corrections and serves TRAIN17/19/20. Outputs are
exclusive combined and native JSON reports. Reuse existing stream and robot APIs;
no promotion, model IO, tuning, probe or final evaluation. Running the CLI spends
training and simulator work and requires a separately declared local allowance.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from scripts.maniskill_continuous_smoke import FrozenRolloutResult, run_frozen_rollout
from scripts.joint_acquisition_robot_learning import (
    _paths as _joint_output_paths,
    run_joint_acquisition_rollouts as run_joint_acquisition_rollouts,
)
from scripts.joint_acquisition_inputs import read_joint_acquisition_inputs
from scripts.run_native_stream_demo import (
    _read_lighting_teacher,
    _read_readiness_teacher,
    _read_teacher,
)
from src.adapters.continuous_learnability_demo import (
    ContinuousLearnabilityResult,
    PHASE_UPDATES,
    run_continuous_learnability,
    validate_phase_updates,
    _require_stage_diagnostics,
    _readiness_checkpoint_counts,
    _require_training_packet_evidence,
)
from src.core.readiness_training_packets import prepare_readiness_packets
from src.adapters.continuous_learnability_modes import (
    FAMILIES,
    ContinuousLearner,
    ContinuousState,
    same_continuous_state,
    validate_learner_family,
)
from src.adapters.continuous_stream_demo import _training_episodes
from src.adapters.numpy_continuous_bc import (
    NumpyContinuousBCLearner,
    _snapshot as validate_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import (
    ContinuousPCSnapshot,
    NumpyContinuousPCLearner,
    _snapshot as validate_pc_snapshot,
)
from src.core.continuous_demonstration import ContinuousEpisode
from src.core.continuous_bc import MIXED_OBJECTIVE, OBJECTIVE
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.pick_cube_rgb_standardization import PREFIX_DIMENSIONS
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core import pick_cube_state_quaternions as state_quaternions
from src.core import pick_cube_action_history as action_history
from src.core.pick_cube_action_decoding import PickCubeBinaryGripperRecipe, validate_gripper_recipe
from src.core.observation_standardization import (
    ScaleFlooredObservationStandardization,
    validate_scale_floors,
)
from src.core.readiness_stage_diagnostics import (
    ReadinessStageBinding,
    copy_readiness_stage_bindings,
    readiness_stage_metadata,
)
from src.infra.maniskill_continuous import ACTION_UNITS


def _output_paths(output: Path) -> tuple[Path, Path]:
    output = Path(output)
    native = output.with_name(output.name + ".rollout.json")
    for path in (output, native):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing learning/rollout evidence: {path}")
    return output, native


def _admit_inputs(warm, new, visual_observation):
    if type(visual_observation) is not bool:
        raise ValueError("visual_observation must be an exact bool")
    warm, new = _training_episodes(warm, new)
    # Why: the native PickCube state/RGB routes have fixed geometry. Refuse a
    # known mismatch before fitting, rather than spending work before reset.
    dimension = pick_cube_rgb_recipe().feature_dim if visual_observation else 42
    if (
        warm.context != pick_cube_training_context(17)
        or warm.spec.observation_dim != dimension
        or warm.spec.action_units != ACTION_UNITS
        or warm.spec.lower_bounds != (-1.0,) * 4
        or warm.spec.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("learning rollout requires default TRAIN17 A and native state42/RGB789")
    if visual_observation and any(
        not 0.0 <= pixel <= 1.0
        for episode in (warm, new)
        for row in episode.transitions
        for observation in (row.observation, row.next_observation)
        for pixel in observation[PREFIX_DIMENSIONS:]
    ):
        raise ValueError("learning rollout requires public RGB features in[0,1]")
    return warm, new


def _candidate_state(candidate, expected=None):
    owner_type = type(candidate)
    if owner_type not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner):
        raise ValueError("learning rollout requires an exact continuous learner owner")
    observed, policy = candidate.snapshot_state(), candidate.policy
    if type(candidate) is not owner_type:
        raise RuntimeError("learned candidate owner type changed during observation")
    # Why: public observers may run instrumentation callbacks. Finish with the
    # exact native-owned state after all public reads, as in frozen PC serving.
    owned = (
        validate_pc_snapshot(candidate._state, candidate._policy)
        if isinstance(candidate, NumpyContinuousPCLearner)
        else validate_bc_snapshot(candidate._state, candidate._policy)
    )
    if (
        observed.policy != policy
        or not same_continuous_state(observed, owned)
        or (expected is not None and not same_continuous_state(expected, owned))
    ):
        raise RuntimeError("learned candidate changed during ownership observation")
    return owned


def _require_binary_gripper_request(family, axis):
    if axis is not None and (type(axis) is not int or axis != 3 or family != "bp"):
        raise ValueError("robot binary gripper requires BP and exact axis3")


def _admit_a_only_history(
    report, warm, phase_updates, family, *, action_history_recipe=None, binary_gripper_axis=None
):
    """Bind the opt-in control to its actual permanently paid A packets.

    Why: equal total updates cannot distinguish A-only from A/B/A when episodes
    have equal lengths. Validate source identity and arrival evidence as well.
    This checks an in-process result; it does not authenticate saved model data.
    """
    _require_binary_gripper_request(family, binary_gripper_axis)
    updates, rows = 3 * phase_updates, len(warm.transitions)
    fit_dimensions = warm.spec.observation_dim
    fit_calls, fit_rows = 1, rows
    if action_history_recipe is not None:
        recipe = action_history.validate_action_history_recipe(action_history_recipe)
        if warm.spec.observation_dim != recipe.feature_dim:
            raise ValueError("causal history candidate requires the declared47D inputs")
        fit_dimensions = recipe.source_dim
        if action_history.is_taskspace_recipe(recipe):
            fit_dimensions, fit_calls, fit_rows = 0, 0, 0
    settings, work, progress = report["settings"], report["work"], report["progress"]
    expected_progress = dict(
        update_attempts=updates,
        updates_completed=updates,
        replay_examples_completed=0,
        next_tick=2 * updates,
    )
    for actual, expected in (
        (
            settings,
            dict(
                phase_updates=phase_updates,
                max_training_updates=updates,
                probe_reads=0,
                final_reads=0,
                guard_reads=0,
                promotion_calls=0,
                simulator_calls=0,
            ),
        ),
        (
            work,
            dict(
                optimizer_updates=updates,
                training_rows=updates * rows,
                unique_training_rows=rows,
                repeated_training_rows=(updates - 1) * rows,
                replay_budget_rows=0,
                fit_calls=fit_calls,
                fit_rows=fit_rows,
                fit_feature_values=fit_rows * fit_dimensions,
            ),
        ),
        (progress, expected_progress),
        (report["checkpoints"][-1], dict(updates=updates, training_examples=updates * rows)),
    ):
        if any(
            type(actual[key]) is not int or actual[key] != value for key, value in expected.items()
        ):
            raise ValueError("A-only candidate counts must retain exact native integer charges")
    if (
        settings.get("a_only_control") is not True
        or settings.get("training_contexts") != ["warm"] * 3
        or settings.get("diagnostic_contexts") != ["warm", "new"]
        or progress != expected_progress
        or len(report["events"]) != updates
        or len(report["applied_updates"]) != updates
    ):
        raise ValueError("A-only candidate settings, exposure or progress differs")
    if family == "bp" or "training_rows_by_context" in work:
        exposure = work.get("training_rows_by_context")
        if exposure != dict(warm=updates * rows, new=0) or any(
            type(value) is not int for value in exposure.values()
        ):
            raise ValueError("A-only work must retain exact zero B optimizer exposure")
    if family != "bp" and (
        len(report["native_updates"]) != updates
        or any(
            type(item["training_rows"]) is not int or item["training_rows"] != rows
            for item in report["native_updates"]
        )
    ):
        raise ValueError("A-only native PC exposure differs from its A packets")
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        _admit_a_only_packet(
            event,
            receipt,
            warm.episode_id,
            rows,
            index,
            phase_updates,
            family,
            binary_gripper_axis=binary_gripper_axis,
        )


def _admit_a_only_packet(
    event, receipt, episode_id, rows, index, phase_updates, family, *, binary_gripper_axis=None
):
    """Check one bounded packet's original clocks, source and successful charge."""
    expected = dict(
        index=index,
        observed_at=2 * index,
        arrived_at=2 * index + 1,
        candidate_updates=index + 1,
        candidate_training_examples=(index + 1) * rows,
    )
    receipt_counts = dict(
        observed_at=2 * index,
        arrived_at=2 * index + 1,
        applied_at=2 * index + 1,
        update_number=index + 1,
    )
    diagnostic = receipt["diagnostic"]
    value = diagnostic["value"]
    if (
        any(type(event[key]) is not int or event[key] != value for key, value in expected.items())
        or any(
            type(receipt[key]) is not int or receipt[key] != value
            for key, value in receipt_counts.items()
        )
        or event["context"] != "warm"
        or event["phase"] != ("warm", "new", "warm_return")[index // phase_updates]
        or any(
            event[key] is not True
            for key in ("deferred_exact", "update_attempted", "success_validated")
        )
        or event["applied"] != [receipt]
        or receipt["episode_id"] != episode_id
        or receipt["sample_id"] != f"mlp-batch-{index}"
        or receipt["event_id"] != f"mlp-label-{index}"
        or receipt["actor_version"] != "actor-0"
        or receipt["learner_version"] != "candidate-0"
        or diagnostic["definition"]
        != (
            (OBJECTIVE if binary_gripper_axis is None else MIXED_OBJECTIVE)
            if family == "bp"
            else CONTINUOUS_PC_ENERGY_ID
        )
        or type(value) not in (int, float)
        or not math.isfinite(value)
    ):
        raise ValueError("A-only candidate lost its validated A receipt history")


def _admit_candidate(
    result,
    warm,
    new,
    family,
    phase_updates,
    *,
    a_only_control=False,
    action_history_recipe=None,
    binary_gripper_axis=None,
    stage_balanced_packets=False,
    diagnostic_stage_bindings=None,
):
    _require_binary_gripper_request(family, binary_gripper_axis)
    if type(a_only_control) is not bool:
        raise ValueError("a_only_control must be an exact bool")
    if type(stage_balanced_packets) is not bool:
        raise ValueError("stage_balanced_packets must be an exact bool")
    if (
        type(result) is not ContinuousLearnabilityResult
        or result.report.get("complete") is not True
    ):
        raise ValueError("learning rollout requires a complete exact learnability result")
    if stage_balanced_packets:
        if not a_only_control or diagnostic_stage_bindings is None:
            raise ValueError("balanced candidate requires declared A-only stage bindings")
        owned = copy_readiness_stage_bindings(diagnostic_stage_bindings, (warm, new))
        plan = prepare_readiness_packets(owned[0], 3 * phase_updates)
        try:
            _require_training_packet_evidence(result.report, plan, 3 * phase_updates)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("balanced candidate lost its original packet evidence") from error
    elif (
        "stage_balanced_packets" in result.report["settings"]
        or "training_packets" in result.report
        or any("training_row_indices" in event for event in result.report["events"])
    ):
        raise ValueError("candidate introduced unrequested stage-balanced packets")
    candidate = result.candidate
    expected_type = NumpyContinuousBCLearner if family == "bp" else NumpyContinuousPCLearner
    if type(candidate) is not expected_type:
        raise ValueError("learning rollout candidate is missing or has the wrong family")
    state = _candidate_state(candidate)
    policy = state.policy.base if isinstance(state, ContinuousPCSnapshot) else state.policy
    if (
        type(policy.binary_gripper_axis) is not type(binary_gripper_axis)
        or policy.binary_gripper_axis != binary_gripper_axis
    ):
        raise ValueError("learned candidate differs from the requested gripper likelihood")
    if isinstance(state, ContinuousPCSnapshot):
        expected_config = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        if state.policy.circadian != expected_config:
            raise ValueError("learning rollout candidate has the wrong circadian configuration")
    updates = 3 * phase_updates
    rows = (
        updates * len(warm.transitions)
        if a_only_control
        else phase_updates * (2 * len(warm.transitions) + len(new.transitions))
    )
    report = result.report
    if not a_only_control and report["settings"].get("a_only_control", False) is not False:
        raise ValueError("default A/B/A candidate unexpectedly declares A-only training")
    if a_only_control:
        _admit_a_only_history(
            report,
            warm,
            phase_updates,
            family,
            action_history_recipe=action_history_recipe,
            binary_gripper_axis=binary_gripper_axis,
        )
    expected_sources = {
        name: dict(episode_id=episode.episode_id, context=episode.context, role="train")
        for name, episode in (("warm", warm), ("new", new))
    }
    if (
        asdict(state.policy) != report["policy"]
        or policy.spec != warm.spec
        or policy.training_role != "train"
        or policy.seed != 17
        or report.get("learner_family", "bp") != family
        or state.completed_updates != updates
        or state.training_examples != rows
        or report["progress"]["updates_completed"] != updates
        or report["source_episodes"] != expected_sources
        or report["checkpoints"][-1]["updates"] != updates
        or report["checkpoints"][-1]["training_examples"] != rows
        or report["checkpoints"][-1]["state_exact"] is not True
        or report["checkpoints"][-1]["owners_exact"] is not True
    ):
        raise ValueError(
            "learning rollout candidate differs from the completed TRAIN recipe/report"
        )
    return candidate, state


def _frozen_return(
    report: dict[str, Any],
    candidate: ContinuousLearner,
    before: ContinuousState,
    native_output: Path,
    visual_observation: bool,
    *,
    observation_projection: state_quaternions.PickCubeStateQuaternionProjection | None = None,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    action_decoding_recipe: PickCubeBinaryGripperRecipe | None = None,
) -> None:
    report["candidate_unchanged"] = False
    _candidate_state(candidate, before)
    report["rollout_attempted"] = True
    error = None
    corrections = None
    projection_options: dict[str, Any] = {}
    if observation_projection is not None:
        projection_options["observation_projection"] = (
            state_quaternions.validate_state_quaternion_projection(observation_projection)
        )
    if action_history_recipe is not None:
        projection_options["action_history_recipe"] = action_history.validate_action_history_recipe(
            action_history_recipe
        )
    decoding_metadata = None
    if action_decoding_recipe is not None:
        decoding_metadata = asdict(validate_gripper_recipe(action_decoding_recipe))
        report["action_decoding_recipe"] = decoding_metadata.copy()
        # Pass the original request so frozen serving can detect a late mutation.
        projection_options["action_decoding_recipe"] = action_decoding_recipe
    try:
        outcome = run_frozen_rollout(
            native_output,
            learner=candidate,
            train_seed=17,
            visual_observation=visual_observation,
            monitor_joint_positions=True,
            **projection_options,
        )
        report["rollout"] = deepcopy(outcome.report)
        if type(outcome) is not FrozenRolloutResult:
            raise ValueError("frozen return requires the exact native result")
        corrections = outcome.corrections
    except Exception as caught:
        error = caught
    # Why: retain the original native failure as well as any subsequent ownership
    # failure. A failed call still needs this full parameter/momentum/wake check.
    try:
        report["candidate_unchanged"] = False
        _candidate_state(candidate, before)
        report["candidate_unchanged"] = True
    except Exception as caught:
        report["state_guard_error"] = f"{type(caught).__name__}: {caught}"
        if error is None:
            error = caught
    if error is not None:
        raise error
    native = report["rollout"]
    if (
        corrections is not None
        or native["complete"] is not True
        or native["frozen_state_exact"] is not True
        or native["original_owner_exact"] is not True
        or native["joint_position_exact"] is not True
        or type(native["optimizer_updates"]) is not int
        or native["optimizer_updates"] != 0
        or native["learner_policy"] != asdict(before.policy)
        or type(native["source_updates"]) is not int
        or native["source_updates"] != before.completed_updates
        or type(native["source_training_rows"]) is not int
        or native["source_training_rows"] != before.training_examples
        or type(native["seed"]) is not int
        or native["seed"] != 17
        or native["context"] != pick_cube_training_context(17)
    ):
        raise RuntimeError("frozen return incomplete or differs from the admitted candidate")
    if observation_projection is not None:
        _admit_state_projection(native, report, observation_projection)
    if action_history_recipe is not None:
        _admit_action_history(native, report, action_history_recipe)
    if action_decoding_recipe is not None and (
        native.get("action_decoding_exact") is not True
        or native.get("action_decoding", {}).get("recipe") != decoding_metadata
        or report.get("action_decoding_recipe") != decoding_metadata
        or asdict(validate_gripper_recipe(action_decoding_recipe)) != decoding_metadata
    ):
        raise ValueError("frozen action decoding differs from the original execution recipe")


def _admit_state_projection(native, report, projection):
    expected = asdict(state_quaternions.validate_state_quaternion_projection(projection))
    for metadata in (report.get("observation_projection"), native.get("observation_projection")):
        if type(metadata) is not dict or set(metadata) != set(expected):
            raise ValueError("canonical state42 projection metadata is missing")
        state_quaternions.validate_state_quaternion_projection(
            state_quaternions.PickCubeStateQuaternionProjection(**metadata)
        )
    if (
        report.get("observation_projection") != expected
        or native.get("observation_projection") != expected
        or native.get("projection_exact") is not True
        or native.get("observation_recipe") is not None
        or native.get("visual_work") is not None
    ):
        raise ValueError("canonical state42 training and serving projection differ")
    steps = native["prediction_attempts"]
    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("canonical state42 serving requires bounded native integer predictions")
    counts = dict(
        observation_attempts=steps,
        observation_returns=steps,
        source_values_upper_bound=steps * 42,
        source_values_validated=steps * 42,
        projected_values=steps * 42,
        projected_numeric_bytes=steps * 42 * 8,
    )
    observed = native.get("projection_work")
    if (
        type(native["prediction_returns"]) is not int
        or native["prediction_returns"] != steps
        or type(observed) is not dict
        or observed != counts
        or any(type(value) is not int for value in observed.values())
    ):
        raise ValueError("canonical state42 serving lost exact projection work")


def _require_action_history_metadata(value, recipe):
    expected = asdict(action_history.validate_action_history_recipe(recipe))
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("causal action history recipe metadata is missing or changed")
    observed = action_history.validate_action_history_recipe(
        type(action_history.validate_action_history_recipe(recipe))(**value)
    )
    if asdict(observed) != expected:
        raise ValueError("causal action history recipe metadata differs from the request")


def _admit_action_history(native, report, recipe):
    config = action_history.validate_action_history_recipe(recipe)
    expected = asdict(config)
    _require_action_history_metadata(report.get("action_history_recipe"), config)
    if native.get("action_history_exact") is not True:
        raise ValueError("training and serving causal action history differ")
    episode = native["episode"]
    # Actual live raw trace remains42D. Rebuild only the small causal state, never
    # a model, fit or hidden teacher phase, from successful native commands.
    state = action_history.ActionHistoryState()
    for record in episode["transitions"]:
        state = action_history.advance_action_history(state, tuple(record["action"]))
    rows = len(episode["transitions"])
    if not 1 <= rows <= 50 or episode["spec"]["observation_dim"] != 42:
        raise ValueError("causal action history serving lost its complete raw trace")
    expected_history = dict(
        recipe=expected,
        state=asdict(state),
        pending=False,
        work=dict(
            encoding_attempts=rows,
            encoding_returns=rows,
            committed_controls=rows,
            source_values_validated=42 * rows,
            model_values_emitted=config.feature_dim * rows,
            model_numeric_bytes_emitted=config.feature_dim * 8 * rows,
        ),
    )
    if (
        json.dumps(native.get("action_history"), sort_keys=True)
        != json.dumps(expected_history, sort_keys=True)
        or type(native["prediction_attempts"]) is not int
        or native["prediction_attempts"] != rows
        or type(native["prediction_returns"]) is not int
        or native["prediction_returns"] != rows
    ):
        raise ValueError("causal action history serving lost its executed-control charges")


def run_learning_rollout(
    output: Path,
    warm_episode: ContinuousEpisode,
    new_episode: ContinuousEpisode,
    *,
    learner_family: str = "bp",
    phase_updates: int = PHASE_UPDATES,
    visual_observation: bool = False,
    a_only_control: bool = False,
    canonical_state_quaternions: bool = False,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    scale_floors: tuple[float, ...] | None = None,
    diagnostic_stage_bindings: tuple[ReadinessStageBinding, ReadinessStageBinding] | None = None,
    binary_gripper_axis: int | None = None,
    stage_balanced_packets: bool = False,
) -> dict[str, Any]:
    """Keep paid training/native reports even when the handoff or return fails.

    The candidate comes directly from the trusted in-process runner, never an
    external saved report/model pair. Matching report metadata is admission, not
    authentication of arbitrary third-party matrices. Native output is separately
    exclusive; a concurrent creator can still refuse it after training is spent.
    """
    if type(a_only_control) is not bool:
        raise ValueError("a_only_control must be an exact bool")
    if type(stage_balanced_packets) is not bool:
        raise ValueError("stage_balanced_packets must be an exact bool")
    if stage_balanced_packets and (
        not a_only_control or diagnostic_stage_bindings is None or visual_observation is not False
    ):
        raise ValueError("balanced robot learning requires nonvisual A-only readiness bindings")
    if type(canonical_state_quaternions) is not bool:
        raise ValueError("canonical_state_quaternions must be an exact bool")
    if canonical_state_quaternions and visual_observation is not False:
        raise ValueError("canonical state quaternions require the original nonvisual state42 route")
    if action_history_recipe is not None:
        action_history_recipe = action_history.validate_action_history_recipe(action_history_recipe)
        if (
            visual_observation is not False
            or (canonical_state_quaternions and not action_history_recipe.canonical_quaternions)
            or (
                action_history.is_taskspace_recipe(action_history_recipe)
                and scale_floors is not None
            )
        ):
            raise ValueError(
                "action history requires nonvisual raw42 with one explicit canonical choice"
            )
    if scale_floors is not None:
        if visual_observation is not False:
            raise ValueError("robot scale floors require the nonvisual state42 route")
        scale_floors = validate_scale_floors(scale_floors, 42)
    learner_family = validate_learner_family(learner_family)
    _require_binary_gripper_request(learner_family, binary_gripper_axis)
    phase_updates = validate_phase_updates(phase_updates)
    warm, new = _admit_inputs(warm_episode, new_episode, visual_observation)
    if diagnostic_stage_bindings is not None:
        diagnostic_stage_bindings = copy_readiness_stage_bindings(
            diagnostic_stage_bindings, (warm, new)
        )
    projection = None
    input_sources = {
        name: dict(episode_id=episode.episode_id, context=episode.context, role=episode.role)
        for name, episode in (("warm", warm), ("new", new))
    }
    if canonical_state_quaternions and action_history_recipe is None:
        projection = state_quaternions.state_quaternion_projection()
        warm, new = (
            state_quaternions.state_quaternion_training_episode(episode, projection)
            for episode in (warm, new)
        )
        if diagnostic_stage_bindings is not None:
            diagnostic_stage_bindings = (
                ReadinessStageBinding(warm, diagnostic_stage_bindings[0].stages),
                ReadinessStageBinding(new, diagnostic_stage_bindings[1].stages),
            )
    if action_history_recipe is not None:
        warm, new = (
            action_history.action_history_training_episode(episode, action_history_recipe)
            for episode in (warm, new)
        )
        if diagnostic_stage_bindings is not None:
            diagnostic_stage_bindings = (
                ReadinessStageBinding(warm, diagnostic_stage_bindings[0].stages),
                ReadinessStageBinding(new, diagnostic_stage_bindings[1].stages),
            )
    robot_packet_plan = None
    if stage_balanced_packets:
        assert diagnostic_stage_bindings is not None
        robot_packet_plan = prepare_readiness_packets(
            diagnostic_stage_bindings[0], 3 * phase_updates
        )
    robot_packet_evidence = (
        json.dumps(asdict(robot_packet_plan), sort_keys=True, allow_nan=False)
        if robot_packet_plan is not None
        else None
    )
    output, native_output = _output_paths(output)
    report: dict[str, Any] = dict(
        schema_version=1,
        protocol_id="continuous_learned_robot_train_v1",
        scope="TRAIN A/B/A then frozen A return; no promotion or final evaluation",
        complete=False,
        training=None,
        rollout=None,
        rollout_output=str(native_output),
        rollout_attempted=False,
        candidate_unchanged=None,
    )
    options: dict[str, Any] = dict(a_only_control=True) if a_only_control else {}
    training_options: dict[str, Any] = {}
    if stage_balanced_packets:
        options["stage_balanced_packets"] = True
        report["stage_balanced_packets"] = True
    if binary_gripper_axis is not None:
        training_options["binary_gripper_axis"] = binary_gripper_axis
        report["binary_gripper_axis"] = binary_gripper_axis
        report["training_objective"] = MIXED_OBJECTIVE
    if action_history_recipe is not None:
        training_options["action_history_recipe"] = action_history_recipe
    if scale_floors is not None:
        training_options["scale_floors"] = scale_floors
    if diagnostic_stage_bindings is not None:
        training_options["diagnostic_stage_bindings"] = diagnostic_stage_bindings
    if a_only_control:
        report.update(
            a_only_control=True,
            protocol_id="continuous_learned_robot_a_only_train_v1",
            scope="TRAIN A-only acquisition control; B read-only TRAIN diagnostics; frozen A return; no promotion or final evaluation",
        )
    projection_options: dict[str, Any] = {}
    if projection is not None:
        observations = 2 * (len(warm.transitions) + len(new.transitions))
        report.update(
            canonical_state_quaternions=True,
            protocol_id=(
                "continuous_learned_robot_canonical_state_a_only_train_v1"
                if a_only_control
                else "continuous_learned_robot_canonical_state_train_v1"
            ),
            scope=report["scope"]
            + "; canonical state42 features; native trace retains original raw state42",
            observation_projection=asdict(projection),
            input_source_episodes=input_sources,
            input_projection_work=dict(
                episode_calls=2,
                observation_attempts=observations,
                observation_returns=observations,
                source_values_upper_bound=observations * 42,
                source_values_validated=observations * 42,
                projected_values=observations * 42,
                projected_numeric_bytes=observations * 42 * 8,
            ),
        )
        projection_options["observation_projection"] = projection
    if action_history_recipe is not None:
        observations = 2 * (len(warm.transitions) + len(new.transitions))
        report.update(
            action_history_recipe=asdict(action_history_recipe),
            input_source_episodes=input_sources,
            input_history_work=dict(
                episode_calls=2,
                observation_rows=observations,
                source_coordinates=42 * observations,
                model_coordinates=action_history_recipe.feature_dim * observations,
                model_numeric_bytes=action_history_recipe.feature_dim * 8 * observations,
            ),
            protocol_id=report["protocol_id"].replace("_train_v1", "_action_history_train_v1"),
            scope=report["scope"]
            + f"; causal executed-action history{action_history_recipe.feature_dim}; original native raw42 trace",
        )
        projection_options["action_history_recipe"] = action_history_recipe
    if scale_floors is not None:
        report.update(
            scale_floors=scale_floors,
            scale_floor_definition="max_frozen_training_scale_and_declared_floor_v1",
            protocol_id=report["protocol_id"].replace("_train_v1", "_scale_floor_train_v1"),
            scope=report["scope"] + "; explicitly declared frozen observation scale floors",
        )
    if binary_gripper_axis is not None:
        report["protocol_id"] = report["protocol_id"].replace(
            "_train_v1", "_binary_gripper_train_v1"
        )
        report["scope"] += "; BP binary gripper likelihood; unchanged soft tanh controller"
    with output.open("x", encoding="utf8") as handle:
        try:
            result = run_continuous_learnability(
                warm,
                new,
                phase_updates=phase_updates,
                learner_family=learner_family,
                **options,
                **training_options,
            )
            if type(result) is ContinuousLearnabilityResult:
                report["training"] = deepcopy(result.report)
            admission_options = dict(options)
            if stage_balanced_packets:
                admission_options["diagnostic_stage_bindings"] = diagnostic_stage_bindings
            if binary_gripper_axis is not None:
                admission_options["binary_gripper_axis"] = binary_gripper_axis
            if action_history_recipe is not None:
                admission_options["action_history_recipe"] = action_history_recipe
            candidate, state = _admit_candidate(
                result, warm, new, learner_family, phase_updates, **admission_options
            )
            base = state.policy.base if isinstance(state, ContinuousPCSnapshot) else state.policy
            transform = base.observation_standardization
            if scale_floors is None and type(transform) is ScaleFlooredObservationStandardization:
                raise ValueError("learned candidate introduced unrequested scale floors")
            if scale_floors is not None:
                if type(
                    transform
                ) is not ScaleFlooredObservationStandardization or transform.scale_floors != (
                    scale_floors + (1.0,) * 5 if action_history_recipe is not None else scale_floors
                ):
                    raise ValueError("learned candidate did not retain requested scale floors")
            if action_history_recipe is not None:
                _require_action_history_metadata(
                    result.report.get("action_history_recipe"), action_history_recipe
                )
                if (
                    (transform is not None)
                    if action_history.is_taskspace_recipe(action_history_recipe)
                    else (
                        transform is None
                        or transform.means[-5:] != (0.0,) * 5
                        or transform.scales[-5:] != (1.0,) * 5
                        or getattr(transform, "effective_scales", transform.scales)[-5:]
                        != (1.0,) * 5
                    )
                ):
                    raise ValueError(
                        "candidate lost its requested causal recipe or fixed history scaling"
                    )
            elif "action_history_recipe" in result.report:
                raise ValueError("candidate introduced unrequested causal action history")
            if diagnostic_stage_bindings is None:
                if "diagnostic_stage_sources" in result.report:
                    raise ValueError("learned candidate introduced unrequested stage diagnostics")
            else:
                _require_stage_diagnostics(
                    result.report,
                    readiness_stage_metadata(diagnostic_stage_bindings),
                    _readiness_checkpoint_counts(warm, new, phase_updates, a_only_control),
                )
            if stage_balanced_packets:
                assert diagnostic_stage_bindings is not None
                assert robot_packet_plan is not None
                current_plan = prepare_readiness_packets(
                    diagnostic_stage_bindings[0], 3 * phase_updates
                )
                if (
                    json.dumps(asdict(current_plan), sort_keys=True, allow_nan=False)
                    != robot_packet_evidence
                ):
                    raise ValueError("balanced original full input changed before native return")
                _require_training_packet_evidence(
                    result.report, robot_packet_plan, 3 * phase_updates
                )
                _require_training_packet_evidence(
                    report["training"], robot_packet_plan, 3 * phase_updates
                )
            _frozen_return(
                report, candidate, state, native_output, visual_observation, **projection_options
            )
            if stage_balanced_packets:
                assert robot_packet_plan is not None and diagnostic_stage_bindings is not None
                current_plan = prepare_readiness_packets(
                    diagnostic_stage_bindings[0], 3 * phase_updates
                )
                if (
                    report.get("stage_balanced_packets") is not True
                    or json.dumps(asdict(current_plan), sort_keys=True, allow_nan=False)
                    != robot_packet_evidence
                ):
                    raise ValueError("balanced original input or request changed during return")
                _require_training_packet_evidence(
                    result.report, robot_packet_plan, 3 * phase_updates
                )
                _require_training_packet_evidence(
                    report["training"], robot_packet_plan, 3 * phase_updates
                )
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-teacher", type=Path, required=True)
    parser.add_argument("--new-teacher", type=Path, required=True)
    parser.add_argument(
        "--joint-acquisition",
        action="store_true",
        help="Run the fixed qualified joint acquisition recipe and frozen TRAIN17/19/20.",
    )
    parser.add_argument("--joint-corrections", type=Path)
    parser.add_argument("--curated-source", type=Path)
    parser.add_argument("--curated-labels", type=Path)
    parser.add_argument("--retained-teacher", type=Path)
    parser.add_argument("--learner-family", choices=FAMILIES, default="bp")
    parser.add_argument("--phase-updates", type=int, default=PHASE_UPDATES)
    parser.add_argument("--visual-observation", action="store_true")
    parser.add_argument(
        "--binary-gripper",
        action="store_true",
        help="Use BP's existing binary likelihood on normalized gripper axis3; BP only.",
    )
    parser.add_argument(
        "--observation-scale-floors",
        type=float,
        nargs=42,
        help="Declare 42 positive physical scale floors before training; nonvisual state42 only.",
    )
    parser.add_argument(
        "--canonical-state-quaternions",
        action="store_true",
        help="Use explicit sign-stable quaternion state42 features for training and frozen serving.",
    )
    parser.add_argument(
        "--action-history",
        choices=("executed", "zero"),
        default=None,
        help="Append causal executed-action4 and bounded closing count, or equal-width zeros; state-only.",
    )
    parser.add_argument("--lighting-drift", action="store_true")
    parser.add_argument(
        "--a-only-control",
        action="store_true",
        help="Train only A; retain B as read-only TRAIN diagnostics before frozen A return.",
    )
    parser.add_argument(
        "--readiness-teachers",
        action="store_true",
        help="read both qualified geometry-readiness TRAIN17/19 demonstrations",
    )
    parser.add_argument(
        "--taskspace-history",
        choices=("executed", "zero"),
        default=None,
        help="Use official WORLD taskspace12 inputs with fixed .1m scaling and causal history.",
    )
    parser.add_argument(
        "--observed-grasp-warm-teacher",
        action="store_true",
        help="Use the qualified observed-grasp TRAIN17 source; keep TRAIN19 read-only.",
    )
    parser.add_argument(
        "--readiness-stage-residuals",
        action="store_true",
        help="Partition existing residual readouts by qualified TRAIN readiness stages.",
    )
    parser.add_argument(
        "--stage-balanced-packets",
        action="store_true",
        help="Use15-row A-only packets with three original examples per recorded stage.",
    )
    args = parser.parse_args(argv)
    joint_paths = (
        args.joint_corrections,
        args.curated_source,
        args.curated_labels,
        args.retained_teacher,
    )
    if args.joint_acquisition:
        if any(path is None for path in joint_paths):
            parser.error("--joint-acquisition requires all four joint receipt paths")
        if (
            args.learner_family != "bp"
            or args.phase_updates != 8
            or args.visual_observation
            or args.binary_gripper
            or args.observation_scale_floors is not None
            or args.canonical_state_quaternions
            or args.action_history is not None
            or args.lighting_drift
            or args.a_only_control
            or args.readiness_teachers
            or args.observed_grasp_warm_teacher
            or args.taskspace_history is not None
            or args.readiness_stage_residuals
            or args.stage_balanced_packets
        ):
            parser.error("--joint-acquisition uses its fixed BP/8 recipe without legacy mode flags")
        try:
            _joint_output_paths(args.output)
            inputs, receipt_metadata = read_joint_acquisition_inputs(
                args.warm_teacher,
                args.new_teacher,
                args.joint_corrections,
                args.curated_source,
                args.curated_labels,
                args.retained_teacher,
            )
            report = run_joint_acquisition_rollouts(
                args.output, *inputs, input_receipts=receipt_metadata
            )
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["complete"] else 1
    if any(path is not None for path in joint_paths):
        parser.error("joint receipt paths require --joint-acquisition")
    if args.stage_balanced_packets and (
        not args.readiness_teachers
        or not args.readiness_stage_residuals
        or not args.a_only_control
        or args.visual_observation
    ):
        parser.error(
            "--stage-balanced-packets requires nonvisual --readiness-teachers, --readiness-stage-residuals and --a-only-control"
        )
    if args.observed_grasp_warm_teacher and (
        not args.readiness_teachers or not args.a_only_control or args.visual_observation
    ):
        parser.error(
            "--observed-grasp-warm-teacher requires --readiness-teachers, "
            "--a-only-control and nonvisual inputs"
        )
    if args.binary_gripper and args.learner_family != "bp":
        parser.error("--binary-gripper requires --learner-family bp")
    if args.taskspace_history is not None and (
        args.action_history is not None
        or args.canonical_state_quaternions
        or args.observation_scale_floors is not None
        or args.visual_observation
    ):
        parser.error("--taskspace-history requires its own nonvisual fixed physical recipe")
    if args.readiness_stage_residuals and not args.readiness_teachers:
        parser.error("--readiness-stage-residuals requires --readiness-teachers")
    if args.action_history is not None and args.visual_observation:
        parser.error("--action-history requires nonvisual state42 inputs")
    if args.canonical_state_quaternions and args.visual_observation:
        parser.error("--canonical-state-quaternions requires nonvisual state42 inputs")
    if args.observation_scale_floors is not None:
        try:
            if args.visual_observation:
                raise ValueError("--observation-scale-floors requires nonvisual state42 inputs")
            validate_scale_floors(tuple(args.observation_scale_floors), 42)
        except ValueError as error:
            parser.error(str(error))
    if args.readiness_teachers and args.lighting_drift:
        parser.error("--readiness-teachers and --lighting-drift require separate inputs")
    if args.lighting_drift and not args.visual_observation:
        parser.error("--lighting-drift requires --visual-observation")
    try:
        validate_phase_updates(args.phase_updates)
        _output_paths(args.output)
        options = dict(visual_observation=args.visual_observation)
        diagnostic_stage_bindings = None
        if args.readiness_teachers:
            # Source episode IDs retain the distinct teacher provenance; the
            # complete input reports remain the acquisition receipts.
            warm_options = dict(options)
            if args.observed_grasp_warm_teacher:
                warm_options["observed_grasp"] = True
            warm, warm_metadata = _read_readiness_teacher(args.warm_teacher, 17, **warm_options)
            new, new_metadata = _read_readiness_teacher(args.new_teacher, 19, **options)
            if args.readiness_stage_residuals:
                diagnostic_stage_bindings = (
                    ReadinessStageBinding(warm, tuple(warm_metadata["stages"])),
                    ReadinessStageBinding(new, tuple(new_metadata["stages"])),
                )
        else:
            warm = _read_teacher(args.warm_teacher, 17, **options)
            new = (
                _read_lighting_teacher(args.new_teacher, **options)[0]
                if args.lighting_drift
                else _read_teacher(args.new_teacher, 19, **options)
            )
        request_options: dict[str, Any] = {}
        if args.stage_balanced_packets:
            request_options["stage_balanced_packets"] = True
        if args.binary_gripper:
            request_options["binary_gripper_axis"] = 3
        if args.a_only_control:
            request_options["a_only_control"] = True
        if args.taskspace_history is not None:
            request_options["action_history_recipe"] = (
                action_history.PickCubeTaskSpaceHistoryRecipe(mode=args.taskspace_history)
            )
        elif args.action_history is not None:
            request_options["action_history_recipe"] = action_history.PickCubeActionHistoryRecipe(
                mode=args.action_history,
                canonical_quaternions=args.canonical_state_quaternions,
            )
        elif args.canonical_state_quaternions:
            request_options["canonical_state_quaternions"] = True
        if args.observation_scale_floors is not None:
            request_options["scale_floors"] = tuple(args.observation_scale_floors)
        if diagnostic_stage_bindings is not None:
            request_options["diagnostic_stage_bindings"] = diagnostic_stage_bindings
        report = run_learning_rollout(
            args.output,
            warm,
            new,
            learner_family=args.learner_family,
            phase_updates=args.phase_updates,
            visual_observation=args.visual_observation,
            **request_options,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
