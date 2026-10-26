"""Read-only reference-stage directions for one already paid correction step.

Inputs are owned TRAIN stage bindings and released native before/after snapshots.
Outputs retain scalar results and attempted/returned costs in a dedicated report.
No learner, update scheduling, fitting, action execution, label release or IO.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from src.adapters.numpy_continuous_bc import (
    ContinuousBCSnapshot,
    ContinuousBCPolicy,
    _snapshot as own_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import ContinuousPCSnapshot, _snapshot as own_pc_snapshot
from src.core import continuous_bc as task
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_step_diagnostics import _StepDirectionWork, diagnose_step_direction
from src.core.pick_cube_action_history import (
    ActionHistoryRecipe,
    validate_action_history_training_episode,
)
from src.core.readiness_stage_diagnostics import (
    ReadinessStageBinding,
    copy_readiness_stage_bindings,
    readiness_stage_metadata,
)

STAGES = ("align", "descend", "grasp", "lift", "settle")


def prepare_readiness_step_bindings(
    bindings: tuple[ReadinessStageBinding, ReadinessStageBinding],
    spec: ContinuousActionSpec,
    action_history_recipe: ActionHistoryRecipe | None = None,
) -> tuple[ReadinessStageBinding, ReadinessStageBinding]:
    """Own existing model-space references; never encode, relabel or admit v2 IDs."""
    if (
        type(bindings) is not tuple
        or len(bindings) != 2
        or any(type(item) is not ReadinessStageBinding for item in bindings)
    ):
        raise ValueError("readiness step diagnostics require exact warm/new stage bindings")
    episodes = (bindings[0].episode, bindings[1].episode)
    owned = copy_readiness_stage_bindings(bindings, episodes)
    expected_spec = task._spec(spec)
    if any(item.episode.spec != expected_spec for item in owned):
        raise ValueError("readiness diagnostic references differ from the model-space spec")
    if action_history_recipe is not None:
        for item in owned:
            validate_action_history_training_episode(item.episode, action_history_recipe)
    return owned


def _own_step(before, after):
    first: ContinuousBCSnapshot | ContinuousPCSnapshot
    final: ContinuousBCSnapshot | ContinuousPCSnapshot
    base: ContinuousBCPolicy
    if type(before) is ContinuousBCSnapshot and type(after) is ContinuousBCSnapshot:
        first = own_bc_snapshot(before, before.policy)
        final = own_bc_snapshot(after, after.policy)
        base = first.policy
        native_objective = (
            task.OBJECTIVE if base.binary_gripper_axis is None else task.MIXED_OBJECTIVE
        )
    elif type(before) is ContinuousPCSnapshot and type(after) is ContinuousPCSnapshot:
        first = own_pc_snapshot(before, before.policy)
        final = own_pc_snapshot(after, after.policy)
        base = first.policy.base
        native_objective = CONTINUOUS_PC_ENERGY_ID
    else:
        raise ValueError("readiness step diagnostics require matching exact native snapshots")
    if final.policy != first.policy or final.completed_updates != first.completed_updates + 1:
        raise ValueError(
            "readiness step diagnostics require one realized update with unchanged policy"
        )
    return first, final, base, native_objective


def _run_group(group, episode, indices, before, after, base):
    progress = _StepDirectionWork()
    group.update(
        rows=len(indices),
        complete=not indices,
        result=None,
        work=asdict(progress),
        input_numeric_bytes=0,
    )
    if not indices:
        return
    try:
        features = np.array(
            [episode.transitions[index].observation for index in indices],
            dtype=np.float64,
            copy=True,
        )
        group["input_numeric_bytes"] += features.nbytes
        targets = np.array(
            [episode.transitions[index].action for index in indices], dtype=np.float64, copy=True
        )
        group["input_numeric_bytes"] += targets.nbytes
        result = diagnose_step_direction(
            features,
            targets,
            deepcopy(before.parameters),
            deepcopy(before.velocity),
            deepcopy(after.parameters),
            deepcopy(base.spec),
            standardization=deepcopy(base.observation_standardization),
            learning_rate=base.learning_rate,
            momentum=base.momentum,
            _work=progress,
        )
        group["result"] = asdict(result)
        group["complete"] = True
    finally:
        group["work"] = asdict(progress)


def _summarize_work(report, dimensions, hidden, parameters):
    groups = [
        group for context in report["contexts"].values() for group in context["groups"].values()
    ]
    counters = {
        key: sum(group["work"][key] for group in groups) for key in asdict(_StepDirectionWork())
    }
    rows = lambda key: sum(group["rows"] * group["work"][key] for group in groups)
    gradient_rows = rows("task_gradient_returns")
    gradient_attempt_rows = rows("task_gradient_attempts")
    cost_per_row = 2 * dimensions * hidden + 3 * hidden * 4
    report["work"] = dict(
        **counters,
        reference_rows=report["reference_rows"],
        groups_recorded=len(groups),
        nonempty_groups_attempted=sum(group["rows"] > 0 for group in groups),
        normalization_rows_attempted=rows("normalization_attempts"),
        normalization_rows_returned=rows("normalization_returns"),
        normalization_values_attempted=rows("normalization_attempts") * dimensions,
        normalization_values_returned=rows("normalization_returns") * dimensions,
        task_gradient_rows_attempted=gradient_attempt_rows,
        task_gradient_rows_returned=gradient_rows,
        gradient_values_returned=counters["task_gradient_returns"] * parameters,
        gradient_numeric_bytes_returned=counters["task_gradient_returns"] * parameters * 8,
        derived_task_dense_multiply_accumulates=gradient_rows * cost_per_row,
        attempted_task_dense_multiply_accumulates_upper_bound=gradient_attempt_rows * cost_per_row,
        input_numeric_bytes=sum(group["input_numeric_bytes"] for group in groups),
        model_creation_calls=0,
        fit_calls=0,
        optimizer_updates=0,
        extra_prediction_calls=0,
        scope="read-only physical BP task gradients; returned work and failed-attempt upper bounds; excludes validation, copies, summaries, elementwise, containers and RSS",
    )


def run_readiness_step_diagnostics(
    report: dict,
    bindings: tuple[ReadinessStageBinding, ReadinessStageBinding],
    before: ContinuousBCSnapshot | ContinuousPCSnapshot,
    after: ContinuousBCSnapshot | ContinuousPCSnapshot,
) -> None:
    """Fill one dedicated report; caller admits only the first correction step.

    A failure propagates so the caller stops more updates, while every attempted
    group and its actual core work remain visible. Gradient dot realized step is
    first-order physical half-MSE evidence, not observed post-update task loss.
    """
    if type(report) is not dict or report:
        raise ValueError("readiness step diagnostics require an empty dedicated report dict")
    report.update(
        complete=False,
        definition="readiness_reference_realized_step_direction_v1",
        metadata=None,
        source_updates=None,
        after_updates=None,
        native_objective_definition=None,
        physical_objective_definition=task.OBJECTIVE,
        reference_rows=0,
        contexts={},
        work={},
    )
    dimensions = hidden = parameters = 0
    try:
        first, final, base, native_objective = _own_step(before, after)
        owned = prepare_readiness_step_bindings(bindings, base.spec)
        report.update(
            metadata=readiness_stage_metadata(owned),
            source_updates=first.completed_updates,
            after_updates=final.completed_updates,
            native_objective_definition=native_objective,
            reference_rows=sum(len(item.episode.transitions) for item in owned),
        )
        dimensions, hidden = base.spec.observation_dim, base.hidden_dim
        parameters = sum(value.size for value in first.parameters)
        for name, binding in zip(("warm", "new"), owned, strict=True):
            episode = binding.episode
            context: dict[str, Any] = dict(
                episode_id=episode.episode_id, context=episode.context, role=episode.role, groups={}
            )
            report["contexts"][name] = context
            for stage in ("overall",) + STAGES:
                indices = tuple(
                    index
                    for index, value in enumerate(binding.stages)
                    if stage == "overall" or value == stage
                )
                group: dict[str, Any] = {}
                context["groups"][stage] = group
                _run_group(group, episode, indices, first, final, base)
                _summarize_work(report, dimensions, hidden, parameters)
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _summarize_work(report, dimensions, hidden, parameters)
