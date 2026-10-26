"""Read-only input/target consistency of released readiness TRAIN evidence.

Own a qualified-boundary teacher binding and provisional v2 correction batch;
compare raw42 and the existing causal taskspace12 view. No file qualification,
learner, fit, update, simulator, clock advancement or IO. Distances have each
view's own physical units and are not cross-view performance scores.
"""

from dataclasses import asdict
from typing import Any

import numpy as np

from src.core.continuous_bc_diagnostics import (
    _InputTargetConsistencyWork,
    continuous_input_target_consistency,
)
from src.core.pick_cube_action_history import action_history_training_episode
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_readiness_corrections import (
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    readiness_correction_definition,
    readiness_correction_provenance,
)
from src.core.pick_cube_readiness_teacher import DEFINITION, OBSERVED_GRASP_DEFINITION
from src.core.pick_cube_taskspace_history import (
    PickCubeTaskSpaceHistoryRecipe,
    validate_taskspace_history_recipe,
)
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


def _admit(teacher, corrections, now, recipe):
    if type(teacher) is not ReadinessStageBinding:
        raise ValueError("label separation requires an exact teacher stage binding")
    if type(corrections) is not GoalConsistentObservedGraspCorrectionBatch:
        raise ValueError("label separation requires exact provisional observed-v2 corrections")
    config = validate_taskspace_history_recipe(recipe)
    if config.mode != "executed":
        raise ValueError("label separation requires actual executed history")
    owned_teacher = ReadinessStageBinding(teacher.episode, teacher.stages)
    owned_query = copy_readiness_correction_batch(corrections)
    if (
        owned_teacher.episode.spec.observation_dim != 42
        or owned_query.episode.spec != owned_teacher.episode.spec
        or owned_query.episode.context != owned_teacher.episode.context
    ):
        raise ValueError("label separation requires same-context raw42 TRAIN evidence")
    # Why: the original release clock is checked on raw evidence before any
    # projection or numerical comparison; a model view grants no early access.
    owned_query.training_pairs(now=now)
    return owned_teacher, owned_query, config


def _rows(teacher, query):
    definition = (
        OBSERVED_GRASP_DEFINITION
        if teacher.episode.episode_id.startswith("pick-cube-observed-grasp-teacher-")
        else DEFINITION
    )
    rows: list[dict[str, Any]] = []
    for row, stage in zip(teacher.episode.transitions, teacher.stages, strict=True):
        rows.append(
            dict(
                index=len(rows),
                source_kind="qualified_teacher",
                episode_id=teacher.episode.episode_id,
                context=teacher.episode.context,
                role=teacher.episode.role,
                step=row.step,
                stage=stage,
                definition=definition,
            )
        )
    query_definition = readiness_correction_definition(query)
    query_provenance = readiness_correction_provenance(query)
    for label in query.labels:
        rows.append(
            dict(
                index=len(rows),
                source_kind="provisional_correction",
                episode_id=query.episode.episode_id,
                context=query.episode.context,
                role=query.episode.role,
                step=label.step,
                stage=label.stage,
                definition=query_definition,
                provenance=query_provenance,
                observed_at=label.observed_at,
                labels_arrived_at=label.labels_arrived_at,
            )
        )
    return rows


def _view(features, targets, spec, group):
    progress = _InputTargetConsistencyWork()
    try:
        result = continuous_input_target_consistency(
            np.array(features, dtype=np.float64),
            np.array(targets, dtype=np.float64),
            spec,
            _work=progress,
        )
        group.update(complete=True, result=asdict(result))
    finally:
        group["work"] = asdict(progress)


def run_readiness_label_separation(
    teacher: ReadinessStageBinding,
    corrections: GoalConsistentObservedGraspCorrectionBatch,
    *,
    now: int,
    recipe: PickCubeTaskSpaceHistoryRecipe,
) -> dict[str, Any]:
    """Compare all bounded pairs; invalid admission raises before projections.

    Numerical/projection failures return an incomplete report with paid progress;
    they stop the next view. This diagnostic never qualifies provisional labels
    or supplies feedback to a deployed learner, benchmark or final evaluation.
    """
    owned_teacher, query, config = _admit(teacher, corrections, now, recipe)
    teacher_rows = owned_teacher.episode.transitions
    targets = tuple(row.action for row in teacher_rows) + tuple(
        label.target for label in query.labels
    )
    raw_features = tuple(row.observation for row in teacher_rows) + tuple(
        label.observation for label in query.labels
    )
    report: dict[str, Any] = dict(
        definition="readiness_raw42_taskspace12_label_separation_v1",
        complete=False,
        error=None,
        rows=_rows(owned_teacher, query),
        targets=targets,
        executed_actions=tuple(row.action for row in teacher_rows + query.episode.transitions),
        recipe=asdict(config),
        projection_work=dict(
            teacher_attempts=0,
            teacher_returns=0,
            correction_attempts=0,
            correction_returns=0,
            attempted_observation_projections_upper_bound=0,
            scope="whole-projection attempts/returns; failed internal work is unknown within the stated upper bound",
        ),
        views={
            name: dict(complete=False, result=None, work=asdict(_InputTargetConsistencyWork()))
            for name in ("raw42", "taskspace12")
        },
    )
    try:
        report["projection_work"]["teacher_attempts"] = 1
        report["projection_work"]["attempted_observation_projections_upper_bound"] += 2 * len(
            teacher_rows
        )
        model_teacher = action_history_training_episode(owned_teacher.episode, config)
        report["projection_work"]["teacher_returns"] = 1
        count = len(teacher_rows)
        report["projection_work"]["teacher"] = dict(
            observation_projections=2 * count,
            source_values_validated=84 * count,
            projected_values=24 * count,
            projected_numeric_bytes=192 * count,
            history_advances=count,
            physical_scale_divisions=12 * count,
        )
        report["projection_work"]["correction_attempts"] = 1
        report["projection_work"]["attempted_observation_projections_upper_bound"] += 2 * len(
            query.labels
        )
        model_query = action_history_correction_view(query, config)
        report["projection_work"]["correction_returns"] = 1
        report["projection_work"]["corrections"] = asdict(model_query.work)
        report["projection_work"]["corrections"]["physical_scale_divisions"] = 12 * len(
            query.labels
        )
        model_features = tuple(row.observation for row in model_teacher.transitions) + tuple(
            label.observation for label in model_query.model.labels
        )
        _view(raw_features, targets, owned_teacher.episode.spec, report["views"]["raw42"])
        _view(model_features, targets, model_teacher.spec, report["views"]["taskspace12"])
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["work"] = {
        key: sum(view["work"][key] for view in report["views"].values())
        for key in asdict(_InputTargetConsistencyWork())
    }
    report["work"].update(
        model_creation_calls=0,
        rng_draws=0,
        fit_calls=0,
        optimizer_updates=0,
        predictions=0,
        gradients=0,
        native_controls=0,
        scope="pair arithmetic and returned projection payload; validation/copies/containers/RSS additional",
    )
    return report
