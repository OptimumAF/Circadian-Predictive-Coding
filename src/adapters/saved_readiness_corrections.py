"""Bounded post-hoc TRAIN corrections on an already recorded raw42 trace.

Inputs are an owned same-setup TRAIN return and explicit NEW curation ticks.
Outputs retain provisional labels, paid partial work and an unchanged source.
No simulator, privileged pose read, model, RNG, role conversion or IO occurs.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from src.core.continuous_demonstration import ContinuousEpisode, copy_continuous_episode
from src.core.curated_readiness_corrections import (
    CuratedReadinessBatch,
    CuratedReadinessLabel,
    DEFINITION,
    GEOMETRY_DEFINITION,
    PROVENANCE,
    copy_curated_readiness_batch,
)
from src.core.experience import require_tick
from src.core.pick_cube_action_history import _native_action, action_history_training_episode
from src.core.pick_cube_readiness_corrections import _observed_grasp_flags
from src.core.pick_cube_readiness_teacher import GoalConsistentObservedGraspPickCubeTeacher
from src.core.pick_cube_taskspace_history import (
    PickCubeTaskSpaceHistoryRecipe,
    validate_taskspace_history_recipe,
)
from src.core.pick_cube_teacher import PickCubeGeometry
from src.core.pick_cube_train_context import pick_cube_training_context

UNITS = ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper")


@dataclass(frozen=True)
class SavedReadinessCurationResult:
    batch: CuratedReadinessBatch | None
    report: dict[str, Any]


def _admit(episode, seed, available, started, observed):
    context = pick_cube_training_context(seed)
    require_tick(available, "source_available_at")
    require_tick(started, "started_at")
    if started < available:
        raise ValueError("curation cannot start before source availability")
    # Inspect exact original flags before the normal episode copier can coerce numbers.
    _observed_grasp_flags(episode)
    source = copy_continuous_episode(episode)
    if (
        source.role != "train"
        or source.context != context
        or source.episode_id != f"pick-cube-smoke-{seed}"
        or source.spec.observation_dim != 42
        or source.spec.action_units != UNITS
        or source.spec.lower_bounds != (-1.0,) * 4
        or source.spec.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("saved corrections require the exact raw42 normalized TRAIN return")
    for row in source.transitions:
        _native_action(row.action)
    times = (None,) * len(source.transitions) if observed is None else observed
    if type(times) is not tuple or len(times) != len(source.transitions):
        raise ValueError("source_observed_at must preserve one original tick or None per row")
    for tick in times:
        if tick is not None:
            require_tick(tick, "source_observed_at")
            if tick > available:
                raise ValueError("source observation cannot follow source availability")
    return source, tuple(times)


def _same_source(original, owned):
    _observed_grasp_flags(original)
    if copy_continuous_episode(original) != owned:
        raise ValueError("original saved source changed during curation")


def _relative_geometry(raw):
    # Why: the pinned official raw42 exports object-TCP and goal-object WORLD
    # displacements. All existing teacher rules are translation invariant. A
    # virtual TCP origin reuses them without reconstructing privileged root poses.
    # World/root AXES align only in the same default Panda table setup as TS12.
    # These are newly curated public-state labels, not bit-identical historical
    # privileged labels. The distinct geometry/provenance definitions preserve that.
    cube = tuple(raw[36:39])
    goal = tuple(a + b for a, b in zip(cube, raw[39:42], strict=True))
    return PickCubeGeometry((0.0, 0.0, 0.0), cube, goal)


def curate_saved_readiness(
    episode: ContinuousEpisode,
    *,
    train_seed: int,
    source_available_at: int,
    started_at: int,
    source_observed_at: tuple[int | None, ...] | None = None,
) -> SavedReadinessCurationResult:
    """Query each saved TRAIN row once, with bounded new logical availability.

    The caller binds the declared default Panda setup and owns query authority.
    started_at+2*i/query+1 are NEW curation ticks, never original control clocks.
    All preadmission errors raise before queries. Paid failures return a partial
    report with batch=None; no source rollback, implicit retry or usable batch.
    """
    source, times = _admit(episode, train_seed, source_available_at, started_at, source_observed_at)
    report: dict[str, Any] = dict(
        complete=False,
        source_episode_id=source.episode_id,
        source_context=source.context,
        source_role=source.role,
        source_available_at=source_available_at,
        started_at=started_at,
        source_observed_at=list(times),
        teacher_definition=DEFINITION,
        provenance=PROVENANCE,
        geometry_definition=GEOMETRY_DEFINITION,
        teacher_queries_attempted=0,
        teacher_queries_returned=0,
        labels=[],
        work=dict(
            geometry_attempts=0,
            geometry_returns=0,
            source_values_validated=0,
            teacher_state_steps=0,
        ),
    )
    labels = []
    batch = None
    try:
        _same_source(episode, source)
        geometries = []
        for row in source.transitions:
            report["work"]["geometry_attempts"] += 1
            geometries.append(_relative_geometry(row.observation))
            report["work"]["geometry_returns"] += 1
            report["work"]["source_values_validated"] += 42
        teacher = GoalConsistentObservedGraspPickCubeTeacher()
        for i, (row, geometry) in enumerate(zip(source.transitions, geometries, strict=True)):
            _same_source(episode, source)
            if type(teacher.steps) is not int or teacher.steps != i:
                raise ValueError("curation teacher step continuation changed")
            report["teacher_queries_attempted"] += 1
            decision = teacher.decide(
                geometry,
                grasped=row.observation[18] == 1.0,
                executed_gripper=source.transitions[i - 1].action[3] if i else 0.0,
            )
            report["teacher_queries_returned"] += 1
            # Own the returned label BEFORE subsequent source/teacher observers.
            label = CuratedReadinessLabel(
                i,
                row.observation,
                decision.action,
                decision.stage,
                times[i],
                started_at + 2 * i,
                started_at + 2 * i + 1,
            )
            labels.append(label)
            report["labels"].append(asdict(label))
            report["work"]["teacher_state_steps"] = teacher.steps
            _same_source(episode, source)
            if type(teacher.steps) is not int or teacher.steps != i + 1:
                raise ValueError("curation teacher did not complete one original decision")
        batch = CuratedReadinessBatch(train_seed, source, tuple(labels), source_available_at)
        _same_source(episode, source)
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        batch = None
    return SavedReadinessCurationResult(batch, deepcopy(report))


def project_curated_training_pairs(
    batch: CuratedReadinessBatch,
    *,
    now: int,
    recipe: PickCubeTaskSpaceHistoryRecipe | None = None,
):
    """Release first, then derive existing TS12 history from actual commands.

    Returns tuple features/targets for the existing caller-owned training port.
    No normalization, array conversion, learner or budget charge happens here.
    The original batch and its curation clocks remain the provenance record.
    """
    owned = copy_curated_readiness_batch(batch)
    _, targets = owned.training_pairs(now=now)
    config = validate_taskspace_history_recipe(
        PickCubeTaskSpaceHistoryRecipe() if recipe is None else recipe
    )
    if config.mode != "executed":
        raise ValueError("curated projection requires the original executed action history")
    projected = action_history_training_episode(owned.episode, config)
    return tuple(row.observation for row in projected.transitions), targets
