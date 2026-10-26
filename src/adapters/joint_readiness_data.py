"""Own arrived joint readiness TRAIN rows for the existing correction runtime.

Inputs are a trusted already-qualified A17 reference binding, provisional B19
labels, optional explicitly supplied later-curated A17 labels and an optional
retained observed-grasp TRAIN20 reference. External files still require their existing
qualification readers. Output rows keep each source's authority and clocks; no
mixed-context episode or new qualification is created. No learner, array math,
geometry/teacher call, normalization fit, RNG, clock advancement or IO belongs here.
"""

from dataclasses import dataclass
from typing import Any

from src.adapters.saved_readiness_corrections import project_curated_training_pairs
from src.core.continuous_demonstration import ContinuousActionSpec, continuous_training_pairs
from src.core.curated_readiness_corrections import (
    CuratedReadinessBatch,
    copy_curated_readiness_batch,
    DEFINITION as CURATED_DEFINITION,
    PROVENANCE as CURATED_PROVENANCE,
    GEOMETRY_DEFINITION,
)
from src.core.experience import require_tick
from src.core.pick_cube_action_history import (
    ActionHistoryState,
    action_history_training_episode,
    advance_action_history,
)
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_readiness_corrections import (
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    _observed_grasp_flags,
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
)
from src.core.pick_cube_readiness_teacher import (
    OBSERVED_GRASP_DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_taskspace_history import (
    PickCubeTaskSpaceHistoryRecipe,
    validate_taskspace_history_recipe,
)
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


OLD_INFORMATION_CONTROL_DEFINITION = "original_A17_15_repetition_control_v1"


@dataclass(frozen=True)
class JointReadinessData:
    """Owned preparation output, not standalone training or qualification authority."""

    spec: ContinuousActionSpec
    rows: tuple[dict[str, Any], ...]
    groups: tuple[tuple[str, int, int], ...]
    work: dict[str, Any]
    recipe: PickCubeTaskSpaceHistoryRecipe


def _admit(reference, corrections, curated, now, recipe):
    require_tick(now, "joint readiness release time")
    if type(reference) is not ReadinessStageBinding:
        raise ValueError("joint reference requires an exact ReadinessStageBinding")
    if type(corrections) is not GoalConsistentObservedGraspCorrectionBatch:
        raise ValueError("joint B requires exact goal-consistent observed-grasp corrections")
    if curated is not None and type(curated) is not CuratedReadinessBatch:
        raise ValueError("joint A curation requires an exact CuratedReadinessBatch")
    # Why: inspect native current/next flags before any copier can normalize aliases.
    _observed_grasp_flags(reference.episode)
    owned_reference = ReadinessStageBinding(reference.episode, reference.stages)
    owned_b = copy_readiness_correction_batch(corrections)
    owned_c = None if curated is None else copy_curated_readiness_batch(curated)
    a, b = owned_reference.episode, owned_b.episode
    c = None if owned_c is None else owned_c.episode
    if (
        a.episode_id != "pick-cube-observed-grasp-teacher-17"
        or a.context != pick_cube_training_context(17)
        or a.role != "train"
        or owned_b.train_seed != 19
        or b.episode_id != "pick-cube-smoke-19"
        or b.context != pick_cube_training_context(19)
        or b.role != "train"
    ):
        raise ValueError("joint readiness requires original A17 reference/B19/A17 TRAIN sources")
    if (
        owned_c is not None
        and c is not None
        and (
            owned_c.train_seed != 17
            or c.episode_id != "pick-cube-smoke-17"
            or c.context != pick_cube_training_context(17)
            or c.role != "train"
        )
    ):
        raise ValueError("joint readiness requires original A17 reference/B19/A17 TRAIN sources")
    if (
        type(a.spec.observation_dim) is not int
        or a.spec.observation_dim != 42
        or a.spec != b.spec
        or (c is not None and a.spec != c.spec)
        or a.spec.action_units != _UNITS
        or a.spec.lower_bounds != (-1.0,) * 4
        or a.spec.upper_bounds != (1.0,) * 4
    ):
        raise ValueError("joint readiness requires the same official raw42 normalized spec")
    sources = (a, b) if c is None else (a, b, c)
    counts = tuple(len(e.transitions) for e in sources)
    if any(not 1 <= count <= 50 for count in counts) or sum(counts) > 128:
        raise ValueError(
            "joint readiness requires one to fifty rows per source and at most128 total"
        )
    config = validate_taskspace_history_recipe(
        PickCubeTaskSpaceHistoryRecipe(mode="executed") if recipe is None else recipe
    )
    if config.mode != "executed":
        raise ValueError("joint readiness requires the fixed executed TS12 recipe")
    # Every present original release gate precedes EVERY observation projection.
    _, b_targets = owned_b.training_pairs(now=now)
    c_targets = () if owned_c is None else owned_c.training_pairs(now=now)[1]
    return owned_reference, owned_b, owned_c, config, b_targets, c_targets


def _admit_retained(retained_reference, raw_spec, base_rows):
    """Own the optional already-qualified TRAIN20 source before any projection."""
    if type(retained_reference) is not ReadinessStageBinding:
        raise ValueError("retained TRAIN20 reference requires an exact ReadinessStageBinding")
    # Inspect original owned flags before a copy can normalize forged int/bool aliases.
    _observed_grasp_flags(retained_reference.episode)
    episode = retained_reference.episode
    if (
        episode.role != "train"
        or episode.episode_id != "pick-cube-observed-grasp-teacher-20"
        or episode.context != pick_cube_training_context(20)
        or episode.spec != raw_spec
        or episode.spec.observation_dim != 42
        or not 1 <= len(episode.transitions) <= 50
        or base_rows + len(episode.transitions) > 130
    ):
        raise ValueError(
            "retained reference requires original raw42 TRAIN20 and at most130 combined rows"
        )
    owned = ReadinessStageBinding(episode, retained_reference.stages)
    # Actual executed actions, not correction targets, define causal history.
    # Validate native32 commands now so bad optional data cannot admit earlier projections.
    state = ActionHistoryState()
    for row in owned.episode.transitions:
        state = advance_action_history(state, row.action)
    return owned


def _row(
    index,
    episode,
    feature,
    target,
    command,
    stage,
    definition,
    provenance,
    source_kind,
    observed_at,
    arrived_at,
    step,
    source_observation,
):
    return dict(
        row_index=index,
        features=list(feature),
        source_observation=list(source_observation),
        target=list(target),
        actual_command=list(command),
        episode_id=episode.episode_id,
        context=episode.context,
        role=episode.role,
        source_kind=source_kind,
        step=step,
        stage=stage,
        teacher_definition=definition,
        provenance=provenance,
        observed_at=observed_at,
        labels_arrived_at=arrived_at,
    )


def _projection_work(count):
    return dict(
        rows=count,
        observation_projections=2 * count,
        source_values_validated=84 * count,
        projected_values=24 * count,
        projected_numeric_bytes=192 * count,
        history_advances=count,
        source_observation_values=42 * count,
        source_observation_numeric_bytes=336 * count,
    )


def prepare_joint_readiness_data(
    reference: ReadinessStageBinding,
    corrections: GoalConsistentObservedGraspCorrectionBatch,
    curated: CuratedReadinessBatch | None,
    *,
    now: int,
    recipe: PickCubeTaskSpaceHistoryRecipe | None = None,
    retained_reference: ReadinessStageBinding | None = None,
    old_information_control: bool = False,
) -> JointReadinessData:
    """Own and release known TRAIN sources, then project actual histories.

    Explicit curated=None selects original A+B; omitted arguments remain errors.
    A supplied curated source must pass its full original admission/release gate.
    Optional retained_reference adds original qualified TRAIN20 information; it
    never expands the original base128-row admission or grants qualification.
    Explicit old_information_control instead repeats all15 original A17 rows in
    their original order; it excludes retained TRAIN20 and changes no labels.

    Provisional targets stay distinct from actually executed commands. Each episode
    starts its own causal history. This helper neither queries a teacher nor gives
    trusted Python reference bindings external-file qualification authority.
    """
    if type(old_information_control) is not bool:
        raise ValueError("old_information_control must be an exact bool")
    if old_information_control and (curated is None or retained_reference is not None):
        raise ValueError("old information control requires curation and excludes retained TRAIN20")
    binding, b, c, config, raw_b_targets, raw_c_targets = _admit(
        reference, corrections, curated, now, recipe
    )
    if old_information_control and (
        len(binding.episode.transitions) != 15
        or c is None
        or len(binding.episode.transitions)
        + len(b.episode.transitions)
        + len(c.episode.transitions)
        > 115
    ):
        raise ValueError(
            "old information control requires original A17 fifteen rows and base at most115"
        )
    if old_information_control:
        # Validate all repeated native commands before any source projection.
        state = ActionHistoryState()
        for row in binding.episode.transitions:
            state = advance_action_history(state, row.action)
    retained = None
    if retained_reference is not None:
        base_rows = len(binding.episode.transitions) + len(b.episode.transitions)
        if c is not None:
            base_rows += len(c.episode.transitions)
        retained = _admit_retained(retained_reference, binding.episode.spec, base_rows)
    a_model = action_history_training_episode(binding.episode, config)
    a_features, a_targets = continuous_training_pairs(a_model, role="train")
    b_view = action_history_correction_view(b, config)
    b_features, b_targets = b_view.model.training_pairs(now=now)
    c_features, c_targets = (
        ((), ()) if c is None else project_curated_training_pairs(c, now=now, recipe=config)
    )
    if b_targets != raw_b_targets or c_targets != raw_c_targets:
        raise ValueError("joint feature projection changed original released targets")
    spec = a_model.spec
    if spec != b_view.model.episode.spec or spec.observation_dim != 12:
        raise ValueError("joint projected source specs differ")
    rows = []
    for i, (record, feature, target, stage) in enumerate(
        zip(a_model.transitions, a_features, a_targets, binding.stages, strict=True)
    ):
        rows.append(
            _row(
                i,
                a_model,
                feature,
                target,
                record.action,
                stage,
                OBSERVED_GRASP_DEFINITION,
                None,
                "qualified_demonstration",
                None,
                None,
                record.step,
                binding.episode.transitions[i].observation,
            )
        )
    n_a = len(rows)
    for i, (record, label, feature, target) in enumerate(
        zip(b_view.model.episode.transitions, b.labels, b_features, b_targets, strict=True)
    ):
        row = _row(
            n_a + i,
            b.episode,
            feature,
            target,
            record.action,
            label.stage,
            GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
            GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
            "provisional_correction",
            label.observed_at,
            label.labels_arrived_at,
            record.step,
            label.observation,
        )
        rows.append(row)
    n_ab = len(rows)
    if c is not None:
        for i, (record, label, feature, target) in enumerate(
            zip(c.episode.transitions, c.labels, c_features, c_targets, strict=True)
        ):
            row = _row(
                n_ab + i,
                c.episode,
                feature,
                target,
                record.action,
                label.stage,
                CURATED_DEFINITION,
                CURATED_PROVENANCE,
                "curated_provisional_correction",
                label.source_observed_at,
                label.arrived_at,
                record.step,
                label.observation,
            )
            row.update(
                step=record.step,
                source_observed_at=label.source_observed_at,
                queried_at=label.queried_at,
                arrived_at=label.arrived_at,
                source_available_at=c.source_available_at,
                geometry_definition=GEOMETRY_DEFINITION,
            )
            rows.append(row)
    n_abc = len(rows)
    if retained is not None:
        retained_model = action_history_training_episode(retained.episode, config)
        retained_features, retained_targets = continuous_training_pairs(
            retained_model, role="train"
        )
        if retained_model.spec != spec:
            raise ValueError("retained TRAIN20 projected spec differs")
        for i, (record, feature, target, stage) in enumerate(
            zip(
                retained_model.transitions,
                retained_features,
                retained_targets,
                retained.stages,
                strict=True,
            )
        ):
            rows.append(
                _row(
                    n_abc + i,
                    retained_model,
                    feature,
                    target,
                    record.action,
                    stage,
                    OBSERVED_GRASP_DEFINITION,
                    None,
                    "retained_qualified_demonstration",
                    None,
                    None,
                    record.step,
                    retained.episode.transitions[i].observation,
                )
            )
    if old_information_control:
        # Why: repeat original information at the same packet shape, but recompute
        # this episode's causal history rather than charging copied features as projections.
        repeated_model = action_history_training_episode(binding.episode, config)
        repeated_features, repeated_targets = continuous_training_pairs(
            repeated_model, role="train"
        )
        if repeated_model.spec != spec:
            raise ValueError("repeated original A17 projected spec differs")
        for i, (record, feature, target, stage) in enumerate(
            zip(
                repeated_model.transitions,
                repeated_features,
                repeated_targets,
                binding.stages,
                strict=True,
            )
        ):
            row = _row(
                n_abc + i,
                repeated_model,
                feature,
                target,
                record.action,
                stage,
                OBSERVED_GRASP_DEFINITION,
                None,
                "old_reference_repetition",
                None,
                None,
                record.step,
                binding.episode.transitions[i].observation,
            )
            row["source_row_index"] = i
            rows.append(row)
    n = len(rows)
    groups: tuple[tuple[str, int, int], ...] = (("a", 0, n_a), ("b", n_a, n_ab))
    if c is not None:
        groups += (("curated_a", n_ab, n_abc),)
    if retained is not None:
        groups += (("retained_train20", n_abc, n),)
    if old_information_control:
        groups += (("repeated_a", n_abc, n),)
    work: dict[str, Any] = _projection_work(n)
    work.update(
        per_group={name: _projection_work(stop - start) for name, start, stop in groups},
        curated_projection_calls=0 if c is None else 1,
        b_label_observation_values=12 * (n_ab - n_a),
        b_label_target_values=4 * (n_ab - n_a),
        b_label_numeric_bytes=128 * (n_ab - n_a),
        curated_label_observation_values=12 * (n_abc - n_ab),
        curated_label_target_values=4 * (n_abc - n_ab),
        curated_label_numeric_bytes=128 * (n_abc - n_ab),
        physical_scale_divisions=12 * n,
        scope="logical returned projections/history and model-space label payloads; "
        "validation/release copies, raw ownership, Python containers/temporaries/RSS additional; "
        "zero learner/fit/RNG/teacher/geometry/IO work",
    )
    if retained is not None:
        work.update(
            retained_reference_rows=n - n_abc,
            retained_reference_projection_calls=1,
            retained_reference_command_validation_advances=n - n_abc,
            retained_reference_observation_values=12 * (n - n_abc),
            retained_reference_target_values=4 * (n - n_abc),
            retained_reference_numeric_bytes=128 * (n - n_abc),
        )
    if old_information_control:
        work.update(
            repeated_reference_rows=15,
            repeated_reference_projection_calls=1,
            repeated_reference_command_validation_advances=15,
            repeated_reference_observation_values=12 * 15,
            repeated_reference_target_values=4 * 15,
            repeated_reference_numeric_bytes=128 * 15,
        )
    return JointReadinessData(spec, tuple(rows), groups, work, config)
