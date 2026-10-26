"""Arrived corrections through the existing private stream.

Inputs are a frozen TRAIN BP/PC/CPC or nearest owner and provisional labels.
Neural learners use eight exposures; nearest appends the batch once. Output is a
bounded report and detached candidate on success. No refit, simulator, promotion,
model archive or experiment selection. Optional read-only TRAIN step diagnostics
use the first paid corrective step; no extra optimizer update. Optional fixed TRAIN
reference rehearsal uses the existing separately charged consolidation helper.
Explicit BP binary-gripper mode keeps endpoint labels and mixed-loss receipts;
raw student commands and the soft serving controller remain unchanged.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import json
from time import monotonic
from typing import Any, cast

import numpy as np

from src.adapters.continuous_learnability_demo import (
    ContinuousLearnabilityResult,
    _same_state as _same_neural_state,
)
from src.adapters.numpy_continuous_bc import (
    NumpyContinuousBCLearner,
    _snapshot as own_bc_snapshot,
    _policy as own_bc_policy,
)
from src.adapters.numpy_continuous_pc import (
    NumpyContinuousPCLearner,
    _snapshot as own_pc_snapshot,
)
from src.adapters.continuous_learnability_modes import RecordedContinuousPC
from src.adapters.continuous_head_learnability import RecordedContinuousHead, head_protocol
from src.adapters.continuous_head_corrections import (
    record_head_work as _record_head_work,
    require_replay_evidence as _require_head_replay_evidence,
)
from src.adapters.readiness_step_diagnostics import (
    prepare_readiness_step_bindings,
    run_readiness_step_diagnostics,
)
from src.adapters.numpy_nearest_continuous_bc import (
    OBJECTIVE as NEAREST_OBJECTIVE,
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
    _snapshot as own_nearest_snapshot,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.continuous_stream_rehearsal import (
    ContinuousReplayPlan,
    ContinuousStreamRehearsal,
    prepare_replay_bank,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_bc import MIXED_OBJECTIVE, OBJECTIVE, _binary_gripper_targets
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PROVENANCE
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ObservedGraspCorrectionBatch,
    GoalConsistentObservedGraspCorrectionBatch,
    copy_readiness_correction_batch,
    readiness_correction_provenance,
    readiness_correction_definition,
)
from src.core.pick_cube_public_corrections import public_vector_correction_view
from src.core.pick_cube_public_vectors import PickCubePublicVectorProjection
from src.core.pick_cube_state_corrections import state_quaternion_correction_view
from src.core.pick_cube_state_quaternions import PickCubeStateQuaternionProjection
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.observation_standardization import _standardization, standardization_numeric_bytes
from src.core.pick_cube_action_history import (
    ActionHistoryRecipe,
    is_taskspace_recipe,
    validate_action_history_recipe,
    validate_action_history_training_episode,
)
from src.core.pick_cube_action_history_corrections import (
    action_history_correction_metadata,
    action_history_correction_view,
)

CORRECTION_UPDATES = 8


@dataclass(frozen=True)
class NearestContinuousCorrectionResult:
    """Owned report and detached nearest learner; incomplete work releases none."""

    report: dict[str, Any]
    candidate: NumpyNearestContinuousBCLearner | None


def _same_state(before, after) -> bool:
    if type(before) is not NearestContinuousBCSnapshot:
        return _same_neural_state(before, after)
    if type(after) is not NearestContinuousBCSnapshot:
        return False
    before = own_nearest_snapshot(before, before.policy)
    after = own_nearest_snapshot(after, after.policy)
    return (
        before.policy == after.policy
        and before.completed_updates == after.completed_updates
        and before.training_examples == after.training_examples
        and np.array_equal(before.prototype_features, after.prototype_features)
        and np.array_equal(before.prototype_actions, after.prototype_actions)
    )


def _require_owners(source, runtime, initial) -> None:
    actor = runtime.actor.snapshot_state()
    if (
        not _same_state(initial, source.snapshot_state())
        or actor.actor_version != runtime.actor.version
        or actor.actor_version != "actor-0"
        or not _same_state(initial, actor.state)
    ):
        raise RuntimeError("continuous correction source/actor changed")


def _require_nearest_append(initial, state, targets) -> None:
    rows = initial.training_examples
    if (
        not np.array_equal(initial.prototype_features, state.prototype_features[:rows])
        or not np.array_equal(initial.prototype_actions, state.prototype_actions[:rows])
        or not np.array_equal(targets, state.prototype_actions[rows:])
    ):
        raise RuntimeError("nearest correction changed original prefix or arrived targets")


def _require_nearest_states(bindings, runtime, initial, expected, readout_expected) -> None:
    source, actor, backend, candidate, readout = bindings
    if (
        runtime._actor is not actor
        or actor._learner is not backend
        or runtime._candidate is not candidate
        or actor._version != "actor-0"
    ):
        raise RuntimeError("nearest correction owner identity changed")
    # Why: a later public snapshot/receipt/clock getter can invalidate an earlier
    # observed copy. Check exact native owners after all such getters, before work
    # or release. This adapter owns this bounded synchronous runtime.
    for model, state in (
        (source, initial),
        (backend, initial),
        (candidate, expected),
        (readout, readout_expected),
    ):
        if type(model) is not NumpyNearestContinuousBCLearner or model._policy != state.policy:
            raise RuntimeError("nearest correction native policy changed")
        if not _same_state(state, own_nearest_snapshot(model._state, state.policy)):
            raise RuntimeError("nearest correction native owner changed after public reads")


def _require_pc_states(bindings, runtime, initial, expected, readout_expected, reports) -> None:
    native_limit = bindings[8] if len(bindings) == 9 else CORRECTION_UPDATES
    source, facade, actor, backend, actor_native, candidate, candidate_native, readout = bindings[
        :8
    ]
    if (
        runtime._actor is not actor
        or actor._learner is not backend
        or runtime._candidate is not candidate
        or runtime._inbox._learner is not candidate
        or actor._version != "actor-0"
        or runtime._candidate_version != "candidate-0"
        or runtime._inbox._learner_version != "candidate-0"
    ):
        raise RuntimeError("PC correction owner identity changed")
    for wrapper, model in (
        (facade, source),
        (backend, actor_native),
        (candidate, candidate_native),
    ):
        if (
            type(wrapper) is not RecordedContinuousPC
            or wrapper._learner is not model
            or wrapper._reports is not reports
            or wrapper._limit != native_limit
        ):
            raise RuntimeError("PC correction native facade changed")
    models = (
        (source, initial),
        (actor_native, initial),
        (candidate_native, expected),
        (readout, readout_expected),
    )
    if len({id(model) for model, _ in models}) != 4:
        raise RuntimeError("PC correction owners must be separate")
    # Public getters above may mutate another owner. Finish with exact native
    # fields and coherent wake validation, without another callback cycle.
    for model, state in models:
        if type(model) is not NumpyContinuousPCLearner or model._policy != state.policy:
            raise RuntimeError("PC correction native policy changed")
        if not _same_state(state, own_pc_snapshot(model._state, state.policy)):
            raise RuntimeError("PC correction native owner changed after public reads")


def _require_pc_evidence(
    runtime, budget, clock, now, receipts, native, expected_native, authority
) -> None:
    limits, timer, started_at, applied_map, original_receipts = authority
    observed_receipts = tuple(runtime._inbox._applied.values())
    if (
        runtime._budget is not budget
        or runtime._inbox._budget is not budget
        or runtime._inbox._clock is not clock
        or budget.budget is not limits
        or budget.clock is not timer
        or type(budget.started_at) is not type(started_at)
        or budget.started_at != started_at
        or type(budget.budget.max_training_updates) is not int
        or budget.budget.max_training_updates != CORRECTION_UPDATES
        or type(budget.updates_completed) is not int
        or budget.updates_completed != len(receipts)
        or type(budget.replay_examples_completed) is not int
        or budget.replay_examples_completed != 0
        or type(clock._time) is not int
        or clock._time != now
        or runtime._inbox._applied is not applied_map
        or len(observed_receipts) != len(original_receipts)
        or any(
            current is not original
            for current, original in zip(observed_receipts, original_receipts)
        )
        or [asdict(item) for item in runtime._inbox._applied.values()] != receipts
        or native != expected_native
    ):
        raise RuntimeError("PC correction preceding receipt, native work or charge changed")


def _admit(
    source,
    batch,
    now,
    observation_projection,
    action_history_recipe=None,
    *,
    binary_gripper_axis: int | None = None,
):
    if action_history_recipe is not None and observation_projection is not None:
        raise ValueError("action history and observation projection are mutually exclusive")
    if type(source) not in (
        NumpyContinuousBCLearner,
        NumpyContinuousPCLearner,
        NumpyNearestContinuousBCLearner,
    ):
        raise ValueError("correction updates require an exact continuous BP, PC or nearest learner")
    mixed = binary_gripper_axis is not None
    if mixed and (
        type(binary_gripper_axis) is not int
        or binary_gripper_axis != 3
        or type(source) is not NumpyContinuousBCLearner
    ):
        raise ValueError("binary gripper corrections require an exact BP learner and native axis3")
    policy = source.policy.base if type(source) is NumpyContinuousPCLearner else source.policy
    if mixed:
        policy = own_bc_policy(policy)
        if type(policy.binary_gripper_axis) is not int or policy.binary_gripper_axis != 3:
            raise ValueError("requested binary gripper axis differs from the owned BP policy")
    if (
        policy.training_role != "train"
        or (
            type(source) is not NumpyNearestContinuousBCLearner
            and policy.binary_gripper_axis is not None
            and not mixed
        )
        or (
            policy.observation_standardization is None
            and not is_taskspace_recipe(action_history_recipe)
        )
    ):
        raise ValueError(
            "correction updates require TRAIN default-MSE (or explicitly selected BP binary "
            "gripper) with frozen statistics"
        )
    if type(batch) not in (
        PickCubeCorrectionBatch,
        ReadinessCorrectionBatch,
        ObservedGraspCorrectionBatch,
        GoalConsistentObservedGraspCorrectionBatch,
    ):
        raise ValueError("correction updates require an exact paired correction batch")
    owned = (
        copy_readiness_correction_batch(batch)
        if type(batch)
        in (
            ReadinessCorrectionBatch,
            ObservedGraspCorrectionBatch,
            GoalConsistentObservedGraspCorrectionBatch,
        )
        else PickCubeCorrectionBatch(batch.train_seed, batch.episode, batch.labels)
    )
    features, targets = owned.training_pairs(now=now)
    projection_metadata = None
    if action_history_recipe is not None:
        if type(source) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner):
            raise ValueError("action history corrections require an exact neural BP or PC learner")
        if type(owned) not in (
            ReadinessCorrectionBatch,
            ObservedGraspCorrectionBatch,
            GoalConsistentObservedGraspCorrectionBatch,
        ):
            raise ValueError("action history corrections require readiness corrections")
        if is_taskspace_recipe(action_history_recipe):
            if policy.spec.observation_dim != 12 or policy.observation_standardization is not None:
                raise ValueError(
                    "taskspace12 corrections require fixed physical scaling and no fitted statistics"
                )
        else:
            stats = _standardization(policy.observation_standardization, 47)
            if (
                policy.spec.observation_dim != 47
                or stats is None
                or stats.means[-5:] != (0.0,) * 5
                or stats.scales[-5:] != (1.0,) * 5
                or getattr(stats, "effective_scales", stats.scales)[-5:] != (1.0,) * 5
            ):
                raise ValueError(
                    "action history corrections require frozen47D statistics with fixed history scaling"
                )
        history_view = action_history_correction_view(
            cast(ReadinessCorrectionBatch, owned), action_history_recipe
        )
        projection_metadata = action_history_correction_metadata(
            history_view.source, history_view.recipe
        )
        owned = history_view.model
        features, targets = owned.training_pairs(now=now)
    elif observation_projection is not None:
        # Why: derive the model view from the arrived raw batch at this boundary;
        # a caller-supplied projected copy is not evidence of raw label pairing.
        if type(observation_projection) is PickCubeStateQuaternionProjection:
            if type(owned) not in (
                ReadinessCorrectionBatch,
                ObservedGraspCorrectionBatch,
                GoalConsistentObservedGraspCorrectionBatch,
            ):
                raise ValueError("canonical state42 projection requires readiness corrections")
            state_view = state_quaternion_correction_view(
                cast(ReadinessCorrectionBatch, owned), observation_projection
            )
            owned = state_view.model
            features, targets = owned.training_pairs(now=now)
            projection_metadata = dict(
                projection=asdict(state_view.projection),
                source_episode_id=state_view.source.episode.episode_id,
                source_spec=asdict(state_view.source.episode.spec),
                model_spec=asdict(owned.episode.spec),
                work=asdict(state_view.work),
                source_encoding="official_state42_raw",
                model_encoding=state_view.projection.definition,
            )
        else:
            view = public_vector_correction_view(
                owned, observation_projection, source_recipe=pick_cube_rgb_recipe()
            )
            owned = view.model
            features, targets = owned.training_pairs(now=now)
            projection_metadata = dict(
                projection=asdict(view.projection),
                source_recipe=asdict(view.source_recipe),
                source_episode_id=view.source.episode.episode_id,
                source_spec=asdict(view.source.episode.spec),
                model_spec=asdict(owned.episode.spec),
                work=asdict(view.work),
            )
    if mixed:
        # The arrived teacher targets are binary; actual recorded student commands
        # may be soft tanh values and must remain untouched as causal evidence.
        _binary_gripper_targets(np.asarray(targets, dtype=np.float64), policy.spec, 3)
    if policy.spec != owned.episode.spec:
        raise ValueError("correction batch differs from frozen learner spec")
    if type(source) is NumpyNearestContinuousBCLearner:
        initial = source.snapshot_state()
        if initial.training_examples == 0:
            raise ValueError("nearest corrections require existing released prototypes")
        if (
            initial.training_examples + len(targets)
            > cast(NearestContinuousBCPolicy, policy).max_prototypes
        ):
            raise ValueError("nearest correction prototype capacity exceeded before private work")
    return (
        owned,
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        projection_metadata,
    )


def _canonical_native_state(owner):
    if type(owner) is RecordedContinuousHead:
        owner.require_reports()
        owner.require_latest()
        return _canonical_native_state(owner._learner)
    if type(owner) is NumpyContinuousBCLearner:
        return own_bc_snapshot(owner._state, owner._policy)
    if type(owner) is NumpyContinuousPCLearner:
        return own_pc_snapshot(owner._state, owner._policy)
    if type(owner) is NumpyNearestContinuousBCLearner:
        return own_nearest_snapshot(owner._state, owner._policy)
    raise ValueError("canonical correction requires an exact continuous learner owner")


def _require_canonical_original(owner, original) -> None:
    observed = owner.snapshot_state()
    policy = (
        _canonical_native_state(owner).policy
        if type(owner) is RecordedContinuousHead
        else owner.policy
    )
    native = _canonical_native_state(owner)
    if (
        policy != original.policy
        or not _same_state(original, observed)
        or not _same_state(original, native)
    ):
        raise RuntimeError("canonical correction owner changed during admission or reporting")


def _require_canonical_bp_states(bindings, runtime, source, initial, expected, readout_expected):
    actor, backend, candidate, readout = bindings[:4]
    head_pins = bindings[4] if len(bindings) == 5 else ()
    if head_pins:
        if (
            head_pins[0][1] is not source
            or head_pins[1][0] is not backend
            or head_pins[2][0] is not candidate
        ):
            raise RuntimeError("GN correction original runtime wrapper changed")
        for wrapper, native, sink, records, latest, original in head_pins:
            if (
                type(wrapper) is not RecordedContinuousHead
                or wrapper._learner is not native
                or wrapper._reports is not sink
                or type(wrapper._limit) is not int
                or wrapper._limit != head_pins[0][5][2]
                or not CORRECTION_UPDATES <= wrapper._limit <= 2 * CORRECTION_UPDATES
                or wrapper._records is not records
                or wrapper._latest is not latest
                or wrapper._original is not original
            ):
                raise RuntimeError("GN correction native facade identity changed")
            wrapper.require_reports(sink)
            wrapper.require_latest()
    if (
        runtime._actor is not actor
        or actor._learner is not backend
        or runtime._candidate is not candidate
        or runtime._inbox._learner is not candidate
        or type(actor._version) is not str
        or actor._version != "actor-0"
        or runtime._candidate_version != "candidate-0"
        or runtime._inbox._learner_version != "candidate-0"
    ):
        raise RuntimeError("canonical BP correction owner identity changed")
    models = (
        (source, initial),
        (backend, initial),
        (candidate, expected),
        (readout, readout_expected),
    )
    native_models = tuple(
        (owner._learner if type(owner) is RecordedContinuousHead else owner, state)
        for owner, state in models
    )
    if len({id(owner) for owner, _ in native_models}) != 4:
        raise RuntimeError("canonical BP correction owners became aliased")
    for owner, state in native_models:
        if type(owner) is not NumpyContinuousBCLearner or owner._policy != state.policy:
            raise RuntimeError("canonical BP correction native policy changed")
        if not _same_state(state, own_bc_snapshot(owner._state, state.policy)):
            raise RuntimeError("canonical BP correction native owner changed")


def _require_canonical_bp_evidence(runtime, budget, clock, now, receipts, originals, authority):
    limits, contents, timer, started_at, inbox, applied_map = authority
    current_limits = asdict(budget.budget)
    current = tuple(runtime._inbox._applied.values())
    records = [asdict(item) for item in current]
    receipt_types_exact = len(records) == len(receipts) and all(
        all(type(record.get(key)) is type(value) for key, value in expected.items())
        and type(record.get("diagnostic")) is dict
        and all(
            type(record["diagnostic"].get(key)) is type(value)
            for key, value in expected["diagnostic"].items()
        )
        for record, expected in zip(records, receipts, strict=True)
    )
    if (
        runtime._budget is not budget
        or runtime._inbox is not inbox
        or inbox._budget is not budget
        or inbox._clock is not clock
        or budget.budget is not limits
        or budget.clock is not timer
        or type(budget.started_at) is not type(started_at)
        or budget.started_at != started_at
        or current_limits != contents
        or any(type(current_limits[key]) is not type(value) for key, value in contents.items())
        or type(budget.updates_completed) is not int
        or budget.updates_completed != len(receipts)
        or type(budget.replay_examples_completed) is not int
        or budget.replay_examples_completed != 0
        or type(clock._time) is not int
        or clock._time != now
        or inbox._applied is not applied_map
        or len(current) != len(originals)
        or any(actual is not original for actual, original in zip(current, originals, strict=True))
        or not receipt_types_exact
        or records != receipts
    ):
        raise RuntimeError("canonical BP correction receipt, budget or clock changed")


def run_continuous_correction_update(
    source,
    batch,
    *,
    now: int,
    observation_projection: (
        PickCubePublicVectorProjection | PickCubeStateQuaternionProjection | None
    ) = None,
    action_history_recipe: ActionHistoryRecipe | None = None,
    replay_plan: ContinuousReplayPlan | None = None,
    reference_episodes: tuple = (),
    binary_gripper_axis: int | None = None,
    diagnostic_stage_bindings: tuple[ReadinessStageBinding, ReadinessStageBinding] | None = None,
    head_gauss_newton: bool = False,
) -> ContinuousLearnabilityResult | NearestContinuousCorrectionResult:
    """Apply eight neural batches or one nearest append, preserving original clocks.

    Admission raises before forks. Runtime failures retain permanent receipts and
    progress, release no learner, and never admit another update after failure.
    """
    # Why: selected mixed loss must retain the same strong native owner/receipt/
    # clock binding as raw-evidence feature views, including ordinary state inputs.
    if type(head_gauss_newton) is not bool:
        raise ValueError("head_gauss_newton must be a native bool")
    head = head_gauss_newton
    if head and (
        type(source) is not NumpyContinuousBCLearner
        or binary_gripper_axis is not None
        or diagnostic_stage_bindings is not None
    ):
        raise ValueError("GN corrections require exact BP common-MSE without step diagnostics")
    mixed = binary_gripper_axis is not None
    step_diagnostics = diagnostic_stage_bindings is not None
    if step_diagnostics and (
        type(source) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner)
        or type(batch)
        not in (
            ReadinessCorrectionBatch,
            ObservedGraspCorrectionBatch,
            GoalConsistentObservedGraspCorrectionBatch,
        )
        or (
            observation_projection is not None
            and type(observation_projection) is not PickCubeStateQuaternionProjection
        )
    ):
        raise ValueError("readiness step diagnostics require neural readiness corrections")
    if mixed and (
        type(binary_gripper_axis) is not int
        or binary_gripper_axis != 3
        or type(source) is not NumpyContinuousBCLearner
    ):
        raise ValueError("binary gripper corrections require an exact BP learner and native axis3")
    replay = replay_plan is not None
    if not replay and (type(reference_episodes) is not tuple or reference_episodes):
        raise ValueError("reference episodes require an explicit replay plan")
    if replay and type(source) is NumpyNearestContinuousBCLearner:
        raise ValueError("reference rehearsal is supported only for neural corrections")
    history = action_history_recipe is not None
    if action_history_recipe is not None:
        action_history_recipe = validate_action_history_recipe(action_history_recipe)
    history_request = (
        None
        if action_history_recipe is None
        else json.dumps(asdict(action_history_recipe), sort_keys=True, allow_nan=False)
    )
    canonical = (
        type(observation_projection) is PickCubeStateQuaternionProjection
        or history
        or replay
        or mixed
        or step_diagnostics
        or head
        or type(batch) in (ObservedGraspCorrectionBatch, GoalConsistentObservedGraspCorrectionBatch)
    )
    # Capture before any public policy/view/clock observer can change the source
    # and make an altered state appear to be the original continuation.
    initial: Any = _canonical_native_state(source) if canonical else None
    if head and (
        initial.policy.training_role != "train"
        or initial.policy.binary_gripper_axis is not None
        or initial.policy.spec.observation_dim > 64
        or initial.policy.hidden_dim > 64
    ):
        raise ValueError(
            "GN corrections require TRAIN common-MSE with input/hidden dimensions at most64"
        )
    if mixed and (
        type(initial.policy.binary_gripper_axis) is not int
        or initial.policy.binary_gripper_axis != 3
    ):
        raise ValueError("requested binary gripper axis differs from the owned BP policy")
    replay_bank: tuple = ()
    replay_events = replay_rows = replay_bytes = 0
    if replay:
        assert replay_plan is not None
        replay_plan, replay_bank, replay_rows, replay_bytes = prepare_replay_bank(
            replay_plan, reference_episodes
        )
        replay_events = CORRECTION_UPDATES // replay_plan.every_updates
        if head and (
            not 1 <= replay_rows <= 128
            or not CORRECTION_UPDATES <= CORRECTION_UPDATES + replay_events <= 16
        ):
            raise ValueError(
                "GN reference rehearsal requires1..128 bank rows and at most16 total updates"
            )
    started = monotonic()
    # Preserve the legacy four-argument admission call when no option is selected.
    if mixed:
        owned, features, targets, projection_metadata = _admit(
            source,
            batch,
            now,
            observation_projection,
            action_history_recipe,
            binary_gripper_axis=binary_gripper_axis,
        )
    else:
        owned, features, targets, projection_metadata = (
            _admit(source, batch, now, observation_projection, action_history_recipe)
            if history
            else _admit(source, batch, now, observation_projection)
        )
    if head and (not 1 <= len(features) <= 128 or len(initial.policy.spec.action_units) > 16):
        raise ValueError("GN corrections require1..128 arrived rows and at most16 action axes")
    if history:
        assert action_history_recipe is not None
        if (
            json.dumps(asdict(action_history_recipe), sort_keys=True, allow_nan=False)
            != history_request
            or json.dumps(projection_metadata["projection"], sort_keys=True, allow_nan=False)
            != history_request
        ):
            raise ValueError("original history correction request changed during admission")
    if replay and history:
        assert action_history_recipe is not None
        replay_bank = tuple(
            validate_action_history_training_episode(episode, action_history_recipe)
            for episode in replay_bank
        )
    if replay and any(episode.spec != owned.episode.spec for episode in replay_bank):
        raise ValueError("reference rehearsal bank differs from the frozen correction model spec")
    step_bindings = (
        prepare_readiness_step_bindings(
            diagnostic_stage_bindings, owned.episode.spec, action_history_recipe
        )
        if diagnostic_stage_bindings is not None
        else None
    )
    if mixed:
        _binary_gripper_targets(targets, initial.policy.spec, 3)
        for episode in replay_bank:
            _binary_gripper_targets(
                np.asarray([row.action for row in episode.transitions], dtype=np.float64),
                initial.policy.spec,
                3,
            )
    if canonical:
        _require_canonical_original(source, initial)
    else:
        initial = source.snapshot_state()
    nearest = type(source) is NumpyNearestContinuousBCLearner
    pc = type(source) is NumpyContinuousPCLearner
    native_updates: list[dict[str, Any]] = []
    validated_native: list[dict[str, Any]] = []
    native_source: Any = (
        RecordedContinuousPC(source, native_updates, CORRECTION_UPDATES + replay_events)
        if pc
        else RecordedContinuousHead(source, native_updates, CORRECTION_UPDATES + replay_events)
        if head
        else source
    )
    head_records = native_source._records if head else None
    sealed_native_updates = []
    updates = 1 if nearest else CORRECTION_UPDATES
    objective = (
        NEAREST_OBJECTIVE
        if nearest
        else CONTINUOUS_PC_ENERGY_ID
        if pc
        else MIXED_OBJECTIVE
        if mixed
        else OBJECTIVE
    )
    clock = LogicalClock(now)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), monotonic)
    birth_budget = (budget.budget, budget.clock, budget.started_at) if pc else ()
    canonical_birth = (
        (budget.budget, asdict(budget.budget), budget.clock, budget.started_at)
        if canonical and not pc and not nearest
        else ()
    )
    if canonical:
        _require_canonical_original(source, initial)
    runtime = ActorShadowRuntime(
        native_source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=updates,
        max_consolidations=replay_events,
    )
    head_pins: tuple[tuple[Any, ...], ...] = ()
    if head:
        wrappers = cast(
            tuple[RecordedContinuousHead, ...],
            (native_source, runtime._actor._learner, runtime._candidate),
        )
        head_pins = tuple(
            (
                wrapper,
                wrapper._learner,
                wrapper._reports,
                wrapper._records,
                wrapper._latest,
                wrapper._original,
            )
            for wrapper in wrappers
        )
    rehearsal = None
    head_replay_budget = None
    head_consolidation_owner = runtime._consolidations if head and replay else None
    head_consolidation_objects: tuple[Any, ...] = ()
    head_consolidation_records: list[dict[str, Any]] = []
    replay_returned_native: list[dict[str, Any]] | None = None
    replay_native_expected: Any = None

    def observe_replay_native_return():
        nonlocal replay_returned_native, sealed_native_updates, replay_native_expected
        # Own paid native work before helper/source observers can change its sink.
        if head:
            assert head_records is not None
            replay_returned_native = [json.loads(payload) for _, payload in head_records]
            sealed_native_updates = deepcopy(replay_returned_native)
            report["sealed_native_updates"] = deepcopy(sealed_native_updates)
            # The helper publishes a copy of this returned replay fork. Pin it
            # before any callback can coherently alter the runtime candidate.
            latest = cast(RecordedContinuousHead, native_source)._latest[0][2]
            replay_native_expected = own_bc_snapshot(latest, initial.policy)
        else:
            replay_returned_native = deepcopy(native_updates)

    if replay:
        assert replay_plan is not None
        helper_options = (
            {"native_return_observer": observe_replay_native_return} if pc or head else {}
        )
        rehearsal = ContinuousStreamRehearsal(
            native_source,
            budget,
            replay_plan,
            replay_budget=ToyBudgetSession(
                ToyExecutionBudget(max_replay_examples=replay_plan.max_replay_examples), monotonic
            ),
            total_updates=CORRECTION_UPDATES,
            fixed_training_episodes=replay_bank,
            states_equal=_same_state,
            expected_definition=objective,
            **helper_options,
        )
        if head:
            head_replay_budget = rehearsal._replay
        if rehearsal.bind(runtime) != replay_events:
            raise RuntimeError("correction reference rehearsal quota differs from fixed plan")
    replay_updates = replay_exposures = 0
    original_receipts: list[Any] = []
    pc_authority = (*birth_budget, runtime._inbox._applied, original_receipts) if pc else None
    pc_owners: tuple[Any, ...] = (
        (
            source,
            native_source,
            runtime._actor,
            runtime._actor._learner,
            cast(RecordedContinuousPC, runtime._actor._learner)._learner,
            runtime._candidate,
            cast(RecordedContinuousPC, runtime._candidate)._learner,
        )
        if pc
        else ()
    )
    if canonical:
        _require_canonical_original(source, initial)
    readout = source.fork()
    if canonical:
        _require_canonical_original(source, initial)
    bindings = (
        (source, runtime._actor, runtime._actor._learner, runtime._candidate, readout)
        if nearest
        else None
    )
    pc_bindings = (
        (*pc_owners, readout, CORRECTION_UPDATES + replay_events)
        if pc and replay
        else (*pc_owners, readout)
        if pc
        else None
    )
    canonical_bp_bindings: tuple[Any, ...] | None = (
        (runtime._actor, runtime._actor._learner, runtime._candidate, readout)
        if canonical and not pc and not nearest
        else None
    )
    if head:
        assert canonical_bp_bindings is not None
        canonical_bp_bindings = (*canonical_bp_bindings, head_pins)
    canonical_bp_authority = (
        (*canonical_birth, runtime._inbox, runtime._inbox._applied)
        if canonical_bp_bindings is not None
        else None
    )
    expected_state = initial
    validated_receipts: list[dict[str, Any]] = []
    observed, arrived = owned.labels[-1].observed_at, owned.labels[-1].labels_arrived_at
    rows = len(features)
    report: dict[str, Any] = dict(
        complete=False,
        protocol_id="nearest_continuous_correction_1_v1"
        if nearest
        else "continuous_correction_8_v1",
        provenance=readiness_correction_provenance(owned)
        if type(owned)
        in (
            ReadinessCorrectionBatch,
            ObservedGraspCorrectionBatch,
            GoalConsistentObservedGraspCorrectionBatch,
        )
        else PROVENANCE,
        source_updates=initial.completed_updates,
        source_training_rows=initial.training_examples,
        policy=asdict(initial.policy),
        observed_at=observed,
        labels_arrived_at=arrived,
        settings=dict(
            max_training_updates=updates,
            reference_replay_rows=0,
            fit_calls=0,
            promotion_calls=0,
            simulator_calls=0,
            guard_reads=0,
            probe_reads=0,
            final_reads=0,
        ),
        events=[],
    )
    if mixed:
        report.update(binary_gripper_axis=3, training_objective=MIXED_OBJECTIVE)
    if type(owned) in (
        ReadinessCorrectionBatch,
        ObservedGraspCorrectionBatch,
        GoalConsistentObservedGraspCorrectionBatch,
    ):
        report["teacher_definition"] = readiness_correction_definition(owned)
    if type(owned) in (ObservedGraspCorrectionBatch, GoalConsistentObservedGraspCorrectionBatch):
        report["observed_grasp"] = list(owned.observed_grasp)
    observed_metadata = (
        (report["provenance"], report["teacher_definition"], tuple(owned.observed_grasp))
        if type(owned) in (ObservedGraspCorrectionBatch, GoalConsistentObservedGraspCorrectionBatch)
        else None
    )
    if projection_metadata is not None:
        report["input_projection"] = projection_metadata
    history_metadata = deepcopy(projection_metadata) if history else None
    if pc:
        report.update(protocol_id="continuous_pc_correction_8_v1", native_updates=native_updates)
    if head:
        report.update(
            protocol_id="continuous_head_gauss_newton_reference_correction_8_v1"
            if replay
            else "continuous_head_gauss_newton_correction_8_v1",
            head_gauss_newton=head_protocol(),
            native_updates=native_updates,
        )
    head_work_evidence = None
    if rehearsal is not None:
        assert replay_plan is not None
        report["settings"].update(
            replay_plan=asdict(replay_plan),
            planned_replay_events=replay_events,
            reference_bank_rows=replay_rows,
            reference_retained_numeric_bytes=replay_bytes,
            reference_replay_rows=replay_events * replay_rows,
            max_reference_replay_rows=replay_plan.max_replay_examples,
        )
        report["replay"] = rehearsal.report
        report["applied_consolidations"] = []

    replay_evidence = json.dumps(report.get("replay"), sort_keys=True, allow_nan=False)
    replay_work_evidence = None
    step_evidence = None
    step_paid_evidence = None

    def require_step_evidence():
        if step_evidence is not None:
            _require_readiness_step_evidence(report, step_evidence, step_paid_evidence)

    def replay_before(wakes):
        nonlocal replay_evidence
        if head:
            if replay:
                _require_head_metadata(report, native_updates, reference=True)
            else:
                _require_head_metadata(report, native_updates)
        if mixed:
            _require_binary_metadata(report)
        if observed_metadata is not None:
            _require_observed_grasp_metadata(report, observed_metadata)
        if rehearsal is not None:
            if json.dumps(report.get("replay"), sort_keys=True, allow_nan=False) != replay_evidence:
                raise RuntimeError("preceding correction replay report changed")
            try:
                rehearsal.before_update(runtime, wakes)
            finally:
                report["replay"] = rehearsal.report
                replay_evidence = json.dumps(report["replay"], sort_keys=True, allow_nan=False)
        if mixed:
            _require_binary_metadata(report)

    try:
        for index in range(updates):
            require_step_evidence()
            replay_before(index)
            if history:
                _require_history_metadata(report, history_metadata)
            # Why: a preceding success is evidence only while its full state,
            # original paid receipts and charge remain unchanged.
            _require_owners(source, runtime, initial)
            if (
                not _same_state(expected_state, runtime.candidate_snapshot().state)
                or [asdict(item) for item in runtime.applied_updates] != validated_receipts
                or budget.updates_completed != index
                or budget.replay_examples_completed != 0
            ):
                raise RuntimeError("preceding correction success/state/charge changed")
            if nearest:
                _require_nearest_states(bindings, runtime, initial, expected_state, initial)
            if canonical_bp_bindings is not None:
                _require_canonical_bp_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    original_receipts,
                    canonical_bp_authority,
                )
                _require_canonical_bp_states(
                    canonical_bp_bindings, runtime, source, initial, expected_state, initial
                )
            if pc:
                _require_pc_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    native_updates,
                    validated_native,
                    pc_authority,
                )
                _require_pc_states(
                    pc_bindings, runtime, initial, expected_state, initial, native_updates
                )
            sample, label = f"corrective-batch-{index}", f"corrective-label-{index}"
            event: dict[str, Any] = dict(
                index=index, update_attempted=False, applied=[], success_validated=False
            )
            report["events"].append(event)
            runtime.record_experience(
                Experience(
                    sample,
                    owned.episode.episode_id,
                    observed,
                    "actor-0",
                    features,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            runtime.record_label(
                LabelArrival(label, sample, owned.episode.episode_id, arrived, "actor-0", targets)
            )
            if canonical_bp_bindings is not None:
                _require_canonical_bp_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    original_receipts,
                    canonical_bp_authority,
                )
                _require_canonical_bp_states(
                    canonical_bp_bindings, runtime, source, initial, expected_state, initial
                )
            if pc:
                _require_pc_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    native_updates,
                    validated_native,
                    pc_authority,
                )
                _require_pc_states(
                    pc_bindings, runtime, initial, expected_state, initial, native_updates
                )
            if history:
                _require_history_metadata(report, history_metadata)
            replay_before(index)
            if rehearsal is not None:
                if canonical_bp_bindings is not None:
                    _require_canonical_bp_states(
                        canonical_bp_bindings, runtime, source, initial, expected_state, initial
                    )
                if pc:
                    _require_pc_states(
                        pc_bindings, runtime, initial, expected_state, initial, native_updates
                    )
            step_owner = pc_owners[6] if pc else runtime._candidate
            step_before = (
                _canonical_native_state(step_owner)
                if step_bindings is not None and index == 0
                else None
            )
            if step_before is not None and not _same_state(step_before, expected_state):
                raise RuntimeError("readiness diagnostic before-state differs from paid sequence")
            event["update_attempted"] = True
            try:
                applied = runtime.train_ready(max_updates=1)
            finally:
                if head:
                    assert head_records is not None
                    # Own the facade's paid serialized returns before any public
                    # snapshot/clock/receipt observer can replace their values.
                    sealed_native_updates = [json.loads(payload) for _, payload in head_records]
                    report["sealed_native_updates"] = deepcopy(sealed_native_updates)
            step_after = _canonical_native_state(step_owner) if step_before is not None else None
            native_bp_state = (
                _canonical_native_state(canonical_bp_bindings[2])
                if canonical_bp_bindings is not None
                else None
            )
            # Freeze returned native work before any later public snapshot can
            # rewrite the shared sink and make a changed record look original.
            returned_native = deepcopy(native_updates) if pc or head else None
            event["applied"] = [asdict(item) for item in applied]
            state = runtime.candidate_snapshot().state
            if step_after is not None and not _same_state(step_after, state):
                raise RuntimeError("readiness diagnostic native step changed during observation")
            if canonical_bp_bindings is not None and not _same_state(native_bp_state, state):
                raise RuntimeError("canonical BP postupdate public snapshot changed native state")
            if len(applied) != 1:
                raise RuntimeError("exactly one corrective exposure must apply")
            receipt = applied[0]
            if (
                receipt.sample_id != sample
                or receipt.event_id != label
                or receipt.episode_id != owned.episode.episode_id
                or receipt.observed_at != observed
                or receipt.arrived_at != arrived
                or receipt.applied_at != now
                or receipt.update_number != index + 1
                or receipt.actor_version != "actor-0"
                or receipt.learner_version != "candidate-0"
                or receipt.diagnostic.definition != objective
                or not math.isfinite(receipt.diagnostic.value)
                or (nearest and receipt.diagnostic.value != rows)
                or state.completed_updates != initial.completed_updates + index + 1 + replay_updates
                or state.training_examples
                != initial.training_examples + (index + 1) * rows + replay_exposures
                or state.policy != initial.policy
                or budget.updates_completed != index + 1
                or budget.replay_examples_completed != 0
                or len(runtime.applied_updates) != index + 1
            ):
                raise RuntimeError("correction receipt/state/charge differs from fixed recipe")
            if nearest:
                _require_nearest_append(initial, state, targets)
            if pc or head:
                if (
                    native_updates != returned_native
                    or len(native_updates) != index + 1 + replay_updates
                    or native_updates[:-1] != validated_native
                    or native_updates[-1]["diagnostic"] != event["applied"][0]["diagnostic"]
                    or native_updates[-1]["training_rows"] != rows
                    or native_updates[-1]["work"]["published_updates"] != 1
                ):
                    raise RuntimeError("PC native result differs from corrective receipt")
            _require_owners(source, runtime, initial)
            if nearest:
                _require_nearest_states(bindings, runtime, initial, state, initial)
            if pc:
                _require_pc_states(pc_bindings, runtime, initial, state, initial, native_updates)
                if native_updates != returned_native:
                    raise RuntimeError("PC returned native work changed during public observations")
            if canonical_bp_bindings is not None:
                _require_canonical_bp_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts + deepcopy(event["applied"]),
                    original_receipts + list(applied),
                    canonical_bp_authority,
                )
                _require_canonical_bp_states(
                    canonical_bp_bindings, runtime, source, initial, native_bp_state, initial
                )
            validated_receipts.extend(
                deepcopy(event["applied"])
                if nearest or pc or canonical_bp_bindings is not None
                else event["applied"]
            )
            if canonical_bp_bindings is not None:
                original_receipts.extend(applied)
            if pc:
                original_receipts.extend(applied)
                assert returned_native is not None
                validated_native = returned_native
                _require_pc_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    native_updates,
                    validated_native,
                    pc_authority,
                )
                _require_pc_states(pc_bindings, runtime, initial, state, initial, native_updates)
            if head:
                assert returned_native is not None
                if native_updates != returned_native or native_updates != sealed_native_updates:
                    raise RuntimeError("GN returned native work changed during public observations")
                validated_native = deepcopy(returned_native)
            expected_state = native_bp_state if canonical_bp_bindings is not None else state
            event["success_validated"] = True
            if step_before is not None:
                assert step_bindings is not None and step_after is not None
                # Why: retain this paid success and diagnostic attempts even if
                # extra math fails; never admit replay or another wake on failure.
                report["step_diagnostics"] = {}
                step_paid_evidence = json.dumps(event, sort_keys=True, allow_nan=False)
                run_readiness_step_diagnostics(
                    report["step_diagnostics"], step_bindings, step_before, step_after
                )
                if report["step_diagnostics"].get("complete") is not True:
                    raise RuntimeError("readiness step diagnostic did not complete")
                step_evidence = json.dumps(
                    report["step_diagnostics"], sort_keys=True, allow_nan=False
                )
                _require_owners(source, runtime, initial)
                if pc:
                    _require_pc_states(
                        pc_bindings, runtime, initial, step_after, initial, native_updates
                    )
                    _require_pc_evidence(
                        runtime,
                        budget,
                        clock,
                        now,
                        validated_receipts,
                        native_updates,
                        validated_native,
                        pc_authority,
                    )
                else:
                    _require_canonical_bp_states(
                        canonical_bp_bindings, runtime, source, initial, step_after, initial
                    )
                    _require_canonical_bp_evidence(
                        runtime,
                        budget,
                        clock,
                        now,
                        validated_receipts,
                        original_receipts,
                        canonical_bp_authority,
                    )
                require_step_evidence()
            if rehearsal is not None:
                assert replay_plan is not None
                preceding = expected_state
                assert preceding is not None
                replay_returned_native = None
                replay_native_expected = None
                try:
                    rehearsal.after_update(runtime, index + 1, expected_state=preceding)
                finally:
                    replay_report_changed = (
                        json.dumps(report.get("replay"), sort_keys=True, allow_nan=False)
                        != replay_evidence
                    )
                    if (pc or head) and replay_returned_native is not None:
                        report["sealed_replay_native_updates"] = deepcopy(replay_returned_native)
                    report["replay"] = rehearsal.report
                    replay_evidence = json.dumps(report["replay"], sort_keys=True, allow_nan=False)
                    if replay_report_changed:
                        report["replay_observation_error"] = (
                            "preceding replay report changed during consolidation"
                        )
                    # Keep the helper's paid report even if receipt observation fails.
                    report["applied_consolidations"] = [
                        asdict(item) for item in runtime.applied_consolidations
                    ]
                if replay_report_changed:
                    raise RuntimeError("preceding replay report changed during consolidation")
                replay_report = report["replay"]
                observed_replay = replay_report["committed_updates"]
                delta = observed_replay - replay_updates
                expected_delta = int((index + 1) % replay_plan.every_updates == 0)
                current = runtime.candidate_snapshot().state
                if (
                    delta != expected_delta
                    or replay_report["charged_exposures"] != observed_replay * replay_rows
                    or current.completed_updates != preceding.completed_updates + delta
                    or current.training_examples
                    != preceding.training_examples + delta * replay_rows
                    or current.policy != initial.policy
                    or (delta == 0 and not _same_state(preceding, current))
                ):
                    raise RuntimeError(
                        "reference replay publication differs from fixed correction sequence"
                    )
                if pc or head:
                    if len(native_updates) != index + 1 + observed_replay:
                        raise RuntimeError(
                            "GN replay native sink length differs from exact sequence"
                            if head
                            else "PC replay native sink length differs from exact sequence"
                        )
                    if delta:
                        if replay_returned_native is None:
                            raise RuntimeError(
                                "GN reference replay lacks its sealed native return"
                                if head
                                else "PC reference replay lacks its sealed native return"
                            )
                        if (
                            native_updates != replay_returned_native
                            or replay_returned_native[: len(validated_native)] != validated_native
                            or replay_returned_native[-1]["training_rows"] != replay_rows
                            or replay_returned_native[-1]["diagnostic"]
                            != report["applied_consolidations"][-1]["diagnostic"]
                            or replay_returned_native[-1]["work"]["published_updates"] != 1
                        ):
                            raise RuntimeError(
                                "GN reference replay native result differs from sealed return"
                                if head
                                else "PC reference replay native result differs from sealed return"
                            )
                        validated_native = deepcopy(replay_returned_native)
                    elif native_updates != validated_native:
                        raise RuntimeError("native sink changed without a scheduled replay")
                    if head:
                        actual_consolidations = tuple(runtime._consolidations)
                        expected_records = deepcopy(head_consolidation_records)
                        if delta:
                            assert replay_returned_native is not None
                            expected_records.append(
                                dict(
                                    event_id=f"continuous-replay-{index + 1}",
                                    actor_version="actor-0",
                                    learner_version="candidate-0",
                                    attempt_number=observed_replay,
                                    diagnostic=deepcopy(replay_returned_native[-1]["diagnostic"]),
                                )
                            )
                        actual_records = [asdict(value) for value in actual_consolidations]
                        if (
                            runtime._consolidations is not head_consolidation_owner
                            or len(actual_consolidations) != observed_replay
                            or any(
                                actual is not original
                                for actual, original in zip(
                                    actual_consolidations[: len(head_consolidation_objects)],
                                    head_consolidation_objects,
                                    strict=True,
                                )
                            )
                            or json.dumps(actual_records, sort_keys=True, allow_nan=False)
                            != json.dumps(expected_records, sort_keys=True, allow_nan=False)
                        ):
                            raise RuntimeError("GN replay original consolidation receipts changed")
                        head_consolidation_objects = actual_consolidations
                        head_consolidation_records = expected_records
                        expected_replay_state = replay_native_expected if delta else preceding
                        if expected_replay_state is None or not _same_state(
                            expected_replay_state, current
                        ):
                            raise RuntimeError(
                                "GN replay candidate differs from sealed native return"
                            )
                        if native_updates != sealed_native_updates:
                            raise RuntimeError(
                                "GN replay native work differs from original serialized returns"
                            )
                replay_updates = observed_replay
                replay_exposures = replay_report["charged_exposures"]
                expected_state = current
                replay_before(index + 1)
                if canonical_bp_bindings is not None:
                    _require_canonical_bp_states(
                        canonical_bp_bindings, runtime, source, initial, expected_state, initial
                    )
                if pc:
                    _require_pc_evidence(
                        runtime,
                        budget,
                        clock,
                        now,
                        validated_receipts,
                        native_updates,
                        validated_native,
                        pc_authority,
                    )
                    _require_pc_states(
                        pc_bindings, runtime, initial, expected_state, initial, native_updates
                    )
        require_step_evidence()
        replay_before(updates)
        if pc:
            _require_pc_evidence(
                runtime,
                budget,
                clock,
                now,
                validated_receipts,
                native_updates,
                validated_native,
                pc_authority,
            )
            _require_pc_states(
                pc_bindings, runtime, initial, expected_state, initial, native_updates
            )
        if (
            runtime.train_ready(max_updates=1) != ()
            or not _same_state(expected_state, runtime.candidate_snapshot().state)
            or budget.updates_completed != updates
            or budget.replay_examples_completed != 0
            or [asdict(item) for item in runtime.applied_updates] != validated_receipts
        ):
            raise RuntimeError("final correction empty poll changed state/receipts/charge")
        _require_owners(source, runtime, initial)
        if canonical_bp_bindings is not None:
            _require_canonical_bp_evidence(
                runtime,
                budget,
                clock,
                now,
                validated_receipts,
                original_receipts,
                canonical_bp_authority,
            )
            _require_canonical_bp_states(
                canonical_bp_bindings, runtime, source, initial, expected_state, initial
            )
        readout.restore_state(expected_state)
        if not _same_state(expected_state, readout.snapshot_state()):
            raise RuntimeError("detached correction readout differs")
        _require_owners(source, runtime, initial)
        if (
            not _same_state(expected_state, runtime.candidate_snapshot().state)
            or budget.updates_completed != updates
            or budget.replay_examples_completed != 0
            or [asdict(item) for item in runtime.applied_updates] != validated_receipts
        ):
            raise RuntimeError("detached readout changed private correction candidate")
        if canonical_bp_bindings is not None:
            _require_canonical_bp_evidence(
                runtime,
                budget,
                clock,
                now,
                validated_receipts,
                original_receipts,
                canonical_bp_authority,
            )
            _require_canonical_bp_states(
                canonical_bp_bindings, runtime, source, initial, expected_state, expected_state
            )
        if nearest:
            d, a = initial.policy.spec.observation_dim, len(initial.policy.spec.action_units)
            report.update(
                complete=True,
                work=dict(
                    prototype_appends=1,
                    optimizer_updates=0,
                    training_rows=rows,
                    unique_correction_rows=rows,
                    repeated_training_rows=0,
                    reference_replay_rows=0,
                    fit_calls=0,
                    forward_passes=0,
                    forward_rows=0,
                    backward_passes=0,
                    backward_rows=0,
                    normalization_calls=1,
                    normalization_rows=rows,
                    normalization_values=rows * d,
                    gradient_values=0,
                    gradient_numeric_bytes=0,
                    dense_multiply_accumulates=0,
                    source_prototypes=initial.training_examples,
                    candidate_prototypes=initial.training_examples + rows,
                    max_prototypes=initial.policy.max_prototypes,
                    prototype_numeric_bytes_added=8 * rows * (d + a),
                    prototype_numeric_bytes_per_candidate=8
                    * (initial.training_examples + rows)
                    * (d + a),
                    statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
                        initial.policy.observation_standardization
                    ),
                    named_model_owners=4,
                    inbox_feature_target_numeric_bytes=8 * rows * (d + a),
                    scope="successful single append; no distance queries; numeric payload excludes copies/validation/elementwise/process overhead",
                ),
            )
        elif pc:
            _record_pc_work(report, initial, rows, native_updates)
        elif head:
            if rehearsal is None:
                _record_head_work(report, initial, rows, native_updates)
            else:
                _record_head_work(report, initial, rows, native_updates, replay=report["replay"])
            head_work_evidence = json.dumps(report["work"], sort_keys=True, allow_nan=False)
        else:
            _record_mlp_work(report, initial, rows)
        if mixed:
            report["work"]["binary_gripper_loss_rows"] = report["work"]["training_rows"]
        if rehearsal is not None and not head:
            _add_reference_replay_work(report, initial, report["replay"], pc)
        if rehearsal is not None or mixed:
            replay_work_evidence = json.dumps(report["work"], sort_keys=True, allow_nan=False)
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        if hasattr(error, "work"):
            report["failed_native_work"] = error.work
    if rehearsal is not None:
        if json.dumps(report.get("replay"), sort_keys=True, allow_nan=False) != replay_evidence:
            report.update(
                complete=False, replay_observation_error="preceding replay report changed"
            )
        report["replay"] = rehearsal.report
        replay_evidence = json.dumps(report["replay"], sort_keys=True, allow_nan=False)
        try:
            report["applied_consolidations"] = [
                asdict(item) for item in runtime.applied_consolidations
            ]
        except Exception as error:
            report["consolidation_observation_error"] = f"{type(error).__name__}: {error}"
            report["complete"] = False
    if pc or canonical:
        # Preserve returned native work when a final observer itself raises.
        report.update(applied_updates=None, progress=None, elapsed_seconds=None)
        try:
            report["applied_updates"] = [asdict(item) for item in runtime.applied_updates]
            report["progress"] = dict(
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                next_tick=clock.now() + 1,
            )
            report["elapsed_seconds"] = monotonic() - started
        except Exception as error:
            report.update(complete=False, metadata_error=f"{type(error).__name__}: {error}")
            report.pop("work", None)
    else:
        report["applied_updates"] = [asdict(item) for item in runtime.applied_updates]
        report["progress"] = dict(
            updates_completed=budget.updates_completed,
            replay_examples_completed=budget.replay_examples_completed,
            next_tick=clock.now() + 1,
        )
        report["elapsed_seconds"] = monotonic() - started
    if pc:
        report["failure_work_may_be_unobserved"] = not report["complete"] and sum(
            event["update_attempted"] for event in report["events"]
        ) + (report["replay"]["attempted_updates"] if rehearsal is not None else 0) > len(
            native_updates
        )
        if report["complete"]:
            try:
                _require_pc_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    native_updates,
                    validated_native,
                    pc_authority,
                )
                if (
                    report["applied_updates"] != validated_receipts
                    or [item for event in report["events"] for item in event["applied"]]
                    != validated_receipts
                ):
                    raise RuntimeError("PC correction returned evidence changed")
                _require_pc_states(
                    pc_bindings, runtime, initial, expected_state, expected_state, native_updates
                )
            except Exception as error:
                report.update(complete=False, error=f"{type(error).__name__}: {error}")
                report.pop("work", None)
    if nearest:
        if report["complete"]:
            try:
                _require_nearest_states(bindings, runtime, initial, expected_state, expected_state)
                if (
                    [asdict(item) for item in runtime._inbox._applied.values()]
                    != validated_receipts
                    or report["applied_updates"] != validated_receipts
                    or report["events"][0]["applied"] != validated_receipts
                    or runtime._budget is not budget
                    or budget.updates_completed != 1
                    or budget.replay_examples_completed != 0
                    or budget.budget.max_training_updates != 1
                    or clock._time != now
                ):
                    raise RuntimeError("nearest correction final evidence or charge changed")
            except Exception as error:
                report.update(complete=False, error=f"{type(error).__name__}: {error}")
                report.pop("work", None)
        if observed_metadata is not None and report["complete"]:
            try:
                _require_observed_grasp_metadata(report, observed_metadata)
            except Exception as error:
                report.update(complete=False, error=f"{type(error).__name__}: {error}")
                report.pop("work", None)
        return NearestContinuousCorrectionResult(report, readout if report["complete"] else None)
    if canonical_bp_bindings is not None and report["complete"]:
        try:
            actor, backend, candidate, detached = canonical_bp_bindings[:4]
            if (
                runtime._actor is not actor
                or actor._learner is not backend
                or runtime._candidate is not candidate
                or readout is not detached
            ):
                raise RuntimeError("canonical correction final owner identity changed")
            owners = (
                (source, initial),
                (backend, initial),
                (candidate, expected_state),
                (detached, expected_state),
            )
            for owner, expected in owners:
                _require_canonical_original(owner, expected)
            # A later public observer can mutate an earlier owner. End with
            # native-owned checks only, after all elapsed/report callbacks.
            if any(
                not _same_state(expected, _canonical_native_state(owner))
                for owner, expected in owners
            ):
                raise RuntimeError("canonical correction final native owner changed")
            _require_canonical_bp_evidence(
                runtime,
                budget,
                clock,
                now,
                validated_receipts,
                original_receipts,
                canonical_bp_authority,
            )
            _require_canonical_bp_states(
                canonical_bp_bindings, runtime, source, initial, expected_state, expected_state
            )
            if (
                report["applied_updates"] != validated_receipts
                or [item for event in report["events"] for item in event["applied"]]
                != validated_receipts
                or report["progress"]["updates_completed"] != CORRECTION_UPDATES
                or report["progress"]["replay_examples_completed"] != 0
                or report["progress"]["next_tick"] != now + 1
                or any(type(value) is not int for value in report["progress"].values())
            ):
                raise RuntimeError("canonical BP returned receipt or progress changed")
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
            report.pop("work", None)
    if history and report["complete"]:
        try:
            _require_history_metadata(report, history_metadata)
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
            report.pop("work", None)
    if rehearsal is not None and report["complete"]:
        try:
            replay_before(updates)
            if pc:
                _require_pc_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    native_updates,
                    validated_native,
                    pc_authority,
                )
                _require_pc_states(
                    pc_bindings, runtime, initial, expected_state, expected_state, native_updates
                )
            else:
                _require_canonical_bp_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    original_receipts,
                    canonical_bp_authority,
                )
                _require_canonical_bp_states(
                    canonical_bp_bindings, runtime, source, initial, expected_state, expected_state
                )
            if (
                report["replay"]["committed_updates"] != replay_events
                or report["replay"]["charged_exposures"] != replay_events * replay_rows
                or report["applied_consolidations"]
                != [asdict(item) for item in runtime._consolidations]
                or json.dumps(report.get("work"), sort_keys=True, allow_nan=False)
                != replay_work_evidence
            ):
                raise RuntimeError("final reference replay receipt or exposure accounting changed")
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
            report.pop("work", None)
    if rehearsal is not None:
        report["replay"] = rehearsal.report
    if (mixed or observed_metadata is not None) and report["complete"]:
        try:
            # The final replay report and owner observers precede the last seals.
            if pc:
                _require_pc_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    native_updates,
                    validated_native,
                    pc_authority,
                )
                _require_pc_states(
                    pc_bindings, runtime, initial, expected_state, expected_state, native_updates
                )
            else:
                _require_canonical_bp_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    original_receipts,
                    canonical_bp_authority,
                )
                _require_canonical_bp_states(
                    canonical_bp_bindings, runtime, source, initial, expected_state, expected_state
                )
            if mixed:
                _require_binary_metadata(report)
                if (
                    type(report["work"]["binary_gripper_loss_rows"]) is not int
                    or report["work"]["binary_gripper_loss_rows"]
                    != updates * rows + replay_exposures
                    or json.dumps(report["work"], sort_keys=True, allow_nan=False)
                    != replay_work_evidence
                ):
                    raise RuntimeError("binary gripper correction work changed")
            if observed_metadata is not None:
                _require_observed_grasp_metadata(report, observed_metadata)
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
            report.pop("work", None)
    if step_bindings is not None and report["complete"]:
        try:
            require_step_evidence()
            if step_evidence is None or report["step_diagnostics"].get("complete") is not True:
                raise RuntimeError("readiness step diagnostic did not complete")
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}")
            report.pop("work", None)
    if head:
        report["sealed_native_updates"] = deepcopy(sealed_native_updates)
        head_attempts: int | None = sum(event["update_attempted"] for event in report["events"])
        if rehearsal is not None:
            observed_attempts = report["replay"].get("attempted_updates")
            head_attempts = (
                head_attempts + observed_attempts
                if head_attempts is not None and type(observed_attempts) is int
                else None
            )
        report["failure_work_may_be_unobserved"] = not report["complete"] and (
            head_attempts is None or head_attempts > len(sealed_native_updates)
        )
        if report["complete"]:
            try:
                _require_canonical_bp_states(
                    canonical_bp_bindings, runtime, source, initial, expected_state, expected_state
                )
                _require_canonical_bp_evidence(
                    runtime,
                    budget,
                    clock,
                    now,
                    validated_receipts,
                    original_receipts,
                    canonical_bp_authority,
                )
                if replay:
                    assert head_replay_budget is not None and rehearsal is not None
                    head_replay_budget.check(0, replay_exposures)
                    if (
                        rehearsal._replay is not head_replay_budget
                        or runtime._consolidations is not head_consolidation_owner
                        or len(runtime._consolidations) != len(head_consolidation_objects)
                        or any(
                            actual is not original
                            for actual, original in zip(
                                runtime._consolidations, head_consolidation_objects, strict=True
                            )
                        )
                        or json.dumps(
                            [asdict(value) for value in runtime._consolidations],
                            sort_keys=True,
                            allow_nan=False,
                        )
                        != json.dumps(head_consolidation_records, sort_keys=True, allow_nan=False)
                    ):
                        raise RuntimeError("GN final reference evidence or budget changed")
                    _require_head_replay_evidence(
                        report, replay_evidence, head_consolidation_records
                    )
                # Final scalar seals follow every facade observer.
                if replay:
                    _require_head_metadata(report, native_updates, reference=True)
                else:
                    _require_head_metadata(report, native_updates)
                if (
                    native_updates != validated_native
                    or native_updates != sealed_native_updates
                    or len(native_updates) != CORRECTION_UPDATES + replay_events
                    or json.dumps(report.get("work"), sort_keys=True, allow_nan=False)
                    != head_work_evidence
                ):
                    raise RuntimeError("GN correction returned work/prefix changed")
            except Exception as error:
                report.update(complete=False, error=f"{type(error).__name__}: {error}")
                report.pop("work", None)
    return ContinuousLearnabilityResult(report, readout if report["complete"] else None)


def _require_readiness_step_evidence(report, step_evidence, paid_evidence) -> None:
    if (
        type(step_evidence) is not str
        or type(paid_evidence) is not str
        or type(report.get("events")) is not list
        or not report["events"]
        or json.dumps(report.get("step_diagnostics"), sort_keys=True, allow_nan=False)
        != step_evidence
        or json.dumps(report["events"][0], sort_keys=True, allow_nan=False) != paid_evidence
    ):
        raise RuntimeError("preceding readiness step diagnostic or paid success changed")


def _require_observed_grasp_metadata(report, expected) -> None:
    feedback = report.get("observed_grasp")
    if (
        report.get("provenance") != expected[0]
        or report.get("teacher_definition") != expected[1]
        or type(feedback) is not list
        or any(type(value) is not bool for value in feedback)
        or tuple(feedback) != expected[2]
    ):
        raise RuntimeError("observed grasp correction metadata changed")


def _require_binary_metadata(report) -> None:
    if (
        type(report.get("binary_gripper_axis")) is not int
        or report["binary_gripper_axis"] != 3
        or type(report.get("training_objective")) is not str
        or report["training_objective"] != MIXED_OBJECTIVE
    ):
        raise RuntimeError("binary gripper correction selection metadata changed")


def _require_history_metadata(report, expected) -> None:
    def same(actual, original):
        if type(actual) is not type(original):
            return False
        if type(original) is dict:
            return actual.keys() == original.keys() and all(
                same(actual[key], value) for key, value in original.items()
            )
        if type(original) in (tuple, list):
            return len(actual) == len(original) and all(
                same(a, b) for a, b in zip(actual, original, strict=True)
            )
        return actual == original

    if not same(report.get("input_projection"), expected):
        raise RuntimeError("action history correction projection metadata changed")


def _record_mlp_work(report, initial, rows) -> None:
    d, h, a = (
        initial.policy.spec.observation_dim,
        initial.policy.hidden_dim,
        len(initial.policy.spec.action_units),
    )
    p = d * h + h + h * a + a
    report.update(
        complete=True,
        work=dict(
            optimizer_updates=8,
            training_rows=8 * rows,
            unique_correction_rows=rows,
            repeated_training_rows=7 * rows,
            reference_replay_rows=0,
            fit_calls=0,
            forward_passes=8,
            forward_rows=8 * rows,
            backward_passes=8,
            backward_rows=8 * rows,
            normalization_calls=8,
            normalization_rows=8 * rows,
            normalization_values=8 * rows * d,
            gradient_values=8 * p,
            gradient_numeric_bytes=64 * p,
            dense_multiply_accumulates=8 * rows * (2 * d * h + 3 * h * a),
            model_numeric_bytes_per_owner=16 * p,
            statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
                initial.policy.observation_standardization
            ),
            named_model_owners=4,
            inbox_feature_target_numeric_bytes=8 * rows * (d + a) * 8,
            scope="successful fixed run; numeric payload/dense work excludes copies/validation/elementwise/process overhead",
        ),
    )


def _record_pc_work(report, initial, rows, native_updates) -> None:
    base = initial.policy.base
    d, h, a = base.spec.observation_dim, base.hidden_dim, len(base.spec.action_units)
    parameters = d * h + h + h * a + a
    counts = {
        key: sum(item["work"][key] for item in native_updates) for key in native_updates[0]["work"]
    }
    weights = {
        key: sum(item["work"][key] * item["training_rows"] for item in native_updates)
        for key in counts
    }
    report.update(
        complete=True,
        work=dict(
            optimizer_updates=CORRECTION_UPDATES,
            training_rows=CORRECTION_UPDATES * rows,
            unique_correction_rows=rows,
            repeated_training_rows=(CORRECTION_UPDATES - 1) * rows,
            reference_replay_rows=0,
            fit_calls=0,
            native_update_counts=counts,
            native_event_row_weights=weights,
            normalization_calls=counts["normalization_returns"],
            normalization_rows=weights["normalization_returns"],
            normalization_values=weights["normalization_returns"] * d,
            model_numeric_bytes_per_owner=16 * parameters,
            statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
                initial.policy.base.observation_standardization
            ),
            wake_numeric_bytes_per_owner=0 if initial.policy.circadian is None else 16 * h,
            named_model_owners=4,
            inbox_feature_target_numeric_bytes=8 * rows * (d + a) * CORRECTION_UPDATES,
            scope="successful fixed PC run; native event counts/row weights, not BP arithmetic; excludes validation recomputation, copies, elementwise math, temporary arrays and process overhead",
        ),
    )


def _add_reference_replay_work(report, initial, replay, pc):
    work = report["work"]
    updates, rows = replay["committed_updates"], replay["charged_exposures"]
    work["optimizer_updates"] += updates
    work["training_rows"] += rows
    work["reference_replay_rows"] = rows
    if "binary_gripper_loss_rows" in work:
        work["binary_gripper_loss_rows"] += rows
    if not pc:
        policy = initial.policy
        d, h, a = policy.spec.observation_dim, policy.hidden_dim, len(policy.spec.action_units)
        parameters = d * h + h + h * a + a
        for key in ("forward_passes", "backward_passes", "normalization_calls"):
            work[key] += updates
        for key in ("forward_rows", "backward_rows", "normalization_rows"):
            work[key] += rows
        work["normalization_values"] += rows * d
        work["gradient_values"] += updates * parameters
        work["gradient_numeric_bytes"] += updates * parameters * 8
        work["dense_multiply_accumulates"] += rows * (2 * d * h + 3 * h * a)
    work.update(
        reference_bank_retained_rows=replay["bank_rows"],
        reference_bank_retained_numeric_bytes=replay["retained_numeric_bytes"],
        reference_bank_signature_bytes=replay["bank_signature_bytes"],
        replay_transient_model_owners=1 if updates else 0,
        replay_transient_model_numeric_bytes_per_owner=work["model_numeric_bytes_per_owner"],
        replay_memory_scope="one temporary learner per event; snapshots/validation/native arrays/RSS additional",
    )


def _require_head_metadata(report, native_updates, *, reference=False) -> None:
    expected = head_protocol()
    actual = report.get("head_gauss_newton")
    protocol = (
        "continuous_head_gauss_newton_reference_correction_8_v1"
        if reference
        else "continuous_head_gauss_newton_correction_8_v1"
    )
    if (
        type(report.get("protocol_id")) is not str
        or report["protocol_id"] != protocol
        or type(actual) is not dict
        or set(actual) != set(expected)
        or any(
            type(actual[key]) is not type(value) or actual[key] != value
            for key, value in expected.items()
        )
        or report.get("native_updates") is not native_updates
    ):
        raise RuntimeError("GN correction selection/native report changed")
