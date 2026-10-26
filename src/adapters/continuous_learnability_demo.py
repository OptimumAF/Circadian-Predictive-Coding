"""Bounded TRAIN A/B/A MLP sequence on the existing actor/shadow runtime.

Inputs are two released complete TRAIN episodes. Outputs are bounded scalar
diagnostics, all training receipts, endpoint predictions and explicit costs.
Python callers can retain the detached final learner; the CLI returns JSON only.
An opt-in A-only update control still diagnoses both complete TRAIN contexts.
No simulator, evaluation selection, promotion, new optimizer or checkpoint IO.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import json
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.continuous_stream_demo import _training_episodes
from src.adapters.continuous_probe import ContinuousProbePlan, copy_probe_plan, run_continuous_probe
from src.adapters.numpy_continuous_bc import (
    ContinuousBCPolicy,
    NumpyContinuousBCLearner,
    fit_training_standardization,
    _snapshot as own_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import ContinuousPCPolicy, NumpyContinuousPCLearner
from src.adapters.numpy_continuous_pc import _snapshot as own_pc_snapshot
from src.adapters.continuous_phase_readouts import PhaseReadoutObserver, PhaseReadouts
from src.adapters.continuous_head_learnability import (
    RecordedContinuousHead,
    head_protocol,
    head_work,
)
from src.adapters.continuous_learnability_modes import (
    ContinuousLearner,
    RecordedContinuousPC,
    diagnose_conditioning,
    diagnose_residual,
    pc_work,
    same_continuous_state as _same_state,
    validate_learner_family,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.continuous_stream_rehearsal import (
    ContinuousReplayPlan,
    ContinuousStreamRehearsal,
    prepare_replay_bank,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_bc import OBJECTIVE
from src.core import continuous_bc as bc
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.continuous_demonstration import continuous_training_pairs
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.observation_standardization import (
    ObservationStandardization,
    _standardization,
    standardization_numeric_bytes,
)
from src.core.pick_cube_rgb_features import PickCubeRGBRecipe, validate_pick_cube_rgb_recipe
from src.core.pick_cube_rgb_standardization import DEFINITION, validate_rgb_range_features
from src.core import pick_cube_action_history as action_history
from src.core.readiness_stage_diagnostics import (
    ReadinessStageBinding,
    copy_readiness_stage_bindings,
    readiness_stage_metadata,
)
from src.core.readiness_training_packets import (
    ReadinessPacketPlan,
    prepare_readiness_packets,
    readiness_packet_metadata,
)

PHASE_UPDATES = 8
MAX_PHASE_UPDATES = 128
PHASES = (("warm", "warm"), ("new", "new"), ("warm_return", "warm"))


def validate_phase_updates(value: object) -> int:
    """Admit a finite packet budget before any episode, fit or model work."""
    if type(value) is not int or not 1 <= value <= MAX_PHASE_UPDATES:
        raise ValueError(f"phase_updates must be an integer in [1,{MAX_PHASE_UPDATES}]")
    return value


@dataclass(frozen=True)
class ContinuousLearnabilityResult:
    """Owned report and detached mutable learner; incomplete work releases no model."""

    report: dict[str, Any]
    candidate: ContinuousLearner | None


def _require_owners(source, runtime, initial) -> None:
    actor = runtime.actor.snapshot_state()
    if (
        not _same_state(initial, source.snapshot_state())
        or actor.actor_version != runtime.actor.version
        or actor.actor_version != "actor-0"
        or not _same_state(initial, actor.state)
    ):
        raise RuntimeError("continuous learnability source/actor changed")


def _require_head_report(report) -> None:
    if report.get("protocol_id") != "continuous_head_gauss_newton_train_aba_v1" or json.dumps(
        report.get("head_gauss_newton"), sort_keys=True, allow_nan=False
    ) != json.dumps(head_protocol(), sort_keys=True, allow_nan=False):
        raise RuntimeError("head Gauss-Newton report metadata changed")


def _head_port(value: object) -> RecordedContinuousHead:
    if type(value) is not RecordedContinuousHead:
        raise RuntimeError("head stream native port was replaced")
    return value


def _phase_native(owner):
    if type(owner) in (RecordedContinuousHead, RecordedContinuousPC):
        owner = owner._learner
    if type(owner) not in (NumpyContinuousBCLearner, NumpyContinuousPCLearner):
        raise RuntimeError("phase readout native owner was replaced")
    return owner


def _phase_private(owner):
    native = _phase_native(owner)
    snapshot = own_pc_snapshot if type(native) is NumpyContinuousPCLearner else own_bc_snapshot
    return snapshot(native._state, native._policy)


def _phase_authority(runtime):
    fields = (
        (
            runtime,
            (
                "_candidate_version",
                "_base_actor_version",
                "_training_role",
                "_original_training_role",
                "_consolidation_limit",
                "_stopped",
                "_retired",
                "_revision",
                "_payload_ready",
            ),
        ),
        (
            runtime._inbox,
            (
                "_learner_version",
                "_capacity",
                "_training_role",
                "_original_training_role",
                "_historical_completed_updates",
                "_draining",
                "_stopped",
                "_last_time",
            ),
        ),
    )
    return json.dumps(
        [[getattr(owner, key) for key in keys] for owner, keys in fields], allow_nan=False
    )


def _phase_callback_guard(source, runtime, readout, initial, expected, budget, clock, report):
    """Pin the existing boundary around external inference, without renewing work."""
    ports = (runtime._actor, runtime._actor._learner, runtime._candidate, runtime._inbox)
    models = (source, _phase_native(ports[1]), _phase_native(ports[2]), readout)
    native_states = tuple(model._state for model in models)
    budget_fields, limit, tick = deepcopy(vars(budget)), budget.budget, clock.now()
    receipts = tuple(runtime.applied_updates)
    receipt_payload = json.dumps([asdict(item) for item in receipts], sort_keys=True)
    payload = json.dumps(report, sort_keys=True, allow_nan=False)
    authority = _phase_authority(runtime)
    wrappers = [
        (owner, owner._learner, owner._reports, owner._limit)
        for owner in ports[1:3]
        if type(owner) in (RecordedContinuousHead, RecordedContinuousPC)
    ]
    inbox_guards = tuple(
        getattr(ports[3], key)
        for key in ("_registration_guard", "_training_guard", "_payload_copy_guard")
    )

    def guard():
        current_ports = (
            runtime._actor,
            runtime._actor._learner,
            runtime._candidate,
            runtime._inbox,
        )
        if (
            any(a is not b for a, b in zip(current_ports, ports, strict=True))
            or _phase_native(ports[1]) is not models[1]
            or _phase_native(ports[2]) is not models[2]
            or ports[3]._learner is not ports[2]
            or _phase_authority(runtime) != authority
            or any(
                owner._learner is not learner
                or owner._reports is not sink
                or type(owner._limit) is not int
                or owner._limit != limit
                for owner, learner, sink, limit in wrappers
            )
            or any(
                getattr(ports[3], key) is not original
                for key, original in zip(
                    ("_registration_guard", "_training_guard", "_payload_copy_guard"),
                    inbox_guards,
                    strict=True,
                )
            )
            or ports[0]._version != "actor-0"
            or runtime._budget is not budget
            or ports[3]._budget is not budget
            or ports[3]._clock is not clock
            or budget.budget is not limit
            or vars(budget) != budget_fields
            or type(clock.now()) is not int
            or clock.now() != tick
            or tuple(map(id, runtime.applied_updates)) != tuple(map(id, receipts))
            or json.dumps([asdict(item) for item in runtime.applied_updates], sort_keys=True)
            != receipt_payload
            or json.dumps(report, sort_keys=True, allow_nan=False) != payload
        ):
            raise RuntimeError("phase observer changed original benchmark authority or evidence")
        for model, state, wanted in zip(
            models, native_states, (initial, initial, expected, expected), strict=True
        ):
            if model._state is not state or not _same_state(_phase_private(model), wanted):
                raise RuntimeError("phase observer changed a protected native model")

    return guard


def _taskspace_normalization_metadata():
    return dict(
        definition="official_state_taskspace_fixed_physical_units_v1",
        displacement_scale=0.1,
        displacement_frame="world",
        empirical_fit=False,
        policy_standardization=None,
        normalization_calls="identity_copy_after_fixed_projection",
    )


def _require_action_history_recipe(report, recipe):
    expected = asdict(action_history.validate_action_history_recipe(recipe))
    if json.dumps(
        report.get("action_history_recipe"), sort_keys=True, allow_nan=False
    ) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise RuntimeError("continuous action history recipe changed")
    if action_history.is_taskspace_recipe(recipe):
        if json.dumps(
            report.get("fixed_input_normalization"), sort_keys=True, allow_nan=False
        ) != json.dumps(_taskspace_normalization_metadata(), sort_keys=True, allow_nan=False):
            raise RuntimeError("taskspace fixed physical normalization metadata changed")
        if "work" in report and any(
            type(report["work"].get(key)) is not int or report["work"][key] != 0
            for key in (
                "fit_calls",
                "fit_rows",
                "fit_feature_values",
                "statistics_numeric_bytes_per_owner",
            )
        ):
            raise RuntimeError("taskspace report invented empirical fitted statistics")


def _readiness_checkpoint_counts(
    warm, new, phase_updates, a_only_control, replay_plan=None, replay_rows=0
):
    warm_rows = phase_updates * len(warm.transitions)
    middle_rows = warm_rows if a_only_control else phase_updates * len(new.transitions)
    counts = (
        (0, 0),
        (phase_updates, warm_rows),
        (2 * phase_updates, warm_rows + middle_rows),
        (3 * phase_updates, 2 * warm_rows + middle_rows),
    )
    if replay_plan is None:
        return counts
    return tuple(
        (
            updates + updates // replay_plan.every_updates,
            rows + (updates // replay_plan.every_updates) * replay_rows,
        )
        for updates, rows in counts
    )


def _require_stage_diagnostics(report, expected, checkpoint_counts) -> None:
    """Keep selected source labels and every returned row partition bound."""
    if checkpoint_counts is None or type(report.get("complete")) is not bool:
        raise RuntimeError("readiness stage checkpoint authority changed")
    if type(report["checkpoints"]) is not list or len(report["checkpoints"]) > len(
        checkpoint_counts
    ):
        raise RuntimeError("readiness stage checkpoint prefix changed")
    if report.get("complete") is True and len(report["checkpoints"]) != 4:
        raise RuntimeError("complete readiness stage diagnostics lost a checkpoint")
    actual_sources = report.get("diagnostic_stage_sources")
    if type(actual_sources) is not dict or set(actual_sources) != set(expected):
        raise RuntimeError("readiness diagnostic stage source metadata changed")
    for name, source in expected.items():
        actual = actual_sources[name]
        if (
            type(actual) is not dict
            or set(actual) != set(source)
            or any(
                type(actual[key]) is not type(value) or actual[key] != value
                for key, value in source.items()
            )
            or any(type(label) is not str for label in actual["stages"])
        ):
            raise RuntimeError("readiness diagnostic stage source metadata changed")
    for index, checkpoint in enumerate(report["checkpoints"]):
        updates, examples = checkpoint_counts[index]
        if (
            type(checkpoint["updates"]) is not int
            or checkpoint["updates"] != updates
            or type(checkpoint["training_examples"]) is not int
            or checkpoint["training_examples"] != examples
            or checkpoint["state_exact"] is not True
            or checkpoint["owners_exact"] is not True
        ):
            raise RuntimeError("readiness stage checkpoint ordinal or row charge changed")
        for name, source in expected.items():
            labels = source["stages"]
            groups = checkpoint["contexts"][name]["residual"]["residuals"]["stages"]
            expected_groups = [
                (label, tuple(i for i, actual in enumerate(labels) if actual == label))
                for label in dict.fromkeys(labels)
            ]
            if type(groups) is not tuple or len(groups) != len(expected_groups):
                raise RuntimeError("readiness diagnostic stage partition changed")
            for actual, (label, indices) in zip(groups, expected_groups, strict=True):
                if (
                    type(actual["label"]) is not str
                    or actual["label"] != label
                    or type(actual["row_indices"]) is not tuple
                    or any(type(i) is not int for i in actual["row_indices"])
                    or actual["row_indices"] != indices
                    or type(actual["summary"]["rows"]) is not int
                    or actual["summary"]["rows"] != len(indices)
                ):
                    raise RuntimeError("readiness diagnostic stage row binding changed")


def _checkpoint(
    report,
    source,
    runtime,
    readout,
    episodes,
    arrays,
    initial,
    *,
    diagnostic_stage_bindings=None,
    diagnostic_checkpoint_counts=None,
) -> None:
    labels_by_name = None
    if diagnostic_stage_bindings is not None:
        owned = copy_readiness_stage_bindings(
            diagnostic_stage_bindings, (episodes["warm"], episodes["new"])
        )
        labels_by_name = dict(zip(("warm", "new"), (v.stages for v in owned), strict=True))
        _require_stage_diagnostics(
            report, readiness_stage_metadata(owned), diagnostic_checkpoint_counts
        )
    candidate = runtime.candidate_snapshot()
    state = candidate.state
    readout.restore_state(state)
    row = dict(
        updates=state.completed_updates,
        training_examples=state.training_examples,
        state_exact=False,
        owners_exact=False,
        contexts={},
    )
    report["checkpoints"].append(row)
    for name, episode in episodes.items():
        result: dict[str, Any] = dict(episode_id=episode.episode_id)
        row["contexts"][name] = result
        # Preserve each returned diagnostic before any subsequent state assertion.
        result["conditioning"] = asdict(diagnose_conditioning(readout, episode))
        if isinstance(readout, NumpyContinuousPCLearner):
            result["conditioning_definition"] = "common_task_head_bp_total_gradients_read_only"
        if not _same_state(state, readout.snapshot_state()):
            raise RuntimeError("conditioning readout changed owner")
        result["residual"] = asdict(
            diagnose_residual(
                readout,
                episode,
                ("all",) * len(episode.transitions)
                if labels_by_name is None
                else labels_by_name[name],
            )
        )
        served = runtime.actor.predict(arrays[name][0])
        result["actor_predictions"] = served.prediction.tolist()
        if served.actor_version != "actor-0" or not _same_state(state, readout.snapshot_state()):
            raise RuntimeError("residual or actor readout changed owner/version")
    if not _same_state(state, runtime.candidate_snapshot().state):
        raise RuntimeError("endpoint diagnostics changed private candidate")
    row["state_exact"] = True
    _require_owners(source, runtime, initial)
    row["owners_exact"] = True
    if diagnostic_stage_bindings is not None:
        _require_stage_diagnostics(
            report, readiness_stage_metadata(owned), diagnostic_checkpoint_counts
        )


def _probe_at_checkpoint(
    plan, report, source, runtime, readout, episodes, initial, expected, budget, clock
):
    if plan is None:
        return
    fields = deepcopy(vars(budget))
    last_clock = fields.pop("last_clock")
    limits, timer, tick = budget.budget, budget.clock, clock.now()

    def guard():
        nonlocal last_clock
        _require_owners(source, runtime, initial)
        if not _same_state(expected, runtime.candidate_snapshot().state) or not _same_state(
            expected, readout.snapshot_state()
        ):
            raise RuntimeError("continuous probe changed benchmark candidate/readout")
        current = vars(budget).copy()
        observed = current.pop("last_clock")
        replace(limits)
        if (
            budget.budget is not limits
            or budget.clock is not timer
            or current != fields
            or type(budget.updates_completed) is not int
            or type(budget.replay_examples_completed) is not int
            or type(observed) not in (int, float)
            or not math.isfinite(observed)
            or observed < last_clock
            or type(budget.started_at) not in (int, float)
            or not math.isfinite(budget.started_at)
            or type(clock.now()) is not int
            or clock.now() != tick
            or runtime._budget is not budget
            or runtime._inbox._budget is not budget
            or runtime._inbox._clock is not clock
        ):
            raise RuntimeError("continuous probe changed original benchmark budget/clock")
        last_clock = observed

    guard()
    entry = dict(benchmark_updates=expected.completed_updates, report=None)
    report["probes"].append(entry)
    result = run_continuous_probe(
        readout, plan, training_readouts=tuple(episodes.values()), check_boundary=guard
    )
    entry["report"] = result
    guard()
    if not result["complete"]:
        raise RuntimeError(
            "continuous fresh-task probe incomplete: " + result.get("error", "unknown failure")
        )


def _apply_packet(runtime, clock, budget, episode, arrays, index, expected_rows, event):
    features, targets = arrays
    before = runtime.candidate_snapshot().state
    observed, arrived = 2 * index, 2 * index + 1
    clock.advance_to(observed)
    identity = f"mlp-batch-{index}"
    runtime.record_experience(
        Experience(
            identity,
            episode.episode_id,
            observed,
            "actor-0",
            features,
            "train",
            ExperiencePermissions(training=True),
        )
    )
    if (
        runtime.train_ready(max_updates=1) != ()
        or not _same_state(before, runtime.candidate_snapshot().state)
        or budget.updates_completed != index
    ):
        raise RuntimeError("candidate changed before label arrival")
    event["deferred_exact"] = True
    clock.advance_to(arrived)
    runtime.record_label(
        LabelArrival(
            f"mlp-label-{index}", identity, episode.episode_id, arrived, "actor-0", targets
        )
    )
    event["update_attempted"] = True
    applied = runtime.train_ready(max_updates=1)
    event["applied"] = [asdict(item) for item in applied]
    paid_evidence = (
        json.dumps(event["applied"], sort_keys=True, allow_nan=False)
        if "training_row_indices" in event
        else None
    )
    state = runtime.candidate_snapshot().state
    event["candidate_updates"] = state.completed_updates
    event["candidate_training_examples"] = state.training_examples
    if len(applied) != 1:
        raise RuntimeError("exactly one packet must apply at arrival")
    receipt = applied[0]
    if (
        receipt.sample_id != identity
        or receipt.event_id != f"mlp-label-{index}"
        or receipt.episode_id != episode.episode_id
        or receipt.observed_at != observed
        or receipt.arrived_at != arrived
        or receipt.applied_at != arrived
        or receipt.update_number != index + 1
        or receipt.actor_version != "actor-0"
        or receipt.learner_version != "candidate-0"
        or receipt.diagnostic.definition
        != (
            CONTINUOUS_PC_ENERGY_ID
            if isinstance(before.policy, ContinuousPCPolicy)
            else OBJECTIVE
            if before.policy.binary_gripper_axis is None
            else bc.MIXED_OBJECTIVE
        )
        or not math.isfinite(receipt.diagnostic.value)
        or state.completed_updates != before.completed_updates + 1
        or state.training_examples != expected_rows
        or state.policy != before.policy
        or budget.updates_completed != index + 1
        or len(runtime.applied_updates) != index + 1
    ):
        raise RuntimeError("successful update receipt/state/budget differs from fixed sequence")
    if paid_evidence is not None and (
        json.dumps(event["applied"], sort_keys=True, allow_nan=False) != paid_evidence
        or json.dumps([asdict(runtime.applied_updates[-1])], sort_keys=True, allow_nan=False)
        != paid_evidence
    ):
        raise RuntimeError("stage-balanced paid receipt changed during public observation")
    event["success_validated"] = True
    return state


def _metrics(checkpoints) -> dict[str, float]:
    values = [
        {
            name: result["residual"]["residuals"]["overall"]["normalized_half_mse"]
            for name, result in row["contexts"].items()
        }
        for row in checkpoints
    ]
    return dict(
        warm_acquisition_gain=values[0]["warm"] - values[1]["warm"],
        new_acquisition_gain=values[1]["new"] - values[2]["new"],
        warm_retention_loss=values[2]["warm"] - values[1]["warm"],
        warm_return_recovery=values[2]["warm"] - values[3]["warm"],
        warm_return_deficit=values[3]["warm"] - values[1]["warm"],
        new_retention_loss=values[3]["new"] - values[2]["new"],
    )


def _work(
    episodes, policy, phase_updates: int = PHASE_UPDATES, *, a_only_control: bool = False
) -> dict[str, Any]:
    warm, new = (len(episodes[name].transitions) for name in ("warm", "new"))
    dimension, hidden, axes = (
        policy.spec.observation_dim,
        policy.hidden_dim,
        len(policy.spec.action_units),
    )
    updates = len(PHASES) * phase_updates
    middle_rows = warm if a_only_control else new
    train_rows, diagnostic_rows = phase_updates * (2 * warm + middle_rows), 4 * (warm + new)
    unique_training_rows = warm if a_only_control else warm + new
    backward_passes = updates + 8
    parameters = dimension * hidden + hidden + hidden * axes + axes
    work = dict(
        fit_calls=1,
        fit_rows=warm,
        fit_feature_values=warm * dimension,
        optimizer_updates=updates,
        training_rows=train_rows,
        unique_source_rows=warm + new,
        repeated_training_rows=train_rows - unique_training_rows,
        replay_budget_rows=0,
        conditioning_calls=8,
        conditioning_rows=diagnostic_rows,
        residual_calls=8,
        residual_rows=diagnostic_rows,
        actor_prediction_calls=8,
        actor_prediction_rows=diagnostic_rows,
        normalization_calls=updates + 24,
        normalization_rows=train_rows + 3 * diagnostic_rows,
        normalization_values=(train_rows + 3 * diagnostic_rows) * dimension,
        forward_passes=updates + 32,
        forward_rows=train_rows + 4 * diagnostic_rows,
        backward_passes=backward_passes,
        backward_rows=train_rows + diagnostic_rows,
        gradient_values=backward_passes * parameters,
        gradient_numeric_bytes=backward_passes * parameters * 8,
        dense_multiply_accumulates=(
            train_rows * (2 * dimension * hidden + 3 * hidden * axes)
            + diagnostic_rows * (3 * dimension * hidden + 4 * hidden * axes)
            + 2 * diagnostic_rows * (dimension * hidden + hidden * axes)
        ),
        parameter_values_per_owner=parameters,
        model_numeric_bytes_per_owner=parameters * 16,
        statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
            policy.observation_standardization
        ),
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=train_rows * (dimension + axes) * 8,
        residual_prediction_bytes=diagnostic_rows * axes * 8,
        actor_prediction_bytes=diagnostic_rows * axes * 8,
        scope="successful fixed run; dense math/numeric payloads exclude copies/validation/elementwise/process overhead",
    )
    if policy.binary_gripper_axis is not None:
        work["binary_gripper_loss_rows"] = train_rows + diagnostic_rows
    if a_only_control:
        work.update(
            unique_training_rows=warm,
            training_rows_by_context=dict(warm=train_rows, new=0),
            diagnostic_rows_by_context=dict(warm=4 * warm, new=4 * new),
        )
    return work


def _label_replay_checkpoint(report, wake_updates):
    """Keep native optimizer history distinct from admitted wake packets."""
    row = report["checkpoints"][-1]
    row.update(wake_updates=wake_updates, native_updates=row["updates"])
    if report.get("probes"):
        report["probes"][-1]["benchmark_wake_updates"] = wake_updates


def _add_replay_work(work, policy, replay, family):
    """Add selected replay math without treating bank rows as inbox storage."""
    updates, rows = replay["committed_updates"], replay["charged_exposures"]
    dimension, hidden, axes = (
        policy.spec.observation_dim,
        policy.hidden_dim,
        len(policy.spec.action_units),
    )
    parameters = dimension * hidden + hidden + hidden * axes + axes
    work.update(
        replay_budget_rows=rows,
        replay_optimizer_updates=updates,
        replay_retained_numeric_bytes=replay["retained_numeric_bytes"],
        replay_input_feature_target_numeric_bytes=rows * (dimension + axes) * 8,
        replay_operation_forks=updates,
        transient_replay_model_owners=1 if updates else 0,
        peak_named_model_owners=work["named_model_owners"] + int(updates > 0),
    )
    if family != "bp":
        # PC's native ledger already includes every returned replay train_batch.
        work["inbox_feature_target_numeric_bytes"] -= rows * (dimension + axes) * 8
        return
    for key in ("optimizer_updates", "normalization_calls", "forward_passes", "backward_passes"):
        work[key] += updates
    for key in (
        "training_rows",
        "repeated_training_rows",
        "normalization_rows",
        "forward_rows",
        "backward_rows",
    ):
        work[key] += rows
    work["normalization_values"] += rows * dimension
    work["gradient_values"] += updates * parameters
    work["gradient_numeric_bytes"] += updates * parameters * 8
    work["dense_multiply_accumulates"] += rows * (2 * dimension * hidden + 3 * hidden * axes)
    if "binary_gripper_loss_rows" in work:
        work["binary_gripper_loss_rows"] += rows
    if "training_rows_by_context" in work:
        work["training_rows_by_context"]["warm"] += rows


def _training_packet_work(plan: ReadinessPacketPlan) -> dict[str, Any]:
    count, spec = len(plan.indices), plan.binding.episode.spec
    return dict(
        source_index_selections=15 * count,
        retained_index_numeric_bytes=15 * count * 8,
        gathered_packets=count,
        gathered_feature_values=15 * count * spec.observation_dim,
        gathered_target_values=15 * count * len(spec.action_units),
        gathered_numeric_bytes=15 * count * (spec.observation_dim + len(spec.action_units)) * 8,
        scope="index selection and copied batch payload; validation, metadata recomputation, Python containers and RSS additional",
    )


def _require_training_packet_evidence(
    report,
    plan: ReadinessPacketPlan,
    completed_packets: int,
    *,
    expected_metadata=None,
    receipts=None,
) -> None:
    """Bind declared sampling to original indices and preceding paid successes."""
    if expected_metadata is None:
        expected_metadata = json.dumps(
            readiness_packet_metadata(plan), sort_keys=True, allow_nan=False
        )
    if (
        type(completed_packets) is not int
        or not 0 <= completed_packets <= len(plan.indices)
        or report["settings"].get("stage_balanced_packets") is not True
        or json.dumps(report.get("training_packets"), sort_keys=True, allow_nan=False)
        != expected_metadata
        or type(report.get("events")) is not list
        or len(report["events"]) != completed_packets
    ):
        raise RuntimeError("stage-balanced packet declaration or paid prefix changed")
    actual_receipts = report.get("applied_updates") if receipts is None else receipts
    if type(actual_receipts) is not list or len(actual_receipts) != completed_packets:
        raise RuntimeError("stage-balanced packet receipt prefix changed")
    if "applied_updates" in report and (
        json.dumps(report["applied_updates"], sort_keys=True, allow_nan=False)
        != json.dumps(actual_receipts, sort_keys=True, allow_nan=False)
    ):
        raise RuntimeError("stage-balanced returned receipt prefix changed")
    spec = plan.binding.episode.spec
    phase_updates = report["settings"]["phase_updates"]
    if type(phase_updates) is not int or 3 * phase_updates != len(plan.indices):
        raise RuntimeError("stage-balanced original phase budget changed")
    for index, event in enumerate(report["events"]):
        indices = event.get("training_row_indices")
        if (
            type(indices) is not list
            or any(type(value) is not int for value in indices)
            or tuple(indices) != plan.indices[index]
            or event.get("packet_gather_attempted") is not True
            or type(event.get("packet_feature_numeric_bytes")) is not int
            or event["packet_feature_numeric_bytes"] != 15 * spec.observation_dim * 8
            or type(event.get("packet_target_numeric_bytes")) is not int
            or event["packet_target_numeric_bytes"] != 15 * len(spec.action_units) * 8
            or event.get("success_validated") is not True
            or json.dumps(event.get("applied"), sort_keys=True, allow_nan=False)
            != json.dumps([actual_receipts[index]], sort_keys=True, allow_nan=False)
        ):
            raise RuntimeError("stage-balanced original rows or preceding paid success changed")
        expected_event = dict(
            index=index,
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            candidate_updates=index + 1,
            candidate_training_examples=15 * (index + 1),
            phase=PHASES[index // phase_updates][0],
            context="warm",
            deferred_exact=True,
            update_attempted=True,
            success_validated=True,
        )
        if any(
            type(event.get(key)) is not type(value) or event[key] != value
            for key, value in expected_event.items()
        ):
            raise RuntimeError("stage-balanced preceding event clocks or state changed")
    if "work" in report:
        work, metadata = report["work"], report["training_packets"]
        unique = metadata["unique_training_rows_actual"]
        if (
            type(work.get("unique_training_rows")) is not int
            or work["unique_training_rows"] != unique
            or type(work.get("repeated_training_rows")) is not int
            or work["repeated_training_rows"] != 15 * len(plan.indices) - unique
            or json.dumps(work.get("stage_balanced_packets"), sort_keys=True, allow_nan=False)
            != json.dumps(_training_packet_work(plan), sort_keys=True, allow_nan=False)
        ):
            raise RuntimeError("stage-balanced exposure or selection work changed")


def run_continuous_learnability(
    warm_episode,
    new_episode,
    *,
    expected_standardization: ObservationStandardization | None = None,
    scale_floors: tuple[float, ...] | None = None,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    diagnostic_stage_bindings: tuple[ReadinessStageBinding, ReadinessStageBinding] | None = None,
    binary_gripper_axis: int | None = None,
    rgb_range_recipe: PickCubeRGBRecipe | None = None,
    phase_updates: int = PHASE_UPDATES,
    a_only_control: bool = False,
    learner_family: str = "bp",
    probe_plan: ContinuousProbePlan | None = None,
    replay_plan: ContinuousReplayPlan | None = None,
    stage_balanced_packets: bool = False,
    head_gauss_newton: bool = False,
    checkpoint_observer: PhaseReadoutObserver | None = None,
) -> ContinuousLearnabilityResult:
    """Run bounded A/B/A (default8 each); release only after complete checks.

    Admission errors raise before model construction. A reference, when supplied,
    gates the single fit BEFORE any training; no data-driven settings are selected.
    a_only_control changes update information, while B labels remain available
    only to the same read-only diagnostics. Both inputs must remain TRAIN data.
    """
    if type(head_gauss_newton) is not bool:
        raise ValueError("head_gauss_newton must be an exact bool")
    phase_readouts = PhaseReadouts(checkpoint_observer) if checkpoint_observer is not None else None
    if phase_readouts is not None and (
        probe_plan is not None
        or replay_plan is not None
        or rgb_range_recipe is not None
        or stage_balanced_packets
        or a_only_control
    ):
        raise ValueError(
            "phase observers require plain A/B/A without probes, replay, RGB or balanced packets"
        )
    if action_history_recipe is not None and action_history.is_taskspace_recipe(
        action_history_recipe
    ):
        # Own this opt-in before a clock/input callback can replace its mode.
        action_history_recipe = action_history.validate_action_history_recipe(action_history_recipe)
    # Why: finite explicit counts permit prospective optimization-depth studies
    # without changing historical8/8/8 defaults or granting experiment authority.
    if type(a_only_control) is not bool:
        raise ValueError("a_only_control must be an exact bool")
    if type(stage_balanced_packets) is not bool:
        raise ValueError("stage_balanced_packets must be an exact bool")
    if stage_balanced_packets and (
        not a_only_control
        or diagnostic_stage_bindings is None
        or replay_plan is not None
        or probe_plan is not None
        or rgb_range_recipe is not None
    ):
        raise ValueError(
            "balanced packets require A-only readiness bindings without replay/probes/RGB"
        )
    if scale_floors is not None and rgb_range_recipe is not None:
        raise ValueError("scale floors and RGB-range normalization require separate recipes")
    learner_family = validate_learner_family(learner_family)
    if head_gauss_newton and (
        learner_family != "bp"
        or binary_gripper_axis is not None
        or replay_plan is not None
        or probe_plan is not None
        or rgb_range_recipe is not None
        or stage_balanced_packets
        or a_only_control
    ):
        raise ValueError(
            "head Gauss-Newton requires plain BP A/B/A without mixed loss, replay, probes, RGB or balanced packets"
        )
    if learner_family != "bp" and binary_gripper_axis is not None:
        raise ValueError("continuous PC/CPC supports only the default continuous likelihood")
    phase_updates = validate_phase_updates(phase_updates)
    total_updates = len(PHASES) * phase_updates
    started = monotonic()
    warm, new = _training_episodes(warm_episode, new_episode)
    if head_gauss_newton and (
        warm.spec.observation_dim > 64
        or any(len(episode.transitions) > 128 for episode in (warm, new))
    ):
        raise ValueError(
            "head Gauss-Newton requires at most64 inputs and128 rows per TRAIN episode"
        )
    if action_history_recipe is not None:
        action_history_recipe = action_history.validate_action_history_recipe(action_history_recipe)
        if action_history.is_taskspace_recipe(action_history_recipe) and (
            expected_standardization is not None
            or scale_floors is not None
            or rgb_range_recipe is not None
        ):
            raise ValueError(
                "taskspace12 uses fixed physical scaling without empirical statistics, floors or RGB"
            )
        warm, new = (
            action_history.validate_action_history_training_episode(value, action_history_recipe)
            for value in (warm, new)
        )
    stage_options: dict[str, Any] = {}
    stage_sources = None
    checkpoint_counts = None
    if diagnostic_stage_bindings is not None:
        diagnostic_stage_bindings = copy_readiness_stage_bindings(
            diagnostic_stage_bindings, (warm, new)
        )
        stage_sources = readiness_stage_metadata(diagnostic_stage_bindings)
        stage_options["diagnostic_stage_bindings"] = diagnostic_stage_bindings
    packet_plan = (
        prepare_readiness_packets(diagnostic_stage_bindings[0], total_updates)
        if stage_balanced_packets and diagnostic_stage_bindings is not None
        else None
    )
    packet_metadata = readiness_packet_metadata(packet_plan) if packet_plan is not None else None
    packet_metadata_evidence = json.dumps(packet_metadata, sort_keys=True, allow_nan=False)
    packet_plan_evidence = (
        json.dumps(asdict(packet_plan), sort_keys=True, allow_nan=False)
        if packet_plan is not None
        else None
    )
    if packet_plan is not None and (
        warm.spec.observation_dim not in (12, 42, 47)
        or (warm.spec.observation_dim in (12, 47) and action_history_recipe is None)
    ):
        raise ValueError("balanced packets require official state42 or declared causal history")
    replay_bank = None
    replay_events = 0
    replay_rows = 0
    if replay_plan is not None:
        replay_plan, replay_bank, replay_rows, _ = prepare_replay_bank(replay_plan, (warm,))
        replay_events = total_updates // replay_plan.every_updates
    if stage_sources is not None:
        checkpoint_counts = _readiness_checkpoint_counts(
            warm, new, phase_updates, a_only_control, replay_plan, replay_rows
        )
        stage_options["diagnostic_checkpoint_counts"] = checkpoint_counts
    probe_plan = copy_probe_plan(probe_plan, (warm, new))
    if rgb_range_recipe is not None:
        rgb_range_recipe = validate_pick_cube_rgb_recipe(rgb_range_recipe)
        for episode in (warm, new):
            features, _ = continuous_training_pairs(episode)
            validate_rgb_range_features(np.asarray(features, dtype=np.float64), rgb_range_recipe)
            validate_rgb_range_features(
                np.asarray([episode.transitions[-1].next_observation], dtype=np.float64),
                rgb_range_recipe,
            )
    if binary_gripper_axis is not None:
        bc._binary_gripper_axis(binary_gripper_axis, warm.spec)
        binary_sources: tuple[tuple[Any, str], ...] = ((warm, "train"), (new, "train"))
        if probe_plan is not None:
            binary_sources += ((probe_plan.training, "probe_train"),)
        for episode, role in binary_sources:
            _, targets = continuous_training_pairs(episode, role=role)
            bc._binary_gripper_targets(
                np.asarray(targets, dtype=np.float64), warm.spec, binary_gripper_axis
            )
    fit_options: dict[str, Any] = {}
    if action_history_recipe is not None:
        fit_options["action_history_recipe"] = action_history_recipe
    if scale_floors is not None:
        fit_options["scale_floors"] = scale_floors
    if rgb_range_recipe is not None:
        fit_options["rgb_range_recipe"] = rgb_range_recipe
    statistics = fit_training_standardization(warm, **fit_options)
    if expected_standardization is not None and (
        statistics != _standardization(expected_standardization, warm.spec.observation_dim)
    ):
        raise ValueError("single A fit differs from expected frozen standardization")
    policy = ContinuousBCPolicy(
        warm.spec, observation_standardization=statistics, binary_gripper_axis=binary_gripper_axis
    )
    native_updates: list[dict[str, Any]] = []
    source: ContinuousLearner
    if learner_family == "bp":
        source = NumpyContinuousBCLearner(policy)
    else:
        chemistry = (
            None
            if learner_family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if learner_family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        source = NumpyContinuousPCLearner(ContinuousPCPolicy(policy, circadian=chemistry))
    initial = source.snapshot_state()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=total_updates), monotonic)
    native_source: Any
    if head_gauss_newton:
        assert isinstance(source, NumpyContinuousBCLearner)
        native_source = RecordedContinuousHead(source, native_updates, total_updates)
    else:
        native_source = (
            source
            if learner_family == "bp"
            else RecordedContinuousPC(source, native_updates, total_updates + replay_events)
        )
    runtime = ActorShadowRuntime(
        native_source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=total_updates,
        max_consolidations=replay_events,
    )
    rehearsal = None
    if replay_plan is not None:
        rehearsal = ContinuousStreamRehearsal(
            native_source,
            budget,
            replay_plan,
            replay_budget=ToyBudgetSession(
                ToyExecutionBudget(max_replay_examples=replay_plan.max_replay_examples), monotonic
            ),
            total_updates=total_updates,
            fixed_training_episodes=replay_bank,
            states_equal=_same_state,
            expected_definition=(
                CONTINUOUS_PC_ENERGY_ID
                if learner_family != "bp"
                else OBJECTIVE
                if binary_gripper_axis is None
                else bc.MIXED_OBJECTIVE
            ),
        )
        if rehearsal.bind(runtime) != replay_events:
            raise RuntimeError("continuous replay event quota differs from fixed sequence")
    readout = source.fork()
    head_owners = (
        (
            runtime._actor,
            runtime._actor._learner,
            runtime._candidate,
            readout,
            _head_port(runtime._actor._learner)._learner,
            _head_port(runtime._candidate)._learner,
        )
        if head_gauss_newton
        else None
    )
    episodes = dict(warm=warm, new=new)
    arrays = {
        name: tuple(
            np.asarray(value, dtype=np.float64) for value in continuous_training_pairs(episode)
        )
        for name, episode in episodes.items()
    }
    report: dict[str, Any] = dict(
        schema_version=1,
        protocol_id="continuous_mlp_train_aba_v1",
        scope="saved_train_imitation",
        complete=False,
        policy=asdict(policy),
        events=[],
        checkpoints=[],
        settings=dict(
            phase_updates=phase_updates,
            phases=[name for name, _ in PHASES],
            max_training_updates=total_updates,
            promotion_calls=0,
            simulator_calls=0,
            probe_reads=0,
            final_reads=0,
            guard_reads=0,
        ),
        source_episodes={
            name: dict(episode_id=episode.episode_id, context=episode.context, role=episode.role)
            for name, episode in episodes.items()
        },
        expected_standardization_checked=expected_standardization is not None,
    )
    if action_history_recipe is not None:
        report["action_history_recipe"] = asdict(action_history_recipe)
        if action_history.is_taskspace_recipe(action_history_recipe):
            report["fixed_input_normalization"] = _taskspace_normalization_metadata()
    if a_only_control:
        report["settings"].update(
            a_only_control=True,
            training_contexts=["warm"] * len(PHASES),
            diagnostic_contexts=list(episodes),
        )
    if stage_sources is not None:
        report["diagnostic_stage_sources"] = deepcopy(stage_sources)
    if packet_plan is not None:
        report["settings"]["stage_balanced_packets"] = True
        report["training_packets"] = deepcopy(packet_metadata)
    if rehearsal is not None:
        report["replay"] = rehearsal.report
    if learner_family != "bp":
        report.update(
            protocol_id="continuous_matched_family_train_aba_v1",
            learner_family=learner_family,
            policy=asdict(source.policy),
            native_updates=native_updates,
        )
    if head_gauss_newton:
        report.update(
            protocol_id="continuous_head_gauss_newton_train_aba_v1",
            head_gauss_newton=head_protocol(),
            native_updates=native_updates,
        )
    if probe_plan is not None:
        report["probes"] = []
        report["settings"].update(
            probe_reads=8,
            probe_plan=dict(
                invocations=4,
                updates_each=probe_plan.updates,
                total_training_update_budget=4 * probe_plan.updates,
            ),
        )
        if probe_plan.zero_velocity_control:
            report["settings"]["probe_reads"] = 16
            report["settings"]["probe_plan"].update(
                zero_velocity_control=True,
                inherited_invocations=4,
                zero_velocity_invocations=4,
                updates_per_scope=probe_plan.updates,
                updates_each=2 * probe_plan.updates,
                total_training_update_budget=8 * probe_plan.updates,
            )
    if rgb_range_recipe is not None:
        report["input_normalization"] = dict(
            definition=DEFINITION,
            recipe=asdict(rgb_range_recipe),
            fitted_prefix_dimensions=21,
            fixed_range_dimensions=768,
            rgb_mean=0.5,
            rgb_scale=0.5,
            range_validation_rows=2 * len(warm.transitions) + len(new.transitions),
            terminal_range_validation_rows=3,
        )
    packet_receipts: list[dict[str, Any]] = []
    head_readout_state = initial
    observer_failure_progress = None
    if phase_readouts is not None:
        report["phase_readouts"] = phase_readouts.report

    def observe_phase():
        nonlocal observer_failure_progress
        if phase_readouts is None:
            return
        index = len(report["checkpoints"]) - 1
        original = deepcopy(report)
        progress = dict(
            update_attempts=sum(event["update_attempted"] for event in report["events"]),
            updates_completed=budget.updates_completed,
            replay_examples_completed=budget.replay_examples_completed,
            next_tick=clock.now() + 1,
        )
        guard = _phase_callback_guard(
            source, runtime, readout, initial, expected_candidate, budget, clock, report
        )
        try:
            phase_readouts.observe(
                readout,
                dict(
                    checkpoint_index=index,
                    phase=("initial", "warm", "new", "warm_return")[index],
                    updates=expected_candidate.completed_updates,
                    training_examples=expected_candidate.training_examples,
                ),
                guard,
            )
            require_head()
        except Exception:
            # Preserve preceding paid evidence, never restore corrupted live authority.
            observer_failure_progress = progress
            report.clear()
            report.update(original)
            raise
        finally:
            report["phase_readouts"] = phase_readouts.report

    def require_head():
        if not head_gauss_newton:
            return
        _require_head_report(report)
        native_source.require_reports(report.get("native_updates"))
        assert head_owners is not None
        if (
            runtime._actor is not head_owners[0]
            or runtime._actor._learner is not head_owners[1]
            or runtime._candidate is not head_owners[2]
            or runtime._actor._version != "actor-0"
            or _head_port(runtime._actor._learner)._learner is not head_owners[4]
            or _head_port(runtime._candidate)._learner is not head_owners[5]
        ):
            raise RuntimeError("head stream native owner was replaced")
        native_source.require_latest(expected_candidate)
        models = (
            (source, initial),
            (_head_port(runtime._actor._learner)._learner, initial),
            (_head_port(runtime._candidate)._learner, expected_candidate),
            (readout, head_readout_state),
        )
        for model, expected in models:
            if type(model) is not NumpyContinuousBCLearner or not _same_state(
                own_bc_snapshot(model._state, model._policy), expected
            ):
                raise RuntimeError("head stream native owner state changed")

    try:

        def require_packets(completed):
            if packet_plan is not None:
                current = [asdict(item) for item in runtime.applied_updates]
                if json.dumps(current, sort_keys=True, allow_nan=False) != json.dumps(
                    packet_receipts, sort_keys=True, allow_nan=False
                ):
                    raise RuntimeError("stage-balanced preceding paid receipt changed")
                _require_training_packet_evidence(
                    report,
                    packet_plan,
                    completed,
                    expected_metadata=packet_metadata_evidence,
                    receipts=packet_receipts,
                )
                if (
                    json.dumps(asdict(packet_plan), sort_keys=True, allow_nan=False)
                    != packet_plan_evidence
                ):
                    raise RuntimeError("stage-balanced original full plan changed")

        expected_candidate = initial
        if head_gauss_newton:
            require_head()
        _checkpoint(report, source, runtime, readout, episodes, arrays, initial, **stage_options)
        require_head()
        observe_phase()
        _probe_at_checkpoint(
            probe_plan,
            report,
            source,
            runtime,
            readout,
            episodes,
            initial,
            expected_candidate,
            budget,
            clock,
        )
        if rehearsal is not None:
            _label_replay_checkpoint(report, 0)
        rows, index, replay_updates = 0, 0, 0
        for phase, phase_context in PHASES:
            # Why: the control changes update information only. Checkpoints keep
            # the actual new task, and receipts identify the true TRAIN source.
            name = "warm" if a_only_control else phase_context
            for _ in range(phase_updates):
                if head_gauss_newton:
                    require_head()
                require_packets(index)
                if action_history_recipe is not None:
                    _require_action_history_recipe(report, action_history_recipe)
                if stage_sources is not None:
                    _require_stage_diagnostics(report, stage_sources, checkpoint_counts)
                if rehearsal is not None:
                    rehearsal.before_update(runtime, index)
                if not _same_state(expected_candidate, runtime.candidate_snapshot().state):
                    raise RuntimeError("candidate changed after preceding validated success")
                require_head()
                require_packets(index)
                event: dict[str, Any] = dict(
                    index=index,
                    phase=phase,
                    context=name,
                    observed_at=2 * index,
                    arrived_at=2 * index + 1,
                    deferred_exact=False,
                    update_attempted=False,
                    applied=[],
                    success_validated=False,
                )
                report["events"].append(event)
                packet_arrays = arrays[name]
                if packet_plan is not None:
                    # Why: gather unchanged encoded rows, never construct a new
                    # episode with fabricated chronological seams or histories.
                    event["training_row_indices"] = list(packet_plan.indices[index])
                    event["packet_gather_attempted"] = True
                    packet_features = np.asarray(
                        [
                            packet_plan.binding.episode.transitions[i].observation
                            for i in packet_plan.indices[index]
                        ],
                        dtype=np.float64,
                    )
                    event["packet_feature_numeric_bytes"] = packet_features.nbytes
                    packet_targets = np.asarray(
                        [
                            packet_plan.binding.episode.transitions[i].action
                            for i in packet_plan.indices[index]
                        ],
                        dtype=np.float64,
                    )
                    event["packet_target_numeric_bytes"] = packet_targets.nbytes
                    packet_arrays = (packet_features, packet_targets)
                rows += len(episodes[name].transitions)
                completed_state = _apply_packet(
                    runtime, clock, budget, episodes[name], packet_arrays, index, rows, event
                )
                if packet_plan is not None:
                    packet_receipts.extend(deepcopy(event["applied"]))
                require_packets(index + 1)
                expected_candidate = (
                    runtime.candidate_snapshot().state
                    if completed_state is None
                    else completed_state
                )
                if (learner_family != "bp" or head_gauss_newton) and (
                    len(native_updates) != index + 1 + replay_updates
                    or native_updates[-1]["diagnostic"] != event["applied"][0]["diagnostic"]
                    or native_updates[-1]["training_rows"] != len(episodes[name].transitions)
                    or native_updates[-1]["work"]["published_updates"] != 1
                ):
                    raise RuntimeError("native report differs from validated runtime success")
                _require_owners(source, runtime, initial)
                if not _same_state(expected_candidate, runtime.candidate_snapshot().state):
                    raise RuntimeError("candidate changed after preceding validated success")
                require_head()
                index += 1
                if rehearsal is not None:
                    assert replay_plan is not None
                    preceding = expected_candidate
                    # Preserve permanently paid work even when publication/checks fail.
                    try:
                        rehearsal.after_update(runtime, index)
                    finally:
                        report["replay"] = rehearsal.report
                    observed_replay = report["replay"]["committed_updates"]
                    delta = observed_replay - replay_updates
                    expected_delta = int(index % replay_plan.every_updates == 0)
                    state = runtime.candidate_snapshot().state
                    if (
                        delta != expected_delta
                        or state.completed_updates != preceding.completed_updates + delta
                        or state.training_examples != rows + delta * report["replay"]["bank_rows"]
                        or state.policy != preceding.policy
                        or (delta == 0 and not _same_state(preceding, state))
                        or (
                            learner_family != "bp"
                            and len(native_updates) != index + observed_replay
                        )
                    ):
                        raise RuntimeError("replay publication differs from validated sequence")
                    rows = state.training_examples
                    replay_updates = observed_replay
                    expected_candidate = state
                    _require_owners(source, runtime, initial)
                    if not _same_state(expected_candidate, runtime.candidate_snapshot().state):
                        raise RuntimeError("candidate changed after validated replay")
            _checkpoint(
                report, source, runtime, readout, episodes, arrays, initial, **stage_options
            )
            head_readout_state = expected_candidate
            require_head()
            observe_phase()
            _probe_at_checkpoint(
                probe_plan,
                report,
                source,
                runtime,
                readout,
                episodes,
                initial,
                expected_candidate,
                budget,
                clock,
            )
            if rehearsal is not None:
                _label_replay_checkpoint(report, index)
                rehearsal.before_update(runtime, index)
        if runtime.train_ready(max_updates=1) != () or budget.replay_examples_completed != 0:
            raise RuntimeError("fixed runtime has unexpected remaining work/replay charge")
        if (
            not _same_state(expected_candidate, runtime.candidate_snapshot().state)
            or not _same_state(readout.snapshot_state(), runtime.candidate_snapshot().state)
            or budget.updates_completed != total_updates
            or readout.snapshot_state().training_examples != rows
        ):
            raise RuntimeError("final empty poll changed candidate/state/budget")
        _require_owners(source, runtime, initial)
        if not _same_state(
            expected_candidate, runtime.candidate_snapshot().state
        ) or not _same_state(expected_candidate, readout.snapshot_state()):
            raise RuntimeError("final owner check changed candidate/readout")
        if rehearsal is not None:
            rehearsal.before_update(runtime, total_updates)
        if head_gauss_newton:
            if len(native_updates) != total_updates:
                raise RuntimeError("head update reports differ from completed sequence")
            work = head_work(episodes, policy, report)
        elif learner_family == "bp":
            work = _work(episodes, policy, phase_updates, a_only_control=a_only_control)
        else:
            if len(native_updates) != total_updates + replay_updates:
                raise RuntimeError("native update reports differ from completed sequence")
            work = pc_work(episodes, source.policy, report, a_only_control=a_only_control)
        if rehearsal is not None:
            _add_replay_work(work, policy, report["replay"], learner_family)
        if action_history_recipe is not None:
            _require_action_history_recipe(report, action_history_recipe)
            if action_history.is_taskspace_recipe(action_history_recipe):
                work.update(fit_calls=0, fit_rows=0, fit_feature_values=0)
                work["fixed_physical_displacement_dimensions"] = 6
            else:
                work["fit_feature_values"] = len(warm.transitions) * 42
            work["fixed_history_dimensions"] = 5
        if rgb_range_recipe is not None:
            work["fit_feature_values"] = len(warm.transitions) * 21
            work["fixed_range_dimensions"] = 768
        if stage_sources is not None:
            _require_stage_diagnostics(report, stage_sources, checkpoint_counts)
            work["stage_residual_summaries"] = 4 * sum(
                len(set(item["stages"])) for item in stage_sources.values()
            )
            work["stage_residual_group_index_visits"] = 4 * sum(
                item["rows"] * len(set(item["stages"])) for item in stage_sources.values()
            )
        if packet_plan is not None:
            require_packets(total_updates)
            assert packet_metadata is not None
            work["unique_training_rows"] = packet_metadata["unique_training_rows_actual"]
            work["repeated_training_rows"] = rows - work["unique_training_rows"]
            work["stage_balanced_packets"] = _training_packet_work(packet_plan)
        report.update(complete=True, metrics=_metrics(report["checkpoints"]), work=work)
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        if hasattr(error, "work"):
            report["failed_native_work"] = error.work
    # A postcommit error can occur before train_ready returns its receipt.
    # The runtime retains at most total_updates receipts, including that charge.
    report["applied_updates"] = (
        [asdict(item) for item in runtime.applied_updates]
        if observer_failure_progress is None
        else deepcopy([item for event in report["events"] for item in event["applied"]])
    )
    if rehearsal is not None:
        report["replay"] = rehearsal.report
        report["applied_consolidations"] = [asdict(item) for item in runtime.applied_consolidations]
    observed_tick = clock.now() if observer_failure_progress is None else 0
    next_tick = (
        observed_tick + 1
        if probe_plan is None or (type(observed_tick) is int and observed_tick >= 0)
        else None
    )
    report["progress"] = dict(
        update_attempts=sum(event["update_attempted"] for event in report["events"]),
        updates_completed=budget.updates_completed,
        replay_examples_completed=budget.replay_examples_completed,
        next_tick=next_tick,
    )
    if observer_failure_progress is not None:
        report["progress"] = observer_failure_progress
    if rehearsal is not None:
        report["progress"].update(
            native_updates_completed=(
                expected_candidate.completed_updates if report["complete"] else None
            ),
            replay_updates_completed=report["replay"]["committed_updates"],
            replay_rows_charged=report["replay"]["charged_exposures"],
        )
    if probe_plan is not None:
        for key, value in tuple(report["progress"].items()):
            if type(value) is not int or value < 0:
                report["progress"][key] = None
    head_final_payload = (
        json.dumps(report, sort_keys=True, allow_nan=False)
        if head_gauss_newton or phase_readouts is not None
        else None
    )
    if head_gauss_newton or phase_readouts is not None:
        elapsed = None
        try:
            final_guard = (
                _phase_callback_guard(
                    source, runtime, readout, initial, expected_candidate, budget, clock, report
                )
                if phase_readouts is not None and report["complete"]
                else None
            )
            elapsed = monotonic() - started
            if final_guard is not None:
                final_guard()
            report["elapsed_seconds"] = elapsed
            current = {key: value for key, value in report.items() if key != "elapsed_seconds"}
            if json.dumps(current, sort_keys=True, allow_nan=False) != head_final_payload:
                raise RuntimeError("head stream final observer changed its scalar report")
            require_head()
        except Exception as error:
            # Keep the original paid report, even if the last observer edited it.
            assert head_final_payload is not None
            report = json.loads(head_final_payload)
            report.update(complete=False, elapsed_seconds=elapsed)
            report.setdefault("error", f"{type(error).__name__}: {error}")
    else:
        report["elapsed_seconds"] = monotonic() - started
    if stage_sources is not None:
        try:
            _require_stage_diagnostics(report, stage_sources, checkpoint_counts)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            report["complete"] = False
            report.setdefault("error", f"{type(error).__name__}: {error}")
    if action_history_recipe is not None:
        try:
            _require_action_history_recipe(report, action_history_recipe)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            report["complete"] = False
            report.setdefault("error", f"{type(error).__name__}: {error}")
    if packet_plan is not None and report["complete"]:
        try:
            require_packets(total_updates)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            report["complete"] = False
            report.setdefault("error", f"{type(error).__name__}: {error}")
    return ContinuousLearnabilityResult(report, readout if report["complete"] else None)


def run_continuous_learnability_demo(
    warm_episode,
    new_episode,
    *,
    expected_standardization: ObservationStandardization | None = None,
    scale_floors: tuple[float, ...] | None = None,
    action_history_recipe: action_history.ActionHistoryRecipe | None = None,
    diagnostic_stage_bindings: tuple[ReadinessStageBinding, ReadinessStageBinding] | None = None,
    binary_gripper_axis: int | None = None,
    rgb_range_recipe: PickCubeRGBRecipe | None = None,
    phase_updates: int = PHASE_UPDATES,
    a_only_control: bool = False,
    learner_family: str = "bp",
    probe_plan: ContinuousProbePlan | None = None,
    replay_plan: ContinuousReplayPlan | None = None,
    stage_balanced_packets: bool = False,
    head_gauss_newton: bool = False,
    checkpoint_observer: PhaseReadoutObserver | None = None,
) -> dict[str, Any]:
    """Existing JSON-only entry point; native callers can retain the owned result."""
    if type(head_gauss_newton) is not bool:
        raise ValueError("head_gauss_newton must be an exact bool")
    if type(a_only_control) is not bool:
        raise ValueError("a_only_control must be an exact bool")
    if type(stage_balanced_packets) is not bool:
        raise ValueError("stage_balanced_packets must be an exact bool")
    learner_family = validate_learner_family(learner_family)
    phase_updates = validate_phase_updates(phase_updates)
    options: dict[str, Any] = {}
    if checkpoint_observer is not None:
        options["checkpoint_observer"] = checkpoint_observer
    if head_gauss_newton:
        options["head_gauss_newton"] = True
    if stage_balanced_packets:
        options["stage_balanced_packets"] = True
    if action_history_recipe is not None:
        options["action_history_recipe"] = action_history_recipe
    if scale_floors is not None:
        options["scale_floors"] = scale_floors
    if diagnostic_stage_bindings is not None:
        options["diagnostic_stage_bindings"] = diagnostic_stage_bindings
    if replay_plan is not None:
        options["replay_plan"] = replay_plan
    if probe_plan is not None:
        options["probe_plan"] = probe_plan
    if learner_family != "bp":
        options["learner_family"] = learner_family
    if a_only_control:
        options["a_only_control"] = True
    if phase_updates != PHASE_UPDATES:
        options["phase_updates"] = phase_updates
    if rgb_range_recipe is not None:
        # Preserve the existing RGB route's explicit binary=None call shape.
        options.update(rgb_range_recipe=rgb_range_recipe, binary_gripper_axis=binary_gripper_axis)
    elif binary_gripper_axis is not None:
        options["binary_gripper_axis"] = binary_gripper_axis
    return run_continuous_learnability(
        warm_episode,
        new_episode,
        expected_standardization=expected_standardization,
        **options,
    ).report
