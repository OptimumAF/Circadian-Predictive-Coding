"""Fixed native BP shadow learning, guarded typed publication and rollback.

Reuses a qualified old-task state and the existing delayed stream/controller.
Development-template guards are not final evaluation or generalization evidence.
Optional --reference selects fixed causal policies through shared native
admission; its finite observation horizon may leave unapplied TRAIN batches.
"""

import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import run_bp_new_contexts as context
from scripts import ticket_workflow_native as native
from scripts.run_ticket_workflow import FEATURES
from scripts.ticket_controller_streams import (
    TicketStreamRecipe,
    build_stream,
    LiveTrainingRecipe,
    LIVE_TRAINING_RECIPES,
    FOLLOWUP_TRAINING_RECIPES,
    require_followup_recipe,
    build_followup_stream,
    build_live_stream,
    build_policy_change_stream,
    POLICY_CHANGE_PHASES,
    POLICY_CHANGE_CHECKPOINTS,
)
from scripts.context_training_streams import CONTEXT_TRAINING_RECIPES
from scripts.ticket_serving_timeline import (
    LiveServingPlan,
    TicketServingTimeline,
    FollowupTrainingPlan,
)
from src.adapters.torch_serving import (
    TorchServingPredictor,
    build_native,
    categorical_guard,
    snapshot_digest,
)
from src.app.coupled_attribution import AttributionStream
from src.app.coupled_attribution import _record_tick, _validated_batches, _same_value
from src.app.actor_shadow import ActorShadowRuntime
from src.app.controller_chronology import ControllerChronology
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import (
    PromotableActor,
    PromotionRejected,
    ServingPromotionController,
)
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.controller_objective import (
    ControllerObjective,
    ControllerProtocol,
    ControllerStream,
    ControllerMeasures,
    ControllerTrial,
    ControllerTrialMetadata,
)
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.contextual_controller import ContextualChoice, FittedController, choose_contextual
from src.core.experience import LogicalClock, Experience, ExperiencePermissions, LabelArrival
from src.core.reference_controller import (
    REFERENCE_MODES,
    ReferencePolicy,
    ReferenceObservation,
    ReferenceDecision,
    ArrivedTrainingFeedback,
    choose_reference,
)
from src.core.resource_sharing import SharingLimits
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import PromotionReceipt, ServingConfiguration
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler


def policy() -> PromotionPolicy:
    return PromotionPolicy(
        "global_categorical_accuracy",
        "complete_native_tensor_bytes",
        1.0,
        0.1,
        0.0,
        0.25,
        4096,
        ACTION_IDS,
    )


def guards(spec: DecisionFeatureSpec, now: int) -> tuple[Any, Any]:
    _, teacher, _ = context._teacher_batches(spec)

    def batch(name: str, start: int) -> Any:
        return GuardBatch(
            name,
            tuple((f"guard-{name}", str(i)) for i in range(6)),
            now,
            teacher.features[start : start + 6].copy(),
            teacher.targets[start : start + 6].copy(),
            "inner_guard",
        )

    return batch("new", 6), batch("old", 0)


def _reference_schedule(
    source,
    actor,
    batches,
    budget,
    mode,
    report,
    require_input,
    midway,
    contextual_policy=None,
    label_delay=1,
    sharing_gate=None,
    live=None,
    on_drain=None,
    followup_plan=None,
    policy_change=False,
    candidate_predict=None,
    candidate_row_limit=84,
    acknowledge_correct=False,
    task_readout=None,
    followup_label_delay=None,
    arrived_shift_predict=None,
):
    """Same native arrival/runtime ports, finite observation horizon; no forced drain."""
    if arrived_shift_predict is not None and (
        not callable(arrived_shift_predict)
        or mode != "drift"
        or contextual_policy is not None
        or candidate_predict is not None
        or acknowledge_correct
        or task_readout is not None
        or policy_change
        or followup_plan is not None
    ):
        raise ValueError("arrived shift feedback requires separate explicit fixed drift mode")
    if type(acknowledge_correct) is not bool or acknowledge_correct and candidate_predict is None:
        raise ValueError("TRAIN acknowledgement requires explicit candidate feedback")
    if followup_label_delay is not None and followup_plan is None:
        raise ValueError("followup label delay requires an explicit bounded second window")
    followup_label_delay = label_delay if followup_label_delay is None else followup_label_delay
    if type(followup_label_delay) is not int or followup_label_delay not in (1, 2):
        raise ValueError("followup label delay requires an exact delay1or2")
    if task_readout is not None and (
        not callable(task_readout)
        or mode != "drift"
        or contextual_policy is not None
        or candidate_predict is not None
        or acknowledge_correct
        or policy_change
        or followup_plan is not None
    ):
        raise ValueError("task feedback requires separate explicit fixed drift mode")
    if (
        type(candidate_row_limit) is not int
        or not 1 <= candidate_row_limit <= 384
        or candidate_predict is None
        and candidate_row_limit != 84
    ):
        raise ValueError("candidate diagnostic row limit requires an explicit bounded callback")
    if candidate_predict is not None and (
        not callable(candidate_predict)
        or mode != "drift"
        or contextual_policy is not None
        or policy_change
        or followup_plan is not None
    ):
        raise ValueError("candidate feedback requires explicit fixed drift without followup")
    batches, _ = _validated_batches(batches, (), training_only=True)
    if type(policy_change) is not bool:
        raise ValueError("policy-change mode must be an exact boolean")
    if policy_change:
        if (
            len(batches) != 12
            or label_delay != 1
            or live is not None
            or on_drain is not None
            or followup_plan is not None
            or any(
                b.context != POLICY_CHANGE_PHASES[i // 4][0]
                or b.features.shape != (6, 7)
                or b.num_classes != 6
                for i, b in enumerate(batches)
            )
        ):
            raise ValueError("policy-change schedule requires the fixed12batch TRAIN recipe")
        report["policy_checkpoints"] = []
        report["policy_capture_complete"] = False
    if not policy_change and followup_plan is None and not 1 <= len(batches) <= 8:
        raise ValueError("reference schedule admits1..8fixed TRAIN batches")
    if type(label_delay) is not int or label_delay not in (1, 2):
        raise ValueError("reference schedule requires declared delay1or2")
    horizon = len(batches) + label_delay if live is None else live.plan.opportunities
    if followup_plan is not None:
        if (
            type(followup_plan) is not FollowupTrainingPlan
            or live is None
            or on_drain is None
            or contextual_policy is not None
            or mode != "periodic"
        ):
            raise ValueError("followup requires explicit live publication and fixed reference")
        followup_plan.__post_init__()
        second = len(batches) - followup_plan.first_batches
        if (
            not 1 <= second <= 8
            or followup_plan.start_tick + second + followup_label_delay > horizon
        ):
            raise ValueError("followup window/labels exceed original finite plan")
        report["followup_training"] = dict(
            plan=asdict(followup_plan), status="planned", rows=[len(b.features) for b in batches]
        )
        if followup_label_delay != label_delay:
            report["followup_training"]["label_delay"] = followup_label_delay
    followup_pin = deepcopy(followup_plan)
    drain_count = len(batches) if followup_plan is None else followup_plan.first_batches
    if horizon < len(batches) + label_delay:
        raise ValueError("live plan must include all original label arrivals")
    report["training_stream"] = dict(
        role="train",
        contexts=[b.context for b in batches],
        first_features=batches[0].features.tolist(),
        first_targets=batches[0].targets.tolist(),
        row_counts=[len(b.features) for b in batches],
        label_delay=label_delay,
    )
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version=actor.version,
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=len(batches),
        max_consolidations=0,
        actor=actor,
    )
    gate = sharing_gate or ServingPriorityGate(
        SharingLimits(1, len(batches), 1), resource_available=lambda: True
    )
    if live is not None and live.gate is not gate:
        raise ValueError("live serving requires the original shared gate")
    if on_drain is not None and (live is None or not callable(on_drain)):
        raise ValueError("drained publication requires the original live schedule")
    shared = ResourceSharedRuntime(
        runtime,
        gate,
        action_envelope=AdaptationEnvelope(
            (AdaptationAction("wait"), AdaptationAction("wake1", 1)), horizon, len(batches)
        ),
    )
    policy = ReferencePolicy(mode) if contextual_policy is None else contextual_policy
    report["reference_policy" if contextual_policy is None else "contextual_policy"] = asdict(
        policy
    )
    report["controller_events"] = []
    paid = report["controller_work"] = dict(
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows=0,
        feedback_rows=0,
        selections_attempted=0,
        selections_returned=0,
        selection_seconds=0.0,
        observation_seconds=0.0,
        admission_seconds=0.0,
        prediction_records_peak=0,
    )
    pending = {}
    feedback = None
    drain_attempted = False
    publication_pin = None
    arrived = 0
    task_reader = None
    shift_reader = None
    if arrived_shift_predict is not None:
        from scripts.arrived_candidate_shift import ArrivedCandidateShift

        shift_reader = ArrivedCandidateShift(arrived_shift_predict)
        report["arrived_shift_feedback"] = shift_reader.report
    if task_readout is not None:
        from scripts.candidate_task_feedback import CandidateTaskFeedback

        task_reader = CandidateTaskFeedback(task_readout, horizon)
        report["task_feedback"] = task_reader.report
    if candidate_predict is not None:
        report["candidate_feedback"] = dict(
            definition="arrived_train_candidate_diagnostic_v1",
            attempts=0,
            returns=0,
            rows_attempted=0,
            rows_returned=0,
            seconds=0.0,
            records=[],
        )
        if candidate_row_limit != 84:
            report["candidate_feedback"]["row_limit"] = candidate_row_limit
    try:
        for tick in range(horizon):
            require_input()
            if shift_reader is not None:
                shift_reader.require_paid(report.get("arrived_shift_feedback"))
            if task_reader is not None:
                task_reader.require_paid(report.get("task_feedback"))
            if followup_plan is not None:
                followup_plan.__post_init__()
            if followup_plan != followup_pin:
                raise ValueError("original followup plan changed")
            clock.advance_to(tick)
            observation_started = monotonic()
            try:
                index = (
                    tick if followup_plan is None else followup_plan.index_at(tick, len(batches))
                )
                if followup_plan is not None and tick >= followup_plan.start_tick:
                    if _require_followup_publication(actor, report) != publication_pin:
                        raise ValueError("original verified publication changed")
                    report["followup_training"]["status"] = "started"
                if index is not None and index < len(batches):
                    batch = batches[index]
                    paid["prediction_attempts"] += 1
                    paid["prediction_rows"] += len(batch.features)
                    observed = actor.predict(batch.features)
                    if followup_plan is not None and observed.actor_version != actor.version:
                        raise ValueError("fresh observation has stale actor version")
                    if followup_plan is not None and tick >= followup_plan.start_tick:
                        if _require_followup_publication(actor, report) != publication_pin:
                            raise ValueError(
                                "original verified publication changed during prediction"
                            )
                    paid["prediction_returns"] += 1
                    p = observed.prediction
                    if (
                        p.shape != (len(batch.features), 6)
                        or not np.isfinite(p).all()
                        or not np.allclose(p.sum(axis=1), 1, atol=1e-6)
                    ):
                        raise ValueError("invalid causal feedback prediction")
                    pending[index] = (observed.actor_version, np.argmax(p, axis=1), tick)
                    if followup_plan is not None:
                        _record_window_source(runtime, batch, index, tick, observed.actor_version)
                    paid["prediction_records_peak"] = max(
                        paid["prediction_records_peak"], len(pending)
                    )
                if followup_plan is None:
                    _record_tick(runtime, batches, tick, label_delay)
                label_index = (
                    tick - label_delay
                    if followup_plan is None
                    else followup_plan.label_index_at(
                        tick, len(batches), label_delay, followup_label_delay
                    )
                )
                if label_index is not None and 0 <= label_index < len(batches):
                    version, predicted, observed_tick = pending[label_index]
                    if followup_plan is not None:
                        if version != actor.version:
                            raise ValueError("fresh label belongs to stale actor version")
                        _record_window_label(
                            runtime, batches[label_index], label_index, tick, version
                        )
                    pending.pop(label_index)
                    arrived += 1
                    targets = batches[label_index].targets[:, 0]
                    paid["feedback_rows"] += len(targets)
                    feedback = ArrivedTrainingFeedback(
                        "train",
                        f"batch-{label_index}",
                        version,
                        observed_tick,
                        tick,
                        int(np.sum(predicted != targets)),
                        len(targets),
                    )
                if candidate_predict is not None:
                    from scripts.candidate_training_feedback import read_candidate_feedback

                    feedback = read_candidate_feedback(
                        runtime,
                        batches,
                        arrived=arrived,
                        completed=budget.updates_completed,
                        tick=tick,
                        label_delay=label_delay,
                        actor_version=actor.version,
                        predict=candidate_predict,
                        paid=report["candidate_feedback"],
                        require_input=require_input,
                        **(
                            dict(
                                consumed=budget.updates_completed
                                + len(runtime.acknowledged_experiences),
                                acknowledge_correct=True,
                            )
                            if acknowledge_correct
                            else {}
                        ),
                    )
                if task_reader is not None:
                    feedback = task_reader.read(
                        runtime,
                        batches,
                        arrived=arrived,
                        completed=budget.updates_completed,
                        tick=tick,
                        label_delay=label_delay,
                        actor=actor,
                        require_input=require_input,
                    )
                    task_reader.require_paid(report.get("task_feedback"))
                if shift_reader is not None:
                    try:
                        feedback = shift_reader.read(
                            runtime,
                            batches,
                            arrived=arrived,
                            completed=budget.updates_completed,
                            tick=tick,
                            label_delay=label_delay,
                            actor_version=actor.version,
                            require_input=require_input,
                        )
                    finally:
                        report["arrived_shift_feedback"] = shift_reader.report
                progress = shared.adaptation_snapshot().progress
                observation = ReferenceObservation(
                    tick,
                    horizon - tick,
                    arrived
                    - budget.updates_completed
                    - (len(runtime.acknowledged_experiences) if acknowledge_correct else 0),
                    len(batches) - progress.wake_admissions,
                    actor.version,
                    feedback,
                )
            finally:
                paid["observation_seconds"] += monotonic() - observation_started
            event: dict[str, Any] = dict(observation=asdict(observation), action_id=None)
            report["controller_events"].append(event)
            paid["selections_attempted"] += 1
            started = monotonic()
            try:
                decision = (
                    _retired_contextual_wait(
                        actor,
                        report,
                        publication_pin,
                        observation,
                        pending=pending,
                        arrived=arrived,
                        consumed=budget.updates_completed,
                        total=len(batches),
                    )
                    if contextual_policy is not None and publication_pin is not None
                    else (
                        choose_reference(policy, observation)
                        if contextual_policy is None
                        else choose_contextual(contextual_policy, observation)
                    )
                )
                paid["selections_returned"] += 1
                event.update(action_id=decision.action_id, reason=decision.reason)
            finally:
                paid["selection_seconds"] += monotonic() - started
            require_input()
            if shift_reader is not None:
                shift_reader.require_paid(report.get("arrived_shift_feedback"))
                shift_reader.require_current(runtime)
            if task_reader is not None:
                task_reader.require_paid(report.get("task_feedback"))
            if followup_plan is not None:
                followup_plan.__post_init__()
            if followup_plan is not None and tick >= followup_plan.start_tick:
                if _require_followup_publication(actor, report) != publication_pin:
                    raise ValueError("original verified publication changed before admission")
            started = monotonic()
            try:
                with live.opportunity(tick) if live is not None else nullcontext():
                    poll = shared.train_action(decision.action_id)
                event["poll"] = asdict(poll)
            finally:
                paid["admission_seconds"] += monotonic() - started
            if (
                on_drain is not None
                and not drain_attempted
                and budget.updates_completed
                + (len(runtime.acknowledged_experiences) if acknowledge_correct else 0)
                == drain_count
                and tick >= drain_count - 1 + label_delay
                and not pending
            ):
                progress = shared.adaptation_snapshot().progress
                admission = gate.snapshot()
                if (
                    len(runtime.applied_updates)
                    + (len(runtime.acknowledged_experiences) if acknowledge_correct else 0)
                    != drain_count
                    or progress.native_completions != budget.updates_completed
                    or admission.training_active
                    or admission.active_serving_requests
                ):
                    raise ValueError(
                        "drained publication requires completed receipts and idle gate"
                    )
                drain_attempted = True
                feedback = _publish_at_drain(
                    actor, runtime, on_drain, tick, feedback, report, budget.updates_completed
                )
                if followup_plan is not None:
                    report["drain_publication"]["scope"] = "first_predeclared_training_window"
                    for row in report["feedback_invalidations"]:
                        row["reason"] = "verified publication after first-window labels consumed"
                if (followup_plan is not None or contextual_policy is not None) and report[
                    "drain_publication"
                ]["accepted"]:
                    publication_pin = _require_followup_publication(actor, report)
            if tick == len(batches) // 2:
                midway()
            if policy_change and tick in dict(POLICY_CHANGE_CHECKPOINTS):
                state = runtime.candidate_snapshot()
                report["policy_checkpoints"].append(
                    dict(
                        opportunity=tick,
                        policy_version=dict(POLICY_CHANGE_CHECKPOINTS)[tick],
                        applied_receipts=len(runtime.applied_updates),
                        checkpoint=native.checkpoint(state.wake_updates, state.state),
                    )
                )
        report["stream"] = dict(
            evaluation_scope="reference_observation_only",
            opportunities=horizon,
            completed_updates=budget.updates_completed,
            registered_batches=len(batches),
            arrived_labels=arrived,
            pending_unapplied=len(batches)
            - budget.updates_completed
            - (len(runtime.acknowledged_experiences) if acknowledge_correct else 0),
            score_forward_calls=0,
        )
        if followup_plan is not None:
            report["followup_training"]["status"] = (
                "complete" if budget.updates_completed == len(batches) else "incomplete"
            )
        if policy_change:
            report["policy_capture_complete"] = True
        return runtime, horizon
    finally:
        if shift_reader is not None:
            report["arrived_shift_feedback"] = shift_reader.report
        if task_reader is not None:
            report["task_feedback"] = task_reader.report
        if acknowledge_correct:
            acknowledgements = runtime.acknowledged_experiences
            report["acknowledged_experiences"] = [asdict(r) for r in acknowledgements]
            report["receipt_progress"] = dict(
                actual_updates=budget.updates_completed,
                acknowledged=len(acknowledgements),
                consumed=budget.updates_completed + len(acknowledgements),
                total=len(batches),
                complete=budget.updates_completed + len(acknowledgements) == len(batches),
            )
        if policy_change:
            report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]
            report["policy_pending_predictions"] = sorted(pending)
        if followup_plan is not None:
            report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]
            report["followup_training"]["pending_prediction_indices"] = sorted(pending)
            if report["followup_training"]["status"] != "complete":
                report["followup_training"]["status"] = "incomplete"
            report.setdefault(
                "stream",
                dict(
                    evaluation_scope="reference_observation_only",
                    opportunities=horizon,
                    completed_updates=budget.updates_completed,
                    registered_batches=len(batches),
                    arrived_labels=arrived,
                    pending_unapplied=len(batches) - budget.updates_completed,
                    score_forward_calls=0,
                ),
            )
        paid["prediction_records_remaining"] = len(pending)
        report["action_snapshot"] = asdict(shared.adaptation_snapshot())
        if live is not None or policy_change:
            report["sharing_snapshot"] = asdict(gate.snapshot())
        if live is not None:
            report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]


def _retired_contextual_wait(actor, report, pin, observation, *, pending, arrived, consumed, total):
    """The owner retires a drained chooser; its old binding never gains authority."""
    observation.__post_init__()
    if (
        _require_followup_publication(actor, report) != pin
        or observation.actor_version != pin[1]
        or observation.feedback is not None
        or observation.ready_updates != 0
        or pending
        or type(arrived) is not int
        or type(consumed) is not int
        or arrived != total
        or consumed != total
    ):
        raise ValueError("retired contextual chooser requires the original verified drained state")
    return ReferenceDecision("wait", "owner_wait_after_verified_contextual_retirement")


def _require_followup_publication(actor, report):
    row = report.get("drain_publication", {})
    if (
        row.get("accepted") is not True
        or row.get("completed") is not True
        or row.get("publication_unknown") is not False
        or type(row.get("generation_before")) is not int
    ):
        raise ValueError("followup requires the verified first publication generation")
    current = actor.serving_snapshot()
    if (
        row.get("accepted") is not True
        or row.get("completed") is not True
        or row.get("publication_unknown") is not False
        or type(current.generation) is not int
        or type(row.get("generation_before")) is not int
        or current.generation != row["generation_before"] + 1
        or current.actor_version != row.get("actor_version_after")
    ):
        raise ValueError("followup requires the verified first publication generation")
    return current.generation, current.actor_version


def _record_window_source(runtime, batch, index, tick, version):
    if batch.role != "train":
        raise ValueError("followup source requires ordinary TRAIN role")
    runtime.record_experience(
        Experience(
            f"batch-{index}",
            "attribution" if batch.episode is None else batch.episode.episode_id,
            tick,
            version,
            batch.features,
            "train",
            ExperiencePermissions(training=True),
        )
    )


def _record_window_label(runtime, batch, index, tick, version):
    if batch.role != "train":
        raise ValueError("followup label requires ordinary TRAIN role")
    runtime.record_label(
        LabelArrival(
            f"label-{index}",
            f"batch-{index}",
            "attribution" if batch.episode is None else batch.episode.episode_id,
            tick,
            version,
            batch.targets,
            role="train",
        )
    )


def _candidate_tasks(spec, runtime, report, require_input):
    state = runtime.candidate_snapshot().state
    report["candidate_checkpoint"] = native.checkpoint(
        runtime.candidate_snapshot().wake_updates, state
    )
    reader = native.readout(spec, state)
    paid = report["candidate_readout_work"] = dict(work=asdict(reader.work))
    report["candidate_episodes"] = []

    def predict(request):
        require_input()
        if not native.same_state(runtime.candidate_snapshot().state, state):
            raise ValueError("candidate changed during isolated task readout")
        try:
            return reader(request)
        finally:
            paid["work"] = asdict(reader.work)

    for concept in context.CONCEPTS:
        episode = run_workflow_episode(
            WorkflowState(f"reference-candidate-{concept}", concept),
            predict,
            capabilities=WorkflowCapabilities(ACTION_IDS),
            scenario="ordinary",
            max_steps=6,
        )
        report["candidate_episodes"].append(
            dict(concept=concept, role="outer_selection", episode=episode)
        )
        reader.assert_unchanged()
        if not episode["complete"]:
            raise ValueError("uncertain paid candidate task work")
    if not native.same_state(runtime.candidate_snapshot().state, state):
        raise ValueError("candidate changed after task readout")
    before = {
        e["concept"]: int(e["episode"]["success"])
        for e in report["episodes"]
        if e["phase"] == "before"
    }
    after = {e["concept"]: int(e["episode"]["success"]) for e in report["candidate_episodes"]}
    count = lambda values, concepts: sum(values[c] for c in concepts)
    report["reference_outcomes"] = dict(
        new_before=count(before, context.NEW),
        new_after=count(after, context.NEW),
        old_before=count(before, context.OLD),
        old_after=count(after, context.OLD),
        new_gain=count(after, context.NEW) - count(before, context.NEW),
        old_drop=count(before, context.OLD) - count(after, context.OLD),
        scope="frozen candidate task readout; higher gain/lower old drop preferred; not deployed/final evaluation",
    )


def _publish_candidate(runtime, controller, spec, now, report, require_input, completed_updates):
    """Reuse the original inner guards, source IDs and publication authority."""
    new, old = guards(spec, now)
    report["guards"] = [
        dict(
            task_id=g.task_id,
            role=g.role,
            keys=g.sample_keys,
            features=g.features.tolist(),
            targets=g.targets.tolist(),
        )
        for g in (new, old)
    ]
    report["guard_qualification"] = (
        "repeated synthetic templates, disjoint IDs; not heldout generalization"
    )
    try:
        require_input()
        ticket = controller.prepare(
            runtime,
            new,
            old,
            now=now,
            training_ids=frozenset(("prior-fit", f"fit-{i}") for i in range(576)),
            metadata={"origin": context.SOURCE_SHA, "native_history": 96 + completed_updates},
        )
    except PromotionRejected as rejection:
        report["guard_report"] = asdict(rejection.report)
        return None
    report["guard_report"] = asdict(ticket.report)
    require_input()
    receipt = controller.commit(runtime, ticket)
    report["promotion_receipt"] = asdict(receipt)
    return receipt


def _publish_at_drain(actor, runtime, on_drain, tick, feedback, report, completed_updates):
    """Consume a single owner callback; never rebind old-version feedback."""
    if "drain_publication" in report:
        raise ValueError("drained publication was already attempted")
    before = actor.serving_snapshot()
    row = report["drain_publication"] = dict(
        tick=tick,
        wake_updates=completed_updates,
        pending_predictions=0,
        actor_version_before=before.actor_version,
        generation_before=before.generation,
        completed=False,
        accepted=False,
        publication_unknown=False,
    )
    report["feedback_invalidations"] = []
    try:
        if (
            type(before.generation) is not int
            or before.generation < 0
            or type(before.actor_version) is not str
        ):
            raise ValueError("drained publication requires native actor generation/version types")
        if feedback is not None and feedback.actor_version != before.actor_version:
            raise ValueError("stale feedback cannot be rebound at drained publication")
        row["publication_unknown"] = True  # A lost callback return may hide a committed swap.
        receipt = on_drain(runtime, tick)
        after = actor.serving_snapshot()
        if type(after.generation) is not int or type(after.actor_version) is not str:
            raise ValueError("drained publication returned non-native generation/version types")
        changed = (after.generation, after.actor_version) != (
            before.generation,
            before.actor_version,
        )
        if receipt is None:
            if changed:
                raise ValueError("unobserved publication has no original commit receipt")
        elif (
            type(receipt) is not PromotionReceipt
            or type(receipt.generation) is not int
            or type(receipt.previous_version) is not str
            or type(receipt.actor_version) is not str
            or not changed
            or receipt.previous_version != before.actor_version
            or receipt.actor_version != after.actor_version
            or receipt.generation != after.generation
            or after.generation != before.generation + 1
            or after.actor_version == before.actor_version
        ):
            raise ValueError("drained publication receipt/version/generation differs")
        else:
            report["feedback_invalidations"].append(
                dict(
                    at=tick,
                    previous_actor_version=before.actor_version,
                    actor_version=after.actor_version,
                    feedback=None if feedback is None else asdict(feedback),
                    reason="verified publication after all original labels consumed",
                )
            )
            feedback = None
            row["accepted"] = True
        row.update(completed=True, publication_unknown=False)
        return feedback
    except BaseException as error:
        row["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        row["actor_version_after"] = actor.version


def _require_followup_input(recipe, recipe_pin, batches, data_pin):
    require_followup_recipe(recipe)
    if recipe != recipe_pin or not _same_value(batches, data_pin):
        raise ValueError("original followup TRAIN recipe or data changed")


def run_case(
    input_path: Path,
    *,
    reference_mode: str | None = None,
    training_context: str = "new_contexts",
    contextual_policy: ContextualChoice | FittedController | None = None,
    base_snapshot: Any = None,
    counterfactual: bool = False,
    stream_recipe: TicketStreamRecipe | None = None,
    deferred_evaluation: bool = False,
    serving_plan: LiveServingPlan | None = None,
    publish_on_drain: bool = False,
    live_training_recipe: LiveTrainingRecipe | None = None,
    followup_plan: FollowupTrainingPlan | None = None,
    policy_change: bool = False,
    followup_training_recipe: LiveTrainingRecipe | None = None,
) -> dict[str, Any]:
    if followup_training_recipe is not None:
        require_followup_recipe(followup_training_recipe)
        if followup_plan is None or live_training_recipe is not None:
            raise ValueError("followup recipe requires a separate explicit post-publication window")
    followup_recipe_pin = deepcopy(followup_training_recipe)
    if type(policy_change) is not bool or (
        policy_change
        and (
            (reference_mode is None and contextual_policy is None)
            or serving_plan is not None
            or publish_on_drain
            or live_training_recipe is not None
            or followup_plan is not None
            or stream_recipe is not None
            or base_snapshot is not None
            or counterfactual
            or training_context != "new_contexts"
        )
    ):
        raise ValueError("policy-change trial requires an isolated scheduled development recipe")
    if followup_plan is not None:
        if (
            type(followup_plan) is not FollowupTrainingPlan
            or serving_plan is None
            or publish_on_drain is not True
            or reference_mode != "periodic"
            or contextual_policy is not None
            or base_snapshot is not None
            or counterfactual
            or deferred_evaluation
            or stream_recipe is not None
            or live_training_recipe is not None
            or training_context != "new_contexts"
        ):
            raise ValueError(
                "followup requires explicit periodic live publication without other recipes"
            )
        followup_plan.__post_init__()
        if followup_plan.first_batches != context.BATCHES_PER_PHASE:
            raise ValueError("followup must preserve original phase partition")
        if (
            type(serving_plan) is not LiveServingPlan
            or followup_plan.start_tick
            + context.BATCHES_PER_PHASE
            + (
                1
                if followup_training_recipe is None
                else followup_training_recipe.recipe.label_delay
            )
            > serving_plan.opportunities
        ):
            raise ValueError("followup labels exceed finite live plan")
    if live_training_recipe is not None:
        if type(live_training_recipe) is not LiveTrainingRecipe or serving_plan is None:
            raise ValueError("live TRAIN recipe requires an explicit original live plan")
        live_training_recipe.__post_init__()
        if stream_recipe is not None or deferred_evaluation:
            raise ValueError("live TRAIN recipes cannot replace comparative deferred evaluation")
    if type(publish_on_drain) is not bool or (publish_on_drain and serving_plan is None):
        raise ValueError("drained publication requires an explicit original live plan")
    if serving_plan is not None:
        if type(serving_plan) is not LiveServingPlan:
            raise ValueError("live serving requires an exact finite plan")
        serving_plan.__post_init__()
        if (
            reference_mode != ("drift" if followup_plan is None else "periodic")
            or training_context != "new_contexts"
            or contextual_policy is not None
            or base_snapshot is not None
            or counterfactual
            or deferred_evaluation
            or stream_recipe is not None
        ):
            raise ValueError("live serving requires the original drift development schedule only")
    serving_pin = deepcopy(serving_plan)
    followup_pin = deepcopy(followup_plan)
    live_recipe_pin = deepcopy(live_training_recipe)
    if live_training_recipe is not None and serving_plan is not None:
        base_recipe = live_training_recipe.recipe
        if serving_plan.opportunities < len(base_recipe.contexts) + base_recipe.label_delay:
            raise ValueError("live plan must contain all declared TRAIN label arrivals")
    scheduled = reference_mode is not None or contextual_policy is not None
    if (
        type(deferred_evaluation) is not bool
        or (stream_recipe is not None and not deferred_evaluation)
        or (deferred_evaluation and stream_recipe is None and not policy_change)
        or (
            deferred_evaluation
            and (not scheduled or counterfactual or training_context != "new_contexts")
        )
    ):
        raise ValueError("custom TRAIN recipe requires explicit scheduled deferred evaluation")
    if stream_recipe is not None:
        if type(stream_recipe) is not TicketStreamRecipe:
            raise ValueError("custom stream requires exact immutable recipe")
        stream_recipe.__post_init__()
    recipe_pin = deepcopy(stream_recipe)
    if (
        type(counterfactual) is not bool
        or (reference_mode is not None and contextual_policy is not None)
        or (contextual_policy is None and (base_snapshot is not None or counterfactual))
        or (counterfactual and base_snapshot is None)
    ):
        raise ValueError("contextual trials require an explicit policy and detached base snapshot")
    if contextual_policy is not None:
        if type(contextual_policy) not in (ContextualChoice, FittedController):
            raise ValueError("contextual execution requires exact frozen policy")
        contextual_policy.__post_init__()
    contextual_pin = deepcopy(contextual_policy)
    if (
        type(training_context) is not str
        or training_context not in ("new_contexts", "old_return")
        or (not scheduled and training_context != "new_contexts")
    ):
        raise ValueError("training context requires an explicit supported reference mode")
    if reference_mode is not None:
        ReferencePolicy(reference_mode)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="bp_ticket_serving_v1",
        complete=False,
        episodes=[],
        publications=[],
        final_examples=0,
        reserved_probe_updates=0,
        replay_examples=0,
        external_actions=0,
    )
    if reference_mode is not None:
        report["definition"] = "reference_bp_ticket_serving_v1"
        report["reference_mode"] = reference_mode
        report["training_context"] = training_context
    if contextual_policy is not None:
        report.update(
            definition="contextual_bp_ticket_serving_v1",
            training_context=training_context,
            counterfactual=counterfactual,
            contextual_policy=asdict(contextual_policy),
        )
    if followup_training_recipe is not None:
        report.update(
            followup_training_recipe=asdict(followup_training_recipe),
            followup_training_corruption=dict(
                definition="explicit_experimental_label_noise_not_verified_teacher",
                role="train",
                window="post_publication",
                rows_per_batch=[7] if followup_training_recipe.retained_invoice_noise else [],
                from_target="inspect" if followup_training_recipe.retained_invoice_noise else None,
                to_target="defer" if followup_training_recipe.retained_invoice_noise else None,
                planned_corrupted_exposures=8
                if followup_training_recipe.retained_invoice_noise
                else 0,
            ),
        )
    if policy_change:
        report.update(
            definition="policy_change_native_reference_v1",
            policy_change=True,
            deferred_evaluation=deferred_evaluation,
            policy_phases=[
                dict(name=p, policy_version=v, concepts=c) for p, v, c in POLICY_CHANGE_PHASES
            ],
            qualification="fixed current-policy development; opportunity axis is not completed updates; no controller/generalization/CPC claim",
        )
    if stream_recipe is not None:
        report.update(
            definition="deferred_controller_stream_v1",
            deferred_evaluation=True,
            stream_recipe=asdict(stream_recipe),
            known_train_label_corruptions=len(stream_recipe.contexts)
            if stream_recipe.noisy_invoice
            else 0,
        )
    if live_training_recipe is not None:
        corrupted_row = (
            7
            if live_training_recipe.retained_invoice_noise
            else 1
            if live_training_recipe.recipe.noisy_invoice
            else None
        )
        report.update(
            live_training_recipe=asdict(live_training_recipe),
            training_corruption=dict(
                definition="explicit_experimental_label_noise_not_verified_teacher",
                role="train",
                rows_per_batch=[] if corrupted_row is None else [corrupted_row],
                from_target=None if corrupted_row is None else "inspect",
                to_target=None if corrupted_row is None else "defer",
                planned_corrupted_exposures=0
                if corrupted_row is None
                else len(live_training_recipe.recipe.contexts),
            ),
        )
    work = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    budget = stream = actor = source = predictor = torch = torch_rng = None
    live = sharing_gate = None
    drain_receipt = None
    published_state = None
    rng = np.random.get_state()
    assert type(rng) is tuple
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            raw, prior = context._source(input_path)
            report.update(
                source_sha256=context.SOURCE_SHA,
                prior_training_work=prior["training_work"],
                prior_costs=prior["costs"],
            )
            torch = native.require_torch()
            torch_rng = torch.get_rng_state().clone()
            initial = context.bp_snapshot_from_record(prior["checkpoints"][-1]["state"])
            if initial.completed_updates != 96 or initial.training_examples != 576:
                raise ValueError("qualified native clock differs")
            digest = snapshot_digest(initial) if contextual_policy is not None else None
            if base_snapshot is not None:
                if snapshot_digest(base_snapshot) != digest:
                    raise ValueError("counterfactual base differs from original complete source")
                initial = base_snapshot
            if type(contextual_policy) is FittedController and (
                contextual_policy.learner_version != "qualified-96"
                or contextual_policy.state_digest != digest
            ):
                raise ValueError("fitted controller requires its original learner version/payload")
            spec = DecisionFeatureSpec(FEATURES, CATALOG)
            _, _, all_batches = context._teacher_batches(spec)
            batches = (
                all_batches[: context.BATCHES_PER_PHASE]
                if training_context == "new_contexts"
                else all_batches[context.BATCHES_PER_PHASE :]
            )
            if stream_recipe is not None:
                batches = build_stream(
                    stream_recipe, all_batches[0], all_batches[context.BATCHES_PER_PHASE]
                )
            if live_training_recipe is not None:
                batches = build_live_stream(
                    live_training_recipe, all_batches[0], all_batches[context.BATCHES_PER_PHASE]
                )
            if followup_plan is not None:
                batches = (
                    all_batches
                    if followup_training_recipe is None
                    else build_followup_stream(followup_training_recipe, all_batches)
                )
                report["followup_plan"] = asdict(followup_plan)
            followup_data_pin = deepcopy(batches) if followup_training_recipe is not None else None
            if policy_change:
                batches = build_policy_change_stream(spec)
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=len(batches),
                    max_wall_seconds=120,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)

            def require_input():
                budget.before_final()
                if followup_training_recipe is not None:
                    _require_followup_input(
                        followup_training_recipe, followup_recipe_pin, batches, followup_data_pin
                    )
                if contextual_policy != contextual_pin:
                    raise ValueError("original frozen contextual policy changed")
                if stream_recipe != recipe_pin:
                    raise ValueError("original frozen TRAIN recipe changed")
                if live_training_recipe != live_recipe_pin:
                    raise ValueError("original frozen live TRAIN recipe changed")
                if followup_plan != followup_pin:
                    raise ValueError("original frozen followup plan changed")
                if input_path.read_bytes() != raw:
                    raise ValueError("original qualified source changed")

            def require_serving_input():
                # Worker source checks do not race the main budget's mutable clock.
                if followup_training_recipe is not None:
                    _require_followup_input(
                        followup_training_recipe, followup_recipe_pin, batches, followup_data_pin
                    )
                if (
                    serving_plan != serving_pin
                    or live_training_recipe != live_recipe_pin
                    or followup_plan != followup_pin
                    or input_path.read_bytes() != raw
                ):
                    raise ValueError("original live serving plan or source changed")

            def before(features):
                work["attempted_updates"] += 1
                work["attempted_rows"] += len(features)
                require_input()
                if live is not None:
                    live.native_started()

            def after():
                work["returned_updates"] += 1

            source = native.counted_learner("bp", 7, 6, before, after)
            source.restore_state(initial)
            actor = PromotableActor(
                source,
                version="qualified-96",
                configuration=ServingConfiguration(3, 2),
                feature_digest=lambda a: __import__("hashlib").sha256(a.tobytes()).hexdigest(),
                metadata={"origin": context.SOURCE_SHA, "native_history": 96},
            )
            predictor = TorchServingPredictor(actor, spec)
            if scheduled:
                sharing_gate = ServingPriorityGate(
                    SharingLimits(1, len(batches), 1), resource_available=lambda: True
                )
            if serving_plan is not None:
                live = TicketServingTimeline(
                    serving_plan, sharing_gate, predictor, require_serving_input
                )
            if not scheduled:
                stream = AttributionStream(
                    source,
                    batches,
                    (),
                    checkpoint_updates=(0, len(batches)),
                    label_delay=1,
                    budget=budget,
                    training_only=True,
                    actor=actor,
                )
            controller = ServingPromotionController(
                actor,
                policy=policy(),
                evaluator=categorical_guard(spec, monotonic),
                build_learner=build_native,
                state_digest=snapshot_digest,
                max_pending=1,
            )

            def publication(phase):
                s = actor.serving_snapshot()
                report["publications"].append(
                    dict(
                        phase=phase,
                        generation=s.generation,
                        actor_version=s.actor_version,
                        model=native.snapshot_record(s.model_state),
                        configuration=asdict(s.configuration),
                        metadata=s.metadata,
                        cache_entries=len(s.cache),
                        last_cache_tick=s.last_cache_tick,
                    )
                )

            def episodes(phase, concepts, scenarios=("ordinary",)):
                if deferred_evaluation or policy_change:
                    return
                for concept in concepts:
                    for scenario in scenarios:
                        require_input()
                        versions = []

                        def predict(request):
                            require_input()
                            with (
                                sharing_gate.serving()
                                if sharing_gate is not None
                                else nullcontext()
                            ):
                                result = predictor.serve(request)
                            versions.append(
                                dict(
                                    generation=result.generation,
                                    actor_version=result.actor_version,
                                    native_updates=result.native_updates,
                                )
                            )
                            return result.result

                        episode = run_workflow_episode(
                            WorkflowState(f"serve-{phase}-{concept}", concept),
                            predict,
                            capabilities=WorkflowCapabilities(ACTION_IDS),
                            scenario=scenario,
                            max_steps=6,
                        )
                        report["episodes"].append(
                            dict(
                                phase=phase,
                                concept=concept,
                                role="outer_selection",
                                publications=versions,
                                episode=episode,
                            )
                        )
                        if not episode["complete"]:
                            raise ValueError("uncertain paid task work")

            publication("before")
            episodes("before", context.CONCEPTS)

            def publish_drained(runtime, tick):
                nonlocal drain_receipt, published_state
                require_input()
                drain_receipt = _publish_candidate(
                    runtime, controller, spec, tick, report, require_input, budget.updates_completed
                )
                publication("stream-promoted" if drain_receipt is not None else "stream-rejected")
                if drain_receipt is not None and followup_plan is not None:
                    published_state = actor.snapshot_state().state
                return drain_receipt

            if not scheduled:
                assert stream is not None
                # Observe stable old serving midway through the existing delayed loop.
                if len(batches) // 2:
                    stream.run_until(len(batches) // 2)
                episodes("during", context.OLD)
                stream.run_until(len(batches))
                candidate = stream.candidate_snapshot()
                report["applied_updates"] = [asdict(u) for u in stream.applied_updates]
                report["stream"] = stream.finish()
                runtime, guard_now = stream._runtime, stream.next_tick
            else:
                runtime, guard_now = _reference_schedule(
                    source,
                    actor,
                    batches,
                    budget,
                    reference_mode,
                    report,
                    require_input,
                    lambda: episodes("during", context.OLD),
                    contextual_policy,
                    (
                        stream_recipe.label_delay
                        if stream_recipe is not None
                        else live_training_recipe.recipe.label_delay
                        if live_training_recipe is not None
                        else 1
                    ),
                    sharing_gate,
                    live,
                    publish_drained if publish_on_drain else None,
                    **({} if followup_plan is None else {"followup_plan": followup_plan}),
                    **(
                        {}
                        if followup_training_recipe is None
                        else {"followup_label_delay": followup_training_recipe.recipe.label_delay}
                    ),
                    **({"policy_change": True} if policy_change else {}),
                )
                candidate = runtime.candidate_snapshot().state
                report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]
                if deferred_evaluation or policy_change:
                    report["candidate_checkpoint"] = native.checkpoint(
                        runtime.candidate_snapshot().wake_updates, candidate
                    )
                    if policy_change and not deferred_evaluation:
                        native.policy_development_readout(spec, report, require_input)
                else:
                    _candidate_tasks(spec, runtime, report, require_input)
            if (
                candidate.completed_updates != 96 + budget.updates_completed
                or not native.same_state(source.snapshot_state(), initial)
                or not native.same_state(
                    actor.snapshot_state().state,
                    initial
                    if drain_receipt is None
                    else (candidate if followup_plan is None else published_state),
                )
            ):
                raise ValueError("shadow/actor/source native clocks or ownership changed")
            if counterfactual:
                # A disposable training trial owns no deployed publication. All
                # candidate/task/monitor work is paid even when its arm loses.
                report.update(
                    complete=True,
                    source_unchanged=True,
                    counterfactual_actor_unchanged=native.same_state(
                        actor.snapshot_state().state, initial
                    ),
                    counterfactual_promotions=0,
                )
                return report
            if deferred_evaluation:
                require_input()
                report.update(
                    complete=True,
                    source_unchanged=True,
                    deferred_actor_unchanged=native.same_state(
                        actor.snapshot_state().state, initial
                    ),
                    deferred_promotions=0,
                )
                return report
            if policy_change:
                require_input()
                report.update(
                    complete=True,
                    source_unchanged=True,
                    policy_actor_unchanged=native.same_state(actor.snapshot_state().state, initial),
                    policy_promotions=0,
                )
                return report
            if publish_on_drain:
                receipt = drain_receipt
                if "drain_publication" not in report:
                    report["publication_skipped"] = "original TRAIN stream did not fully drain"
            else:
                receipt = _publish_candidate(
                    runtime,
                    controller,
                    spec,
                    guard_now,
                    report,
                    require_input,
                    budget.updates_completed,
                )
                publication("promoted" if receipt is not None else "rejected")
            if receipt is not None:
                episodes("promoted", context.CONCEPTS)
                episodes("promoted-faults", context.NEW, ("stale_once", "revoke_commit"))
                require_input()
                report["rollback_generation"] = controller.rollback(receipt)
                publication("rollback")
                episodes("rollback", context.CONCEPTS)
                if not native.same_state(actor.snapshot_state().state, initial):
                    raise ValueError("rollback did not restore complete native state")
            report["complete"] = True
            require_input()
            report["source_unchanged"] = native.same_state(source.snapshot_state(), initial)
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        finally:
            if live is not None:
                live.close()
                report["live_serving"] = live.snapshot()

            report["training_work"] = dict(
                **work,
                unreturned_work_unknown=work["attempted_updates"] != work["returned_updates"],
            )
            if predictor is not None:
                report["serving_work"] = dict(predictor.work)
            if budget is not None:
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    updates_completed=budget.updates_completed,
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                )
            current = np.random.get_state()
            assert type(current) is tuple
            report["numpy_rng_unchanged"] = bool(
                rng[0] == current[0]
                and np.array_equal(rng[1], current[1])
                and rng[2:] == current[2:]
            )
            report["torch_rng_unchanged"] = (
                None if torch is None else bool(torch.equal(torch_rng, torch.get_rng_state()))
            )
            if not report["numpy_rng_unchanged"] or report["torch_rng_unchanged"] is False:
                report.update(complete=False, error="caller RNG changed")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
    return report


def _violation_count(episodes):
    count = 0
    for episode in episodes:
        if not episode["complete"]:
            raise ValueError("incomplete task evidence has no utility")
        for step in episode["steps"]:
            if step["transition"]["status"] == "applied":
                permitted = (
                    step["result"]["selected_id"] in step["capabilities_at_effect"]["allowed"]
                )
                current = step["effect_state"]["revision"] == step["observation_state"]["revision"]
                count += int(not permitted or not current)
    return count


def _require_legacy_task_evaluation(report):
    if (
        "policy_change" in report
        or "policy_checkpoints" in report
        or report.get("definition") == "policy_change_native_reference_v1"
    ):
        raise ValueError("legacy task evaluation cannot score policy-change development trials")


def _case_measures(result, seconds):
    _require_legacy_task_evaluation(result)
    if not result["complete"] or result["training_work"]["unreturned_work_unknown"]:
        raise ValueError("failed or unknown native work has no outer utility")
    outcomes = result["reference_outcomes"]
    episodes = [e["episode"] for e in result["episodes"] + result["candidate_episodes"]]
    return ControllerMeasures(
        outcomes["new_before"],
        outcomes["new_after"],
        len(context.NEW),
        outcomes["old_before"],
        outcomes["old_after"],
        len(context.OLD),
        seconds,
        result["costs"]["sampled_process_rss"]["peak_bytes"],
        _violation_count(episodes),
        True,
    )


def _sealed_cases():
    # Construct only after the chronology's freeze check. Same public grammar,
    # fresh episode identities; this is not evidence of statistical novelty.
    return tuple((f"sealed-controller-episode-{i}", c) for i, c in enumerate(context.CONCEPTS))


def _sealed_readout(training_report, report, clock):
    _require_legacy_task_evaluation(training_report)
    started = clock.now()
    then = monotonic()
    outcomes: dict[str, dict[str, int]] = {}
    readers = []
    episodes = []
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        clock.advance_to(started + 1)
        report["sealed_source_reads"].append(dict(at=clock.now(), role="sealed_evaluation"))
        cases = _sealed_cases()
        states = (
            training_report["publications"][0]["model"],
            training_report["candidate_checkpoint"]["state"],
        )
        spec = DecisionFeatureSpec(FEATURES, CATALOG)
        for name, state in zip(("before", "after"), states, strict=True):
            # The existing decoder intentionally accepts canonical JSON IDs.
            # These fresh in-memory observation dictionaries still have int keys.
            canonical = json.loads(json.dumps(state, allow_nan=False))
            reader = native.readout(spec, context.bp_snapshot_from_record(canonical))
            readers.append(reader)
            outcomes[name] = {}
            for key, concept in cases:
                episode = run_workflow_episode(
                    WorkflowState(f"{key}-{name}", concept),
                    reader,
                    capabilities=WorkflowCapabilities(ACTION_IDS),
                    scenario="ordinary",
                    max_steps=6,
                )
                episodes.append(
                    dict(
                        case_id=key, role="final_test", model=name, concept=concept, episode=episode
                    )
                )
                if not episode["complete"]:
                    raise ValueError("incomplete sealed readout")
                reader.assert_unchanged()
                outcomes[name][concept] = int(episode["success"])
        elapsed = monotonic() - then
        peak = sampler.snapshot().peak_bytes
    report["sealed_episodes"] = episodes
    report["sealed_readout_work"] = [asdict(r.work) for r in readers]
    clock.advance_to(started + 2)
    count = lambda phase, concepts: sum(outcomes[phase][c] for c in concepts)
    return ControllerMeasures(
        count("before", context.NEW),
        count("after", context.NEW),
        len(context.NEW),
        count("before", context.OLD),
        count("after", context.OLD),
        len(context.OLD),
        elapsed,
        peak,
        _violation_count([e["episode"] for e in episodes]),
        True,
    )


def run_outer_study(input_path: Path) -> dict[str, Any]:
    started = monotonic()
    clock = LogicalClock()
    protocol = ControllerProtocol(
        "native_controller_outer_chronology_v1",
        ControllerObjective(),
        ReferencePolicy("periodic"),
        (
            ControllerStream(
                "controller-development-new", "controller_train", "new-context-source"
            ),
            ControllerStream(
                "controller-validation-return", "controller_validation", "old-return-source"
            ),
            ControllerStream(
                "controller-sealed-tasks", "sealed_evaluation", "sealed-task-episodes"
            ),
        ),
    )
    chronology = ControllerChronology(protocol, clock)
    report: dict[str, Any] = dict(
        definition="controller_outer_study_v1",
        complete=False,
        protocol=asdict(protocol),
        declared_at=clock.now(),
        phases={},
        sealed_source_reads=[],
        sealed_training_updates=0,
        sealed_promotions=0,
    )
    torch = native.require_torch()
    torch_rng = torch.get_rng_state().clone()
    rng = np.random.get_state()
    assert type(rng) is tuple
    try:
        for stream, kind in zip(protocol.streams[:2], ("new_contexts", "old_return"), strict=True):

            def run(policy):
                begin = clock.now()
                then = monotonic()
                result = run_case(input_path, reference_mode=policy.mode, training_context=kind)
                seconds = monotonic() - then
                report["phases"][stream.role] = result
                clock.advance_to(begin + 1)
                measures = _case_measures(result, seconds)
                clock.advance_to(begin + 2)
                return ControllerTrial(
                    ControllerTrialMetadata(stream, policy, begin, begin + 1, begin + 2),
                    measures,
                )

            chronology.run_development(stream.stream_id, run)
            if stream.role == "controller_train":
                report["training_rows_before_validation"] = [
                    asdict(r) for r in chronology.training_rows()
                ]
        report["validation_rows_before_freeze"] = [asdict(r) for r in chronology.validation_rows()]
        frozen = chronology.freeze()
        report["freeze"] = asdict(frozen)

        def final(policy):
            begin = clock.now()
            measures = _sealed_readout(report["phases"]["controller_train"], report, clock)
            return ControllerTrial(
                ControllerTrialMetadata(protocol.streams[2], policy, begin, begin + 1, begin + 2),
                measures,
            )

        chronology.evaluate_final(frozen, final)
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:512]
    finally:
        report["chronology"] = asdict(protocol) | {
            k: ([asdict(r) for r in v] if k == "rows" else v)
            for k, v in chronology.snapshot().items()
            if k != "protocol"
        }
        current = np.random.get_state()
        assert type(current) is tuple
        report["numpy_rng_unchanged"] = (
            rng[0] == current[0] and np.array_equal(rng[1], current[1]) and rng[2:] == current[2:]
        )
        report["torch_rng_unchanged"] = bool(torch.equal(torch_rng, torch.get_rng_state()))
        if not report["numpy_rng_unchanged"] or not report["torch_rng_unchanged"]:
            report.update(complete=False, error="outer study caller RNG changed")
        report["elapsed_seconds"] = monotonic() - started
    return report


def _reported_work_fields(record, fields):
    """Bound known scalar telemetry; null means unavailable, never zero work."""
    record = record if type(record) is dict else {}
    result = {}
    for field in fields:
        value = record.get(field)
        continuous = field in {
            "elapsed_seconds",
            "started_at",
            "last_clock",
            "interval_seconds",
            "max_wall_seconds",
        }
        valid = type(value) is int and 0 <= value <= 2**63 - 1
        if continuous and type(value) is float:
            valid = isfinite(value) and 0 <= value <= 2**63 - 1
        result[field] = value if valid else None
    return result


def _task_report_failure(report, report_bytes, byte_limit):
    """Keep scalar paid prefixes when detailed task-trial output cannot be saved.

    Why this: an oversized report must fail without erasing all evidence of work.
    These producer-reported totals are not validated trial receipts or a fit.
    Fixed fields/four trials bound storage without traversing outcome/state trees.
    """
    work_fields = ("attempted_updates", "attempted_rows", "returned_updates")
    memory_fields = ("start_bytes", "peak_bytes", "sample_count", "interval_seconds")

    def costs(record):
        record = record if type(record) is dict else {}
        error = record.get("error")
        error = error[:512] if type(error) is str else None
        # JSON escapes can expand one Unicode character to twelve ASCII bytes.
        while error and len(json.dumps(error)) > 1024:
            error = error[: len(error) // 2]
        return dict(
            training_work=_reported_work_fields(record.get("training_work"), work_fields),
            memory=_reported_work_fields(record.get("memory"), memory_fields),
            **_reported_work_fields(record, ("elapsed_seconds",)),
            producer_error=error,
        )

    trials = report.get("trials")
    reported = costs(report)
    reported.update(
        trial_count=len(trials) if type(trials) is list else None,
        omitted_trial_count=max(0, len(trials) - 4) if type(trials) is list else None,
        trials=[],
    )
    for trial in trials[:4] if type(trials) is list else ():
        result = trial.get("result") if type(trial) is dict else None
        result = result if type(result) is dict else {}
        budget = result.get("budget")
        budget = budget if type(budget) is dict else {}
        readouts = result.get("task_readout_work")
        readouts = readouts if type(readouts) is dict else {}
        row = costs(result)
        row["budget"] = dict(
            **_reported_work_fields(budget, ("updates", "started_at", "last_clock")),
            limits=_reported_work_fields(
                budget.get("limits"),
                (
                    "max_training_updates",
                    "max_wall_seconds",
                    "max_replay_examples",
                    "max_hidden_width",
                    "max_process_rss_bytes",
                ),
            ),
        )
        row["task_readout_work"] = {
            phase: _reported_work_fields(
                readouts.get(phase), ("attempts", "returns", "readouts", "typed_attempts")
            )
            for phase in ("before", "after")
        }
        reported["trials"].append(row)
    return dict(
        definition="context_controller_task_trials_output_failure_v1",
        complete=False,
        error="ValueError: serving report exceeds declared byte limit",
        report_bytes=report_bytes,
        report_byte_limit=byte_limit,
        detailed_report_available=False,
        task_outcomes_available=False,
        fitted_policy_available=False,
        trial_receipts_available=False,
        work_summary_status="producer-reported only; null is unavailable; not validated evidence",
        reported=reported,
    )


def main(argv=None):
    with ExitStack() as sources:
        return _main(argv, sources)


def _main(argv, sources):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument(
        "--example-context-source",
        action="store_true",
        help="bundled qualified CPU source; requires --context-head-serving",
    )
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference", choices=REFERENCE_MODES)
    mode.add_argument("--outer-study", action="store_true")
    mode.add_argument("--learned-controller", action="store_true")
    mode.add_argument("--controller-comparison", action="store_true")
    mode.add_argument("--policy-change-controller-comparison", action="store_true")
    mode.add_argument("--serving-during-updates", action="store_true")
    mode.add_argument("--publish-during-serving", action="store_true")
    mode.add_argument("--continue-training-after-publication", action="store_true")
    mode.add_argument("--policy-change-reference", choices=REFERENCE_MODES)
    mode.add_argument("--context-head-serving", action="store_true")
    mode.add_argument("--context-head-serving-live", action="store_true")
    mode.add_argument("--context-controller-comparison", action="store_true")
    mode.add_argument("--context-controller-learning", action="store_true")
    mode.add_argument("--context-controller-task-trials", action="store_true")
    mode.add_argument("--context-recipe-validation", action="store_true")
    mode.add_argument("--context-stream-protocol", action="store_true")
    mode.add_argument("--context-stream-comparison", action="store_true")
    parser.add_argument("--context-live-reference", choices=REFERENCE_MODES)
    parser.add_argument("--context-candidate-feedback", action="store_true")
    parser.add_argument("--context-acknowledge-correct", action="store_true")
    parser.add_argument("--context-task-feedback", action="store_true")
    parser.add_argument("--context-arrived-shift", action="store_true")
    parser.add_argument(
        "--context-shift-diagnostics",
        action="store_true",
        help="informational arrived K6 metrics; requires --context-head-serving",
    )
    parser.add_argument("--context-withheld-orders", action="store_true")
    parser.add_argument(
        "--followup-training-recipe", choices=tuple(n for n, _ in FOLLOWUP_TRAINING_RECIPES)
    )
    parser.add_argument(
        "--context-stream-recipe", choices=tuple(r.recipe_id for r in CONTEXT_TRAINING_RECIPES)
    )
    parser.add_argument("--periodic-report", type=Path)
    parser.add_argument("--fitted-report", type=Path)
    parser.add_argument(
        "--live-training-recipe", choices=tuple(r.recipe.recipe_id for r in LIVE_TRAINING_RECIPES)
    )
    args = parser.parse_args(argv)
    if args.followup_training_recipe is not None and not args.continue_training_after_publication:
        parser.error("--followup-training-recipe requires --continue-training-after-publication")
    if args.example_context_source and not args.context_head_serving:
        parser.error("--example-context-source requires --context-head-serving")
    if args.context_shift_diagnostics and not args.context_head_serving:
        parser.error("--context-shift-diagnostics requires --context-head-serving")
    if args.context_arrived_shift and (
        not args.context_head_serving_live
        or args.context_live_reference != "drift"
        or args.context_candidate_feedback
        or args.context_task_feedback
        or args.context_acknowledge_correct
    ):
        parser.error("--context-arrived-shift requires separate explicit context live drift")
    if (args.periodic_report is not None) != (
        args.context_controller_comparison
        or args.context_stream_comparison
        and not args.context_withheld_orders
    ):
        parser.error("--periodic-report is required only with an explicit context comparison")
    if args.context_withheld_orders and not (
        args.context_stream_protocol or args.context_stream_comparison
    ):
        parser.error("--context-withheld-orders requires context stream protocol or comparison")
    if args.context_live_reference is not None and not args.context_head_serving_live:
        parser.error("--context-live-reference requires --context-head-serving-live")
    if args.context_stream_recipe is not None and not args.context_head_serving_live:
        parser.error("--context-stream-recipe requires --context-head-serving-live")
    if args.context_candidate_feedback and (
        not args.context_head_serving_live or args.context_live_reference != "drift"
    ):
        parser.error("--context-candidate-feedback requires explicit context live drift")
    if args.context_acknowledge_correct and not args.context_candidate_feedback:
        parser.error("--context-acknowledge-correct requires --context-candidate-feedback")
    if args.context_task_feedback and (
        not args.context_head_serving_live
        or args.context_live_reference != "drift"
        or args.context_candidate_feedback
        or args.context_acknowledge_correct
    ):
        parser.error("--context-task-feedback requires separate explicit context live drift")
    if (
        args.controller_comparison
        or args.policy_change_controller_comparison
        or args.context_stream_comparison
    ) != (args.fitted_report is not None):
        parser.error("--fitted-report is required only with an explicit controller comparison")
    if args.live_training_recipe is not None and not (
        args.serving_during_updates or args.publish_during_serving
    ):
        parser.error("--live-training-recipe requires an explicit live serving mode")
    live_kwargs: dict[str, Any] = (
        {}
        if args.live_training_recipe is None
        else dict(
            live_training_recipe=next(
                r for r in LIVE_TRAINING_RECIPES if r.recipe.recipe_id == args.live_training_recipe
            )
        )
    )
    followup_kwargs: dict[str, Any] = (
        {}
        if args.followup_training_recipe is None
        else {
            "followup_training_recipe": next(
                r for n, r in FOLLOWUP_TRAINING_RECIPES if n == args.followup_training_recipe
            )
        }
    )
    if args.example_context_source:
        from src.infra.context_head_example import example_context_source

        args.source = sources.enter_context(example_context_source())
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    if args.context_stream_protocol:
        from scripts.context_stream_comparison import protocol_snapshot

        report = protocol_snapshot(**(dict(withheld=True) if args.context_withheld_orders else {}))
    elif args.context_stream_comparison:
        from scripts.context_stream_comparison import run_comparison as run_stream_comparison

        report = run_stream_comparison(
            args.source,
            args.periodic_report,
            args.fitted_report,
            **(dict(withheld=True) if args.context_withheld_orders else {}),
        )
    elif args.context_recipe_validation:
        from scripts.context_recipe_validation import run_validation

        report = run_validation(args.source)
    elif args.context_controller_task_trials:
        from scripts.context_controller_learning import run_study as run_context_task_trials

        report = run_context_task_trials(args.source, task_outcomes=True)
    elif args.context_controller_learning:
        from scripts.context_controller_learning import run_study as run_context_learning

        report = run_context_learning(args.source)
    elif args.context_controller_comparison:
        from scripts.context_reference_comparison import run_comparison as run_context_comparison

        report = run_context_comparison(args.source, args.periodic_report)
    elif args.context_head_serving_live:
        from scripts.context_head_live import run_case as run_context_live

        context_kwargs: dict[str, Any] = {}
        if args.context_live_reference is not None:
            context_kwargs["reference_mode"] = args.context_live_reference
        if args.context_candidate_feedback:
            context_kwargs["candidate_feedback"] = True
        if args.context_acknowledge_correct:
            context_kwargs["acknowledge_correct"] = True
        if args.context_task_feedback:
            context_kwargs["task_feedback"] = True
        if args.context_arrived_shift:
            context_kwargs["arrived_shift"] = True
        if args.context_stream_recipe is not None:
            context_kwargs["stream_recipe"] = next(
                r for r in CONTEXT_TRAINING_RECIPES if r.recipe_id == args.context_stream_recipe
            )
        report = run_context_live(args.source, **context_kwargs)
    elif args.context_head_serving:
        from scripts.context_head_serving import run_case as run_context_serving

        report = run_context_serving(
            args.source, **({"shift_diagnostics": True} if args.context_shift_diagnostics else {})
        )
    elif args.controller_comparison:
        from scripts.ticket_controller_comparison import run_comparison

        report = run_comparison(args.source, args.fitted_report)
    elif args.policy_change_controller_comparison:
        from scripts.ticket_controller_comparison import run_comparison

        report = run_comparison(args.source, args.fitted_report, policy_change=True)
    elif args.continue_training_after_publication:
        report = run_case(
            args.source,
            reference_mode="periodic",
            serving_plan=LiveServingPlan(opportunities=48),
            publish_on_drain=True,
            followup_plan=FollowupTrainingPlan(),
            **followup_kwargs,
        )
    elif args.policy_change_reference is not None:
        report = run_case(
            args.source, reference_mode=args.policy_change_reference, policy_change=True
        )
    elif args.publish_during_serving:
        report = run_case(
            args.source,
            reference_mode="drift",
            serving_plan=LiveServingPlan(),
            publish_on_drain=True,
            **live_kwargs,
        )
    elif args.serving_during_updates:
        report = run_case(
            args.source, reference_mode="drift", serving_plan=LiveServingPlan(), **live_kwargs
        )
    elif args.learned_controller:
        from scripts.ticket_controller_learning import run_study

        report = run_study(args.source)
    elif args.outer_study:
        report = run_outer_study(args.source)
    else:
        report = (
            run_case(args.source)
            if args.reference is None
            else run_case(args.source, reference_mode=args.reference)
        )
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    limit = (
        8
        if args.controller_comparison or args.policy_change_controller_comparison
        else (
            4
            if args.learned_controller
            else (
                4
                if args.continue_training_after_publication
                else (
                    2
                    if args.outer_study
                    or args.serving_during_updates
                    or args.publish_during_serving
                    else 1
                )
            )
        )
    ) * 1024**2
    if args.policy_change_reference is not None:
        limit = 2 * 1024**2
    if args.context_controller_comparison:
        limit = 4 * 1024**2
    if args.context_controller_learning:
        limit = 2 * 1024**2
    if args.context_controller_task_trials:
        limit = 3 * 1024**2
    if args.context_task_feedback:
        limit = 3 * 1024**2
    if args.context_recipe_validation:
        limit = 3 * 1024**2
    if args.context_stream_comparison:
        limit = (24 if args.context_withheld_orders else 12) * 1024**2
    if len(data) > limit:
        if args.context_controller_task_trials:
            failure = _task_report_failure(report, len(data), limit)
            summary = (json.dumps(failure, indent=2, allow_nan=False) + "\n").encode()
            if len(summary) <= min(limit, 32 * 1024):
                args.output.write_bytes(summary)
        raise ValueError("serving report exceeds declared byte limit")
    args.output.write_bytes(data)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
