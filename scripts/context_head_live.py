"""Public-context serving on the existing finite reference/worker schedule.

One availability integration case, not a new learning comparison. Reuses known
TRAIN packets/guards and source77 weights; no probe/final access or new runtime.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from time import monotonic
from typing import Any

import numpy as np

from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes, head_states
from scripts.context_head_serving import prepare_data, policy, _episodes
from scripts.context_head_serving import TASKS, PAIRED_TASKS
from scripts.context_training_streams import (
    ContextTrainingRecipe,
    build_context_stream,
    CONTEXT_PAIRED_REHEARSAL_RECIPES,
)
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from scripts.ticket_serving_timeline import LiveServingPlan, TicketServingTimeline
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import (
    ContextServingPredictor,
    build_context,
    categorical_guard,
    snapshot_digest,
    snapshot_record_bytes,
)
from src.app.coupled_attribution import _same_value
from src.app.resource_sharing import ServingPriorityGate
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.contextual_controller import ContextualChoice, FittedController
from src.core.context_perceptron import predict_context_perceptron
from src.core.resource_sharing import SharingLimits
from src.core.reference_controller import ReferencePolicy
from src.core.serving_ports import ServingConfiguration
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler

LIVE_TASKS = (
    ("v1", "invoice"),
    ("v1", "legacy"),
    ("v1", "password"),
    ("v2", "invoice"),
    ("v2", "parcel"),
    ("v1", "legacy"),
    ("v1", "invoice"),
    ("v1", "invoice"),
    ("v1", "password"),
    ("v2", "invoice"),
    ("v2", "parcel"),
    ("v1", "legacy"),
    ("v2", "legacy"),
    ("v1", "legacy"),
)


def live_plan():
    return LiveServingPlan(14, (2,), context_tasks=LIVE_TASKS)


def recipe_plan(*, head_task_transfer=False):
    if type(head_task_transfer) is not bool:
        raise ValueError("paired serving plan requires an exact boolean opt-in")
    tasks: tuple[tuple[str, str], ...] = (
        *LIVE_TASKS,
        ("v1", "invoice"),
        ("v2", "invoice"),
        ("v1", "password"),
        *TASKS,
        ("v1", "legacy"),
    )
    if head_task_transfer:
        # Why: preserve tick/fault opportunities, changing only declared tasks.
        new = dict((v, c) for v, c in PAIRED_TASKS[-2:])
        tasks = tuple((v, new[v] if c == "legacy" else c) for v, c in tasks)
        # Cover every ordinary task after the six periodic updates can drain;
        # v2's new task otherwise appears only in stale/revoked scenarios.
        late = {13: ("v2", "password"), 16: ("v2", "parcel"), 23: ("v2", "password")}
        tasks = tuple(late.get(i, task) for i, task in enumerate(tasks))
    return LiveServingPlan(24, (2,), context_tasks=tasks)


def overlap_readback(records, calls):
    """Observed interval intersections; not evidence of parallel CPU instructions."""
    reads = [p for row in records for p in row["requests"]]
    complete = [p for p in reads if p.get("complete") is True and "seconds" in p]
    intervals = [c for c in calls if "model_finished_at" in c]
    return dict(
        definition="observed_request_and_original_update_interval_v1",
        request_attempts=len(reads),
        complete_requests=len(complete),
        started_during_model=sum(
            any(
                c["model_started_at"] <= p["started_at"] <= c["model_finished_at"]
                for c in intervals
            )
            for p in complete
        ),
        finished_during_model=sum(
            any(
                c["model_started_at"] <= p["started_at"] + p["seconds"] <= c["model_finished_at"]
                for c in intervals
            )
            for p in complete
        ),
        intersected_model=sum(
            any(
                p["started_at"] <= c["model_finished_at"]
                and p["started_at"] + p["seconds"] >= c["model_started_at"]
                for c in intervals
            )
            for p in complete
        ),
        scope="wall-clock intervals only; no forced rendezvous, CPU parallelism or speedup claim",
    )


def run_case(
    input_path,
    *,
    reference_mode="periodic",
    contextual_policy=None,
    candidate_feedback=False,
    stream_recipe=None,
    acknowledge_correct=False,
    task_feedback=False,
    arrived_shift=False,
    catalog_trial=None,
    catalog_policy=None,
    catalog_base=None,
    resume_catalog_wake=False,
):
    from scripts.context_catalog_control import ContextCatalogControl

    catalog_control = None
    if (
        type(resume_catalog_wake) is not bool
        or resume_catalog_wake
        and (catalog_trial is None and catalog_policy is None)
    ):
        raise ValueError("wake continuation requires explicit contextual catalog mode")
    if catalog_trial is not None or catalog_policy is not None or catalog_base is not None:
        if (
            reference_mode != "periodic"
            or contextual_policy is not None
            or candidate_feedback
            or stream_recipe is not None
            or acknowledge_correct
            or task_feedback
            or arrived_shift
        ):
            raise ValueError("context catalog requires its separate private original live mode")
        catalog_control = ContextCatalogControl(
            catalog_trial,
            catalog_policy,
            base_snapshot=catalog_base,
            resume_wake=resume_catalog_wake,
        )
    if (
        type(arrived_shift) is not bool
        or arrived_shift
        and (
            reference_mode != "drift"
            or contextual_policy is not None
            or candidate_feedback
            or acknowledge_correct
            or task_feedback
        )
    ):
        raise ValueError("arrived shift feedback requires separate explicit fixed drift mode")
    if (
        type(task_feedback) is not bool
        or task_feedback
        and (
            reference_mode != "drift"
            or contextual_policy is not None
            or candidate_feedback
            or acknowledge_correct
        )
    ):
        raise ValueError("task feedback requires separate explicit fixed drift mode")
    if type(acknowledge_correct) is not bool or acknowledge_correct and not candidate_feedback:
        raise ValueError("TRAIN acknowledgement requires explicit candidate feedback")
    ReferencePolicy(reference_mode)  # Validate before source, sampler or model access.
    if stream_recipe is not None:
        if type(stream_recipe) is not ContextTrainingRecipe:
            raise ValueError("context live stream requires an exact contextual recipe")
        stream_recipe.__post_init__()
    paired = stream_recipe in CONTEXT_PAIRED_REHEARSAL_RECIPES
    if (
        stream_recipe is not None
        and any(c.endswith("-v2") for c in stream_recipe.contexts)
        and not paired
    ):
        raise ValueError("paired live TRAIN requires an exact named paired recipe")
    if paired and (
        reference_mode not in ("periodic", "no_learning")
        or contextual_policy is not None
        or candidate_feedback
        or acknowledge_correct
        or task_feedback
        or arrived_shift
        or catalog_control is not None
    ):
        raise ValueError(
            "paired TRAIN currently supports periodic/no_learning without extra feedback"
        )
    recipe_pin = deepcopy(stream_recipe)
    planned_updates = 2 if stream_recipe is None else len(stream_recipe.contexts)
    if catalog_control is not None:
        planned_updates = catalog_control._batches
    if (
        type(candidate_feedback) is not bool
        or candidate_feedback
        and (reference_mode != "drift" or contextual_policy is not None)
    ):
        raise ValueError("candidate feedback requires explicit fixed drift mode")
    if contextual_policy is not None:
        if type(contextual_policy) not in (ContextualChoice, FittedController):
            raise ValueError("contextual serving requires an exact frozen policy")
        contextual_policy.__post_init__()
        if reference_mode != "periodic":
            raise ValueError("select either a fixed reference or contextual policy")
    contextual_pin = deepcopy(contextual_policy)
    from scripts.run_bp_ticket_serving import _reference_schedule

    started = monotonic()
    report: dict[str, Any] = dict(
        definition="live_context_serving_v1",
        complete=False,
        source_sha256=HEAD_SOURCE_SHA,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        episodes=[],
        native_calls=[],
        training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
        qualification="one availability integration case; known TRAIN templates, no frozen control rerun, heldout/CPC/efficiency claim",
    )
    live = budget = source = actor = predictor = runtime = None
    receipt = None
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("qualified contextual source replaced")
            initial = head_states(json.loads(raw))[0]
            if catalog_control is not None:
                catalog_control.require_source(
                    initial, catalog_base, digest=snapshot_digest, same_state=_same_value
                )
            if type(contextual_policy) is FittedController and (
                contextual_policy.learner_version != "qualified-12"
                or contextual_policy.state_digest != snapshot_digest(initial)
            ):
                raise ValueError("fitted controller requires original contextual version/state")
            spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
            data = prepare_data(spec, **(dict(head_task_transfer=True) if paired else {}))
            if catalog_control is not None:
                data = (catalog_control.prepare_packets(data[0]), data[1])
                report.update(
                    definition="live_context_catalog_v1",
                    role="controller_train"
                    if catalog_control.trial
                    else "frozen_application_development",
                    catalog_control=catalog_control.report,
                    task_outcomes={"before": [], "after": []},
                    task_readout_work={},
                )
            if stream_recipe is not None:
                batches, noise = build_context_stream(
                    stream_recipe, *data[0][:2], **(dict(second_head=data[0][2:]) if paired else {})
                )
                data = (batches, data[1])
                report.update(
                    definition="live_context_recipe_v1",
                    context_recipe=asdict(stream_recipe),
                    synthetic_label_noise=noise,
                )
                if paired:
                    report.update(
                        train_task_family="paired_head_tasks_v1", declared_tasks=PAIRED_TASKS
                    )
            pin = deepcopy(data)
            rng = deepcopy(np.random.get_state())
            plan = (
                live_plan()
                if stream_recipe is None
                else recipe_plan(**(dict(head_task_transfer=True) if paired else {}))
            )
            plan_pin = deepcopy(plan)
            report.update(
                live_plan=asdict(plan),
                promotion_policy=asdict(policy()),
                input_bytes=sum(b.features.nbytes + b.targets.nbytes for b in (*data[0], *data[1])),
                coefficient_bytes=768,
                source_record_bytes=snapshot_record_bytes(initial),
            )
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=planned_updates,
                    max_wall_seconds=45 if stream_recipe is None else 60,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            budget_pin = (id(budget.budget), budget.budget, budget.started_at, id(budget.clock))
            if catalog_control is not None:
                catalog_control.prepare(budget, sampler)

            def require_source():
                # Shared worker read-only checks never mutate the main budget clock.
                if monotonic() - started > (120 if stream_recipe is None else 180):
                    raise ValueError("original live case deadline exhausted")
                if stream_recipe is not None:
                    stream_recipe.__post_init__()
                if stream_recipe != recipe_pin:
                    raise ValueError("original contextual stream recipe changed")
                if contextual_policy is not None:
                    contextual_policy.__post_init__()
                if contextual_policy != contextual_pin:
                    raise ValueError("original contextual policy changed")
                if catalog_control is not None:
                    catalog_control.require_pin()
                if (
                    _source_bytes(input_path) != raw
                    or not _same_value(data, pin)
                    or plan != plan_pin
                ):
                    raise ValueError("original source/TRAIN/guard/plan changed")
                if not _same_value(np.random.get_state(), rng):
                    raise ValueError("caller NumPy RNG changed")

            def require():
                require_source()
                budget.before_final()
                if source is None:
                    raise ValueError("original source owner unavailable")
                if source.snapshot_state() != initial or budget_pin != (
                    id(budget.budget),
                    budget.budget,
                    budget.started_at,
                    id(budget.clock),
                ):
                    raise ValueError("original source owner or budget changed")

            class CountedLearner(ContextPerceptronLearner):
                def fork(self):
                    child = CountedLearner(self.policy)
                    child.restore_state(self.snapshot_state())
                    return child

                def train_batch(self, features, targets):
                    require_source()
                    paid: dict[str, Any] = dict(
                        rows=len(features), entered_at=monotonic(), complete=False
                    )
                    report["native_calls"].append(paid)
                    report["training_work"]["attempted_updates"] += 1
                    report["training_work"]["attempted_rows"] += len(features)
                    try:
                        if live is None:
                            raise ValueError("original live timeline unavailable")
                        live.native_started()
                        paid["model_started_at"] = monotonic()
                        try:
                            result = super().train_batch(features, targets)
                        finally:
                            paid["model_finished_at"] = monotonic()
                        paid.update(complete=True, state=asdict(self.snapshot_state()))
                        report["training_work"]["returned_updates"] += 1
                        return result
                    finally:
                        paid["returned_at"] = monotonic()

            source = CountedLearner(initial.policy)
            source.restore_state(initial)
            actor = PromotableActor(
                source,
                version="qualified-12",
                configuration=ServingConfiguration(3, 2),
                feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
                metadata={"source": HEAD_SOURCE_SHA, "history_updates": 12},
            )
            predictor = ContextServingPredictor(actor, spec)
            gate = ServingPriorityGate(
                SharingLimits(
                    1, planned_updates if catalog_control is None else catalog_control.admissions, 1
                ),
                resource_available=lambda: True,
            )
            live = TicketServingTimeline(plan, gate, predictor, require_source)
            controller = ServingPromotionController(
                actor,
                policy=policy(),
                evaluator=categorical_guard(spec, monotonic),
                build_learner=build_context,
                state_digest=snapshot_digest,
                max_pending=1,
            )

            def publish(owner, now):
                nonlocal runtime, receipt
                runtime = owner  # Retain receipts even if later publication or serving fails.
                require()
                new, old = data[1]
                report["candidate_state"] = asdict(owner.candidate_snapshot().state)
                try:
                    ticket = controller.prepare(owner, new, old, now=now, training_ids=frozenset())
                except PromotionRejected as error:
                    report["guard_report"] = asdict(error.report)
                    return None
                report["guard_report"] = asdict(ticket.report)
                receipt = controller.commit(owner, ticket)
                report["promotion_receipt"] = asdict(receipt)
                return receipt

            require()
            if catalog_control is not None:
                from scripts.context_controller_learning import _task_readout

                report["before"] = _task_readout(initial, "before", require, report)

            def predict_candidate(state, features):
                learner = build_context(state)
                learner.restore_state(state)
                return learner.predict(features)

            task_kwargs = {}
            if task_feedback:
                from scripts.context_controller_learning import _task_readout

                task_kwargs["task_readout"] = _task_readout

            runtime, _ = _reference_schedule(
                source,
                actor,
                data[0],
                budget,
                reference_mode,
                report,
                require,
                lambda: None,
                contextual_policy=contextual_policy,
                sharing_gate=gate,
                live=live,
                on_drain=publish if catalog_control is None else None,
                **(dict(catalog_control=catalog_control) if catalog_control is not None else {}),
                **task_kwargs,
                **(dict(arrived_shift_predict=predict_context_perceptron) if arrived_shift else {}),
                **(dict(candidate_predict=predict_candidate) if candidate_feedback else {}),
                **(dict(acknowledge_correct=True) if acknowledge_correct else {}),
                **(
                    dict(
                        label_delay=stream_recipe.label_delay,
                        candidate_row_limit=144 if candidate_feedback else 84,
                    )
                    if stream_recipe is not None
                    else {}
                ),
            )
            live.close()
            require()
            if catalog_control is not None:
                candidate = runtime.candidate_snapshot().state
                catalog_control.require_history(candidate, budget, runtime.applied_updates)
                native = catalog_control.report["native_history"]
                expected_calls = native["wake_updates"] + native["replay_updates"]
                if (
                    report["training_work"]["attempted_updates"] != expected_calls
                    or report["training_work"]["returned_updates"] != expected_calls
                    or report["training_work"]["attempted_rows"]
                    != native["wake_rows"] + native["replay_rows"]
                ):
                    raise ValueError(
                        "context catalog has unknown or incomplete physical model work"
                    )
                report["candidate_state"] = asdict(candidate)
                report["after"] = _task_readout(candidate, "after", require, report)
                catalog_control.require_candidate()
            if receipt is not None:
                report["rollback_generation"] = controller.rollback(receipt)
                after = actor.serving_snapshot()
                report["rollback_equal_source"] = after.model_state == initial
                report["rollback"] = asdict(after)
                _episodes(
                    predictor,
                    "rollback",
                    require,
                    report["episodes"],
                    **(dict(tasks=PAIRED_TASKS) if paired else {}),
                )
            if (
                contextual_policy is None
                and catalog_control is None
                and reference_mode == "periodic"
                and budget.updates_completed != planned_updates
            ):
                raise ValueError("finite live case did not complete planned updates")
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            if catalog_control is not None:
                report["catalog_control"] = catalog_control.capture()
                if runtime is None and catalog_control.replay is not None:
                    runtime = catalog_control.replay._runtime
                if catalog_control.replay_budget is not None:
                    report["replay_examples"] = (
                        catalog_control.replay_budget.replay_examples_completed
                    )
            if live is not None:
                live.close()
                report["live_serving"] = live.snapshot()
                report["overlap"] = overlap_readback(
                    live.snapshot()["records"], report["native_calls"]
                )
            if budget is not None:
                report["training_complete"] = budget.updates_completed == planned_updates
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    updates=budget.updates_completed,
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                )
            if predictor is not None:
                report["typed_work"] = dict(predictor.work)
            if source is not None:
                report["source_unchanged"] = source.snapshot_state() == initial
            if runtime is not None:
                report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]
            if actor is not None:
                final_actor = actor.serving_snapshot()
                report["final_actor"] = asdict(final_actor)
                if catalog_control is not None:
                    report["actor_unchanged"] = (
                        final_actor.model_state == initial
                        and final_actor.generation == 0
                        and final_actor.actor_version == "qualified-12"
                    )
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(sampler.snapshot()))
    return report
