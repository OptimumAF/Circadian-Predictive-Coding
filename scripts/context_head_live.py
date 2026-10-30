"""Public-context serving on the existing finite reference/worker schedule.

One availability integration case, not a new learning comparison. Reuses known
TRAIN packets/guards and source77 weights; no probe/final access or new runtime.
"""

from copy import deepcopy
from contextlib import nullcontext
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
    CONTEXT_REHEARSAL_RECIPES,
    WITHHELD_CONTEXT_RECIPES,
    CONTEXT_INCIDENT_RECIPES,
)
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from scripts.ticket_serving_timeline import (
    LiveServingPlan,
    TicketServingTimeline,
    FollowupTrainingPlan,
)
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
from src.core.shift_calibration import ShiftCalibration
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


def _commit_after_guard(controller, owner, ticket, before_commit):
    # Why: a readonly-guard violation must refuse the swap, not merely fail
    # the report after a mutated budget/history has already reached serving.
    if before_commit is not None:
        before_commit()
    return controller.commit(owner, ticket)


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


def run_owned_case(
    state,
    spec,
    source_report,
    source_budget,
    *,
    control,
    check_boundary,
    memory,
    linear_fit=False,
    adaptive=False,
    incident_stream=None,
    calibrated=False,
    catalog_action=None,
    catalog_context=None,
    probe_rehearsal=None,
):
    """Serve ordinary TRAIN from the current workflow; never load probe weights."""
    from scripts.context_head_rehearsal import prepare_owned_source

    report: dict[str, Any] = dict(
        definition="fresh_context_head_paired_serving_v1",
        complete=False,
        source_provenance="current_workflow_train_checkpoint12",
        training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
    )
    delegated = returned = False
    try:
        if type(linear_fit) is not bool:
            raise ValueError("owned LS serving requires an exact boolean opt-in")
        if type(adaptive) is not bool or adaptive and not linear_fit:
            raise ValueError("adaptive serving requires the explicit owned LS source")
        if (
            type(calibrated) is not bool
            or calibrated
            and incident_stream is None
            or incident_stream is not None
            and (
                type(incident_stream) is not str
                or incident_stream not in ("stationary", "shift")
                or not adaptive
                or not linear_fit
                or control != "joint"
            )
        ):
            raise ValueError("incident serving requires explicit adaptive owned LS joint mode")
        study = control == "catalog_study"
        validation = control == "reference_validation"
        retained = control in ("retained_joint", "retained_joint_publish", "retained_joint_probe")
        probe_retained = control == "retained_joint_probe"
        if probe_rehearsal is not None and (
            type(probe_rehearsal) is not str
            or probe_rehearsal not in ("joint", "current_only")
            or not probe_retained
            or linear_fit is not True
        ):
            raise ValueError("probe rehearsal requires explicit owned LS retained probe")
        publish_retained = control == "retained_joint_publish"
        if retained and (not linear_fit or adaptive or calibrated or incident_stream is not None):
            raise ValueError("retained joint serving requires separate ordinary periodic LS mode")
        if validation and (
            not linear_fit
            or adaptive
            or incident_stream is not None
            or catalog_action is not None
            or catalog_context is not None
        ):
            raise ValueError("reference validation requires its separate ordinary LS mode")
        catalog = control == "catalog"
        same_context = control == "same_context_joint"
        if same_context and not linear_fit:
            raise ValueError("same-context joint serving requires ordinary owned LS mode")
        if study and (
            not linear_fit
            or adaptive
            or calibrated
            or incident_stream is not None
            or catalog_action is not None
            or catalog_context is not None
        ):
            raise ValueError("owned catalog study requires its separate ordinary LS mode")
        if (
            catalog
            and (
                not linear_fit
                or adaptive
                or incident_stream is not None
                or calibrated
                or type(catalog_action) is not str
                or catalog_action not in ("wait", "wake1", "replay6-wake1")
                or type(catalog_context) is not str
                or catalog_context not in ("clean", "mistakes")
            )
            or not catalog
            and (catalog_action is not None or catalog_context is not None)
        ):
            raise ValueError("owned catalog requires explicit ordinary LS action and context")
        if type(control) is not str or control not in (
            "current",
            "joint",
            "catalog",
            "catalog_study",
            "same_context_joint",
            "reference_validation",
            "retained_joint",
            "retained_joint_publish",
            "retained_joint_probe",
        ):
            raise ValueError("owned serving requires the exact current or joint control")
        if (
            type(source_budget) is not ToyBudgetSession
            or source_budget.process_rss_sampler is not memory
            or memory is None
        ):
            raise ValueError("owned serving requires the original source sampler and budget")
        state, guard = prepare_owned_source(
            state,
            spec,
            source_report,
            source_budget,
            check_boundary=check_boundary,
            training_key="serving_source_training",
            **(dict(linear_fit=True) if linear_fit else {}),
            **(dict(catalog_study=True) if study else {}),
            **(dict(reference_validation=True) if validation else {}),
            **(dict(retained_probe=True) if probe_retained else {}),
        )
        delegated = True
        catalog_options = {}
        if catalog:
            from src.app.counterfactual_controller import CatalogCounterfactualRequest
            from src.adapters.context_linear_serving import snapshot_digest as linear_digest
            from scripts.context_catalog_control import context_catalog

            catalog_options = dict(
                catalog_trial=CatalogCounterfactualRequest(
                    "context-ls-12",
                    linear_digest(state),
                    catalog_context,
                    catalog_action,
                    0,
                    catalog=context_catalog(),
                ),
                catalog_base=state,
                resume_catalog_wake=True,
            )
        if validation:
            from scripts.context_stream_comparison import run_comparison

            report = run_comparison(
                None,
                None,
                None,
                withheld=True,
                _owned_source=(state, guard),
                _memory=memory,
                _linear_fit=True,
            )
        elif study:
            from scripts.context_catalog_learning import run_catalog_study

            report = run_catalog_study(
                None, _owned_source=(state, guard), _memory=memory, _linear_fit=True
            )
        else:
            report = run_case(
                None,
                stream_recipe=(
                    None
                    if catalog
                    else (
                        CONTEXT_REHEARSAL_RECIPES[1]
                        if same_context
                        else WITHHELD_CONTEXT_RECIPES[0]
                        if retained
                        else CONTEXT_INCIDENT_RECIPES[0 if incident_stream == "stationary" else 1]
                        if incident_stream is not None
                        else CONTEXT_PAIRED_REHEARSAL_RECIPES[0 if control == "current" else 1]
                    )
                ),
                publish_after_learning=not adaptive and not catalog and not retained,
                **(
                    dict(
                        _retained_joint=True,
                        _publish_retained=publish_retained,
                        _probe_retained=probe_retained,
                        **(
                            dict(_probe_rehearsal=probe_rehearsal)
                            if probe_rehearsal is not None
                            else {}
                        ),
                    )
                    if retained
                    else {}
                ),
                _owned_source=(state, guard),
                _memory=memory,
                **(dict(_linear_fit=True) if linear_fit else {}),
                **catalog_options,
                **(
                    dict(incident_stream=incident_stream, calibrated_serving=calibrated)
                    if incident_stream is not None
                    else {}
                ),
                **(
                    dict(reference_mode="drift", candidate_feedback=True, acknowledge_correct=True)
                    if adaptive
                    else {}
                ),
            )
        returned = True
        guard()
    except Exception as error:
        if delegated and not returned:
            # A lost result can follow native work or a committed publication.
            # Only failure BEFORE delegation proves that no child work occurred.
            report["training_work"] = dict(
                attempted_updates=None,
                attempted_rows=None,
                returned_updates=None,
                unreturned_work_unknown=True,
            )
            report["delegated_work_unknown"] = True
        message = f"{type(error).__name__}: {error}"[:512]
        report.update(complete=False, owned_source_verified=False, owned_source_error=message)
        report.setdefault("error", message)
        if "source_unchanged" in report:
            report["source_unchanged"] = None
    else:
        report["owned_source_verified"] = True
    return report


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
    publish_after_learning=False,
    _owned_source=None,
    _memory=None,
    _linear_fit=False,
    incident_stream=None,
    calibrated_serving=False,
    _reference_validation=False,
    _capture_candidate=None,
    _retained_joint=False,
    _publish_retained=False,
    _probe_retained=False,
    _probe_rehearsal=None,
):
    from scripts.context_catalog_control import ContextCatalogControl

    if _probe_rehearsal is not None and (
        type(_probe_rehearsal) is not str
        or _probe_rehearsal not in ("joint", "current_only")
        or _probe_retained is not True
    ):
        raise ValueError("probe rehearsal requires explicit private retained probe")
    if (
        type(_probe_retained) is not bool
        or _probe_retained
        and (_retained_joint is not True or _publish_retained is not False)
    ):
        raise ValueError("retained probe requires separate explicit private retained mode")

    if type(_publish_retained) is not bool or _publish_retained and not _retained_joint:
        raise ValueError("retained publication requires explicit retained joint mode")

    if (
        type(_retained_joint) is not bool
        or _retained_joint
        and (
            _linear_fit is not True
            or _owned_source is None
            or stream_recipe != WITHHELD_CONTEXT_RECIPES[0]
            or reference_mode != "periodic"
            or publish_after_learning
            or candidate_feedback
            or acknowledge_correct
            or task_feedback
            or arrived_shift
            or catalog_trial is not None
            or catalog_policy is not None
            or catalog_base is not None
            or contextual_policy is not None
            or _reference_validation
            or incident_stream is not None
        )
    ):
        raise ValueError("retained joint requires exact private ordinary periodic owned LS mode")

    if (
        type(_reference_validation) is not bool
        or _reference_validation
        and (
            _linear_fit is not True
            or _owned_source is None
            or stream_recipe not in WITHHELD_CONTEXT_RECIPES
            or publish_after_learning is not False
            or not callable(_capture_candidate)
            or reference_mode not in ("no_learning", "periodic", "drift")
            or (candidate_feedback, acknowledge_correct)
            != ((True, True) if reference_mode == "drift" else (False, False))
            or contextual_policy is not None
            or catalog_trial is not None
            or catalog_policy is not None
            or catalog_base is not None
            or incident_stream is not None
        )
        or not _reference_validation
        and _capture_candidate is not None
    ):
        raise ValueError("reference validation requires its exact owned private LS composition")
    adaptive_owned = (
        _linear_fit is True
        and _owned_source is not None
        and reference_mode == "drift"
        and candidate_feedback is True
        and acknowledge_correct is True
        and publish_after_learning is False
    )
    catalog_requested = any(
        value is not None for value in (catalog_trial, catalog_policy, catalog_base)
    )
    owned_catalog = (
        _linear_fit is True
        and _owned_source is not None
        and (catalog_trial is not None or catalog_policy is not None)
    )
    incident_owned = incident_stream is not None
    if (
        type(calibrated_serving) is not bool
        or calibrated_serving
        and not incident_owned
        or incident_owned
        and (
            type(incident_stream) is not str
            or incident_stream not in ("stationary", "shift")
            or not adaptive_owned
            or stream_recipe
            != CONTEXT_INCIDENT_RECIPES[0 if incident_stream == "stationary" else 1]
        )
    ):
        raise ValueError("incident serving requires its exact owned adaptive LS TRAIN recipe")

    if (
        type(_linear_fit) is not bool
        or _linear_fit
        and (
            _owned_source is None
            or catalog_requested
            and not owned_catalog
            or contextual_policy is not None
            or arrived_shift
            or candidate_feedback
            and not adaptive_owned
            or task_feedback
            or acknowledge_correct
            and not adaptive_owned
        )
    ):
        raise ValueError("LS live serving requires the explicit owned paired source path")
    from src.core.context_perceptron import copy_context_snapshot

    copy_owned_state: Any = copy_context_snapshot
    digest_state, record_size = snapshot_digest, snapshot_record_bytes
    build_state_learner, make_guard = build_context, categorical_guard
    predictor_type = ContextServingPredictor
    source_version = "qualified-12"
    if _linear_fit:
        from src.adapters import context_linear_serving as linear_binding
        from src.core.context_linear_state import copy_context_linear_state

        copy_owned_state = copy_context_linear_state
        digest_state, record_size = (
            linear_binding.snapshot_digest,
            linear_binding.snapshot_record_bytes,
        )
        build_state_learner, make_guard = (
            linear_binding.build_context,
            linear_binding.categorical_guard,
        )
        predictor_type = linear_binding.ContextLinearServingPredictor
        source_version = "context-ls-12"

    if type(publish_after_learning) is not bool:
        raise ValueError("learning-boundary publication requires an exact boolean opt-in")
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
            **(dict(linear_fit=True) if _linear_fit else {}),
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
    paired = stream_recipe in (*CONTEXT_PAIRED_REHEARSAL_RECIPES, *CONTEXT_INCIDENT_RECIPES)
    same_context = (
        _linear_fit is True
        and _owned_source is not None
        and stream_recipe == CONTEXT_REHEARSAL_RECIPES[1]
    )
    # Why: the existing same-context recipe has two joint learning packets,
    # whereas the paired-head recipe has four. Return work keeps its old tick.
    learning_plan = FollowupTrainingPlan(2 if same_context else 4, 12)
    if stream_recipe in CONTEXT_INCIDENT_RECIPES and not incident_owned:
        raise ValueError("incident TRAIN recipe requires an explicit incident stream opt-in")
    same_context_ack = same_context and adaptive_owned
    if adaptive_owned and not (paired or same_context or _reference_validation):
        raise ValueError("adaptive owned serving requires an exact paired recipe")
    if publish_after_learning and (not (paired or same_context) or reference_mode != "periodic"):
        raise ValueError("learning-boundary publication requires an exact paired periodic recipe")
    if (
        _owned_source is not None
        and (
            input_path is not None
            or type(_owned_source) is not tuple
            or len(_owned_source) != 2
            or not callable(_owned_source[1])
            or _memory is None
            or not publish_after_learning
            and not adaptive_owned
            and not owned_catalog
            and not _reference_validation
            and not _retained_joint
        )
        or _owned_source is None
        and _memory is not None
    ):
        raise ValueError("owned source transport requires the guarded paired serving composition")
    if (
        stream_recipe is not None
        and any(c.endswith("-v2") for c in stream_recipe.contexts)
        and not paired
    ):
        raise ValueError("paired live TRAIN requires an exact named paired recipe")
    if paired and (
        reference_mode not in ("periodic", "no_learning")
        and not adaptive_owned
        or contextual_policy is not None
        or candidate_feedback
        and not adaptive_owned
        or acknowledge_correct
        and not adaptive_owned
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
    if adaptive_owned:
        report["training_control"] = "arrived_candidate_error_ack_v1"
    if _reference_validation:
        report.update(
            definition="fresh_context_linear_reference_validation_case_v1",
            role="reference_validation_train",
        )
    if incident_owned:
        report["incident_stream"] = incident_stream
        report["training_control"] = (
            "calibrated_onset_arrived_fifo_v1"
            if calibrated_serving
            else "arrived_candidate_error_ack_v1"
        )
    if _linear_fit:
        report["prior_training"] = dict(updates=12, rows=72, mistakes=None)
    live = budget = source = actor = runtime = rehearsal = None
    predictor: Any = None
    receipt = None
    if _owned_source is not None:
        report.pop("source_sha256")
        report.update(
            source_provenance="current_workflow_train_checkpoint12",
            memory_scope="shared_source_and_serving",
        )
    with (
        ProcessRssSampler(interval_seconds=0.02) if _memory is None else nullcontext(_memory)
    ) as sampler:
        try:
            if _owned_source is None:
                raw = _source_bytes(input_path)
                if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                    raise ValueError("qualified contextual source replaced")
                initial = head_states(json.loads(raw))[0]
            else:
                from src.core.context_perceptron import copy_context_snapshot

                _owned_source[1]()
                initial = copy_owned_state(_owned_source[0])
            if catalog_control is not None:
                catalog_control.require_source(
                    initial, catalog_base, digest=digest_state, same_state=_same_value
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
                if same_context:
                    report.update(train_task_family="legacy_tasks_v1", declared_tasks=TASKS)
                if _owned_source is not None:
                    report["definition"] = (
                        "fresh_context_linear_same_context_serving_v1"
                        if same_context
                        else "fresh_context_linear_paired_serving_v1"
                        if _linear_fit
                        else "fresh_context_head_paired_serving_v1"
                    )
                if _reference_validation:
                    report["definition"] = "fresh_context_linear_reference_validation_case_v1"
            if _retained_joint:
                from scripts.context_catalog_replay import compile_context_packet

                data = (tuple(compile_context_packet(spec, b) for b in data[0]), data[1])
                report["definition"] = "fresh_context_linear_retained_joint_serving_v1"
                report["retained_publication_requested"] = _publish_retained
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
                source_record_bytes=record_size(initial),
            )
            if publish_after_learning or same_context_ack:
                report["learning_boundary_plan"] = asdict(learning_plan)
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
                if _owned_source is not None:
                    _owned_source[1]()
                if (
                    (
                        _source_bytes(input_path) != raw
                        if _owned_source is None
                        else not _same_value(_owned_source[0], initial)
                    )
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

            if _owned_source is not None:
                require_source()
            if _linear_fit:
                from scripts.ticket_workflow_linear import counted_learner

                def before_linear_fit(features):
                    require_source()
                    paid = dict(rows=len(features), entered_at=monotonic(), complete=False)
                    report["native_calls"].append(paid)
                    report["training_work"]["attempted_updates"] += 1
                    report["training_work"]["attempted_rows"] += len(features)
                    if live is None:
                        raise ValueError("original live timeline unavailable")
                    live.native_started()
                    paid["model_started_at"] = monotonic()

                def after_linear_fit():
                    paid = report["native_calls"][-1]
                    paid.update(
                        model_finished_at=monotonic(), returned_at=monotonic(), complete=True
                    )
                    report["training_work"]["returned_updates"] += 1

                source = counted_learner(
                    before_linear_fit, after_linear_fit, separate=True, spec=spec
                )
            else:
                source = CountedLearner(initial.policy)
            source.restore_state(initial)
            if _retained_joint:
                from src.app.stream_rehearsal import StreamRehearsal, RecentStreamReplayPolicy
                from src.core.circadian_predictive_coding import ReplayRetentionBudget
                from src.core.context_joint_replay import JointContextReplayPolicy
                from src.core.demonstration_episode import EpisodeReplaySchema

                replay_budget = ToyBudgetSession(
                    ToyExecutionBudget(
                        max_replay_examples=12 * planned_updates,
                        max_wall_seconds=60,
                        max_process_rss_bytes=512 * 1024**2,
                    ),
                    monotonic,
                )
                replay_budget.attach_memory(sampler)
                rehearsal = StreamRehearsal(
                    source,
                    budget,
                    RecentStreamReplayPolicy(1, 12, ReplayRetentionBudget(30, 65536)),
                    replay_budget=replay_budget,
                    total_updates=planned_updates,
                    input_dim=9,
                    episode_schema=EpisodeReplaySchema(9, 3, 6),
                    joint_context=JointContextReplayPolicy(9, 6, (4, 5)),
                    check_boundary=require,
                )
            actor = PromotableActor(
                source,
                version=source_version,
                configuration=ServingConfiguration(3, 2),
                feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
                metadata={
                    "source": HEAD_SOURCE_SHA
                    if _owned_source is None
                    else "current_workflow_train_checkpoint12",
                    "history_updates": 12,
                },
            )
            predictor = predictor_type(actor, spec)
            gate = ServingPriorityGate(
                SharingLimits(
                    1,
                    planned_updates * (2 if _retained_joint else 1)
                    if catalog_control is None
                    else catalog_control.admissions,
                    1,
                ),
                resource_available=lambda: True,
            )
            live = TicketServingTimeline(plan, gate, predictor, require_source)
            controller = ServingPromotionController(
                actor,
                policy=policy(),
                evaluator=make_guard(spec, monotonic),
                build_learner=build_state_learner,
                state_digest=digest_state,
                max_pending=1,
            )

            def publish(owner, now, *, before_commit=None):
                nonlocal runtime, receipt
                runtime = owner  # Retain receipts even if later publication or serving fails.
                require()
                first_boundary = (publish_after_learning or same_context_ack) and (
                    len(owner.applied_updates) + len(owner.acknowledged_experiences)
                    if same_context_ack
                    else owner.candidate_snapshot().wake_updates
                ) == learning_plan.first_batches
                new, old = data[1]
                report["candidate_state"] = asdict(owner.candidate_snapshot().state)
                try:
                    ticket = controller.prepare(owner, new, old, now=now, training_ids=frozenset())
                except PromotionRejected as error:
                    report["guard_report"] = asdict(error.report)
                    if publish_after_learning or same_context_ack or _publish_retained:
                        if first_boundary:
                            report["learning_boundary_guard"] = deepcopy(report["guard_report"])
                            report["learning_boundary_tick"] = now
                        raise
                    return None
                report["guard_report"] = asdict(ticket.report)
                if first_boundary:
                    report["learning_boundary_guard"] = deepcopy(report["guard_report"])
                    report["learning_boundary_tick"] = now
                receipt = _commit_after_guard(controller, owner, ticket, before_commit)
                report["promotion_receipt"] = asdict(receipt)
                if publish_after_learning or adaptive_owned or _publish_retained:
                    report["publication_tick"] = now
                return receipt

            require()
            if catalog_control is not None:
                from scripts.context_controller_learning import _task_readout

                report["before"] = _task_readout(
                    initial,
                    "before",
                    require,
                    report,
                    **(dict(linear_fit=True) if _linear_fit else {}),
                )

            def predict_candidate(state, features):
                learner = build_state_learner(state)
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
                on_drain=publish
                if catalog_control is None
                and not _reference_validation
                and (not _retained_joint or _publish_retained)
                else None,
                **(dict(wake_rehearsal=rehearsal) if rehearsal is not None else {}),
                **(dict(catalog_control=catalog_control) if catalog_control is not None else {}),
                **(
                    dict(
                        followup_plan=learning_plan,
                        rejected_guard_digest=digest_state,
                    )
                    if publish_after_learning or same_context_ack
                    else dict(rejected_guard_digest=digest_state)
                    if _publish_retained
                    else {}
                ),
                **task_kwargs,
                **(dict(arrived_shift_predict=predict_context_perceptron) if arrived_shift else {}),
                **(dict(candidate_predict=predict_candidate) if candidate_feedback else {}),
                **(
                    dict(
                        arrived_shift_predict=predict_candidate,
                        shift_calibration=ShiftCalibration(),
                        shift_model_id=source_version,
                        shift_prefix_rows=tuple(b.row_ids for b in data[0][:6]),
                    )
                    if calibrated_serving
                    else {}
                ),
                **(dict(acknowledge_correct=True) if acknowledge_correct else {}),
                **(
                    dict(
                        label_delay=stream_recipe.label_delay,
                        candidate_row_limit=216
                        if incident_owned
                        else 144
                        if candidate_feedback
                        else 84,
                    )
                    if stream_recipe is not None
                    else {}
                ),
            )
            live.close()
            require()
            if _retained_joint:
                from scripts.context_controller_learning import _task_readout

                candidate = runtime.candidate_snapshot().state
                retained_readout: dict[str, Any] = dict(
                    task_outcomes={"after": []}, task_readout_work={}
                )
                report["candidate_return_readout"] = retained_readout
                report["candidate_state"] = asdict(candidate)
                retained_readout["counts"] = _task_readout(
                    candidate, "after", require, retained_readout, linear_fit=True
                )
                if runtime.candidate_snapshot().state != candidate:
                    raise ValueError("retained candidate changed during terminal TRAIN readout")
                if _probe_retained:
                    from scripts.context_retained_probe import run_probe

                    probe_record: dict[str, Any] = {}
                    report["isolated_transfer_probe"] = probe_record
                    run_probe(
                        runtime,
                        spec,
                        rehearsal,
                        require,
                        sampler,
                        report,
                        probe_record,
                        **(
                            dict(rehearsal_control=_probe_rehearsal)
                            if _probe_rehearsal is not None
                            else {}
                        ),
                    )
            if _reference_validation:
                candidate = runtime.candidate_snapshot().state
                _capture_candidate(candidate)
                require()
                if runtime.candidate_snapshot().state != candidate:
                    raise ValueError("validation capture changed the actual candidate")
                report["candidate_state"] = asdict(candidate)
            if adaptive_owned:
                progress = report.get("receipt_progress", {})
                if (
                    progress.get("complete") is not True
                    or progress.get("consumed") != planned_updates
                ):
                    raise ValueError("adaptive serving left original TRAIN receipts unconsumed")
                report["candidate_state"] = asdict(runtime.candidate_snapshot().state)
                if same_context_ack:
                    from scripts.context_controller_learning import _task_readout

                    candidate = runtime.candidate_snapshot().state
                    readout: dict[str, Any] = dict(
                        task_outcomes={"after": []}, task_readout_work={}
                    )
                    report["candidate_return_readout"] = readout
                    readout["counts"] = _task_readout(
                        candidate, "after", require, readout, linear_fit=True
                    )
                    if runtime.candidate_snapshot().state != candidate:
                        raise ValueError("final candidate changed during private TRAIN readout")
            if publish_after_learning or same_context_ack:
                report["candidate_state"] = asdict(runtime.candidate_snapshot().state)
                report["return_publication_attempted"] = "return_publication" in report
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
                report["after"] = _task_readout(
                    candidate,
                    "after",
                    require,
                    report,
                    **(dict(linear_fit=True) if _linear_fit else {}),
                )
                catalog_control.require_candidate()
            if receipt is not None:
                if publish_after_learning or adaptive_owned or _publish_retained:
                    _episodes(
                        predictor,
                        "published",
                        require,
                        report["episodes"],
                        **(dict(role="controller_train") if _publish_retained else {}),
                        **(dict(tasks=PAIRED_TASKS) if paired else {}),
                    )
                report["rollback_generation"] = controller.rollback(receipt)
                after = actor.serving_snapshot()
                report["rollback_equal_source"] = after.model_state == initial
                report["rollback"] = asdict(after)
                _episodes(
                    predictor,
                    "rollback",
                    require,
                    report["episodes"],
                    **(dict(role="controller_train") if _publish_retained else {}),
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
            from src.core.learner_ports import TrainingUpdateFailed

            if isinstance(error, TrainingUpdateFailed):
                report["native_failure_definition"] = error.definition
                report["native_failure_work"] = deepcopy(error.work)
            for paid in report["native_calls"]:
                if "model_started_at" in paid and "model_finished_at" not in paid:
                    paid["model_interval_unknown"] = True
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
                report["training_complete"] = (
                    report.get("receipt_progress", {}).get("complete") is True
                    if adaptive_owned
                    else budget.updates_completed == planned_updates
                )
                if adaptive_owned:
                    report["training_complete_scope"] = (
                        "all original TRAIN consumed by real fits or verified correct ACK; ACK is not a fit"
                    )
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    updates=budget.updates_completed,
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                )
            if predictor is not None:
                report["typed_work"] = dict(predictor.work)
                if _linear_fit:
                    report["typed_native_work"] = asdict(predictor.native_work)
            if source is not None:
                report["source_unchanged"] = source.snapshot_state() == initial
            if runtime is not None:
                report["applied_updates"] = [asdict(u) for u in runtime.applied_updates]
            if actor is not None:
                final_actor = actor.serving_snapshot()
                report["final_actor"] = asdict(final_actor)
                if catalog_control is not None or _reference_validation or _retained_joint:
                    report["actor_unchanged"] = (
                        final_actor.model_state == initial
                        and final_actor.generation == 0
                        and final_actor.actor_version == source_version
                    )
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(sampler.snapshot()))
    return report
