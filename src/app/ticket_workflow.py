"""Bounded local typed-policy episodes with independent effect admission.

Consumes a trusted predictor callback and immutable ticket/capability records.
Returns paid step prefixes and verified local task outcomes. No fitting, final
selection, external handlers or containment of arbitrary callback code occurs.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from src.app.decision_reference_evaluation import _observed_probability_return, _observed_return
from src.core.ticket_workflow import (
    CATALOG,
    NoWorkflowCandidates,
    WorkflowCapabilities,
    WorkflowState,
    _exact,
    dispatch_workflow,
    invalidate_workflow,
    workflow_request,
    workflow_success,
)
from src.core.typed_decision import (
    DecisionResult,
    ProbabilityDecisionResult,
    validate_decision_result,
    validate_probability_decision_result,
)


def _validate_return(request, supplied, result):
    if supplied != request:
        raise ValueError("predictor changed workflow request")
    if type(result) is DecisionResult:
        validate_decision_result(request, result)
    elif type(result) is ProbabilityDecisionResult:
        validate_probability_decision_result(request, result)
    else:
        raise ValueError("workflow predictor must return a typed decision")


def _reported_work(work, result):
    if type(result) is DecisionResult:
        work["reported_token_visits"] += result.token_visits
        work["reported_candidate_comparisons"] += result.candidate_comparisons
    else:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(result, "reported_" + name)
            work["reported_" + name] += 0 if value is None else value
            work[name + "_unknown_calls"] += value is None


def _environment_event(state, capabilities, scenario, selected, used):
    if not used and scenario == "stale_once" and state.phase == "ready":
        return invalidate_workflow(state), capabilities, "invalidate_ticket"
    if (
        not used
        and scenario == "revoke_commit"
        and state.phase == "staged"
        and selected == "commit"
    ):
        revoked = WorkflowCapabilities(
            tuple(a for a in capabilities.allowed if a != "commit"), capabilities.version + 1
        )
        return state, revoked, "revoke_commit"
    return state, capabilities, None


def build_public_workflow_request(
    state, capabilities, request_id, last_status="initial", *, include_policy_context=False
):
    """Build the actual public model input, without teacher targets or private IDs.

    Why: opt-in TRAIN observations must share live text construction, including
    the checkout's existing suffix, rather than reproduce it in another recipe.
    """
    request = workflow_request(
        state, capabilities, request_id, last_status, include_policy_context=include_policy_context
    )
    return replace(request, query=request.query + " please")


def run_workflow_episode(
    initial_state,
    predict,
    *,
    capabilities,
    max_steps=6,
    scenario="ordinary",
    include_policy_context=False,
) -> dict[str, Any]:
    if type(include_policy_context) is not bool:
        raise ValueError("include_policy_context must be an exact bool")
    _exact(initial_state, WorkflowState)
    _exact(capabilities, WorkflowCapabilities)
    if initial_state.phase != "sealed" or initial_state.revision != 0:
        raise ValueError("episodes require a fresh sealed ticket")
    if type(max_steps) is not int or not 1 <= max_steps <= 12:
        raise ValueError("max_steps must be an integer in1..12")
    if type(scenario) is not str or scenario not in ("ordinary", "stale_once", "revoke_commit"):
        raise ValueError("unknown workflow scenario")
    if not callable(predict):
        raise ValueError("predict must be callable")
    state, current_caps = replace(initial_state), replace(capabilities)
    work = dict.fromkeys(
        (
            "candidate_construction_attempts",
            "catalog_entries_considered",
            "candidate_entries_returned",
            "prediction_calls_attempted",
            "prediction_calls_returned",
            "prediction_calls_validated",
            "dispatch_attempts",
            "dispatches_completed",
            "effects_applied",
            "environment_events",
            "reported_token_visits",
            "reported_candidate_comparisons",
            "reported_input_tokens",
            "reported_output_tokens",
            "input_tokens_unknown_calls",
            "output_tokens_unknown_calls",
        ),
        0,
    )
    report: dict[str, Any] = dict(
        complete=False,
        success=False,
        scenario=scenario,
        initial_state=asdict(state),
        steps=[],
        work=work,
        stopped_reason="max_steps",
        external_actions=0,
        training_updates=0,
        max_steps=max_steps,
    )
    last_status, used = "initial", False
    context_options = {"include_policy_context": True} if include_policy_context else {}
    if include_policy_context:
        report["policy_context_in_query"] = True
    try:
        for attempt in range(max_steps):
            row: dict[str, Any] = dict(
                attempt=attempt + 1,
                observation_state=asdict(state),
                capabilities_observed=asdict(current_caps),
            )
            report["steps"].append(row)
            work["candidate_construction_attempts"] += 1
            work["catalog_entries_considered"] += len(CATALOG)
            try:
                # Ticket identity can contain private content; it is not a model input.
                request = build_public_workflow_request(
                    state, current_caps, f"workflow-step{attempt}", last_status, **context_options
                )
            except NoWorkflowCandidates:
                report["stopped_reason"] = "no_permitted_actions"
                break
            row["request"] = asdict(request)
            work["candidate_entries_returned"] += len(request.allowed_candidates)
            supplied = deepcopy(request)
            work["prediction_calls_attempted"] += 1
            result = predict(supplied)
            work["prediction_calls_returned"] += 1
            row["returned"] = (
                _observed_probability_return(result)
                if type(result) is ProbabilityDecisionResult
                else _observed_return(result)
            )
            _validate_return(request, supplied, result)
            row["result"] = asdict(result)
            work["prediction_calls_validated"] += 1
            _reported_work(work, result)
            observed_revision = state.revision
            state, current_caps, event = _environment_event(
                state, current_caps, scenario, result.selected_id, used
            )
            if event:
                used = True
                work["environment_events"] += 1
                row["environment_event"] = dict(
                    kind=event, state=asdict(state), capabilities=asdict(current_caps)
                )
            row["effect_state"] = asdict(state)
            row["capabilities_at_effect"] = asdict(current_caps)
            work["dispatch_attempts"] += 1
            transition = dispatch_workflow(
                state, result.selected_id, observed_revision, current_caps
            )
            work["dispatches_completed"] += 1
            state, last_status = transition.state, transition.status
            row["transition"] = asdict(transition)
            work["effects_applied"] += transition.status == "applied"
            if transition.status == "permission_denied":
                report["stopped_reason"] = "permission_denied"
                break
            if state.phase == "committed":
                report["stopped_reason"] = (
                    "success" if workflow_success(state) else "wrong_workflow"
                )
                break
        report["success"] = workflow_success(state)
        report["complete"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
        report["stopped_reason"] = "error"
    finally:
        report["final_state"] = asdict(state)
        report["final_capabilities"] = asdict(current_caps)
        report["error_recovery_demonstrated"] = report["success"] and any(
            s.get("transition", {}).get("status") not in (None, "applied") for s in report["steps"]
        )
        report["unreturned_prediction_work_unknown"] = (
            work["prediction_calls_attempted"] > work["prediction_calls_returned"]
        )
        report["unvalidated_return_work_unknown"] = (
            work["prediction_calls_returned"] > work["prediction_calls_validated"]
        )
        report["cost_scope"] = (
            "fixed catalog scans/typed calls and local transitions; provider counters are assertions; copies/validation/metrics/Python extra"
        )
    return report
