"""In-memory workflow transitions and scalar policies; no fitting or tool IO."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.app.ticket_workflow import run_workflow_episode
from src.core.ticket_workflow import (
    NoWorkflowCandidates,
    WorkflowCapabilities,
    WorkflowState,
    dispatch_workflow,
    invalidate_workflow,
    workflow_request,
    workflow_success,
)
from src.core.typed_decision import (
    CandidateScore,
    DecisionRequest,
    DecisionResult,
    ProbabilityDecisionResult,
)


ALL_TOOLS = ("commit", "defer", "inspect", "open", "retire", "route")
ACTIVE_TOOLS = ("commit", "defer", "inspect", "open", "route")
RULES = {"invoice": "inspect", "legacy": "defer", "parcel": "defer", "password": "inspect"}


def capabilities():
    return WorkflowCapabilities(ALL_TOOLS)


def initial(concept="invoice"):
    return WorkflowState("ticket-1", concept)


def apply(state, action, caps=None):
    return dispatch_workflow(
        state, action, state.revision, capabilities() if caps is None else caps
    )


@pytest.mark.parametrize("concept,target", list(RULES.items()))
def test_should_require_three_explicit_effects_and_verify_committed_current_truth(concept, target):
    source = initial(concept)
    opened = apply(source, "open")
    assert opened.status == "applied" and opened.state.phase == "ready"
    staged = apply(opened.state, target)
    assert staged.status == "applied" and staged.state.staged_tool == target
    assert staged.state.committed_tool is None and workflow_success(staged.state) is False
    committed = apply(staged.state, "commit")
    assert committed.status == "applied" and committed.state.committed_tool == target
    assert workflow_success(committed.state) is True
    assert source.phase == "sealed" and source.revision == 0
    assert source.staged_tool is source.committed_tool is None


def test_should_preserve_wrong_but_permitted_workflow_choice_as_terminal_task_failure():
    ready = apply(initial(), "open").state
    staged = apply(ready, "route")
    final = apply(staged.state, "commit")
    assert staged.status == final.status == "applied"
    assert final.state.staged_tool == final.state.committed_tool == "route"
    assert workflow_success(final.state) is False


def test_should_expose_every_active_capability_at_each_phase_without_solving_the_next_action():
    sealed = initial()
    ready = apply(sealed, "open").state
    staged = apply(ready, "inspect").state
    for index, state in enumerate((sealed, ready, staged)):
        request = workflow_request(state, capabilities(), f"decision-{index}")
        assert (
            tuple(candidate.candidate_id for candidate in request.allowed_candidates)
            == ACTIVE_TOOLS
        )
        assert set(vars(request)) == {"request_id", "query", "allowed_candidates"}
    sealed_request = workflow_request(sealed, capabilities(), "hidden")
    ready_request = workflow_request(ready, capabilities(), "visible", last_status="stale_revision")
    assert "invoice" not in sealed_request.query
    assert "invoice" in ready_request.query and "stale_revision" in ready_request.query


def test_should_enforce_current_permissions_even_when_observed_request_admitted_the_action():
    ready = apply(initial(), "open").state
    observed = workflow_request(ready, capabilities(), "observed")
    assert "inspect" in {candidate.candidate_id for candidate in observed.allowed_candidates}
    revoked = WorkflowCapabilities(
        tuple(tool for tool in ALL_TOOLS if tool != "inspect"), version=1
    )
    denied = dispatch_workflow(ready, "inspect", ready.revision, revoked)
    assert denied.status == "permission_denied" and denied.state == ready
    assert denied.state.staged_tool is None


def test_should_refuse_stale_action_after_explicit_reseal_without_repairing_it():
    ready = apply(initial(), "open").state
    invalidated = invalidate_workflow(ready)
    refused = dispatch_workflow(invalidated, "inspect", ready.revision, capabilities())
    assert invalidated.phase == "sealed" and invalidated.revision == ready.revision + 1
    assert refused.status == "stale_revision" and refused.state == invalidated
    assert refused.state.staged_tool is refused.state.committed_tool is None


@pytest.mark.parametrize(
    "action,status",
    [("commit", "wrong_phase"), ("inspect", "wrong_phase"), ("retire", "inactive_tool")],
)
def test_should_refuse_wrong_phase_or_inactive_tool_without_effect(action, status):
    state = initial()
    refused = apply(state, action)
    assert refused.status == status and refused.state == state


def test_should_not_allow_completion_to_be_reopened_or_invalidated():
    final = apply(apply(apply(initial(), "open").state, "inspect").state, "commit").state
    repeated = apply(final, "open")
    assert repeated.status == "already_complete" and repeated.state == final
    with pytest.raises(ValueError):
        invalidate_workflow(final)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_concept",
        "boolean_revision",
        "unknown_policy",
        "unsorted_capabilities",
        "duplicate_capabilities",
        "unknown_capability",
        "boolean_capability_version",
    ],
)
def test_should_refuse_invalid_state_or_authority_metadata_before_any_effect(fault):
    with pytest.raises(ValueError):
        if fault == "unknown_concept":
            WorkflowState("ticket-1", "unknown")
        elif fault == "boolean_revision":
            WorkflowState("ticket-1", "invoice", revision=True)
        elif fault == "unknown_policy":
            WorkflowState("ticket-1", "invoice", policy_version="v3")
        elif fault == "unsorted_capabilities":
            WorkflowCapabilities(tuple(reversed(ALL_TOOLS)))
        elif fault == "duplicate_capabilities":
            WorkflowCapabilities(("open", "open"))
        elif fault == "unknown_capability":
            WorkflowCapabilities(("network",))
        else:
            WorkflowCapabilities(ALL_TOOLS, version=True)


def selected_tool(request: DecisionRequest) -> str:
    words = request.query.split()
    return {"sealed": "open", "staged": "commit"}.get(words[0], RULES.get(words[1], "route"))


def decision(request: DecisionRequest, selected: str | None = None) -> DecisionResult:
    chosen = selected_tool(request) if selected is None else selected
    return DecisionResult(
        request.request_id,
        chosen,
        tuple(
            CandidateScore(
                candidate.candidate_id,
                float(candidate.candidate_id == chosen),
                float(candidate.candidate_id == chosen),
            )
            for candidate in request.allowed_candidates
        ),
        len(request.query.split()),
        len(request.allowed_candidates),
    )


EPISODE_CASES = tuple(
    (scenario, concept) for scenario in ("ordinary", "stale_once") for concept in RULES
) + (("revoke_commit", "invoice"),)


@pytest.fixture(scope="module")
def completed_episodes():
    reports = {}
    for scenario, concept in EPISODE_CASES:
        source, authority = initial(concept), capabilities()
        before = deepcopy((source, authority))
        report = run_workflow_episode(source, decision, capabilities=authority, scenario=scenario)
        assert (source, authority) == before
        reports[scenario, concept] = report
    return reports


@pytest.mark.parametrize("scenario,concept", EPISODE_CASES)
def test_should_finish_or_refuse_bounded_local_scenarios_without_hidden_action_repair(
    completed_episodes, scenario, concept
):
    report = completed_episodes[scenario, concept]
    assert report["complete"] is True
    assert report["external_actions"] == report["training_updates"] == 0
    assert report["unreturned_prediction_work_unknown"] is False
    assert report["unvalidated_return_work_unknown"] is False
    statuses = [row["transition"]["status"] for row in report["steps"]]
    if scenario == "revoke_commit":
        assert report["success"] is False and report["stopped_reason"] == "permission_denied"
        assert statuses == ["applied", "applied", "permission_denied"]
        last = report["steps"][-1]
        assert "commit" in last["capabilities_observed"]["allowed"]
        assert "commit" not in last["capabilities_at_effect"]["allowed"]
        assert last["environment_event"]["kind"] == "revoke_commit"
        assert report["final_capabilities"]["version"] == 1
        assert report["final_state"]["committed_tool"] is None
    else:
        assert report["success"] is True and report["stopped_reason"] == "success"
        assert report["final_state"]["committed_tool"] == RULES[concept]
        assert statuses == (
            ["applied"] * 3
            if scenario == "ordinary"
            else ["applied", "stale_revision", "applied", "applied", "applied"]
        )
        if scenario == "stale_once":
            stale = report["steps"][1]
            assert stale["environment_event"]["kind"] == "invalidate_ticket"
            assert stale["effect_state"]["revision"] == stale["observation_state"]["revision"] + 1
            assert report["steps"][2]["request"]["query"] == "sealed feedback stale_revision please"
    assert report["error_recovery_demonstrated"] is (scenario == "stale_once")


def test_should_charge_catalog_generation_separately_and_send_only_observed_typed_inputs(
    completed_episodes,
):
    totals = dict(calls=0, catalog=0, candidates=0)
    for (scenario, concept), report in completed_episodes.items():
        rows, work = report["steps"], report["work"]
        n = len(rows)
        for key in (
            "candidate_construction_attempts",
            "prediction_calls_attempted",
            "prediction_calls_returned",
            "prediction_calls_validated",
            "dispatch_attempts",
            "dispatches_completed",
        ):
            assert work[key] == n
        assert work["catalog_entries_considered"] == 6 * n
        assert work["candidate_entries_returned"] == work["reported_candidate_comparisons"] == 5 * n
        assert work["reported_token_visits"] == sum(
            len(row["request"]["query"].split()) for row in rows
        )
        assert (
            work["effects_applied"]
            == {"ordinary": 3, "stale_once": 4, "revoke_commit": 2}[scenario]
        )
        assert work["environment_events"] == int(scenario != "ordinary")
        for index, row in enumerate(rows):
            wire = row["request"]
            assert set(wire) == {"request_id", "query", "allowed_candidates"}
            assert wire["request_id"] == f"workflow-step{index}"
            assert (
                tuple(item["candidate_id"] for item in wire["allowed_candidates"]) == ACTIVE_TOOLS
            )
            assert "ticket-1" not in json.dumps(wire)
            assert "target_id" not in wire and "policy_version" not in wire
            assert row["observation_state"]["concept"] == concept
        totals["calls"] += n
        totals["catalog"] += work["catalog_entries_considered"]
        totals["candidates"] += work["candidate_entries_returned"]
    assert totals == dict(calls=35, catalog=210, candidates=175)


@pytest.mark.parametrize("tokens", [None, 0], ids=["unknown", "reported-zero"])
def test_should_accept_provider_probabilities_and_keep_unknown_costs_distinct_from_zero(tokens):
    def predict(request):
        selected = selected_tool(request)
        return ProbabilityDecisionResult(
            request.request_id,
            selected,
            tuple(
                (candidate.candidate_id, float(candidate.candidate_id == selected))
                for candidate in request.allowed_candidates
            ),
            reported_confidence=0.4,
            reported_input_tokens=tokens,
            reported_output_tokens=tokens,
        )

    report = run_workflow_episode(initial(), predict, capabilities=capabilities())
    assert report["complete"] is report["success"] is True
    assert report["work"]["reported_input_tokens"] == report["work"]["reported_output_tokens"] == 0
    assert (
        report["work"]["input_tokens_unknown_calls"]
        == report["work"]["output_tokens_unknown_calls"]
        == (3 if tokens is None else 0)
    )
    assert report["work"]["reported_candidate_comparisons"] == 0
    assert all(row["result"]["reported_confidence"] == 0.4 for row in report["steps"])


def test_should_report_wrong_committed_workflow_as_complete_execution_and_failed_task():
    def wrong(request):
        return decision(
            request, "route" if request.query.startswith("ready") else selected_tool(request)
        )

    report = run_workflow_episode(initial(), wrong, capabilities=capabilities())
    assert report["complete"] is True and report["success"] is False
    assert report["stopped_reason"] == "wrong_workflow"
    assert report["final_state"]["committed_tool"] == "route"
    assert report["work"]["effects_applied"] == 3


def test_should_exhaust_the_declared_attempt_budget_without_opening_or_repairing_automatically():
    report = run_workflow_episode(
        initial(), lambda request: decision(request, "commit"), capabilities=capabilities()
    )
    assert report["complete"] is True and report["success"] is False
    assert report["stopped_reason"] == "max_steps" and len(report["steps"]) == 6
    assert all(row["transition"]["status"] == "wrong_phase" for row in report["steps"])
    assert report["final_state"] == report["initial_state"]
    assert report["work"]["effects_applied"] == 0


def test_should_honor_a_shorter_budget_without_adding_unpaid_actions():
    report = run_workflow_episode(initial(), decision, capabilities=capabilities(), max_steps=1)
    assert report["complete"] is True and report["success"] is False
    assert report["stopped_reason"] == "max_steps" and report["final_state"]["phase"] == "ready"
    assert report["work"]["prediction_calls_attempted"] == report["work"]["effects_applied"] == 1


def test_should_stop_without_prediction_when_no_capability_is_available():
    def forbidden(_request):
        pytest.fail("empty authority must stop before prediction")

    report = run_workflow_episode(initial(), forbidden, capabilities=WorkflowCapabilities(()))
    assert report["complete"] is True and report["success"] is False
    assert report["stopped_reason"] == "no_permitted_actions"
    assert report["work"]["catalog_entries_considered"] == 6
    assert (
        report["work"]["candidate_entries_returned"]
        == report["work"]["prediction_calls_attempted"]
        == report["work"]["dispatch_attempts"]
        == 0
    )
    assert "request" not in report["steps"][0]


@pytest.mark.parametrize(
    "fault",
    ["unreturned", "unsupported", "invalid-selection", "changed-request", "nonfinite-probability"],
)
def test_should_preserve_paid_prediction_prefix_and_refuse_effects_after_invalid_or_missing_return(
    fault,
):
    def predict(request):
        result: ProbabilityDecisionResult | DecisionResult
        if fault == "unreturned":
            raise RuntimeError("injected predictor failure")
        if fault == "unsupported":
            return object()
        if fault == "nonfinite-probability":
            result = ProbabilityDecisionResult(
                request.request_id,
                "open",
                tuple(
                    (candidate.candidate_id, float(candidate.candidate_id == "open"))
                    for candidate in request.allowed_candidates
                ),
            )
            object.__setattr__(result, "probabilities", (("open", float("nan")),))
            return result
        result = decision(request)
        if fault == "invalid-selection":
            object.__setattr__(result, "selected_id", "network")
        else:
            object.__setattr__(request, "query", "changed callback input")
        return result

    report = run_workflow_episode(initial(), predict, capabilities=capabilities())
    assert report["complete"] is report["success"] is False
    assert report["stopped_reason"] == "error" and report["error"]
    assert report["work"]["prediction_calls_attempted"] == 1
    assert report["work"]["prediction_calls_returned"] == int(fault != "unreturned")
    assert report["work"]["prediction_calls_validated"] == report["work"]["dispatch_attempts"] == 0
    assert report["unreturned_prediction_work_unknown"] is (fault == "unreturned")
    assert report["unvalidated_return_work_unknown"] is (fault != "unreturned")
    assert ("returned" in report["steps"][0]) is (fault != "unreturned")
    assert report["steps"][0]["request"]["query"] == "sealed feedback initial please"
    assert report["final_state"] == report["initial_state"]
    json.dumps(report, allow_nan=False)


def test_should_retain_an_applied_prefix_when_the_next_prediction_does_not_return():
    calls = []

    def predict(request):
        calls.append(request.request_id)
        if len(calls) == 2:
            raise RuntimeError("second prediction unavailable")
        return decision(request)

    report = run_workflow_episode(initial(), predict, capabilities=capabilities())
    assert calls == ["workflow-step0", "workflow-step1"]
    assert report["complete"] is False and report["success"] is False
    assert report["work"]["prediction_calls_attempted"] == 2
    assert (
        report["work"]["prediction_calls_returned"]
        == report["work"]["prediction_calls_validated"]
        == 1
    )
    assert report["work"]["dispatch_attempts"] == report["work"]["effects_applied"] == 1
    assert report["steps"][0]["transition"]["status"] == "applied"
    assert report["steps"][0]["result"]["selected_id"] == "open"
    assert "returned" not in report["steps"][1]
    assert report["final_state"]["phase"] == "ready" and report["final_state"]["revision"] == 1
    assert report["unreturned_prediction_work_unknown"] is True
    assert report["unvalidated_return_work_unknown"] is False


def test_should_use_a_detached_admission_snapshot_when_callback_mutates_original_caller_records():
    source, authority = initial(), capabilities()

    def predict(request):
        object.__setattr__(source, "concept", "parcel")
        object.__setattr__(authority, "allowed", ())
        return decision(request)

    report = run_workflow_episode(source, predict, capabilities=authority)
    assert report["success"] is True and report["final_state"]["concept"] == "invoice"
    assert report["final_state"]["committed_tool"] == "inspect"
    assert report["final_capabilities"]["allowed"] == ALL_TOOLS
    assert source.concept == "parcel" and authority.allowed == ()
    before = deepcopy(report)
    object.__setattr__(source, "ticket_id", "later caller mutation")
    assert report == before


@pytest.mark.parametrize(
    "fault",
    [
        "zero-budget",
        "boolean-budget",
        "oversized-budget",
        "scenario",
        "noncallable",
        "nonfresh",
        "changed-state",
        "changed-capabilities",
    ],
)
def test_should_refuse_invalid_episode_admission_before_predicting(fault):
    source, authority = initial(), capabilities()
    options: dict[str, Any] = {}

    def forbidden(_request):
        pytest.fail("invalid admission must not call predictor")

    predict: Any = forbidden
    if fault in {"zero-budget", "boolean-budget", "oversized-budget"}:
        options["max_steps"] = {"zero-budget": 0, "boolean-budget": True, "oversized-budget": 13}[
            fault
        ]
    elif fault == "scenario":
        options["scenario"] = "unknown"
    elif fault == "noncallable":
        predict = None
    elif fault == "nonfresh":
        source = apply(source, "open").state
    elif fault == "changed-state":
        object.__setattr__(source, "revision", True)
    else:
        object.__setattr__(authority, "allowed", ("network",))
    with pytest.raises(ValueError):
        run_workflow_episode(source, predict, capabilities=authority, **options)


def test_should_reject_empty_candidate_construction_and_unknown_commands_explicitly():
    with pytest.raises(NoWorkflowCandidates):
        workflow_request(initial(), WorkflowCapabilities(()), "empty")
    with pytest.raises(ValueError):
        dispatch_workflow(initial(), "network", 0, capabilities())
    with pytest.raises(ValueError):
        dispatch_workflow(initial(), "open", True, capabilities())


@pytest.mark.parametrize("complete,exit_code", [(True, 0), (False, 1)], ids=["complete", "partial"])
def test_should_publish_a_fresh_cli_report_without_repeating_the_experiment(
    monkeypatch, tmp_path, complete, exit_code
):
    from scripts import run_ticket_workflow as cli

    calls = []
    report = {"complete": complete, "qualification": "scalar CLI fixture", "external_actions": 0}

    def run_once():
        calls.append("run")
        return report

    monkeypatch.setattr(cli, "run_example", run_once)
    output = tmp_path / "workflow.json"
    assert cli.main(["--output", str(output)]) == exit_code
    assert calls == ["run"]
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_should_refuse_an_existing_cli_output_before_fitting_or_running_episodes(
    monkeypatch, tmp_path
):
    from scripts import run_ticket_workflow as cli

    def forbidden():
        pytest.fail("existing output must refuse before the experiment")

    monkeypatch.setattr(cli, "run_example", forbidden)
    output = tmp_path / "existing.json"
    output.write_bytes(b"original paid report")
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output)])
    assert output.read_bytes() == b"original paid report"


@pytest.mark.parametrize("fault", ["nonfinite", "oversized"])
def test_should_refuse_invalid_cli_serialization_without_publishing_partial_bytes(
    monkeypatch, tmp_path, fault
):
    from scripts import run_ticket_workflow as cli

    value = float("nan") if fault == "nonfinite" else "x" * (512 * 1024)
    monkeypatch.setattr(cli, "run_example", lambda: {"complete": True, "value": value})
    output = tmp_path / "refused.json"
    with pytest.raises(ValueError):
        cli.main(["--output", str(output)])
    assert not output.exists()


def test_should_keep_cli_json_bounded_when_whitespace_would_exceed_storage(monkeypatch, tmp_path):
    from scripts import run_ticket_workflow as cli

    # This represents bounded repeated records, without running an episode/model.
    report = {"complete": True, "steps": [{"observed": 1, "result": 2, "state": 3}] * 12000}
    assert len(json.dumps(report, indent=2).encode()) > 512 * 1024
    assert len(json.dumps(report, separators=(",", ":")).encode()) < 512 * 1024
    monkeypatch.setattr(cli, "run_example", lambda: report)
    output = tmp_path / "compact.json"
    assert cli.main(["--output", str(output)]) == 0
    assert output.stat().st_size <= 512 * 1024
    assert json.loads(output.read_bytes()) == report
