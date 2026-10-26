"""Four-arm information, causal cost and paid-prefix checks with scalar owners.

The imported recording learner is an opaque scalar fake. These tests perform no
model construction, fitting, tensor operations or released-model inference.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any

import pytest

from scripts import run_open_jev_memory_comparison as comparison
from scripts import run_open_jev_ticket_workflow as ticket
from scripts.run_ticket_workflow import training_cases
from src.core.typed_decision import ProbabilityDecisionResult
from tests.test_native_ticket_workflow import WorkflowLearner, request


ARMS = ("frozen_context", "retrieval_only", "adapter_only", "hybrid")


@pytest.fixture(scope="module")
def completed():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}
    events: list[Any] = []
    comparison.run_comparison(owner, report, events.append)
    return owner, report, events


def test_should_fit_once_before_all_four_arms_and_never_train_from_development(completed):
    owner, report, _ = completed
    original = training_cases()
    expected = [
        (replace(case.request, request_id=f"pass{p}-{case.request.request_id}"), case.target_id)
        for p in range(2)
        for case in original
    ]
    assert owner.training == expected and report["app_committed_updates"] == 24
    first = next(i for i, entry in enumerate(owner.timeline) if entry[0] == "predict")
    assert sum(entry[0] == "train" for entry in owner.timeline[:first]) == 24
    assert report["sequence_complete"] is True and report["logical_tick"] == 48
    assert report["native_work"] == dict(
        training_attempts=24,
        forward_calls=57,
        backward_calls=24,
        optimizer_steps=24,
        committed_updates=24,
    )
    assert all(query.query.endswith(" train") for query, _ in owner.training)
    assert all(
        query.query.split(" evidence ", 1)[0].endswith(" please") for query in owner.predictions
    )
    json.dumps(report, allow_nan=False)


def test_should_build_the_same_causal_consented_information_from_actual_twenty_four_receipts(
    completed,
):
    _, report, _ = completed
    events, rows, examples = report["evidence_events"], report["delivery"], report["training_cases"]
    assert len(events) == len(rows) == len(examples) == 24
    assert len({event["evidence"]["evidence_id"] for event in events}) == 24
    for index, (event, row, case) in enumerate(zip(events, rows, examples, strict=True)):
        evidence = event["evidence"]
        assert event["revision"] == 0
        assert evidence["evidence_id"] == row["exposure_id"] == row["source_example_id"]
        assert evidence["request"] == case["request"] and evidence["target_id"] == case["target_id"]
        assert evidence["observed_at"] == row["observed_at"] == 2 * index
        assert evidence["available_at"] == row["receipt"]["arrived_at"] == 2 * index + 1
        assert evidence["recorded_at"] == row["receipt"]["applied_at"] == 2 * index + 1
        assert (
            evidence["permissions"]
            == row["permissions"]
            == dict(training=True, replay=True, evaluation=False)
        )
        assert evidence["policy_version"] == "v2" and evidence["role"] == "train"
        assert evidence["source_group"] == case["source_group"] == "train-workflow-traces"
    assert report["evidence_cutoff"] == 48
    encoded = json.dumps(events, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    assert report["evidence_storage_bytes"] == len(encoded) <= 32768
    assert (
        report["evidence_build_work"]["writes_attempted"]
        == report["evidence_build_work"]["writes_completed"]
        == 24
    )
    assert report["evidence_build_work"]["retrieval_attempts"] == 0


def test_should_restore_only_native_sources_and_keep_every_realized_episode_for_each_arm(completed):
    owner, report, _ = completed
    assert owner.captures == [0, 24]
    assert owner.restores == [
        {"fixture_checkpoint": 0},
        {"fixture_checkpoint": 24},
        {"fixture_checkpoint": 24},
    ]
    assert (
        report["snapshot_work"]["restore_attempts"]
        == report["snapshot_work"]["restores_completed"]
        == 3
    )
    assert report["snapshot_work"]["released_snapshots"] == 2
    assert [row["name"] for row in report["arms"]] == list(ARMS)
    assert [row["source_updates"] for row in report["arms"]] == [0, None, 24, 24]
    assert [row["control"] for row in report["episodes"]] == [arm for arm in ARMS for _ in range(3)]
    assert [row["episode_indices"] for row in report["arms"]] == [
        list(range(start, start + 3)) for start in (0, 3, 6, 9)
    ]
    for arm in report["arms"]:
        episodes = [report["episodes"][index]["episode"] for index in arm["episode_indices"]]
        assert [row["scenario"] for row in episodes] == ["ordinary", "stale_once", "revoke_commit"]
        assert [len(row["steps"]) for row in episodes] == [3, 5, 3]
        assert [row["success"] for row in episodes] == [True, True, False]
        assert all(row["complete"] for row in episodes)
        assert episodes[1]["error_recovery_demonstrated"] is True
        assert episodes[-1]["stopped_reason"] == "permission_denied"
        assert report["summary"][arm["name"]] == dict(
            episodes=3,
            successes=2,
            successful_recoveries=1,
            permission_denials=1,
            stops=["success", "success", "permission_denied"],
        )


def test_should_charge_native_and_evidence_work_per_arm_without_double_counting_the_shared_fit(
    completed,
):
    owner, report, _ = completed
    assert report["prediction_work"] == dict(attempted=33, returned=33, validated=33)
    assert len(owner.predictions) == len(report["native_predictions"]) == 33
    for arm in report["arms"]:
        native, evidence = arm["native_work_delta"], arm["evidence_work_delta"]
        assert native["forward_calls"] == (0 if arm["name"] == "retrieval_only" else 11)
        assert all(
            native[key] == 0
            for key in (
                "training_attempts",
                "backward_calls",
                "optimizer_steps",
                "committed_updates",
            )
        )
        assert evidence["writes_attempted"] == evidence["writes_completed"] == 0
        count = 0 if arm["name"] == "adapter_only" else 11
        assert evidence["retrieval_attempts"] == evidence["retrievals_completed"] == count
        assert evidence["metadata_scans"] == 24 * count
        assert arm["training_rows_available"] == 24 and arm["context_edits"] == 0
        assert arm["elapsed_seconds"] >= 0
    assert report["evidence_work"]["writes_completed"] == 24
    assert report["evidence_work"]["retrievals_completed"] == 33
    assert report["evidence_work"]["metadata_scans"] == 792
    assert (
        sum(arm["native_work_delta"]["forward_calls"] for arm in report["arms"]) + 24
        == report["native_work"]["forward_calls"]
    )


def test_should_augment_only_context_arms_and_keep_candidate_identity_outside_editable_json(
    completed,
):
    _, report, _ = completed
    original_candidates = asdict(request())["allowed_candidates"]
    for arm in report["arms"]:
        if arm["name"] == "adapter_only":
            assert "evidence_predictor" not in arm
            continue
        calls = arm["evidence_predictor"]["calls"]
        assert len(calls) == 11
        expected_callback = 0 if arm["name"] == "retrieval_only" else 11
        assert arm["evidence_predictor"]["work"]["callback_attempted"] == expected_callback
        assert arm["evidence_predictor"]["work"]["callback_returned"] == expected_callback
        for call in calls:
            assert call["request"]["allowed_candidates"] == original_candidates
            assert len(call["selection"]["items"]) == 1
            assert len(call["selection"]["context"].encode("utf-8")) <= 768
            if expected_callback:
                supplied = call["supplied"]
                assert supplied["allowed_candidates"] == call["request"]["allowed_candidates"]
                assert supplied["request_id"] == call["request"]["request_id"]
                assert (
                    supplied["query"]
                    == call["request"]["query"] + " evidence " + call["selection"]["context"]
                )
                data = json.loads(call["selection"]["context"])["decision_evidence"][0]
                assert "instructions" not in data and "allowed_candidates" not in data
            else:
                assert "supplied" not in call
    for index, row in enumerate(report["native_predictions"]):
        assert (" evidence " in row["request"]["query"]) is (index < 11 or index >= 22)


def test_should_preserve_a_context_models_wrong_choice_instead_of_replacing_it_with_retrieved_truth():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}

    def ignore_context(_owner, supplied, result):
        if " evidence " in supplied.query and supplied.query.startswith("ready "):
            return ProbabilityDecisionResult(
                supplied.request_id,
                "route",
                tuple(
                    (candidate.candidate_id, float(candidate.candidate_id == "route"))
                    for candidate in supplied.allowed_candidates
                ),
            )
        return result

    owner.on_prediction = ignore_context
    comparison.run_comparison(owner, report, lambda _event: None)
    assert report["sequence_complete"] is True
    assert (
        report["summary"]["frozen_context"]["successes"]
        == report["summary"]["hybrid"]["successes"]
        == 0
    )
    assert (
        report["summary"]["retrieval_only"]["successes"]
        == report["summary"]["adapter_only"]["successes"]
        == 2
    )
    for arm in (report["arms"][0], report["arms"][-1]):
        ready = [
            call
            for call in arm["evidence_predictor"]["calls"]
            if call["request"]["query"].startswith("ready ")
        ]
        assert ready and all(call["result"]["selected_id"] == "route" for call in ready)
    assert len(owner.training) == 24 and len(owner.predictions) == 33


@pytest.mark.parametrize(
    "fault", ["consent", "observed-time", "arrival-time", "source-id", "boolean-update"]
)
def test_should_refuse_unbound_evidence_metadata_without_touching_the_original_receipts(
    completed, fault
):
    _, report, _ = completed
    rows = deepcopy(report["delivery"])
    if fault == "consent":
        rows[0]["permissions"]["replay"] = False
    elif fault == "observed-time":
        rows[0]["observed_at"] = 1
    elif fault == "arrival-time":
        rows[0]["receipt"]["arrived_at"] = 2
    elif fault == "source-id":
        rows[0]["source_example_id"] = "foreign"
    else:
        rows[0]["receipt"]["update_number"] = True
    before = deepcopy(report["delivery"])
    with pytest.raises(ValueError):
        comparison.build_evidence(ticket._training(), rows)
    assert report["delivery"] == before


@pytest.mark.parametrize("fault", ["short", "role", "group", "query"])
def test_should_admit_the_fixed_training_exposures_before_any_snapshot_or_update(fault):
    examples = list(deepcopy(ticket._training()))
    if fault == "short":
        examples.pop()
    elif fault == "role":
        object.__setattr__(examples[0], "role", "outer_selection")
    elif fault == "group":
        object.__setattr__(examples[0], "source_group", "development-workflow-traces")
    else:
        examples[0] = replace(
            examples[0], request=replace(examples[0].request, query="foreign please")
        )
    owner = WorkflowLearner()
    with pytest.raises(ValueError):
        comparison.run_comparison(owner, {}, lambda _event: None, training_examples=tuple(examples))
    assert owner.training == owner.predictions == owner.captures == []


@pytest.mark.parametrize("options", [{"evidence_reuse": 1}, {"evaluate_sources": False}])
def test_should_refuse_invalid_posttrain_hook_options_before_any_work(options):
    owner = WorkflowLearner()
    with pytest.raises(ValueError):
        ticket.run_sequence(owner, {}, lambda _event: None, **options)
    assert owner.training == owner.predictions == owner.captures == []


def test_should_keep_default_observation_consent_unchanged_when_using_the_optional_hook():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}
    seen: list[Any] = []

    def observe(learner, states, observed, _progress, verify):
        verify()
        seen.append((learner, tuple(states), len(observed["delivery"])))
        assert learner.work["committed_updates"] == 24

    ticket.run_sequence(owner, report, lambda _event: None, evaluate_sources=observe)
    assert seen == [(owner, (0, 24), 24)]
    assert all("permissions" not in row for row in report["delivery"])
    assert report["sequence_complete"] is True and report["episodes"] == []
    assert owner.predictions == owner.restores == []
    assert report["snapshot_work"]["released_snapshots"] == 2


@pytest.mark.parametrize("fault", ["unreturned", "postreturn-state"])
def test_should_keep_the_first_incomplete_context_episode_and_stop_before_later_arms(fault):
    owner = WorkflowLearner()
    report: dict[str, Any] = {}

    def unavailable(_owner, _supplied, _result):
        raise RuntimeError("controlled first context return failure")

    if fault == "unreturned":
        owner.on_prediction = unavailable

    def progress(event):
        if fault == "postreturn-state" and event == "typed_prediction_validated":
            owner.version += 1

    with pytest.raises(RuntimeError):
        comparison.run_comparison(owner, report, progress)
    assert len(owner.training) == 24 and len(owner.predictions) == 1
    assert (
        report["sequence_complete"] is False and len(report["arms"]) == len(report["episodes"]) == 1
    )
    assert report["episodes"][0]["episode"]["complete"] is False
    assert report["arms"][0]["native_work_delta"]["forward_calls"] == 1
    assert report["arms"][0]["evidence_work_delta"]["retrieval_attempts"] == 1
    assert report["snapshot_work"]["released_snapshots"] == 2
    assert report["prediction_work"] == dict(
        attempted=1, returned=int(fault != "unreturned"), validated=0
    )
    assert ("returned" in report["native_predictions"][0]) is (fault != "unreturned")


def test_should_refuse_mutated_paid_receipts_before_the_next_episode():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}

    def progress(event):
        if event == "comparison_episode_finished":
            report["delivery"][0]["receipt"]["update_number"] = 99

    with pytest.raises(ValueError):
        comparison.run_comparison(owner, report, progress)
    assert len(owner.training) == 24 and len(owner.predictions) == 3
    assert len(report["episodes"]) == 1 and report["episodes"][0]["episode"]["complete"] is True
    assert report["arms"][0]["native_work_delta"]["forward_calls"] == 3
    assert report["evidence_work"]["retrievals_completed"] == 3
    assert report["sequence_complete"] is False


def test_should_stop_at_a_later_restore_failure_without_losing_completed_native_and_retrieval_arms():
    owner = WorkflowLearner(fail_restore=2)
    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError):
        comparison.run_comparison(owner, report, lambda _event: None)
    assert len(owner.training) == 24 and len(owner.predictions) == 11
    assert len(report["episodes"]) == 6 and len(report["arms"]) == 3
    assert report["arms"][0]["native_work_delta"]["forward_calls"] == 11
    assert report["arms"][1]["native_work_delta"]["forward_calls"] == 0
    assert report["evidence_work"]["retrievals_completed"] == 22
    assert (
        report["snapshot_work"]["restore_attempts"] == 2
        and report["snapshot_work"]["restores_completed"] == 1
    )
    assert (
        report["sequence_complete"] is False and report["snapshot_work"]["released_snapshots"] == 2
    )


def test_should_refuse_missing_reuse_consent_before_constructing_evidence_or_restoring_sources():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}

    def progress(event):
        if event == "training_finished" and len(report["delivery"]) == 24:
            report["delivery"][0]["permissions"]["replay"] = False

    with pytest.raises(ValueError):
        comparison.run_comparison(owner, report, progress)
    assert len(owner.training) == 24 and owner.predictions == owner.restores == []
    assert "evidence_events" not in report and report["episodes"] == []
    assert report["app_committed_updates"] == 24 and report["sequence_complete"] is False


def test_should_preserve_completed_evidence_writes_when_a_later_add_return_is_lost(monkeypatch):
    original_add = comparison.DecisionEvidenceStore.add

    def lose_sixth_return(store, value):
        result = original_add(store, value)
        if len(store.events) == 6:
            raise RuntimeError("paid sixth evidence return unavailable")
        return result

    monkeypatch.setattr(comparison.DecisionEvidenceStore, "add", lose_sixth_return)
    owner = WorkflowLearner()
    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError):
        comparison.run_comparison(owner, report, lambda _event: None)
    assert len(owner.training) == 24 and owner.predictions == owner.restores == []
    assert len(report["evidence_events"]) == 6
    assert (
        report["evidence_work"]["writes_attempted"]
        == report["evidence_work"]["writes_completed"]
        == 6
    )
    assert report["evidence_storage_bytes"] > 0
    assert report["episodes"] == [] and report["sequence_complete"] is False
    assert report["snapshot_work"]["released_snapshots"] == 2


def test_should_refuse_existing_cli_output_before_loading_any_native_owner(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("existing result must refuse before model loading")

    monkeypatch.setattr(comparison, "_load", forbidden)
    output = tmp_path / "paid.json"
    output.write_bytes(b"original paid output")
    with pytest.raises(FileExistsError):
        comparison.main(
            ["--source", str(tmp_path), "--checkpoint", str(tmp_path), "--output", str(output)]
        )
    assert output.read_bytes() == b"original paid output"


def boundary_owner():
    # Declared scalar birth history isolates the readout gate; no TRAIN is run.
    owner = WorkflowLearner()
    owner._work = dict.fromkeys(owner._work, 24)
    owner.live_checkpoint = owner.version = 24
    return owner


@pytest.mark.parametrize("maximum", [0, True, 129])
def test_should_refuse_invalid_explicit_prediction_caps_before_native_work(maximum):
    owner = boundary_owner()
    with pytest.raises(ValueError):
        ticket.BoundedWorkflowPredictor(owner, {}, lambda _event: None, max_predictions=maximum)
    assert owner.predictions == []


def test_should_preserve_a_spent_limit_and_bind_the_larger_cap_only_to_a_fresh_history():
    owner = boundary_owner()
    owner._work["forward_calls"] = 60
    report: dict[str, Any] = {"prediction_work": dict(attempted=36, returned=36, validated=36)}
    default = ticket.BoundedWorkflowPredictor(owner, report, lambda _event: None)
    with pytest.raises(ValueError):
        default(request())
    assert owner.predictions == []
    with pytest.raises(ValueError):
        ticket.BoundedWorkflowPredictor(owner, report, lambda _event: None, max_predictions=54)
    assert report["prediction_work"] == dict(attempted=36, returned=36, validated=36)
    assert report["prediction_limit"] == 36 and owner.predictions == []
    fresh_owner = boundary_owner()
    fresh_report: dict[str, Any] = {}
    selected = ticket.BoundedWorkflowPredictor(
        fresh_owner, fresh_report, lambda _event: None, max_predictions=54
    )
    assert selected(request()).selected_id == "open"
    assert fresh_report["prediction_work"] == dict(attempted=1, returned=1, validated=1)
    assert fresh_report["prediction_limit"] == 54
    assert len(fresh_owner.predictions) == 1 and fresh_owner.work["forward_calls"] == 25
    fresh_report["prediction_limit"] = 36
    with pytest.raises(ValueError):
        selected(request())
    assert len(fresh_owner.predictions) == 1


def test_should_refuse_larger_caps_for_spent_legacy_history_without_limit_metadata():
    owner = boundary_owner()
    owner._work["forward_calls"] = 60
    report: dict[str, Any] = {"prediction_work": dict(attempted=36, returned=36, validated=36)}
    with pytest.raises(ValueError):
        ticket.BoundedWorkflowPredictor(owner, report, lambda _event: None, max_predictions=54)
    assert "prediction_limit" not in report and owner.predictions == []
    default = ticket.BoundedWorkflowPredictor(owner, report, lambda _event: None)
    with pytest.raises(ValueError):
        default(request())
    assert report["prediction_limit"] == 36 and owner.predictions == []


def test_should_refuse_boolean_bound_limit_even_when_it_equals_the_requested_integer():
    owner = boundary_owner()
    with pytest.raises(ValueError):
        ticket.BoundedWorkflowPredictor(
            owner, {"prediction_limit": True}, lambda _event: None, max_predictions=1
        )
    report: dict[str, Any] = {}
    selected = ticket.BoundedWorkflowPredictor(
        owner, report, lambda _event: None, max_predictions=1
    )
    report["prediction_limit"] = True
    with pytest.raises(ValueError):
        selected(request())
    assert owner.predictions == []
