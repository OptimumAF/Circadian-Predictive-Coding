"""Frozen reserved-text delivery and disposable ownership qualification.

Scalar fakes make no learning claim. One tiny native fixture exercises the real
fork/inbox/predict path; only the root executor imports its optional Torch backend.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from math import log
from typing import Any

import pytest

from scripts import run_open_jev_reserved_probe as runner
from src.core.native_choice_probe import probe_examples, validate_probe_examples
from src.core.native_choice_sequence import sequence_examples
from src.core.typed_decision import ProbabilityDecisionResult
from tests.test_native_choice_sequence import RecordingLearner


@pytest.fixture(scope="module")
def corpus():
    return probe_examples()


def test_should_freeze_two_new_train_two_test_and_four_original_development_records(corpus):
    assert type(corpus) is tuple and len(corpus) == len({item.example_id for item in corpus}) == 8
    assert len({item.request.request_id for item in corpus}) == 8
    new = [item for item in corpus if item.context == "C"]
    assert {(item.role, item.request.query, item.target_id) for item in new} == {
        ("probe_train", "Case marker: tavrix", "billing"),
        ("probe_train", "Intake code tavrix. Route this request.", "billing"),
        ("probe_test", "Case marker: tavrix; please assist.", "billing"),
        ("probe_test", "A support ticket marked tavrix needs routing.", "billing"),
    }
    original = sequence_examples()
    retained = tuple(
        item
        for item in original
        if item.role == "outer_selection" and item.context in ("A", "anchor")
    )
    assert tuple(item for item in corpus if item.role == "outer_selection") == retained
    assert not {item.request.query for item in new}.intersection(
        item.request.query for item in original
    )
    validate_probe_examples(corpus)
    with pytest.raises(FrozenInstanceError):
        setattr(corpus[0], "target_id", "password")


@pytest.mark.parametrize("fault", ["final", "duplicate", "target", "query", "group", "catalog"])
def test_should_refuse_altered_frozen_roles_targets_queries_groups_or_catalog(corpus, fault):
    changed = deepcopy(corpus)
    target = next(item for item in changed if item.role == "probe_train")
    if fault == "final":
        object.__setattr__(target, "role", "final_test")
    elif fault == "duplicate":
        changed = (changed[0],) + changed[:-1]
    elif fault == "target":
        object.__setattr__(target, "target_id", "password")
    elif fault == "query":
        object.__setattr__(target.request, "query", "Case marker: an-unreserved-marker")
    elif fault == "group":
        object.__setattr__(target, "source_group", "foreign_examples")
    else:
        object.__setattr__(
            target.request.allowed_candidates[0], "description", "foreign description"
        )
    with pytest.raises(ValueError):
        validate_probe_examples(changed)


class ProbeChild(RecordingLearner):
    """Explicit inherited6 baseline, then canned scalar probe observations."""

    def __init__(self, *, fail_train=None, fail_predict=None, invalid_predict=False):
        super().__init__([], fail_at=fail_train)
        self._work = dict.fromkeys(self._work, 6)
        self.last_update = dict(loss=0.5, gradient_norm_before_clip=0.25)
        self.inputs: list[Any] = []
        self.fail_predict, self.invalid_predict = fail_predict, invalid_predict
        self.report: dict[str, Any] | None = None

    @property
    def optimizer_reset_work(self):
        return dict(optimizer_reset_attempts=0, optimizer_resets_completed=0)

    @property
    def update_health_work(self):
        return dict(
            update_health_attempts=0,
            parameter_copy_elements_attempted=0,
            gradient_reduction_elements_attempted=0,
            step_reduction_elements_attempted=0,
            update_health_reports_completed=0,
        )

    def predict(self, features):
        assert set(vars(features)) == {"request_id", "query", "allowed_candidates"}
        if self.report is not None:
            assert all(
                not {"target", "correct", "nll", "nll_status"}.intersection(row)
                for row in self.report["readouts"]
            )
        self.inputs.append(deepcopy(features))
        self._work["forward_calls"] += 1
        if len(self.inputs) == self.fail_predict:
            raise RuntimeError("controlled unreturned typed prediction")
        query = features.query
        if "tavrix" in query:
            selected = "billing" if len(self.training) == 6 else "password"
        elif "zorbex" in query:
            selected = "shipping"
        else:
            selected = "password" if "password" in query else "billing"
        result = ProbabilityDecisionResult(
            features.request_id,
            selected,
            tuple(
                (item.candidate_id, 0.8 if item.candidate_id == selected else 0.1)
                for item in features.allowed_candidates
            ),
        )
        if self.invalid_predict:
            object.__setattr__(result, "selected_id", "foreign")
        return result


@pytest.fixture(scope="module")
def completed():
    child = ProbeChild()
    report: dict[str, Any] = {}
    child.report = report
    events: list[Any] = []
    runner.run_probe_updates(child, report, events.append)
    return child, report, events


def test_should_score_before_and_after_but_train_only_the_two_reserved_train_templates(
    completed, corpus
):
    child, report, _ = completed
    assert [(request.query, target) for request, target in child.training] == [
        ("Case marker: tavrix", "billing"),
        ("Intake code tavrix. Route this request.", "billing"),
    ] * 3
    selected = [item for item in corpus if item.role != "probe_train"]
    assert [request.query for request in child.inputs] == [
        item.request.query for item in selected
    ] * 2
    assert len(report["readouts"]) == 12 and report["probe_complete"] is True
    assert (
        report["prediction_work"]["attempted"]
        == report["prediction_work"]["returned"]
        == report["prediction_work"]["validated"]
        == 12
    )
    assert report["probe_app_committed_updates"] == 6
    assert child.work == dict(
        training_attempts=12,
        forward_calls=24,
        backward_calls=12,
        optimizer_steps=12,
        committed_updates=12,
    )


def test_should_keep_fresh_six_update_scope_delayed_receipts_and_explicit_probe_train_role(
    completed,
):
    _, report, _ = completed
    rows = report["probe_delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 6
    assert [row["observed_at"] for row in rows] == list(range(0, 12, 2))
    assert all(row["prelabel_updates"] == 0 for row in rows)
    receipts = [row["receipt"] for row in rows]
    assert [row["update_number"] for row in receipts] == list(range(1, 7))
    assert [row["arrived_at"] for row in receipts] == list(range(1, 12, 2))
    assert all(row["arrived_at"] == row["applied_at"] for row in receipts)
    assert all(row["role"] == "probe_train" for row in rows)
    assert all("replay_source_exposure_id" not in row for row in rows)


def test_should_score_after_all_observations_with_acquisition_distinct_from_zero_shot_success(
    completed,
):
    _, report, _ = completed
    summary = report["summary"]
    assert summary["by_phase"]["before"]["C"]["accuracy"] == 0.0
    assert summary["by_phase"]["after"]["C"]["accuracy"] == 1.0
    assert summary["by_phase"]["before"]["C"]["nll_mean"] == pytest.approx(-log(0.1))
    assert summary["by_phase"]["after"]["C"]["nll_mean"] == pytest.approx(-log(0.8))
    assert all(
        {"target", "correct", "nll", "nll_status"}.issubset(row) for row in report["readouts"]
    )
    assert summary["criteria"] == dict(
        C_endpoint_success=True, C_acquired=True, A_retained=True, anchors_preserved=True
    )
    assert summary["contrasts"]["C"]["accuracy"] == 1.0
    assert summary["contrasts"]["C"]["nll_mean"] == pytest.approx(log(0.1) - log(0.8))
    assert summary["recovery"] == "not_tested"


def set_probabilities(row, selected, values):
    row["result"]["selected_id"] = selected
    row["result"]["probabilities"] = tuple(
        (key, values[key]) for key, _ in row["result"]["probabilities"]
    )


def test_should_distinguish_successful_c_zero_shot_endpoint_from_new_acquisition(completed):
    rows = deepcopy(completed[1]["readouts"])
    for row in rows:
        if row["phase"] == "before" and row["context"] == "C":
            set_probabilities(row, "billing", dict(billing=0.8, password=0.1, shipping=0.1))
    summary = runner.summarize_probe(rows)
    assert summary["criteria"]["C_endpoint_success"] is True
    assert summary["criteria"]["C_acquired"] is False
    assert summary["contrasts"]["C"]["accuracy"] == 0.0


@pytest.mark.parametrize("probability", [0.0, 1e-300], ids=["zero", "tiny-unclipped"])
def test_should_keep_zero_or_tiny_target_probability_without_clipping_or_dropping_cases(
    completed, probability
):
    rows = deepcopy(completed[1]["readouts"])
    for row in rows:
        if row["phase"] == "before" and row["context"] == "C":
            set_probabilities(
                row, "password", dict(billing=probability, password=1.0, shipping=0.0)
            )
    summary = runner.summarize_probe(rows)
    group = summary["by_phase"]["before"]["C"]
    assert group["examples"] == 2 and group["accuracy"] == 0.0
    if probability == 0:
        assert group["nll_mean"] is None and group["nll_status"] == "infinite"
        assert summary["contrasts"]["C"]["nll_mean"] is None
        assert summary["contrasts"]["C"]["nll_status"] == "nonfinite_comparison_unavailable"
    else:
        assert group["nll_mean"] == pytest.approx(-log(probability))
        assert group["nll_mean"] > 600
    import json

    json.dumps(summary, allow_nan=False)


def test_should_keep_a_retention_and_anchor_drift_failures_even_when_c_is_acquired(completed):
    rows = deepcopy(completed[1]["readouts"])
    first_a = next(row for row in rows if row["phase"] == "after" and row["context"] == "A")
    set_probabilities(first_a, "billing", dict(billing=0.8, password=0.1, shipping=0.1))
    for row in rows:
        if row["phase"] == "after" and row["context"] == "anchor":
            target = row["target"]
            set_probabilities(
                row,
                target,
                {key: 0.6 if key == target else 0.2 for key, _ in row["result"]["probabilities"]},
            )
    summary = runner.summarize_probe(rows)
    assert summary["criteria"]["C_acquired"] is True
    assert summary["criteria"]["A_retained"] is False
    assert summary["criteria"]["anchors_preserved"] is False
    assert summary["by_phase"]["after"]["A"]["examples"] == 2


@pytest.mark.parametrize("fault", ["partial", "duplicate", "role", "request", "result"])
def test_should_refuse_incomplete_or_replaced_reserved_readout_evidence(completed, fault):
    rows = deepcopy(completed[1]["readouts"])
    if fault == "partial":
        rows.pop()
    elif fault == "duplicate":
        rows[1]["example_id"] = rows[0]["example_id"]
    elif fault == "role":
        rows[0]["role"] = "outer_selection"
    elif fault == "request":
        rows[0]["request"]["query"] = "unreserved replacement"
    else:
        rows[0]["result"]["selected_id"] = "foreign"
    with pytest.raises(ValueError):
        runner.summarize_probe(rows)


@pytest.mark.parametrize("fault", ["unreturned", "invalid_return", "fourth_training"])
def test_should_keep_paid_probe_prefix_and_distinguish_unreturned_from_unvalidated_predictions(
    fault,
):
    child = ProbeChild(
        fail_train=4 if fault == "fourth_training" else None,
        fail_predict=2 if fault == "unreturned" else None,
        invalid_predict=fault == "invalid_return",
    )
    report: dict[str, Any] = {}
    child.report = report
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_probe_updates(child, report, lambda event: None)
    attempted, returned, validated, trains, committed = {
        "unreturned": (2, 1, 1, 0, 0),
        "invalid_return": (1, 1, 0, 0, 0),
        "fourth_training": (6, 6, 6, 4, 3),
    }[fault]
    assert report["prediction_work"] == dict(
        attempted=attempted, returned=returned, validated=validated
    )
    assert len(child.inputs) == attempted and len(child.training) == trains
    assert report["probe_app_committed_updates"] == committed
    assert report["child_incremental_work"]["training_attempts"] == trains
    assert report["child_incremental_work"]["committed_updates"] == committed
    assert report["child_incremental_work"]["forward_calls"] == attempted + trains
    assert report["probe_complete"] is False and "summary" not in report
    assert all("target" not in row for row in report["readouts"])
    if fault == "invalid_return":
        assert report["readouts"][0]["result"]["selected_id"] == "foreign"
    elif fault == "unreturned":
        assert "result" not in report["readouts"][-1]
    else:
        assert sum("receipt" in row for row in report["probe_delivery"]) == 3


def test_should_keep_first_paid_probe_receipt_when_resource_callback_stops_before_later_work():
    child = ProbeChild()
    report: dict[str, Any] = {}
    child.report = report

    def progress(event):
        if event == "probe_training_finished":
            raise RuntimeError("controlled postcommit probe stop")

    with pytest.raises(RuntimeError, match="controlled postcommit probe stop"):
        runner.run_probe_updates(child, report, progress)
    assert len(child.inputs) == 6 and len(child.training) == 1
    assert report["probe_app_committed_updates"] == 1
    assert report["probe_delivery"][0]["receipt"]["update_number"] == 1
    assert report["probe_delivery"][0]["training"] == child.last_update
    assert report["probe_complete"] is False


def test_should_refuse_changed_reserved_corpus_before_inbox_or_child_work(monkeypatch, corpus):
    changed = deepcopy(corpus)
    object.__setattr__(changed[0], "role", "final_test")
    monkeypatch.setattr(runner, "probe_examples", lambda: changed)
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("changed reserved corpus reached child work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    child = ProbeChild()
    with pytest.raises(ValueError):
        runner.run_probe_updates(child, {}, lambda event: None)
    assert calls == [] and child.training == child.inputs == []


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip("torch", reason="native probe backend unavailable; not a Torch pass")


def native_source(torch, *, source_history="after_A6"):
    from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
    from src.core.native_choice_sequence import INSTRUCTIONS
    from tests.test_open_jev_training import constant_model

    model = constant_model(torch)
    model.register_buffer("running", torch.tensor([0.25, 0.5], dtype=torch.float32))
    model.eval()
    original_forward = type(model).forward

    def forward(self, records):
        converted = deepcopy(records)
        converted[0]["answer_keys"] = [
            {"billing": "a", "password": "b", "shipping": "c"}[key]
            for key in converted[0]["answer_keys"]
        ]
        # An unbound function preserves the fork's self/parameter ownership.
        return original_forward(self, converted)

    type(model).forward = forward

    def compile_request(state, questions):
        question = questions["decision"]
        return [dict(kind=question["type"], state=state, answer_keys=list(question["criteria"]))]

    source = OpenJevTrainingLearner(
        model,
        compile_request,
        instructions=INSTRUCTIONS,
        policy=OpenJevTrainingPolicy(6 if source_history == "cold" else 12, 1, max_forks=1),
    )
    training = [
        item for item in sequence_examples() if item.role == "train" and item.context == "A"
    ]
    for _ in range(0 if source_history == "cold" else 3):
        for example in training:
            source.train_batch(example.request, example.target_id)
    return source, model


def test_should_run_one_real_tiny_native_fork_probe_and_leave_all_source_state_unchanged(
    torch_backend,
):
    from tests.test_open_jev_training import assert_tensor_tree_equal

    torch = torch_backend
    source, model = native_source(torch)
    before = source.snapshot_state()
    buffer = model.running.detach().clone()
    caller_rng = torch.get_rng_state().clone()
    last_update, records, work = deepcopy(source.last_update), deepcopy(model.records), source.work
    report: dict[str, Any] = {}
    runner.run_fresh_probe(source, report, lambda event: None)
    assert report["probe_complete"] is report["source_preserved"] is True
    assert source.work == report["source_native_work"] == work == dict.fromkeys(work, 6)
    assert report["child_birth_work"] == work
    assert report["child_incremental_work"] == dict(
        training_attempts=6,
        forward_calls=18,
        backward_calls=6,
        optimizer_steps=6,
        committed_updates=6,
    )
    assert report["child_native_work"] == dict(
        training_attempts=12,
        forward_calls=24,
        backward_calls=12,
        optimizer_steps=12,
        committed_updates=12,
    )
    assert report["probe_app_committed_updates"] == 6 and len(report["readouts"]) == 12
    after = source.snapshot_state()
    for name in ("weights", "gradients", "optimizer", "rng"):
        assert_tensor_tree_equal(torch, getattr(before, name), getattr(after, name))
    assert torch.equal(model.running, buffer) and torch.equal(torch.get_rng_state(), caller_rng)
    assert source.last_update == last_update and model.records == records
    assert model.training is False
    assert report["fork_work"]["fork_attempts"] == report["fork_work"]["forks_completed"] == 1
    assert 0 < report["source_preservation_copy_elements_attempted"] <= 20_000_000
    assert 0 < report["source_preservation_comparison_elements_attempted"] <= 40_000_000


def test_should_run_source_proof_even_when_callback_mutates_source_and_stops_child_before_probe_work(
    torch_backend,
):
    torch = torch_backend
    source, model = native_source(torch)
    report: dict[str, Any] = {}
    boundaries: list[str] = []

    def corrupt(event):
        boundaries.append(event)
        if event == "probe_fork_created":
            with torch.no_grad():
                model.head.weight.add_(0.125)
            raise RuntimeError("controlled source mutation after native fork")

    with pytest.raises(ValueError, match="changed source"):
        runner.run_fresh_probe(source, report, corrupt)
    assert boundaries == ["probe_fork_created"]
    assert report["probe_complete"] is report["source_preserved"] is False
    assert "controlled source mutation" in report["probe_error"]
    assert "changed source" in report["source_preservation_error"]
    assert not report.get("probe_delivery") and not report.get("readouts")
    assert report["child_native_work"] == report["child_birth_work"] == source.work
    assert report["child_incremental_work"] == dict.fromkeys(source.work, 0)
    assert report["fork_work"]["forks_completed"] == 1
