"""Bounded workflow orchestration with scalar owners and one literal port check.

The separately selected tensor test has no randomized initializer or TRAIN work.
All other tests use opaque scalar snapshots and no tensor backend.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import exp
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_open_jev_ticket_workflow as runner
from scripts.run_ticket_workflow import training_cases
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.ticket_workflow import WorkflowCapabilities, WorkflowState, workflow_request
from src.core.typed_decision import ProbabilityDecisionResult


TOOLS = ("commit", "defer", "inspect", "open", "route")
RULES = {"invoice": "inspect", "legacy": "defer", "parcel": "defer", "password": "inspect"}


class ScalarRng:
    """A cloneable scalar stand-in, never a tensor or a random generator."""

    def __init__(self, value=17):
        self.value = value

    def clone(self):
        return ScalarRng(self.value)


class ScalarBackend:
    def __init__(self):
        self.global_rng = ScalarRng()

    def get_rng_state(self):
        return self.global_rng

    @staticmethod
    def equal(left, right):
        return left.value == right.value


class WorkflowLearner:
    """Records native-shaped work; the controlled policy is not learned."""

    def __init__(self, *, fail_train=None, fail_capture=None, fail_restore=None):
        self.timeline: list[Any] = []
        self.training: list[Any] = []
        self.predictions: list[Any] = []
        self.captures: list[Any] = []
        self.restores: list[Any] = []
        self._work = dict.fromkeys(
            (
                "training_attempts",
                "forward_calls",
                "backward_calls",
                "optimizer_steps",
                "committed_updates",
            ),
            0,
        )
        self.last_update: dict[str, float] | None = None
        self.live_checkpoint = 0
        self.version = 0
        self._torch = ScalarBackend()
        self._rng = ScalarRng()
        self.model = SimpleNamespace(training=False)
        self.fail_train, self.fail_capture, self.fail_restore = (
            fail_train,
            fail_capture,
            fail_restore,
        )
        self.on_prediction: Any = None
        self._poisoned = False

    @property
    def work(self):
        return dict(self._work)

    def _guard(self):
        if self._poisoned:
            raise ValueError("fixture source is poisoned")

    def _trainable_versions(self):
        return (("fixture", self.version),)

    def train_batch(self, request, target):
        self.timeline.append(("train", request.request_id))
        self.training.append((deepcopy(request), target))
        for key in ("training_attempts", "forward_calls", "backward_calls", "optimizer_steps"):
            self._work[key] += 1
        if len(self.training) == self.fail_train:
            raise TrainingUpdateFailed(
                "controlled workflow TRAIN failure", "scalar_workflow_fixture", self.work
            )
        self._work["committed_updates"] += 1
        self.live_checkpoint = self._work["committed_updates"]
        self.version += 1
        self.last_update = {"loss": 0.5, "gradient_norm_before_clip": 0.25}
        return TrainingDiagnostic("scalar_workflow_fixture", 0.5)

    def snapshot_state(self):
        self.timeline.append(("capture", self.live_checkpoint))
        self.captures.append(self.live_checkpoint)
        if len(self.captures) == self.fail_capture:
            raise RuntimeError("controlled snapshot failure")
        return {"fixture_checkpoint": self.live_checkpoint}

    def restore_state(self, snapshot):
        self.restores.append(deepcopy(snapshot))
        self.timeline.append(("restore", snapshot["fixture_checkpoint"]))
        if len(self.restores) == self.fail_restore:
            raise RuntimeError("controlled restore failure")
        self.live_checkpoint = snapshot["fixture_checkpoint"]
        self.version += 1

    def predict(self, request):
        self.timeline.append(("predict", self.live_checkpoint))
        self.predictions.append(deepcopy(request))
        self._work["forward_calls"] += 1
        words = request.query.split()
        selected = (
            "open"
            if words[0] == "sealed"
            else "commit"
            if words[0] == "staged"
            else RULES[words[1]]
        )
        result = ProbabilityDecisionResult(
            request.request_id,
            selected,
            tuple(
                (candidate.candidate_id, float(candidate.candidate_id == selected))
                for candidate in request.allowed_candidates
            ),
        )
        if self.on_prediction is not None:
            return self.on_prediction(self, request, result)
        return result


def request():
    return workflow_request(
        WorkflowState("fixture-ticket", "invoice"),
        WorkflowCapabilities(("commit", "defer", "inspect", "open", "retire", "route")),
        "fixture-decision",
    )


@pytest.fixture(scope="module")
def completed():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}
    events: list[Any] = []
    runner.run_sequence(owner, report, events.append)
    return owner, report, events


def test_should_complete_twenty_four_delayed_train_exposures_before_any_development_prediction(
    completed,
):
    owner, report, _ = completed
    original = training_cases()
    expected = [
        (replace(case.request, request_id=f"pass{p}-{case.request.request_id}"), case.target_id)
        for p in range(2)
        for case in original
    ]
    assert owner.training == expected
    assert len(report["training_cases"]) == 24
    assert report["training_cases"] == [
        dict(asdict(case), request=asdict(query))
        for (query, _), case in zip(expected, original * 2)
    ]
    first_predict = next(i for i, event in enumerate(owner.timeline) if event[0] == "predict")
    assert sum(event[0] == "train" for event in owner.timeline[:first_predict]) == 24
    assert report["sequence_complete"] is True and report["app_committed_updates"] == 24
    assert report["logical_tick"] == 48 and report["inbox_stopped"] is False
    assert report["native_work"] == {
        "training_attempts": 24,
        "forward_calls": 46,
        "backward_calls": 24,
        "optimizer_steps": 24,
        "committed_updates": 24,
    }
    json.dumps(report, allow_nan=False)


def test_should_bind_unique_receipts_to_train_only_observations_and_keep_development_unlabeled(
    completed,
):
    owner, report, _ = completed
    rows = report["delivery"]
    assert len({row["exposure_id"] for row in rows}) == 24
    assert [row["observed_at"] for row in rows] == list(range(0, 48, 2))
    assert [row["receipt"]["arrived_at"] for row in rows] == list(range(1, 48, 2))
    assert [row["receipt"]["update_number"] for row in rows] == list(range(1, 25))
    assert all(row["prelabel_updates"] == 0 and row["policy_version"] == "v2" for row in rows)
    assert all(
        row["source_group"] == "train-workflow-traces" and row["phase"] == "workflow"
        for row in rows
    )
    assert all(row["receipt"]["arrived_at"] == row["receipt"]["applied_at"] for row in rows)
    for query in owner.predictions:
        assert set(vars(query)) == {"request_id", "query", "allowed_candidates"}
        assert tuple(candidate.candidate_id for candidate in query.allowed_candidates) == TOOLS
        assert query.query.endswith(" please") and " train" not in query.query
        assert "ticket-" not in query.request_id
    assert all(query.query.endswith(" train") for query, _ in owner.training)


def test_should_restore_two_sources_without_refunding_work_and_release_snapshot_storage(completed):
    owner, report, _ = completed
    assert owner.captures == [0, 24]
    assert owner.restores == [{"fixture_checkpoint": 0}, {"fixture_checkpoint": 24}]
    assert report["snapshot_work"] == {
        "capture_attempts": 2,
        "captures_completed": 2,
        "restore_attempts": 2,
        "restores_completed": 2,
        "retained_numeric_bytes": None,
        "released_snapshots": 2,
    }
    assert owner.live_checkpoint == 24 and owner.work["committed_updates"] == 24
    assert [row["source_updates"] for row in report["episodes"]] == [0] * 3 + [24] * 3
    assert [row["control"] for row in report["episodes"]] == ["unchanged"] * 3 + ["learned"] * 3


def test_should_keep_full_verified_effects_recovery_and_denial_for_both_control_owners(completed):
    _, report, _ = completed
    for control in ("unchanged", "learned"):
        rows = [row for row in report["episodes"] if row["control"] == control]
        assert [row["episode"]["scenario"] for row in rows] == [
            "ordinary",
            "stale_once",
            "revoke_commit",
        ]
        assert all(
            row["role"] == "outer_selection"
            and row["source_group"] == "development-workflow-traces"
            for row in rows
        )
        assert [row["episode"]["success"] for row in rows] == [True, True, False]
        assert [len(row["episode"]["steps"]) for row in rows] == [3, 5, 3]
        assert rows[1]["episode"]["error_recovery_demonstrated"] is True
        denied = rows[2]["episode"]
        assert denied["complete"] is True and denied["stopped_reason"] == "permission_denied"
        assert denied["steps"][-1]["transition"]["status"] == "permission_denied"
        assert denied["final_state"]["committed_tool"] is None
        assert report["summary"][control] == dict(
            episodes=3, successes=2, recoveries=1, permission_denials=1
        )


def test_should_record_batched_native_forward_cost_without_inventing_provider_usage(completed):
    owner, report, _ = completed
    assert report["prediction_work"] == dict(attempted=22, returned=22, validated=22)
    assert len(report["native_predictions"]) == len(owner.predictions) == 22
    assert report["episode_work"]["candidate_entries_returned"] == 110
    assert report["episode_work"]["catalog_entries_considered"] == 132
    assert (
        report["episode_work"]["reported_input_tokens"]
        == report["episode_work"]["reported_output_tokens"]
        == 0
    )
    assert (
        report["episode_work"]["input_tokens_unknown_calls"]
        == report["episode_work"]["output_tokens_unknown_calls"]
        == 22
    )
    assert report["episode_work"]["reported_candidate_comparisons"] == 0
    before = deepcopy(report)
    owner._work["forward_calls"] += 100
    assert report == before
    owner._work["forward_calls"] -= 100


def ready_boundary():
    # This is a declared scalar birth-history double, not twenty-four executed updates.
    owner = WorkflowLearner()
    owner._work = dict.fromkeys(owner._work, 24)
    owner.live_checkpoint = owner.version = 24
    report: dict[str, Any] = {}
    return owner, report


@pytest.mark.parametrize(
    "fault",
    [
        "unreturned",
        "unsupported",
        "nonfinite",
        "request",
        "version",
        "mode",
        "caller-rng",
        "private-rng",
        "training-counter",
        "forward-counter",
    ],
)
def test_should_stop_after_uncertain_native_prediction_and_retain_any_observed_return(fault):
    owner, report = ready_boundary()
    original = request()

    def corrupt(owner, supplied, result):
        if fault == "unreturned":
            raise RuntimeError("controlled native prediction failure")
        if fault == "unsupported":
            return object()
        if fault == "nonfinite":
            object.__setattr__(result, "probabilities", (("open", float("nan")),))
        elif fault == "request":
            object.__setattr__(supplied, "query", "changed native request")
        elif fault == "version":
            owner.version += 1
        elif fault == "mode":
            owner.model.training = True
        elif fault == "caller-rng":
            owner._torch.global_rng.value += 1
        elif fault == "private-rng":
            owner._rng.value += 1
        elif fault == "training-counter":
            owner._work["training_attempts"] += 1
        elif fault == "forward-counter":
            owner._work["forward_calls"] += 1
        return result

    owner.on_prediction = corrupt
    predict = runner.BoundedWorkflowPredictor(owner, report, lambda _event: None)
    with pytest.raises((ValueError, RuntimeError)):
        predict(original)
    assert report["prediction_work"] == dict(
        attempted=1, returned=int(fault != "unreturned"), validated=0
    )
    row = report["native_predictions"][0]
    assert ("returned" in row) is (fault != "unreturned")
    assert row["request"] == asdict(original)
    assert original.query == "sealed feedback initial"
    assert row["error"] and predict.failed is True
    with pytest.raises(ValueError):
        predict(original)
    assert len(owner.predictions) == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["candidate-subset", "wrong-type", "training-history", "spent", "uncertain-prefix"]
)
def test_should_refuse_prediction_admission_before_entering_the_native_owner(fault):
    owner, report = ready_boundary()
    query: Any = request()
    if fault == "candidate-subset":
        query = replace(query, allowed_candidates=query.allowed_candidates[:-1])
    elif fault == "wrong-type":
        query = {"query": "sealed"}
    elif fault == "training-history":
        owner._work["committed_updates"] = 23
    elif fault == "spent":
        report["prediction_work"] = dict.fromkeys(("attempted", "returned", "validated"), 36)
    else:
        report["prediction_work"] = dict(attempted=1, returned=0, validated=0)
    predict = runner.BoundedWorkflowPredictor(owner, report, lambda _event: None)
    with pytest.raises(ValueError):
        predict(query)
    assert owner.predictions == [] and report["native_predictions"] == []


def test_should_refuse_a_late_progress_state_mutation_before_a_second_native_prediction():
    owner, report = ready_boundary()

    def progress(event):
        if event == "typed_prediction_validated":
            owner.version += 1

    predict = runner.BoundedWorkflowPredictor(owner, report, progress)
    with pytest.raises(ValueError):
        predict(request())
    assert report["prediction_work"] == dict(attempted=1, returned=1, validated=0)
    assert report["native_predictions"][0]["result"]["selected_id"] == "open"
    with pytest.raises(ValueError):
        predict(request())
    assert len(owner.predictions) == 1


@pytest.mark.parametrize("fault", ["short", "duplicate", "role", "target"])
def test_should_refuse_changed_demonstration_corpus_before_snapshot_train_or_predict(
    monkeypatch, fault
):
    cases = list(deepcopy(training_cases()))
    if fault == "short":
        cases.pop()
    elif fault == "duplicate":
        cases[-1] = cases[0]
    elif fault == "role":
        object.__setattr__(cases[0], "role", "final")
    else:
        cases[0] = replace(cases[0], target_id="commit")
    monkeypatch.setattr(runner, "training_cases", lambda: tuple(cases))
    owner = WorkflowLearner()
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        runner.run_sequence(owner, report, lambda _event: None)
    assert owner.training == owner.predictions == owner.captures == []


@pytest.mark.parametrize("fault", ["train", "capture-birth", "capture-final", "restore"])
def test_should_keep_partial_native_work_and_snapshot_prefix_without_starting_development(fault):
    owner = WorkflowLearner(
        fail_train=5 if fault == "train" else None,
        fail_capture=1 if fault == "capture-birth" else 2 if fault == "capture-final" else None,
        fail_restore=1 if fault == "restore" else None,
    )
    report: dict[str, Any] = {}
    with pytest.raises((ValueError, RuntimeError, TrainingUpdateFailed)):
        runner.run_sequence(owner, report, lambda _event: None)
    expected_attempts = 5 if fault == "train" else 0 if fault == "capture-birth" else 24
    assert owner.work["training_attempts"] == expected_attempts
    assert report["app_committed_updates"] == (4 if fault == "train" else expected_attempts)
    assert report["native_work"] == owner.work
    assert report["sequence_complete"] is False and report["episodes"] == []
    assert owner.predictions == []
    if fault == "train":
        assert len(report["delivery"]) == 5
        assert report["delivery"][3]["receipt"]["update_number"] == 4
        assert "receipt" not in report["delivery"][4]


@pytest.mark.parametrize("fault", ["receipt", "last-update", "boolean-work", "callback"])
def test_should_preserve_paid_first_receipt_and_refuse_corrupted_prefix_before_more_training(fault):
    owner = WorkflowLearner()
    report: dict[str, Any] = {}

    def progress(event):
        if event != "training_finished":
            return
        if fault == "receipt":
            report["delivery"][0]["receipt"]["update_number"] = 99
        elif fault == "last-update":
            assert owner.last_update is not None
            owner.last_update["loss"] = 0.9
        elif fault == "boolean-work":
            owner._work["committed_updates"] = True
        else:
            raise RuntimeError("paid training callback failure")

    with pytest.raises((ValueError, RuntimeError)):
        runner.run_sequence(owner, report, progress)
    assert len(owner.training) == 1 and owner.predictions == []
    assert report["app_committed_updates"] == 1
    assert report["delivery"][0]["training"] == {"loss": 0.5, "gradient_norm_before_clip": 0.25}
    assert report["delivery"][0]["receipt"]
    assert report["sequence_complete"] is False


def test_should_retain_incomplete_first_episode_and_stop_before_any_later_scene():
    owner = WorkflowLearner()
    report: dict[str, Any] = {}

    def unavailable(_owner, _request, _result):
        raise RuntimeError("first native readout unavailable")

    owner.on_prediction = unavailable
    with pytest.raises(RuntimeError):
        runner.run_sequence(owner, report, lambda _event: None)
    assert len(owner.training) == 24 and len(owner.predictions) == 1
    assert len(report["episodes"]) == 1 and report["sequence_complete"] is False
    episode = report["episodes"][0]["episode"]
    assert episode["complete"] is False and episode["unreturned_prediction_work_unknown"] is True
    assert report["prediction_work"] == dict(attempted=1, returned=0, validated=0)
    assert report["native_work"]["forward_calls"] == 25
    assert report["snapshot_work"]["released_snapshots"] == 2


def test_should_refuse_existing_cli_output_before_loading_native_resources(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("existing output must refuse before native loading")

    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "paid.json"
    output.write_bytes(b"keep existing")
    with pytest.raises(SystemExit) as error:
        runner.main(
            ["--source", str(tmp_path), "--checkpoint", str(tmp_path), "--output", str(output)]
        )
    assert error.value.code == 2 and output.read_bytes() == b"keep existing"


def test_should_match_five_candidate_native_port():
    """Separately selected WSL-only nine-scalar model; no initializer or TRAIN."""
    torch = pytest.importorskip(
        "torch", reason="literal native port requires the declared tensor boundary"
    )
    from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy

    Module: Any = torch.nn.Module

    class Factor(Module):
        def __init__(self, values):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    class Backbone(Module):
        def __init__(self):
            super().__init__()
            self.base = Factor([[1.0, 0.0], [0.0, 1.0]])
            self.lora_A = Factor([[0.1]])
            self.lora_B = Factor([[0.3]])

        def gradient_checkpointing_enable(self):
            pass

        def enable_input_require_grads(self):
            pass

        def forward(self, features):
            base = features @ self.base.weight.T
            correction = (features[:, :1] @ self.lora_A.weight.T) @ self.lora_B.weight.T
            return torch.cat((base[:, :1] + correction, base[:, 1:]), dim=1)

    class Head(Factor):
        def __init__(self):
            super().__init__([[0.2, -0.1]])
            self.bias = torch.nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

        def forward(self, features):
            return (features @ self.weight.T).squeeze(-1) + self.bias

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.lora_rank = 1
            self.backbone = Backbone()
            self.head = Head()
            self.records: list[Any] = []

        def forward(self, records):
            self.records.append(deepcopy(records))
            table = {
                "commit": [1.0, 0.0],
                "defer": [0.0, 1.0],
                "inspect": [-1.0, 1.0],
                "open": [2.0, -1.0],
                "route": [0.5, 0.5],
            }
            features = torch.tensor(
                [table[key] for key in records[0]["answer_keys"]], dtype=torch.float32
            )
            return [self.head(self.backbone(features))]

    compiled: list[Any] = []

    def compile_request(state, questions):
        question = questions["decision"]
        compiled.append(deepcopy((state, questions)))
        return [
            dict(
                kind=question["type"],
                state=state,
                question=question["instructions"],
                answer_keys=list(question["criteria"]),
                options=list(question["criteria"].values()),
            )
        ]

    model = Model()
    model.eval()
    owner = OpenJevTrainingLearner(
        model, compile_request, instructions=runner.INSTRUCTIONS, policy=OpenJevTrainingPolicy(1, 1)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 9
    before = owner.snapshot_state()
    caller_rng = torch.get_rng_state().clone()
    original = request()
    forward = owner.predict(original)
    reversed_result = owner.predict(
        replace(original, allowed_candidates=tuple(reversed(original.allowed_candidates)))
    )
    raw = {"commit": 0.256, "defer": -0.05, "inspect": -0.256, "open": 0.562, "route": 0.103}
    mass = sum(exp(value / 1.518796342858676) for value in raw.values())
    expected = {key: exp(value / 1.518796342858676) / mass for key, value in raw.items()}
    assert dict(forward.probabilities) == pytest.approx(expected, abs=1e-7)
    assert dict(reversed_result.probabilities) == pytest.approx(expected, abs=1e-7)
    assert forward.selected_id == reversed_result.selected_id == "open"
    assert tuple(key for key, _ in forward.probabilities) == TOOLS
    assert tuple(key for key, _ in reversed_result.probabilities) == tuple(reversed(TOOLS))
    assert (
        forward.reported_confidence
        is forward.reported_input_tokens
        is forward.reported_output_tokens
        is None
    )
    assert owner.work == dict(
        training_attempts=0,
        forward_calls=2,
        backward_calls=0,
        optimizer_steps=0,
        committed_updates=0,
    )
    assert model.training is False and torch.equal(caller_rng, torch.get_rng_state())
    after = owner.snapshot_state()
    for name in before.weights:
        assert torch.equal(before.weights[name], after.weights[name])
        assert before.gradients[name] is after.gradients[name] is None
    assert torch.equal(before.rng, after.rng)
    assert before.optimizer == after.optimizer and before.optimizer["state"] == {}
    assert len(model.records) == len(compiled) == 2
    assert all("target" not in records[0] for records in model.records)
    assert all(
        question["decision"]["instructions"] == runner.INSTRUCTIONS for _, question in compiled
    )
