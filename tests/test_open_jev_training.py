"""Constant-tensor native update qualification; no released model or tokenizer.

Torch is optional and imported only by backend fixtures. The synthetic rank-one
module has literal parameters; no randomized module initializer is used.
"""

from copy import deepcopy
from dataclasses import replace
import json
from math import exp, log
from typing import Any

import pytest

from src.adapters.open_jev_training import (
    DEFINITION,
    OpenJevTrainingLearner,
    OpenJevTrainingPolicy,
    native_choice_loss,
)
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from src.core.typed_decision import DecisionCandidate, DecisionRequest


def request() -> DecisionRequest:
    return DecisionRequest(
        "train-1",
        "declared synthetic input",
        (
            DecisionCandidate("a", "first"),
            DecisionCandidate("b", "second"),
            DecisionCandidate("c", "third"),
        ),
    )


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip(
        "torch", reason="native training backend unavailable; not a Torch pass"
    )


def constant_model(torch, *, dropout: bool = False):
    class Factor(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = Factor([[1.0, 0.0], [0.0, 1.0]])
            self.lora_A = Factor([[0.1, -0.2]])
            self.lora_B = Factor([[0.3], [0.1]])
            self.checkpoint_calls = 0
            self.input_gradient_calls = 0

        def gradient_checkpointing_enable(self):
            self.checkpoint_calls += 1

        def enable_input_require_grads(self):
            self.input_gradient_calls += 1

        def forward(self, features):
            return (
                features @ self.base.weight.T
                + (features @ self.lora_A.weight.T) @ self.lora_B.weight.T
            )

    class Head(Factor):
        def __init__(self):
            super().__init__([[0.2, -0.1]])
            self.bias = torch.nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

        def forward(self, features):
            return (features @ self.weight.T).squeeze(-1) + self.bias

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_rank = 1
            self.backbone = Backbone()
            self.head = Head()
            self.records: list[Any] = []
            self.fail_forward = False

        def forward(self, records):
            self.records.append(deepcopy(records))
            if self.fail_forward:
                raise RuntimeError("controlled native forward failure")
            table = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [-1.0, -1.0]}
            features = torch.tensor(
                [table[key] for key in records[0]["answer_keys"]], dtype=torch.float32
            )
            hidden = self.backbone(features)
            if dropout:
                # One prospectively authorized constant-input dropout fixture.
                hidden = torch.nn.functional.dropout(hidden, p=0.5, training=self.training)
            return [self.head(hidden)]

    return Model()


def learner(torch, *, dropout: bool = False, maximum: int = 1):
    model = constant_model(torch, dropout=dropout)
    compiled: list[dict[str, Any]] = []

    def compile_request(state, questions):
        question = questions["decision"]
        record = {
            "kind": question["type"],
            "state": state,
            "question": question["instructions"],
            "answer_keys": list(question["criteria"]),
            "options": list(question["criteria"].values()),
        }
        compiled.append(deepcopy(record))
        return [record]

    owner = OpenJevTrainingLearner(
        model,
        compile_request,
        instructions="Choose a synthetic class.",
        policy=OpenJevTrainingPolicy(maximum, 1),
    )
    return owner, model, compiled


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_updates": 0},
        {"max_updates": 129},
        {"max_updates": True},
        {"expected_lora_rank": 0},
        {"expected_lora_rank": 65},
        {"expected_lora_rank": 1.0},
    ],
)
def test_should_refuse_invalid_policy_without_a_torch_backend(arguments):
    with pytest.raises(ValueError):
        OpenJevTrainingPolicy(**arguments)


def test_should_keep_the_native_policy_defaults_explicit():
    policy = OpenJevTrainingPolicy()
    assert policy.max_updates == 1 and policy.expected_lora_rank == 8


def test_should_match_independent_native_loss_and_gradient_oracle(torch_backend):
    torch = torch_backend
    logits = torch.tensor([0.0, log(3)], dtype=torch.float64, requires_grad=True)
    loss = native_choice_loss(logits, 0)
    assert loss.item() == pytest.approx(log(4) + 0.1125, abs=1e-6)
    loss.backward()
    assert logits.grad.tolist() == pytest.approx([-0.80625, 0.80625], abs=1e-6)


def test_should_apply_one_native_update_to_only_lora_and_head_and_restore_serving_mode(
    torch_backend,
):
    torch = torch_backend
    owner, model, compiled = learner(torch)
    model.eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    prediction = owner.predict(request())
    raw = [0.255, -0.06, -0.045]
    mass = sum(exp(value / 1.518796342858676) for value in raw)
    assert dict(prediction.probabilities) == pytest.approx(
        {key: exp(value / 1.518796342858676) / mass for key, value in zip(("a", "b", "c"), raw)}
    )
    assert prediction.reported_input_tokens is prediction.reported_output_tokens is None
    assert model.training is False
    diagnostic = owner.train_batch(request(), "c")
    probabilities = [exp(value) / sum(exp(item) for item in raw) for value in raw]
    expected_loss = -log(probabilities[2]) + 0.1 * sum(
        (value - (index == 2)) ** 2 for index, value in enumerate(probabilities)
    )
    assert diagnostic.definition == DEFINITION
    assert diagnostic.value == pytest.approx(expected_loss, abs=1e-6)
    assert model.training is False
    assert model.backbone.checkpoint_calls == model.backbone.input_gradient_calls == 1
    assert torch.equal(model.backbone.base.weight, before["backbone.base.weight"])
    assert model.backbone.base.weight.requires_grad is False
    for fragment in ("head.", ".lora_A.", ".lora_B."):
        assert any(
            not torch.equal(parameter, before[name])
            for name, parameter in model.named_parameters()
            if fragment in name
        )
    assert compiled and all("target" not in record for record in compiled)
    assert model.records[-1][0]["target"] == [0.0, 0.0, 1.0]
    assert owner.work == {
        "training_attempts": 1,
        "forward_calls": 2,
        "backward_calls": 1,
        "optimizer_steps": 1,
        "committed_updates": 1,
    }
    assert owner.last_update["gradient_norm_before_clip"] >= 0
    detached_work = owner.work
    detached_work["training_attempts"] = 0
    assert owner.work["training_attempts"] == 1


def test_should_restore_trainable_state_without_renewing_permanent_update_allowance(torch_backend):
    torch = torch_backend
    owner, model, _ = learner(torch)
    original = owner.snapshot_state()
    detached = owner.snapshot_state()
    detached.weights["head.weight"].add_(1.0)
    assert not torch.equal(detached.weights["head.weight"], model.head.weight)
    owner.train_batch(request(), "c")
    assert owner.snapshot_state().optimizer["state"]
    owner.restore_state(original)
    assert all(
        torch.equal(parameter, original.weights[name])
        for name, parameter in model.named_parameters()
        if name in original.weights
    )
    assert owner.snapshot_state().optimizer["state"] == {}
    assert owner.work["training_attempts"] == owner.work["committed_updates"] == 1
    before = owner.work
    with pytest.raises(ValueError, match="allowance"):
        owner.train_batch(request(), "c")
    assert owner.work == before


def assert_tensor_tree_equal(torch, first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_tensor_tree_equal(torch, first[name], second[name])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tensor_tree_equal(torch, left, right)
    else:
        assert first == second


def test_should_restore_nonempty_adamw_gradients_and_rng_and_refuse_corrupt_snapshots_atomically(
    torch_backend,
):
    torch = torch_backend
    owner, _, _ = learner(torch, maximum=2)
    owner.train_batch(request(), "c")
    saved = owner.snapshot_state()
    assert saved.optimizer["state"] and all(value is not None for value in saved.gradients.values())
    with torch.no_grad():
        for parameter in owner._trainable.values():
            parameter.add_(0.25)
            parameter.grad.add_(0.5)
        owner._rng.zero_()
    owner.restore_state(saved)
    actual = owner.snapshot_state()
    for name in ("weights", "gradients", "optimizer", "rng"):
        assert_tensor_tree_equal(torch, getattr(actual, name), getattr(saved, name))
    work = owner.work
    bad_optimizer = deepcopy(saved.optimizer)
    next(iter(bad_optimizer["state"].values()))["exp_avg_sq"].fill_(-1.0)
    with pytest.raises(ValueError):
        owner.restore_state(replace(saved, optimizer=bad_optimizer))
    bad_weights = deepcopy(saved.weights)
    next(iter(bad_weights.values())).fill_(float("nan"))
    with pytest.raises(ValueError):
        owner.restore_state(replace(saved, weights=bad_weights))
    after = owner.snapshot_state()
    for name in ("weights", "gradients", "optimizer", "rng"):
        assert_tensor_tree_equal(torch, getattr(after, name), getattr(saved, name))
    assert owner.work == work


def test_should_refuse_a_snapshot_from_another_model_owner_before_mutation(torch_backend):
    torch = torch_backend
    first, _, _ = learner(torch)
    second, model, _ = learner(torch)
    before = second.snapshot_state()
    with pytest.raises(ValueError, match="another"):
        second.restore_state(first.snapshot_state())
    assert all(
        torch.equal(parameter, before.weights[name])
        for name, parameter in model.named_parameters()
        if name in before.weights
    )
    assert second.work["training_attempts"] == 0


def test_should_advance_private_dropout_rng_without_changing_caller_rng_or_mode(torch_backend):
    torch = torch_backend
    owner, model, _ = learner(torch, dropout=True)
    model.eval()
    global_before = torch.get_rng_state().clone()
    private_before = owner.snapshot_state().rng
    owner.train_batch(request(), "c")
    assert torch.equal(torch.get_rng_state(), global_before)
    assert not torch.equal(owner.snapshot_state().rng, private_before)
    assert model.training is False


def test_should_refuse_unknown_target_and_labeled_compiler_output_before_any_forward(torch_backend):
    owner, _, _ = learner(torch_backend)
    with pytest.raises(ValueError):
        owner.train_batch(request(), "outside")
    original = owner._compile

    def compile_labeled(state, questions):
        records = original(state, questions)
        records[0]["target"] = [1.0, 0.0, 0.0]
        return records

    owner._compile = compile_labeled
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    assert owner.work == {
        "training_attempts": 0,
        "forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "committed_updates": 0,
    }


def test_should_pin_policy_identity_and_scalar_values_before_additional_work(torch_backend):
    owner, _, _ = learner(torch_backend)
    birth = owner.policy
    owner.policy = OpenJevTrainingPolicy(1, 1)
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    owner.policy = birth
    object.__setattr__(birth, "max_updates", 2)
    with pytest.raises(ValueError):
        owner.predict(request())
    assert owner.work["training_attempts"] == owner.work["forward_calls"] == 0


def test_should_refuse_equal_parameter_owner_replacement_before_training(torch_backend):
    torch = torch_backend
    owner, model, _ = learner(torch)
    model.head.weight = torch.nn.Parameter(model.head.weight.detach().clone())
    with pytest.raises(ValueError, match="replacement"):
        owner.train_batch(request(), "c")
    assert owner.work["training_attempts"] == owner.work["forward_calls"] == 0


def test_should_recheck_frozen_base_after_the_compiler_callback(torch_backend):
    torch = torch_backend
    owner, model, _ = learner(torch)
    original = owner._compile

    def change_base(state, questions):
        records = original(state, questions)
        with torch.no_grad():
            model.backbone.base.weight.add_(1.0)
        return records

    owner._compile = change_base
    with pytest.raises(ValueError, match="frozen base"):
        owner.train_batch(request(), "c")
    assert owner.work["training_attempts"] == owner.work["forward_calls"] == 0
    assert model.backbone.base.weight[0, 0].item() == 2.0


def test_should_refuse_a_base_write_after_returned_prediction_forward(torch_backend):
    torch = torch_backend
    owner, model, _ = learner(torch)

    def change_base(module, arguments, output):
        with torch.no_grad():
            model.backbone.base.weight.add_(1.0)

    handle = model.register_forward_hook(change_base)
    try:
        with pytest.raises(ValueError, match="frozen base"):
            owner.predict(request())
    finally:
        handle.remove()
    assert owner.work["forward_calls"] == 1 and owner.work["training_attempts"] == 0


def test_should_keep_failed_forward_work_restore_mode_and_refuse_retry(torch_backend):
    owner, model, _ = learner(torch_backend)
    model.eval()
    model.fail_forward = True
    with pytest.raises(TrainingUpdateFailed) as captured:
        owner.train_batch(request(), "c")
    assert captured.value.work == {
        "training_attempts": 1,
        "forward_calls": 1,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "committed_updates": 0,
    }
    assert model.training is False
    before = owner.work
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    with pytest.raises(ValueError):
        owner.predict(request())
    assert owner.work == before


def test_should_keep_post_backward_mutation_and_paid_optimizer_work_without_publishing_update(
    torch_backend,
):
    torch = torch_backend
    owner, model, _ = learner(torch)
    before_head = model.head.weight.detach().clone()

    def change_base(gradient):
        with torch.no_grad():
            model.backbone.base.weight.add_(1.0)
        return gradient

    handle = model.head.weight.register_hook(change_base)
    try:
        with pytest.raises(TrainingUpdateFailed) as captured:
            owner.train_batch(request(), "c")
    finally:
        handle.remove()
    assert captured.value.work == {
        "training_attempts": 1,
        "forward_calls": 1,
        "backward_calls": 1,
        "optimizer_steps": 1,
        "committed_updates": 0,
    }
    assert not torch.equal(model.head.weight, before_head)
    assert model.backbone.base.weight[0, 0].item() == 2.0
    before = owner.work
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    with pytest.raises(ValueError):
        owner.predict(request())
    assert owner.work == before


def test_should_apply_only_after_delayed_label_arrival_through_existing_opaque_inbox(torch_backend):
    from src.app.experience_inbox import ExperienceInbox
    from src.app.toy_execution_budget import ToyExecutionBudget, ToyBudgetSession
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock

    owner, _, _ = learner(torch_backend)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), clock=lambda: 0.0)
    inbox = ExperienceInbox(
        owner, clock=clock, budget=budget, learner_version="tiny", max_experiences=1
    )
    source = Experience(
        "sample", "episode", 0, "tiny", request(), "train", ExperiencePermissions(training=True)
    )
    inbox.record_experience(source)
    assert inbox.drain(max_updates=1) == ()
    assert owner.work["training_attempts"] == 0
    inbox.record_label(LabelArrival("label", "sample", "episode", 1, "tiny", "c"))
    assert inbox.drain(max_updates=1) == ()
    clock.advance_to(1)
    receipts = inbox.drain(max_updates=1)
    assert len(receipts) == 1 and receipts[0].arrived_at == receipts[0].applied_at == 1
    assert receipts[0].diagnostic.definition == DEFINITION
    assert budget.updates_completed == owner.work["committed_updates"] == 1
    assert inbox.drain(max_updates=1) == ()


@pytest.mark.parametrize("fault", ["rank", "dtype"])
def test_should_refuse_wrong_native_rank_or_dtype_before_forward(torch_backend, fault):
    torch = torch_backend
    model = constant_model(torch)
    if fault == "rank":
        model.lora_rank = 2
    else:
        model.head.weight = torch.nn.Parameter(model.head.weight.detach().double())
    with pytest.raises(ValueError):
        OpenJevTrainingLearner(
            model,
            lambda state, questions: [],
            instructions="Choose.",
            policy=OpenJevTrainingPolicy(1, 1),
        )
    assert model.records == []


class FakeLearner:
    """Metadata-only NativeLearner for runner delivery; no tensors or model."""

    def __init__(self, events: list[str], *, fail: bool = False):
        self.events, self.fail = events, fail
        self._work = {
            "training_attempts": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "committed_updates": 0,
        }
        self.last_update: dict[str, float] | None = None
        self.training: list[Any] = []

    @property
    def work(self):
        return dict(self._work)

    def train_batch(self, features, targets):
        self.events.append("train")
        self.training.append((deepcopy(features), targets))
        self._work.update(training_attempts=1, forward_calls=1, backward_calls=1, optimizer_steps=1)
        if self.fail:
            raise TrainingUpdateFailed("controlled fake update failure", DEFINITION, self.work)
        self._work["committed_updates"] = 1
        self.last_update = {"loss": 0.25, "gradient_norm_before_clip": 0.5}
        return TrainingDiagnostic(DEFINITION, 0.25)


def runner_reply(wire):
    keys = list(wire["questions"]["decision"]["criteria"])
    selected = "shipping" if "zorbex" in wire["state"] else "billing"
    return {
        "model": "Qwen/Qwen3.5-2B",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": selected,
                "probabilities": {key: 0.8 if key == selected else 0.1 for key in keys},
                "confidence": 0.7,
            }
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "metadata": {
            "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "method": "lora_decision_head",
            "temperature": 1.518796342858676,
            "max_length": 256,
            "backbone_parameter_dtypes": ["torch.float32"],
            "candidate_sequences": 3,
            "prefix_cache": {"enabled": False, "mode": "independent_candidates"},
        },
    }


def test_should_deliver_one_train_label_between_four_distinct_outer_selection_readouts_without_model_math():
    from scripts.run_open_jev_native_update import run_update

    events: list[str] = []
    owner = FakeLearner(events)
    wires: list[Any] = []
    progress: list[str] = []

    def predict(wire):
        events.append("predict")
        wires.append(deepcopy(wire))
        assert set(wire) == {"model", "state", "questions"}
        return runner_reply(wire)

    report: dict[str, Any] = {}
    run_update(owner, predict, report, progress.append)
    assert events == ["predict", "predict", "train", "predict", "predict"]
    assert len(wires) == 4 and len(owner.training) == 1
    source, target = owner.training[0]
    assert source.request_id == "train-zorbex" and source.query == "Case marker: zorbex"
    assert target == "shipping"
    assert report["delivery"][0] == {"event": "pre_label_poll", "updates": 0, "tick": 0}
    receipt = report["delivery"][1]["receipt"]
    assert receipt["observed_at"] == 0 and receipt["arrived_at"] == receipt["applied_at"] == 1
    assert report["sequence_complete"] is True
    assert report["app_committed_updates"] == 1
    assert all(row["role"] == "outer_selection" for row in report["readouts"])
    assert [row["phase"] for row in report["readouts"]] == ["before", "before", "after", "after"]


def test_should_retain_before_readouts_and_failed_training_work_without_after_retry():
    from scripts.run_open_jev_native_update import run_update

    events: list[str] = []
    owner = FakeLearner(events, fail=True)

    def predict(wire):
        events.append("predict")
        return runner_reply(wire)

    report: dict[str, Any] = {}
    with pytest.raises(TrainingUpdateFailed):
        run_update(owner, predict, report, lambda event: None)
    assert events == ["predict", "predict", "train"]
    assert len(report["readouts"]) == 2 and len(owner.training) == 1
    assert report["inbox_stopped"] is True
    assert report["native_work"]["optimizer_steps"] == 1
    assert report["native_work"]["committed_updates"] == report["app_committed_updates"] == 0
    assert "sequence_complete" not in report


@pytest.mark.parametrize("fault", ["nan", "unsupported", "changed-wire", "changed-order"])
def test_should_preserve_invalid_paid_runner_returns_and_refuse_more_work_before_training(fault):
    from scripts.run_open_jev_native_update import run_update

    owner = FakeLearner([])
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        reply = runner_reply(wire)
        if fault == "nan":
            reply["metadata"]["invalid"] = float("nan")
        elif fault == "unsupported":
            reply["metadata"]["invalid"] = object()
        elif fault == "changed-wire":
            wire["state"] = "changed callback request"
        else:
            question = wire["questions"]["decision"]
            question["criteria"] = dict(reversed(tuple(question["criteria"].items())))
        return reply

    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        run_update(owner, predict, report, lambda event: None)
    assert calls == [None] and owner.training == []
    assert len(report["readouts"]) == 1
    row = report["readouts"][0]
    assert row["request"]["query"] == "invoice refund"
    if fault == "nan":
        assert "NaN" in row["response_json"]
    elif fault == "unsupported":
        assert isinstance(row["response_unavailable"], str)
    else:
        assert row["response"]["answers"]["decision"]["type"] == "choice"
    json_report = json.dumps(report, allow_nan=False)
    assert json_report
