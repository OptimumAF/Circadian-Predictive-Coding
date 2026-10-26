"""Realized-update diagnostics on tiny literal parameters, never a released model.

The executor alone imports Torch through fixtures. Independent Python reductions
check the reported gradients and actual AdamW deltas; diagnostics are not a fit.
"""

from copy import deepcopy
import hashlib
import json
from math import fsum, sqrt
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.adapters import open_jev_training as native
from src.adapters import open_jev_update_health as health
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_open_jev_training import assert_tensor_tree_equal, constant_model, request


HEALTH_KEYS = (
    "update_health_attempts",
    "parameter_copy_elements_attempted",
    "gradient_reduction_elements_attempted",
    "step_reduction_elements_attempted",
    "update_health_reports_completed",
)


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip(
        "torch", reason="native health backend unavailable; not a Torch pass"
    )


def compile_request(state, questions):
    question = questions["decision"]
    return [dict(kind=question["type"], state=state, answer_keys=list(question["criteria"]))]


def health_learner(torch, *, enabled=True, updates=2):
    model = constant_model(torch)
    model.eval()
    owner = native.OpenJevTrainingLearner(
        model,
        compile_request,
        instructions="Choose a synthetic class.",
        policy=native.OpenJevTrainingPolicy(updates, 1, 1, record_update_health=enabled),
    )
    return owner, model


def group_of(name):
    return "head" if name.startswith("head.") else "lora_A" if ".lora_A." in name else "lora_B"


def python_norm(values):
    return sqrt(fsum(value * value for value in values))


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_should_refuse_nonboolean_health_policy_without_constructing_a_backend(value):
    with pytest.raises(ValueError):
        native.OpenJevTrainingPolicy(record_update_health=value)


def test_should_leave_health_disabled_in_the_original_policy():
    assert native.OpenJevTrainingPolicy().record_update_health is False


def test_should_measure_preclip_gradient_and_realized_parameter_delta_with_independent_scalar_oracle(
    torch_backend, monkeypatch
):
    owner, model = health_learner(torch_backend)
    observed: dict[str, Any] = {}
    actual_capture = native.capture_update_health

    def capture(parameters, work):
        observed["weights"] = {
            name: parameter.detach().reshape(-1).tolist() for name, parameter in parameters.items()
        }
        observed["gradients"] = {
            name: parameter.grad.detach().reshape(-1).tolist()
            for name, parameter in parameters.items()
        }
        return actual_capture(parameters, work)

    monkeypatch.setattr(native, "capture_update_health", capture)
    owner.train_batch(request(), "c")
    report = owner.last_update["update_health"]
    assert report["definition"] == "open_jev_train_group_update_health_v1"
    assert report["work"] == owner.update_health_work
    assert set(observed["weights"]) == set(owner._trainable)
    assert "backbone.base.weight" not in observed["weights"]
    assert set(report["groups"]) == {"head", "lora_A", "lora_B"}
    for group, counts in {"head": (2, 3), "lora_A": (1, 2), "lora_B": (1, 2)}.items():
        names = [name for name in observed["weights"] if group_of(name) == group]
        gradients = [value for name in names for value in observed["gradients"][name]]
        deltas = [
            after - before
            for name in names
            for after, before in zip(
                owner._trainable[name].detach().reshape(-1).tolist(), observed["weights"][name]
            )
        ]
        entry = report["groups"][group]
        assert entry["parameter_tensors"] == entry["gradient_tensors"] == counts[0]
        assert entry["parameter_elements"] == counts[1] and entry["missing_gradient_tensors"] == 0
        assert entry["gradient_l2_before_clip"] == pytest.approx(
            python_norm(gradients), rel=1e-12, abs=1e-12
        )
        assert entry["parameter_step_l2"] == pytest.approx(
            python_norm(deltas), rel=1e-12, abs=1e-12
        )
    assert model.training is False
    assert owner.update_health_work == {
        "update_health_attempts": 1,
        "parameter_copy_elements_attempted": 7,
        "gradient_reduction_elements_attempted": 7,
        "step_reduction_elements_attempted": 7,
        "update_health_reports_completed": 1,
    }
    json.dumps(report, allow_nan=False)


def test_should_leave_two_step_parameters_gradients_moments_rng_and_mode_identical_when_enabled(
    torch_backend,
):
    torch = torch_backend
    disabled, disabled_model = health_learner(torch, enabled=False)
    enabled, enabled_model = health_learner(torch, enabled=True)
    caller_rng = torch.get_rng_state().clone()
    published: list[dict[str, Any]] = []
    for target in ("c", "a"):
        disabled.train_batch(request(), target)
        enabled.train_batch(request(), target)
        left, right = disabled.snapshot_state(), enabled.snapshot_state()
        for name in ("weights", "gradients", "optimizer", "rng"):
            assert_tensor_tree_equal(torch, getattr(left, name), getattr(right, name))
        assert disabled.work == enabled.work
        assert disabled_model.training is enabled_model.training is False
        assert torch.equal(torch.get_rng_state(), caller_rng)
        assert "update_health" not in disabled.last_update
        assert disabled.last_update == {
            key: value for key, value in enabled.last_update.items() if key != "update_health"
        }
        published.append(enabled.last_update["update_health"])
    assert all(value == 0 for value in disabled.update_health_work.values())
    assert enabled.update_health_work["update_health_reports_completed"] == 2
    assert [report["work"]["update_health_reports_completed"] for report in published] == [1, 2]
    assert set(disabled.work) == {
        "training_attempts",
        "forward_calls",
        "backward_calls",
        "optimizer_steps",
        "committed_updates",
    }


def test_should_distinguish_missing_gradient_from_present_zero_without_copying_a_frozen_base(
    torch_backend,
):
    torch = torch_backend
    parameters = {
        "head.weight": torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        "backbone.lora_A.weight": torch.nn.Parameter(
            torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        ),
        "backbone.lora_B.weight": torch.nn.Parameter(
            torch.tensor([[5.0], [6.0]], dtype=torch.float32)
        ),
    }
    parameters["head.weight"].grad = torch.zeros_like(parameters["head.weight"])
    parameters["backbone.lora_B.weight"].grad = torch.tensor([[3.0], [4.0]], dtype=torch.float32)
    work = dict.fromkeys(HEALTH_KEYS, 0)
    before, groups = health.capture_update_health(parameters, work)
    initial = {
        name: parameter.detach().reshape(-1).tolist() for name, parameter in parameters.items()
    }
    with torch.no_grad():
        parameters["backbone.lora_B.weight"].add_(0.1)
    report = health.finish_update_health(parameters, before, groups, work)
    entries = report["groups"]
    assert (
        entries["head"]["gradient_tensors"] == 1
        and entries["head"]["missing_gradient_tensors"] == 0
    )
    assert entries["head"]["gradient_l2_before_clip"] == 0.0
    assert (
        entries["lora_A"]["gradient_tensors"] == 0
        and entries["lora_A"]["missing_gradient_tensors"] == 1
    )
    assert (
        entries["lora_A"]["gradient_l2_before_clip"]
        == entries["lora_A"]["parameter_step_l2"]
        == 0.0
    )
    assert entries["lora_B"]["gradient_l2_before_clip"] == 5.0
    delta = [
        after - previous
        for after, previous in zip(
            parameters["backbone.lora_B.weight"].detach().reshape(-1).tolist(),
            initial["backbone.lora_B.weight"],
        )
    ]
    assert entries["lora_B"]["parameter_step_l2"] == pytest.approx(python_norm(delta), rel=1e-12)
    assert work == {
        "update_health_attempts": 1,
        "parameter_copy_elements_attempted": 6,
        "gradient_reduction_elements_attempted": 4,
        "step_reduction_elements_attempted": 6,
        "update_health_reports_completed": 1,
    }
    json.dumps(report, allow_nan=False)


def test_should_charge_capture_before_a_gradient_reduction_refuses(torch_backend, monkeypatch):
    torch = torch_backend
    parameters = {
        "head.weight": torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        "backbone.lora_A.weight": torch.nn.Parameter(
            torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        ),
        "backbone.lora_B.weight": torch.nn.Parameter(
            torch.tensor([[5.0], [6.0]], dtype=torch.float32)
        ),
    }
    parameters["head.weight"].grad = torch.zeros_like(parameters["head.weight"])
    parameters["backbone.lora_B.weight"].grad = torch.zeros_like(
        parameters["backbone.lora_B.weight"]
    )
    original = health._squared_norm
    calls: list[None] = []

    def refuse_second(value):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("controlled second gradient reduction failure")
        return original(value)

    monkeypatch.setattr(health, "_squared_norm", refuse_second)
    work = dict.fromkeys(HEALTH_KEYS, 0)
    with pytest.raises(RuntimeError, match="controlled second gradient reduction failure"):
        health.capture_update_health(parameters, work)
    assert work == {
        "update_health_attempts": 1,
        "parameter_copy_elements_attempted": 6,
        "gradient_reduction_elements_attempted": 4,
        "step_reduction_elements_attempted": 0,
        "update_health_reports_completed": 0,
    }


def test_should_charge_attempted_step_reduction_and_refuse_nonfinite_report(torch_backend):
    torch = torch_backend
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    parameter.grad = torch.zeros_like(parameter)
    parameters = {"head.weight": parameter}
    work = dict.fromkeys(HEALTH_KEYS, 0)
    before, groups = health.capture_update_health(parameters, work)
    with torch.no_grad():
        parameter.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        health.finish_update_health(parameters, before, groups, work)
    assert work == {
        "update_health_attempts": 1,
        "parameter_copy_elements_attempted": 2,
        "gradient_reduction_elements_attempted": 2,
        "step_reduction_elements_attempted": 2,
        "update_health_reports_completed": 0,
    }


def test_should_keep_detached_permanent_health_costs_through_prediction_reset_and_restore(
    torch_backend,
):
    owner, _ = health_learner(torch_backend, updates=1)
    original = owner.snapshot_state()
    owner.train_batch(request(), "c")
    cost = owner.update_health_work
    assert set(cost) == set(HEALTH_KEYS)
    detached = owner.update_health_work
    detached["update_health_attempts"] = 99
    assert owner.update_health_work == cost
    owner.predict(request())
    owner.reset_optimizer_state()
    owner.restore_state(original)
    assert owner.update_health_work == cost
    assert set(vars(original)) == {"owner", "weights", "gradients", "optimizer", "rng"}
    with pytest.raises(ValueError, match="allowance"):
        owner.train_batch(request(), "c")
    assert owner.update_health_work == cost


@pytest.mark.parametrize("stage", ["capture", "finish"])
def test_should_keep_paid_diagnostic_prefix_and_poison_without_crediting_a_failed_update(
    torch_backend, monkeypatch, stage
):
    owner, _ = health_learner(torch_backend)
    if stage == "finish":
        owner.train_batch(request(), "c")
    original = owner.snapshot_state()
    last = deepcopy(owner.last_update)
    actual_capture = native.capture_update_health

    def capture(parameters, work):
        actual_capture(parameters, work)
        raise RuntimeError("controlled paid health capture failure")

    def finish(*args, **kwargs):
        raise RuntimeError("controlled post-AdamW health failure")

    monkeypatch.setattr(
        native,
        "capture_update_health" if stage == "capture" else "finish_update_health",
        capture if stage == "capture" else finish,
    )
    with pytest.raises(TrainingUpdateFailed) as error:
        owner.train_batch(request(), "a")
    paid = 1 if stage == "capture" else 2
    assert (
        owner.work["training_attempts"]
        == owner.work["forward_calls"]
        == owner.work["backward_calls"]
        == paid
    )
    assert owner.work["optimizer_steps"] == (0 if stage == "capture" else 2)
    assert owner.work["committed_updates"] == (0 if stage == "capture" else 1)
    assert owner.update_health_work["update_health_attempts"] == paid
    assert owner.update_health_work["parameter_copy_elements_attempted"] == 7 * paid
    assert owner.update_health_work["gradient_reduction_elements_attempted"] == 7 * paid
    assert owner.update_health_work["step_reduction_elements_attempted"] == (
        0 if stage == "capture" else 7
    )
    assert owner.update_health_work["update_health_reports_completed"] == (
        0 if stage == "capture" else 1
    )
    assert error.value.work == {**owner.work, **owner.update_health_work}
    assert owner.last_update == last
    work, health_work = owner.work, owner.update_health_work
    for operation in (
        lambda: owner.train_batch(request(), "c"),
        lambda: owner.restore_state(original),
        owner.reset_optimizer_state,
    ):
        with pytest.raises(ValueError):
            operation()
    assert owner.work == work and owner.update_health_work == health_work


def test_should_pin_health_policy_scalar_before_any_training_or_diagnostic_cost(torch_backend):
    owner, _ = health_learner(torch_backend)
    object.__setattr__(owner.policy, "record_update_health", False)
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    assert all(value == 0 for value in owner.work.values())
    assert all(value == 0 for value in owner.update_health_work.values())


def test_should_transport_enabled_health_through_cli_real_inbox_receipt_and_failed_run_final_work(
    torch_backend, monkeypatch, tmp_path
):
    from scripts import run_open_jev_optimizer_control as runner
    from tests.test_native_optimizer_reset_control import reset_reference

    torch = torch_backend
    model = constant_model(torch)
    # Rank is declarative fixture metadata, not a claim that these tiny factors
    # are the released rank-eight graph. The CLI's actual policy is exercised.
    model.lora_rank = 8
    model.eval()
    original_forward = model.forward

    def forward(records):
        converted = deepcopy(records)
        converted[0]["answer_keys"] = [
            {"billing": "a", "password": "b", "shipping": "c"}[key]
            for key in converted[0]["answer_keys"]
        ]
        return original_forward(converted)

    model.forward = forward
    reference = reset_reference()
    selected_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("head.") or ".lora_A." in name or ".lora_B." in name
    }
    reference["baseline_weights"] = runner._weight_summary(model, selected_before)
    reference["initial_rng_sha256"] = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()
    ).hexdigest()
    api = ModuleType("jev.api")
    setattr(api, "compile_request", compile_request)
    resource = ModuleType("resource")
    setattr(resource, "RUSAGE_SELF", 0)
    setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=1))
    monkeypatch.setitem(sys.modules, "jev.api", api)
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(runner, "load_reference", lambda path, **kwargs: deepcopy(reference))
    owners: list[Any] = []
    actual_learner = runner.OpenJevTrainingLearner

    def build(*args, **kwargs):
        owner = actual_learner(*args, **kwargs)
        owners.append(owner)
        return owner

    monkeypatch.setattr(runner, "OpenJevTrainingLearner", build)
    loads: list[dict[str, Any]] = []

    def load(source, checkpoint, progress, **options):
        loads.append(options)
        options["model_observer"](model)

        def forbidden_prediction(wire):
            raise AssertionError("first-exposure fixture reached native readout")

        return forbidden_prediction

    monkeypatch.setattr(runner, "_load", load)
    actual_run = runner.run_optimizer_control

    def run(learner, predict, inherited, report, progress, **options):
        def stop_after_paid_receipt(event):
            progress(event)
            if event == "training_finished":
                raise RuntimeError("controlled stop after first CLI health receipt")

        return actual_run(learner, predict, inherited, report, stop_after_paid_receipt, **options)

    monkeypatch.setattr(runner, "run_optimizer_control", run)
    output = tmp_path / "health.json"
    result = runner.main(
        [
            "--reference",
            str(tmp_path / "synthetic-reference.json"),
            "--source",
            str(tmp_path / "unused-source"),
            "--checkpoint",
            str(tmp_path / "unused-checkpoint"),
            "--output",
            str(output),
            "--update-health",
        ]
    )
    assert result == 1 and len(owners) == len(loads) == 1
    owner = owners[0]
    assert owner.policy.record_update_health is True
    report = json.loads(output.read_text())
    assert report["complete"] is False and report["update_health_enabled"] is True
    assert "controlled stop after first CLI health receipt" in report["error"]
    assert len(report["delivery"]) == 1 and report["readouts"] == []
    assert report["delivery"][0]["receipt"]["update_number"] == 1
    assert report["delivery"][0]["training"]["update_health"] == owner.last_update["update_health"]
    assert report["update_health_work"] == owner.update_health_work
    assert report["update_health_work"]["update_health_reports_completed"] == 1
    assert report["app_committed_updates"] == report["native_work"]["committed_updates"] == 1
    assert (
        owner.work["forward_calls"]
        == owner.work["backward_calls"]
        == owner.work["optimizer_steps"]
        == 1
    )
