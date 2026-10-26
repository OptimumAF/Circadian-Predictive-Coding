"""One bounded optimizer-history reset on tiny literal native parameters.

Torch is imported only by the executor's fixture. No released model, tokenizer,
random initializer, inference, dropout or native package is used here.
"""

from copy import deepcopy

import pytest

from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_open_jev_training import assert_tensor_tree_equal, constant_model, request


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip("torch", reason="native reset backend unavailable; not a Torch pass")


def reset_learner(torch, *, updates=2, resets=1):
    model = constant_model(torch)

    def compile_request(state, questions):
        question = questions["decision"]
        return [dict(kind=question["type"], answer_keys=list(question["criteria"]), state=state)]

    owner = OpenJevTrainingLearner(
        model,
        compile_request,
        instructions="Choose a synthetic class.",
        policy=OpenJevTrainingPolicy(updates, 1, max_optimizer_resets=resets),
    )
    return owner, model


@pytest.mark.parametrize("value", [-1, 2, True, 1.0])
def test_should_refuse_invalid_reset_allowances_without_constructing_a_backend(value):
    with pytest.raises(ValueError):
        OpenJevTrainingPolicy(max_optimizer_resets=value)


def test_should_keep_optimizer_reset_disabled_in_the_original_policy():
    policy = OpenJevTrainingPolicy()
    assert policy.max_optimizer_resets == 0
    assert policy.max_updates == 1 and policy.expected_lora_rank == 8


def test_should_clear_only_adamw_history_and_keep_owned_old_moments_after_future_training(
    torch_backend,
):
    torch = torch_backend
    owner, model = reset_learner(torch)
    model.eval()
    owner.train_batch(request(), "c")
    before = owner.snapshot_state()
    original_optimizer = owner._optimizer
    original_state = deepcopy(original_optimizer.state_dict())
    parameter_owners = dict(owner._trainable)
    gradient_owners = {name: parameter.grad for name, parameter in parameter_owners.items()}
    rng_owner = owner._rng
    caller_rng = torch.get_rng_state().clone()
    native_work = owner.work
    result = owner.reset_optimizer_state()
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": 1,
        "optimizer_resets_completed": 1,
    }
    assert result == owner.optimizer_reset_work
    result["optimizer_reset_attempts"] = 99
    assert owner.optimizer_reset_work["optimizer_reset_attempts"] == 1
    assert owner._optimizer is not original_optimizer
    after = owner.snapshot_state()
    assert before.optimizer["state"] and after.optimizer["state"] == {}
    assert before.optimizer["param_groups"] == after.optimizer["param_groups"]
    for name in ("weights", "gradients", "rng"):
        assert_tensor_tree_equal(torch, getattr(after, name), getattr(before, name))
    assert all(owner._trainable[name] is parameter for name, parameter in parameter_owners.items())
    assert all(
        parameter.grad is gradient_owners[name] for name, parameter in parameter_owners.items()
    )
    assert owner._rng is rng_owner and torch.equal(torch.get_rng_state(), caller_rng)
    assert model.training is False and owner.work == native_work
    assert_tensor_tree_equal(torch, original_optimizer.state_dict(), original_state)
    owner.train_batch(request(), "c")
    assert owner.work["committed_updates"] == 2 and owner._optimizer.state_dict()["state"]
    assert all(
        state["step"].item() == 1 for state in owner._optimizer.state_dict()["state"].values()
    )
    assert_tensor_tree_equal(torch, original_optimizer.state_dict(), original_state)
    assert_tensor_tree_equal(torch, before.optimizer, original_state)


def test_should_not_renew_permanent_training_or_reset_allowance_by_restoring_old_snapshot(
    torch_backend,
):
    owner, _ = reset_learner(torch_backend, updates=1)
    original = owner.snapshot_state()
    owner.train_batch(request(), "c")
    populated = owner.snapshot_state()
    owner.reset_optimizer_state()
    owner.restore_state(populated)
    assert owner.snapshot_state().optimizer["state"]
    owner.restore_state(original)
    assert owner.snapshot_state().optimizer["state"] == {}
    assert set(vars(original)) == {"owner", "weights", "gradients", "optimizer", "rng"}
    training_work, reset_work = owner.work, owner.optimizer_reset_work
    with pytest.raises(ValueError):
        owner.reset_optimizer_state()
    with pytest.raises(ValueError, match="allowance"):
        owner.train_batch(request(), "c")
    assert owner.work == training_work and owner.optimizer_reset_work == reset_work


def test_should_refuse_default_disabled_reset_before_creating_an_optimizer(
    torch_backend, monkeypatch
):
    owner, _ = reset_learner(torch_backend, resets=0)
    calls: list[None] = []

    def forbidden():
        calls.append(None)
        raise AssertionError("disabled reset constructed optimizer")

    monkeypatch.setattr(owner, "_new_optimizer", forbidden)
    with pytest.raises(ValueError):
        owner.reset_optimizer_state()
    assert calls == []
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": 0,
        "optimizer_resets_completed": 0,
    }
    assert all(value == 0 for value in owner.work.values())


def test_should_pin_reset_policy_scalar_before_any_reset_work(torch_backend):
    owner, _ = reset_learner(torch_backend)
    object.__setattr__(owner.policy, "max_optimizer_resets", 0)
    with pytest.raises(ValueError):
        owner.reset_optimizer_state()
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": 0,
        "optimizer_resets_completed": 0,
    }
    assert all(value == 0 for value in owner.work.values())


@pytest.mark.parametrize(
    "fault",
    [
        "raise",
        "weight",
        "gradient",
        "private_rng",
        "caller_rng",
        "mode",
        "state",
        "foreign_parameter",
    ],
)
def test_should_charge_and_poison_uncertain_factory_work_without_retry_or_snapshot_recovery(
    torch_backend, monkeypatch, fault
):
    torch = torch_backend
    owner, model = reset_learner(torch)
    model.eval()
    # Literal buffers make gradient preservation observable without an extra update.
    for parameter in owner._trainable.values():
        parameter.grad = torch.zeros_like(parameter)
    saved = owner.snapshot_state()
    original_factory = owner._new_optimizer
    caller_rng = torch.get_rng_state().clone()
    calls: list[None] = []

    def factory():
        calls.append(None)
        if fault == "raise":
            raise RuntimeError("controlled optimizer factory failure")
        optimizer = original_factory()
        with torch.no_grad():
            if fault == "weight":
                model.head.weight.add_(0.25)
            elif fault == "gradient":
                model.head.weight.grad.add_(0.25)
            elif fault == "private_rng":
                owner._rng.zero_()
            elif fault == "caller_rng":
                changed = torch.get_rng_state().clone()
                changed[0] = changed[0] ^ 1
                torch.set_rng_state(changed)
            elif fault == "mode":
                model.train()
            elif fault == "foreign_parameter":
                parameter = optimizer.param_groups[0]["params"][0]
                optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(
                    parameter.detach().clone()
                )
            else:
                optimizer.state[model.head.weight] = {"untrusted": True}
        return optimizer

    monkeypatch.setattr(owner, "_new_optimizer", factory)
    try:
        with pytest.raises(TrainingUpdateFailed) as error:
            owner.reset_optimizer_state()
        assert owner.optimizer_reset_work == {
            "optimizer_reset_attempts": 1,
            "optimizer_resets_completed": 0,
        }
        assert error.value.definition == "open_jev_adamw_state_reset_v1"
        assert error.value.work == {**owner.work, **owner.optimizer_reset_work}
        assert all(value == 0 for value in owner.work.values())
        for operation in (
            owner.reset_optimizer_state,
            lambda: owner.restore_state(saved),
            lambda: owner.train_batch(request(), "c"),
        ):
            with pytest.raises(ValueError):
                operation()
        assert calls == [None]
    finally:
        # The failed boundary is not permitted to contaminate other fixtures.
        torch.set_rng_state(caller_rng)
