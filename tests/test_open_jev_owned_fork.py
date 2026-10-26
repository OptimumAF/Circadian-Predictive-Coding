"""Disposable native ownership on tiny literal modules, never released weights.

Only backend fixtures import Torch during the root executor's test run. The
worker authors source and checks AST; no model/test execution occurs here.
"""

from copy import deepcopy
import pytest

from src.adapters import open_jev_training as native
from src.adapters import open_jev_owned_fork as owned
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_open_jev_training import assert_tensor_tree_equal, constant_model, request


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip("torch", reason="owned-fork backend unavailable; not a Torch pass")


def fork_learner(torch, *, updates=3, forks=1, health=False, resets=0):
    model = constant_model(torch)
    model.register_buffer("running", torch.tensor([0.25, 0.5], dtype=torch.float32))
    model.eval()

    def compile_request(state, questions):
        question = questions["decision"]
        return [dict(kind=question["type"], state=state, answer_keys=list(question["criteria"]))]

    owner = native.OpenJevTrainingLearner(
        model,
        compile_request,
        instructions="Choose a synthetic class.",
        policy=native.OpenJevTrainingPolicy(
            updates, 1, resets, record_update_health=health, max_forks=forks
        ),
    )
    return owner, model


def assert_snapshot_equal(torch, left, right):
    for field in ("weights", "gradients", "optimizer", "rng"):
        assert_tensor_tree_equal(torch, getattr(left, field), getattr(right, field))


@pytest.mark.parametrize("value", [-1, 2, True, 1.0])
def test_should_refuse_invalid_fork_policy_without_a_backend(value):
    with pytest.raises(ValueError):
        native.OpenJevTrainingPolicy(max_forks=value)


def test_should_keep_fork_disabled_in_the_original_policy():
    assert native.OpenJevTrainingPolicy().max_forks == 0


def test_should_fork_complete_warm_state_without_constructor_forward_or_hook_reinitialization(
    torch_backend, monkeypatch
):
    torch = torch_backend
    source, model = fork_learner(torch)
    source.train_batch(request(), "c")
    before = source.snapshot_state()
    native_work, reset_work, health_work = (
        source.work,
        source.optimizer_reset_work,
        source.update_health_work,
    )
    diagnostic = deepcopy(source.last_update)
    records = deepcopy(model.records)
    rng_elements = source._rng.numel()

    def forbidden(*args, **kwargs):
        raise AssertionError("fork re-ran a constructor")

    with monkeypatch.context() as patch:
        for module_type in {type(module) for module in model.modules()}:
            patch.setattr(module_type, "__init__", forbidden)
        patch.setattr(native.OpenJevTrainingLearner, "__init__", forbidden)
        child = source.fork()
    assert type(child) is type(source) and child is not source
    assert child.model is not source.model and child._compile is source._compile
    assert child.policy == source.policy
    assert child.policy is not source.policy
    assert_snapshot_equal(torch, before, child.snapshot_state())
    assert source.work == child.work == native_work
    assert source.optimizer_reset_work == child.optimizer_reset_work == reset_work
    assert source.update_health_work == child.update_health_work == health_work
    for field in ("_work", "_optimizer_reset_work", "_update_health_work", "_fork_work"):
        assert getattr(source, field) is not getattr(child, field)
    assert source.last_update == child.last_update == diagnostic
    assert child.last_update is not source.last_update
    assert (
        model.records == child.model.records == records and child.model.records is not model.records
    )
    assert model.backbone.checkpoint_calls == child.model.backbone.checkpoint_calls == 1
    assert model.backbone.input_gradient_calls == child.model.backbone.input_gradient_calls == 1
    assert all(left is not right for left, right in zip(model.modules(), child.model.modules()))
    assert model.backbone.base.weight is child.model.backbone.base.weight
    assert (
        model.backbone.base.weight.requires_grad is False
        and model.backbone.base.weight.grad is None
    )
    assert (
        source._optimizer is not child._optimizer
        and source._rng.data_ptr() != child._rng.data_ptr()
    )
    assert model.running.data_ptr() != child.model.running.data_ptr()
    for name, parameter in source._trainable.items():
        copied = child._trainable[name]
        assert parameter is not copied and parameter.data_ptr() != copied.data_ptr()
        assert (
            parameter.grad is not copied.grad
            and parameter.grad.data_ptr() != copied.grad.data_ptr()
        )
        original_state, copied_state = (
            source._optimizer.state[parameter],
            child._optimizer.state[copied],
        )
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert original_state[field].data_ptr() != copied_state[field].data_ptr()
    assert_snapshot_equal(torch, source.snapshot_state(), before)
    assert (
        source.fork_work
        == child.fork_work
        == {
            "fork_attempts": 1,
            "parameter_copy_elements_attempted": 7,
            "gradient_copy_elements_attempted": 7,
            "buffer_copy_elements_attempted": 2,
            "optimizer_copy_elements_attempted": 18,
            "rng_copy_elements_attempted": 3 * rng_elements,
            "forks_completed": 1,
        }
    )
    source.train_batch(request(), "a")
    child.train_batch(request(), "a")
    assert_snapshot_equal(torch, source.snapshot_state(), child.snapshot_state())
    saved_child = child.snapshot_state()
    with pytest.raises(ValueError, match="another"):
        child.restore_state(source.snapshot_state())
    assert_snapshot_equal(torch, child.snapshot_state(), saved_child)


def test_should_isolate_alternate_child_update_modes_buffers_moments_gradients_and_rng_from_source(
    torch_backend,
):
    torch = torch_backend
    source, model = fork_learner(torch, health=True)
    source.train_batch(request(), "c")
    before = source.snapshot_state()
    native_work, health_work = source.work, source.update_health_work
    diagnostic = deepcopy(source.last_update)
    child = source.fork()
    assert child.last_update["update_health"] is not source.last_update["update_health"]
    assert (
        child.last_update["update_health"]["groups"]
        is not source.last_update["update_health"]["groups"]
    )
    child.train_batch(request(), "b")
    assert child.work["committed_updates"] == 2 and source.work == native_work
    assert child.update_health_work["update_health_reports_completed"] == 2
    assert source.update_health_work == health_work
    assert source.last_update == diagnostic
    with torch.no_grad():
        child.model.running.add_(1.0)
        child.model.head.weight.grad.fill_(123.0)
        child._rng.zero_()
        child._optimizer.state[child.model.head.weight]["exp_avg"].fill_(123.0)
    child.model.train()
    assert model.training is False and child.model.training is True
    assert model.running.tolist() == [0.25, 0.5]
    assert_snapshot_equal(torch, source.snapshot_state(), before)
    before_work = source.work, child.work
    with torch.no_grad():
        child.model.backbone.base.weight.add_(1.0)
    for owner in (source, child):
        with pytest.raises(ValueError, match="frozen base"):
            owner.train_batch(request(), "c")
    assert (source.work, child.work) == before_work


def test_should_inherit_spent_training_reset_health_and_fork_limits_without_restore_refund(
    torch_backend,
):
    source, _ = fork_learner(torch_backend, updates=1, resets=1, health=True)
    original = source.snapshot_state()
    source.train_batch(request(), "c")
    source.reset_optimizer_state()
    child = source.fork()
    copied = child.snapshot_state()
    assert source.work == child.work and source.optimizer_reset_work == child.optimizer_reset_work
    assert source.update_health_work == child.update_health_work
    assert source.fork_work == child.fork_work
    assert (
        child.policy.max_updates == 1
        and child.policy.max_optimizer_resets == child.policy.max_forks == 1
    )
    with pytest.raises(ValueError, match="allowance"):
        child.train_batch(request(), "c")
    with pytest.raises(ValueError):
        child.reset_optimizer_state()
    source.restore_state(original)
    child.restore_state(copied)
    source_work, child_work = source.fork_work, child.fork_work
    for owner in (source, child):
        with pytest.raises(ValueError):
            owner.fork()
    assert source.fork_work == source_work and child.fork_work == child_work
    detached = child.fork_work
    detached[next(iter(detached))] = 999
    assert child.fork_work == child_work


def test_should_refuse_default_disabled_fork_without_copying_or_changing_native_work(torch_backend):
    owner, _ = fork_learner(torch_backend, forks=0)
    snapshot = owner.snapshot_state()
    work = owner.work
    with pytest.raises(ValueError):
        owner.fork()
    assert all(value == 0 for value in owner.fork_work.values())
    assert owner.work == work
    assert_snapshot_equal(torch_backend, owner.snapshot_state(), snapshot)


@pytest.mark.parametrize(
    "fault", ["frozen_gradient", "elements", "bytes", "parameter_buffer", "storage_alias"]
)
def test_should_refuse_unshareable_frozen_gradients_and_unbounded_or_aliased_buffers_before_copy(
    torch_backend, fault
):
    torch = torch_backend
    owner, model = fork_learner(torch)
    if fault == "frozen_gradient":
        model.backbone.base.weight.grad = torch.zeros_like(model.backbone.base.weight)
    elif fault == "elements":
        # One allocated scalar; the oversized view is rejected from metadata.
        model.register_buffer(
            "oversized", torch.tensor([1.0], dtype=torch.float32).expand(5_000_001)
        )
    elif fault == "bytes":
        # Below the element cap, above40MiB, still only one allocated scalar.
        model.register_buffer(
            "oversized", torch.tensor([1.0 + 0.0j], dtype=torch.complex128).expand(3_000_000)
        )
    elif fault == "parameter_buffer":
        model.register_buffer("alias_parameter", model.head.weight)
    else:
        model.register_buffer("alias_storage", model.running.view_as(model.running))
    work = owner.work
    with pytest.raises(ValueError):
        owner.fork()
    assert owner.work == work and all(value == 0 for value in owner.fork_work.values())
    assert owner._poisoned is False


def test_should_pin_fork_policy_scalar_before_any_copy_work(torch_backend):
    owner, _ = fork_learner(torch_backend)
    object.__setattr__(owner.policy, "max_forks", 0)
    with pytest.raises(ValueError):
        owner.fork()
    assert all(value == 0 for value in owner.work.values())
    assert all(value == 0 for value in owner.fork_work.values())


def test_should_keep_failed_copy_charges_poison_source_and_refuse_snapshot_recovery_or_retry(
    torch_backend, monkeypatch
):
    owner, _ = fork_learner(torch_backend)
    saved = owner.snapshot_state()
    work = owner.work
    rng_elements = owner._rng.numel()
    original = owned.deepcopy
    copies: list[None] = []

    def fail_owner(value, memo=None):
        if value is owner:
            copies.append(None)
            raise RuntimeError("controlled native owner copy failure")
        return original(value) if memo is None else original(value, memo)

    monkeypatch.setattr(owned, "deepcopy", fail_owner)
    with pytest.raises(TrainingUpdateFailed) as error:
        owner.fork()
    assert copies == [None] and owner.work == work
    assert owner.fork_work == {
        "fork_attempts": 1,
        "parameter_copy_elements_attempted": 7,
        "gradient_copy_elements_attempted": 0,
        "buffer_copy_elements_attempted": 2,
        "optimizer_copy_elements_attempted": 0,
        "rng_copy_elements_attempted": 2 * rng_elements,
        "forks_completed": 0,
    }
    assert error.value.definition == "open_jev_owned_fork_v1"
    assert error.value.work == {**owner.work, **owner.fork_work}
    paid = owner.fork_work
    for operation in (owner.fork, owner.snapshot_state, lambda: owner.restore_state(saved)):
        with pytest.raises(ValueError):
            operation()
    assert copies == [None] and owner.fork_work == paid
