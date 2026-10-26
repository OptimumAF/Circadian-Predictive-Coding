"""Fake-native staged zero-loss SGD and explicit momentum-only release tests.

No Torch runtime is loaded: tensor/optimizer ports record the staged lifecycle.
Real numerical qualification belongs to the separately budgeted native tests.
"""

from copy import deepcopy
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import torch_demonstration_reward_learner as typed
from src.adapters import torch_recurrent_learner as native
from src.adapters import torch_recurrent_zero_loss as control
from src.adapters.torch_reward_update_learner import RewardUpdate
from src.core.demonstration_episode import DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode
from test_torch_demonstration_reward_learner import Native as PreviousNative, demonstration
from test_torch_reward_update_learner import packet


class Result:
    def __init__(self, value):
        self.value = np.asarray(value)

    def all(self):
        return Result(self.value.all())

    def item(self):
        return self.value.item()


class Tensor:
    def __init__(self, values, work):
        self.values = np.array(values, dtype=np.float32, copy=True)
        self.work = work
        self.grad: Any = None
        self.is_sparse = False
        self.layout = "strided"

    @property
    def shape(self):
        return self.values.shape

    def __mul__(self, scalar):
        assert scalar == 0.0
        self.work["events"].append("elementwise_zero")
        return Tensor(self.values * scalar, self.work)

    def sum(self):
        self.work["events"].append("sum")
        with np.errstate(over="ignore", invalid="ignore"):
            value = self.values.sum(dtype=np.float32)
        return Tensor(value, self.work)

    def item(self):
        return self.values.item()

    def backward(self):
        self.work["events"].append("backward")
        parameters = self.work["staged_parameters"]
        for value in parameters.values():
            value.grad = Tensor(np.zeros_like(value.values), self.work)
        fault = self.work.get("gradient_fault")
        selected = parameters["bias_output"]
        if fault == "none":
            selected.grad = None
        elif fault == "sparse":
            selected.grad.is_sparse = True
        elif fault == "csr":
            selected.grad.layout = "sparse_csr"
        elif fault in ("nan", "inf", "nonzero"):
            selected.grad.values.flat[0] = {"nan": np.nan, "inf": np.inf, "nonzero": 0.125}[fault]
        if self.work.get("failure") == "backward":
            raise RuntimeError("staged backward failed")


class Optimizer:
    def __init__(self, parameters, work):
        self.parameters, self.work = parameters, work
        self.momentum = {key: np.full_like(value.values, 0.25) for key, value in parameters.items()}

    def zero_grad(self, *, set_to_none):
        assert set_to_none is True
        self.work["events"].append("zero_grad")
        for value in self.parameters.values():
            value.grad = None
        if self.work.get("failure") == "zero_grad":
            raise RuntimeError("staged zero_grad failed")

    def step(self):
        self.work["events"].append("step")
        for key, parameter in self.parameters.items():
            assert parameter.grad is not None and not parameter.grad.is_sparse
            assert (
                np.isfinite(parameter.grad.values).all()
                and np.count_nonzero(parameter.grad.values) == 0
            )
            self.momentum[key] = 0.9 * self.momentum[key]
            parameter.values -= 0.03 * self.momentum[key]
        if self.work.get("failure") == "step":
            raise RuntimeError("staged optimizer step failed")


def trajectory(*, role="train", dimension=3, classes=2):
    transitions = tuple(
        DemonstrationTransition(
            index,
            (0.5 * index,) + (0.0,) * (dimension - 1),
            index % classes,
            0.75 if index == 0 else -0.75,
            (0.5 * (index + 1),) + (0.0,) * (dimension - 1),
            index == 1,
            classes,
        )
        for index in range(2)
    )
    return PolicyGradientEpisode(
        "observed-reward",
        "A",
        role,
        transitions,
        ((1.0 / classes,) * classes,) * 2,
        "terminal",
        classes,
    )


@pytest.fixture
def staged(monkeypatch):
    work: dict[str, Any] = {"events": [], "feature_inputs": [], "captures": []}
    policy = native.TorchRecurrentPolicy(3, 2, 2, 17)
    learner = object.__new__(native.TorchRecurrentLearner)
    learner._policy = policy
    learner._payload = ({"live": [1.0]}, {"momentum": [0.5]}, 3, 6)
    saved = SimpleNamespace(policy=policy, completed_updates=3, training_examples=6)

    def snapshot(owner):
        assert owner is learner
        work["events"].append("snapshot")
        return deepcopy(saved)

    def features(values, bound_policy):
        assert bound_policy == policy
        work["events"].append("features")
        work["feature_inputs"].append(values.copy())
        return Tensor(values, work)

    def restore(state):
        assert state.completed_updates == saved.completed_updates
        work["events"].append("restore")
        shapes = dict(
            weight_input=(3, 2),
            weight_recurrent=(2, 2),
            bias_hidden=(1, 2),
            weight_output=(2, 2),
            bias_output=(1, 2),
        )
        parameters = {key: Tensor(np.full(shape, 0.5), work) for key, shape in shapes.items()}
        work["staged_parameters"] = parameters
        optimizer = Optimizer(parameters, work)
        work["staged_optimizer"] = optimizer
        return parameters, optimizer

    def forward(parameters, values, hidden):
        work["events"].append("forward")
        assert set(parameters) == {
            "weight_input",
            "weight_recurrent",
            "bias_hidden",
            "weight_output",
            "bias_output",
        }
        np.testing.assert_array_equal(hidden.values, np.zeros((1, 2), dtype=np.float32))
        if work.get("failure") == "forward":
            raise RuntimeError("staged forward failed")
        value = work.get("logit", np.finfo(np.float32).max / 2)
        return Tensor(np.full((len(values.values), 2), value), work), hidden

    def capture(parameters, optimizer, bound_policy, updates, examples):
        work["events"].append("capture")
        assert bound_policy == policy
        work["captures"].append((parameters, optimizer, updates, examples))
        assert all(
            value.grad is not None and np.count_nonzero(value.grad.values) == 0
            for value in parameters.values()
        )
        if work.get("failure") == "capture":
            raise ValueError("staged full snapshot refused")
        return SimpleNamespace(completed_updates=updates, training_examples=examples)

    def tensor(value, shape, name):
        assert type(value) is Tensor and value.shape == shape
        if not np.isfinite(value.values).all():
            raise ValueError("staged gradient is nonfinite")

    def forbidden(*args, **kwargs):
        pytest.fail("the zero-loss control must not call an existing reward or CE objective")

    torch = SimpleNamespace(
        float32=np.float32,
        strided="strided",
        zeros=lambda shape, **options: Tensor(np.zeros(shape, dtype=np.float32), work),
        isfinite=lambda value: Result(np.isfinite(value.values)),
        count_nonzero=lambda value: Result(np.count_nonzero(value.values)),
    )
    monkeypatch.setattr(native.TorchRecurrentLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(native.TorchRecurrentLearner, "train_reward_episode", forbidden)
    monkeypatch.setattr(native.TorchRecurrentLearner, "train_episode", forbidden)
    monkeypatch.setattr(native, "_features", features)
    monkeypatch.setattr(native, "_restore", restore)
    monkeypatch.setattr(native, "_forward", forward)
    monkeypatch.setattr(native, "_capture", capture)
    monkeypatch.setattr(native, "_tensor", tensor)
    monkeypatch.setattr(control, "require_torch", lambda: torch)
    return SimpleNamespace(learner=learner, saved=saved, work=work)


@pytest.mark.parametrize("entry", ["function", "method"])
def test_should_stage_dense_zero_gradients_before_sgd_and_publish_only_validated_state(
    staged, entry
):
    episode = trajectory()
    original = pickle.dumps(episode)
    before = staged.learner._payload
    result = (
        control.train_zero_loss_episode(staged.learner, episode)
        if entry == "function"
        else staged.learner.train_zero_loss_episode(episode)
    )
    assert result == TrainingDiagnostic(control.TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)
    assert staged.work["events"] == [
        "snapshot",
        "features",
        "restore",
        "zero_grad",
        "forward",
        "elementwise_zero",
        "sum",
        "backward",
        "step",
        "capture",
    ]
    parameters, optimizer, updates, examples = staged.learner._payload
    assert (updates, examples) == (4, 8)
    assert staged.learner._payload is not before
    assert staged.work["captures"] == [(parameters, optimizer, 4, 8)]
    for name, parameter in parameters.items():
        assert parameter.grad is not None and parameter.grad.values.dtype == np.float32
        np.testing.assert_array_equal(parameter.grad.values, np.zeros_like(parameter.values))
        np.testing.assert_array_equal(
            optimizer.momentum[name], np.full_like(parameter.values, 0.225)
        )
    np.testing.assert_array_equal(
        staged.work["feature_inputs"][0], [step.observation for step in episode.transitions]
    )
    assert staged.work["feature_inputs"][0].dtype == np.float64
    assert pickle.dumps(episode) == original
    assert [step.reward for step in episode.transitions] == [0.75, -0.75]


@pytest.mark.parametrize("fault", ["none", "sparse", "csr", "nan", "inf", "nonzero"])
def test_should_refuse_missing_sparse_or_nonzero_dense_gradients_without_publishing(staged, fault):
    staged.work["gradient_fault"] = fault
    before = staged.learner._payload
    with pytest.raises(ValueError):
        control.train_zero_loss_episode(staged.learner, trajectory())
    assert staged.learner._payload is before
    assert "step" not in staged.work["events"] and staged.work["captures"] == []


@pytest.mark.parametrize("failure", ["zero_grad", "forward", "backward", "step", "capture"])
def test_should_keep_live_payload_and_counters_atomic_after_any_staged_failure(staged, failure):
    staged.work["failure"] = failure
    before = staged.learner._payload
    original = pickle.dumps(before)
    with pytest.raises((ValueError, RuntimeError), match="staged"):
        control.train_zero_loss_episode(staged.learner, trajectory())
    assert staged.learner._payload is before and pickle.dumps(before) == original
    assert staged.learner.completed_updates == 3 and staged.learner.training_examples == 6


@pytest.mark.parametrize("logit", [np.nan, np.inf])
def test_should_refuse_nonfinite_forward_before_backward_or_optimizer_step(staged, logit):
    staged.work["logit"] = logit
    before = staged.learner._payload
    with pytest.raises(FloatingPointError):
        control.train_zero_loss_episode(staged.learner, trajectory())
    assert staged.learner._payload is before
    assert "backward" not in staged.work["events"] and "step" not in staged.work["events"]


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_non_train_episodes_before_snapshot_or_forward(staged, role):
    with pytest.raises(ValueError):
        control.train_zero_loss_episode(staged.learner, trajectory(role=role))
    assert staged.work["events"] == []


@pytest.mark.parametrize("kind", ["foreign", "subclass"])
def test_should_require_exact_native_learner_before_any_staged_operation(staged, kind):
    class Derived(native.TorchRecurrentLearner):
        pass

    owner: Any = SimpleNamespace() if kind == "foreign" else object.__new__(Derived)
    with pytest.raises(ValueError, match="exact TorchRecurrentLearner"):
        control.train_zero_loss_episode(owner, trajectory())
    assert staged.work["events"] == []


@pytest.mark.parametrize("fault", ["width", "classes", "length", "extra_fields", "wrong_type"])
def test_should_refuse_wrong_episode_contract_before_native_staging(staged, fault):
    episode: Any = trajectory(
        dimension=2 if fault == "width" else 3, classes=3 if fault == "classes" else 2
    )
    if fault == "length":
        object.__setattr__(staged.learner._policy, "max_episode_steps", 1)
    elif fault == "extra_fields":
        object.__setattr__(episode, "unobserved", True)
    elif fault == "wrong_type":
        episode = {"episode": episode}
    with pytest.raises(ValueError):
        control.train_zero_loss_episode(staged.learner, episode)
    assert staged.work["events"] == []


@pytest.mark.parametrize("field", ["completed_updates", "training_examples"])
def test_should_refuse_exhausted_native_counters_before_restore(staged, field):
    setattr(staged.saved, field, native._MAX_COUNT)
    before = staged.learner._payload
    with pytest.raises(ValueError):
        control.train_zero_loss_episode(staged.learner, trajectory())
    assert staged.learner._payload is before
    assert staged.work["events"] == ["snapshot"]


class PortNative(PreviousNative):
    def train_zero_loss_episode(self, episode):
        self.calls.append(("momentum_only", episode))
        if self.failure is not None:
            raise self.failure
        self.state.optimizer[0] *= self.policy.momentum
        self.state.parameters[0] -= self.policy.learning_rate * self.state.optimizer[0]
        self.state.gradients[0] = 0.0
        self.state.completed_updates += 1
        self.state.training_examples += len(episode.transitions)
        return TrainingDiagnostic(control.TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)


@pytest.fixture
def typed_case(monkeypatch):
    monkeypatch.setattr(typed, "TorchRecurrentLearner", PortNative)
    learner = typed.TorchDemonstrationRewardLearner(native.TorchRecurrentPolicy(2, 3, 3, 17))
    return SimpleNamespace(learner=learner, native=learner._learner)


def test_should_copy_original_episode_without_relabeling_and_release_only_momentum_objective(
    typed_case,
):
    episode = packet()
    update = typed.MomentumOnlyUpdate(episode)
    original = pickle.dumps(update)
    diagnostic = typed_case.learner.train_batch(update, episode.episode_id)
    assert diagnostic == TrainingDiagnostic(control.TORCH_RECURRENT_ZERO_LOSS_ID, 0.0)
    name, owned = typed_case.native.calls[-1]
    assert name == "momentum_only" and owned == episode and owned is not update.episode
    assert (
        update.episode is not episode
        and update.episode.transitions[0] is not episode.transitions[0]
    )
    assert owned.transitions[0].reward == 1.0 and owned.behavior_probabilities == ((0.2, 0.3, 0.5),)
    assert set(vars(update)) == {"episode"} and pickle.dumps(update) == original
    object.__setattr__(episode.transitions[0], "reward", -999.0)
    assert update.episode.transitions[0].reward == owned.transitions[0].reward == 1.0


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_non_train_momentum_packet_at_construction(typed_case, role):
    with pytest.raises(ValueError, match="train-role"):
        typed.MomentumOnlyUpdate(packet(role))
    assert typed_case.native.calls == []


@pytest.mark.parametrize("target", [None, True, "wrong", " reward-1", np.zeros((1, 1))])
def test_should_require_exact_matching_release_id_without_native_entry(typed_case, target):
    update = typed.MomentumOnlyUpdate(packet())
    with pytest.raises(ValueError, match="release identifier"):
        typed_case.learner.train_batch(update, target)
    assert typed_case.native.calls == []


@pytest.mark.parametrize("fault", ["extra", "episode_role", "class_count", "wrong_type"])
def test_should_revalidate_complete_typed_packet_before_native_entry(typed_case, fault):
    update: Any = typed.MomentumOnlyUpdate(packet())
    if fault == "extra":
        object.__setattr__(update, "behavior_state", typed_case.learner.snapshot_state())
    elif fault == "episode_role":
        object.__setattr__(update.episode, "role", "final_test")
    elif fault == "class_count":
        source = trajectory(dimension=2)
        update = typed.MomentumOnlyUpdate(source)
    else:
        update = update.episode
    release_id = (
        update.episode.episode_id if type(update) is typed.MomentumOnlyUpdate else "reward-1"
    )
    with pytest.raises(ValueError):
        typed_case.learner.train_batch(update, release_id)
    assert typed_case.native.calls == []


def test_should_keep_existing_demonstration_and_reward_dispatch_on_same_native_owner(typed_case):
    learner, owner = typed_case.learner, typed_case.native
    demonstration_update = typed.DemonstrationUpdate(demonstration())
    first = learner.train_batch(demonstration_update, demonstration_update.episode.episode_id)
    momentum = typed.MomentumOnlyUpdate(packet())
    second = learner.train_batch(momentum, momentum.episode.episode_id)
    reward = RewardUpdate(packet(), learner.snapshot_state())
    third = learner.train_batch(reward, reward.episode.episode_id)
    assert learner._learner is owner
    assert [first.definition, second.definition, third.definition] == [
        "fake_episode_ce_mean",
        control.TORCH_RECURRENT_ZERO_LOSS_ID,
        "fake_reward_sum",
    ]
    assert owner.calls[0][0] == "demonstration" and owner.calls[1][0] == "momentum_only"
    assert owner.calls[2][0] == reward.episode and owner.calls[2][2] == 1.0
    assert learner.completed_updates == 3 and learner.training_examples == 3


def test_should_refuse_subclass_or_forged_momentum_dto():
    class Derived(typed.MomentumOnlyUpdate):
        pass

    with pytest.raises(ValueError, match="exact complete"):
        Derived(packet())


def test_should_preserve_atomic_native_refusal_at_typed_boundary(typed_case):
    typed_case.native.failure = RuntimeError("zero-loss native stage refused")
    before = typed_case.learner.snapshot_state()
    with pytest.raises(RuntimeError, match="native stage refused"):
        typed_case.learner.train_batch(typed.MomentumOnlyUpdate(packet()), "reward-1")
    assert typed_case.learner.snapshot_state() == before
    assert len(typed_case.native.calls) == 1 and typed_case.native.calls[0][0] == "momentum_only"
