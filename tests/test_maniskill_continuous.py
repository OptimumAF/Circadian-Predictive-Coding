"""Boundary checks use a fake native environment, not robot learning evidence."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.continuous_demonstration import continuous_training_pairs
from src.infra import maniskill_continuous as port


class ForeignArray(np.ndarray):
    pass


class ForeignTuple(tuple):
    pass


class FakeEnvironment:
    """A deterministic boundary double; deliberately no simulator dynamics."""

    def __init__(self, *, steps=2, terminated=True, truncated=False):
        self.unwrapped = self
        self.num_envs = 1
        self.obs_mode = "state"
        self.control_mode = "pd_ee_delta_pos"
        self.device = SimpleNamespace(type="cpu")
        self.sim_freq = 100
        self.control_freq = 20
        self.single_action_space = SimpleNamespace(
            low=np.full(4, -1, dtype=np.float32), high=np.ones(4, dtype=np.float32), shape=(4,)
        )
        self.agent = SimpleNamespace(uid="panda")
        self.spec = SimpleNamespace(id="PickCube-v1", max_episode_steps=50)
        self.steps = steps
        self.terminated = terminated
        self.truncated = truncated
        self.reset_calls = []
        self.commands = []
        self.close_calls = 0
        self.reset_error = None
        self.step_error = None
        self.reset_observation = np.array([[100, -50, 0.25]], dtype=np.float32)
        self.last_observation = None
        self.output_override = None

    def reset(self, *, seed):
        self.reset_calls.append(seed)
        if self.reset_error is not None:
            raise self.reset_error
        return self.reset_observation, {}

    def step(self, command):
        self.commands.append(command.copy())
        if self.step_error is not None:
            raise self.step_error
        if self.output_override is not None:
            return self.output_override
        index = len(self.commands)
        self.last_observation = np.array([[100 + index, -50, 0.25]], dtype=np.float32)
        done = index == self.steps
        return (
            self.last_observation,
            np.array([37.5], dtype=np.float32),
            np.array([done and self.terminated], dtype=np.bool_),
            np.array([done and self.truncated], dtype=np.bool_),
            {},
        )

    def close(self):
        self.close_calls += 1


@pytest.fixture
def fake_environment(monkeypatch):
    environment = FakeEnvironment()
    calls = []

    def factory():
        calls.append(True)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    return environment, calls


def open_episode():
    return port.ManiSkillPickCubeEpisode("robot-1", "pick-cube", seed=17)


@pytest.mark.parametrize("model", [None, False, 0, "unavailable"])
def test_should_explain_missing_cpu_kinematics_before_robot_creation(monkeypatch, model):
    monkeypatch.setattr(port, "import_module", lambda name: SimpleNamespace(PinocchioModel=model))
    with pytest.raises(RuntimeError, match="compatible Pinocchio installation"):
        port._require_cpu_kinematics()


def test_should_accept_available_cpu_kinematics_without_creating_a_model(monkeypatch):
    called = []

    def model(*args):
        called.append(args)

    monkeypatch.setattr(port, "import_module", lambda name: SimpleNamespace(PinocchioModel=model))
    port._require_cpu_kinematics()
    assert called == []


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dimension", [1, 3, 1024])
def test_should_encode_owned_native_observations_without_toy_bounds(dtype, dimension):
    source = np.full((1, dimension), 100.25, dtype=dtype)
    actual = port.encode_state_observation(source)
    source.fill(0)
    assert type(actual) is tuple
    assert actual == (100.25,) * dimension
    assert all(type(value) is float for value in actual)


@pytest.mark.parametrize(
    "source",
    [
        [[1.0, 2.0]],
        (1.0, 2.0),
        np.ones(2, dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((1, 0), dtype=np.float32),
        np.ones((1, 1025), dtype=np.float32),
        np.ones((1, 2), dtype=np.int64),
        np.ones((1, 2), dtype=np.float16),
        np.ones((1, 2), dtype=np.float32).view(ForeignArray),
        np.array([[np.nan]], dtype=np.float64),
        np.array([[np.inf]], dtype=np.float32),
        np.array([[-np.inf]], dtype=np.float64),
    ],
)
def test_should_refuse_ambiguous_or_invalid_state_arrays(source):
    with pytest.raises(ValueError):
        port.encode_state_observation(source)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_should_encode_cpu_torch_state_without_retaining_tensor_storage(dtype_name):
    import torch

    source = torch.tensor([[101.25, -50.5]], dtype=getattr(torch, dtype_name))
    actual = port.encode_state_observation(source)
    source.fill_(0)
    assert actual == (101.25, -50.5)
    assert all(type(value) is float for value in actual)


@pytest.mark.parametrize("case", ["integer", "gradient", "batch"])
def test_should_refuse_torch_state_that_would_change_contract_or_detach_training(case):
    import torch

    source = torch.ones((1, 3))
    if case == "integer":
        source = source.to(dtype=torch.int64)
    elif case == "gradient":
        source.requires_grad_(True)
    else:
        source = source.expand(2, 3)
    with pytest.raises(ValueError):
        port.encode_state_observation(source)


def test_should_convert_cpu_tensor_step_outputs_without_implicit_squeezing(fake_environment):
    import torch

    environment, _ = fake_environment
    environment.reset_observation = torch.tensor([[100.0, -50.0, 0.25]])
    session = open_episode()
    native_after = torch.tensor([[101.0, -50.0, 0.25]])
    environment.output_override = (
        native_after,
        torch.tensor([2.5]),
        torch.tensor([False]),
        torch.tensor([True]),
        {},
    )
    record = session.step((0.0,) * 4)
    native_after.fill_(999)
    assert record.next_observation == (101.0, -50.0, 0.25)
    assert record.reward == 2.5
    assert record.truncated and not record.terminated
    assert session.episode().transitions == (record,)
    assert session.work.completed_control_physics_steps == 5
    session.close()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("success", [False, True])
def test_should_preserve_native_sparse_boolean_reward_as_numeric_zero_or_one(
    fake_environment, backend, success
):
    environment, _ = fake_environment
    reward = np.array([success], dtype=np.bool_)
    if backend == "torch":
        import torch

        reward = torch.tensor([success], dtype=torch.bool)
    environment.output_override = (
        np.array([[101.0, -50.0, 0.25]], dtype=np.float32),
        reward,
        np.array([True], dtype=np.bool_),
        np.array([False], dtype=np.bool_),
        {},
    )
    session = open_episode()
    record = session.step((0.0,) * 4)
    assert type(record.reward) is float
    assert record.reward == float(success)
    assert record.terminated is True and record.truncated is False
    assert session.episode().transitions == (record,)
    session.close()


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_should_capture_actual_commands_and_separate_completion_flags(
    fake_environment, terminated, truncated
):
    environment, calls = fake_environment
    environment.terminated = terminated
    environment.truncated = truncated
    session = open_episode()
    assert calls == [True]
    assert environment.reset_calls == [17]
    assert session.observation == (100.0, -50.0, 0.25)
    assert session.spec.observation_dim == 3
    assert session.spec.lower_bounds == (-1.0,) * 4
    assert session.spec.upper_bounds == (1.0,) * 4
    assert len(session.spec.action_units) == 4
    assert all("norm" in unit.lower() for unit in session.spec.action_units)

    first = session.step((0.123456789, -1.0, 1.0, 0.0))
    second = session.step((0.0, 0.0, 0.0, 0.0))
    packet = session.episode()
    assert first.action == tuple(float(value) for value in environment.commands[0][0])
    assert first.action[0] != 0.123456789
    assert environment.commands[0].shape == (1, 4)
    assert environment.commands[0].dtype == np.float32
    assert first.next_observation == second.observation
    assert first.step == 0 and second.step == 1
    assert first.reward == 37.5
    assert not first.terminated and not first.truncated
    assert second.terminated is terminated and second.truncated is truncated
    assert packet.transitions == (first, second)
    assert packet.role == "train"
    assert session.observation == second.next_observation
    assert session.work.step_attempts == 2
    assert session.work.completed_steps == 2
    assert session.work.completed_control_physics_steps == 10
    assert session.work.possible_failed_physics_steps == 0
    assert continuous_training_pairs(packet)[1] == (first.action, second.action)

    with pytest.raises((RuntimeError, ValueError)):
        session.step((0.0,) * 4)
    assert len(environment.commands) == 2
    session.close()
    session.close()
    assert environment.close_calls == 1


@pytest.mark.parametrize("role", ["train", "probe_train", "probe_test"])
def test_should_preserve_explicit_permitted_role(fake_environment, role):
    environment, _ = fake_environment
    environment.steps = 1
    with port.ManiSkillPickCubeEpisode("robot-1", "pick", role=role) as session:
        session.step((0.0,) * 4)
        packet = session.episode()
    assert packet.role == role
    assert environment.close_calls == 1
    if role == "probe_test":
        with pytest.raises(ValueError):
            continuous_training_pairs(packet)


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": ""},
        {"episode_id": " robot"},
        {"episode_id": "x" * 65},
        {"context": ""},
        {"context": "x" * 65},
        {"role": "final_test"},
        {"role": "outer_selection"},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": np.int64(17)},
    ],
)
def test_should_refuse_invalid_identity_role_or_seed_before_environment_creation(
    fake_environment, changes
):
    _, calls = fake_environment
    values = dict(episode_id="robot", context="pick", role="train", seed=17)
    values.update(changes)
    with pytest.raises(ValueError):
        port.ManiSkillPickCubeEpisode(**values)
    assert calls == []


@pytest.mark.parametrize(
    "action",
    [
        [0.0] * 4,
        np.zeros(4),
        ForeignTuple((0.0,) * 4),
        (),
        (0.0,) * 3,
        (0.0,) * 5,
        (True, 0.0, 0.0, 0.0),
        (np.float32(0), 0.0, 0.0, 0.0),
        (np.nan, 0.0, 0.0, 0.0),
        (np.inf, 0.0, 0.0, 0.0),
        (1.000000001, 0.0, 0.0, 0.0),
        (-1.000000001, 0.0, 0.0, 0.0),
    ],
)
def test_should_reject_invalid_action_before_quantization_and_native_work(fake_environment, action):
    environment, _ = fake_environment
    session = open_episode()
    with pytest.raises(ValueError):
        session.step(action)
    assert environment.commands == []
    assert session.work.step_attempts == 0
    session.step((0.0,) * 4)
    assert len(environment.commands) == 1
    session.close()


def test_should_own_observations_and_immutable_work_receipts(fake_environment):
    environment, _ = fake_environment
    session = open_episode()
    initial = session.observation
    before = session.work
    environment.reset_observation.fill(999)
    first = session.step((0.0,) * 4)
    environment.last_observation.fill(-999)
    second = session.step((0.0,) * 4)
    assert initial == first.observation == (100.0, -50.0, 0.25)
    assert first.next_observation == second.observation == (101.0, -50.0, 0.25)
    assert before.step_attempts == 0
    with pytest.raises((FrozenInstanceError, AttributeError)):
        before.step_attempts = 100
    assert session.episode().transitions == (first, second)
    session.close()


def test_should_refuse_incomplete_capture_and_closed_environment_without_native_work(
    fake_environment,
):
    environment, _ = fake_environment
    session = open_episode()
    with pytest.raises((RuntimeError, ValueError)):
        session.episode()
    session.step((0.0,) * 4)
    with pytest.raises((RuntimeError, ValueError)):
        session.episode()
    session.close()
    with pytest.raises((RuntimeError, ValueError)):
        session.step((0.0,) * 4)
    assert len(environment.commands) == 1


def test_should_charge_and_poison_native_step_exception_without_reusing_prior_trace(
    fake_environment,
):
    environment, _ = fake_environment
    session = open_episode()
    session.step((0.0,) * 4)
    environment.step_error = RuntimeError("native failure after possible physics")
    with pytest.raises(RuntimeError, match="native failure"):
        session.step((0.0,) * 4)
    assert session.work.step_attempts == 2
    assert session.work.completed_steps == 1
    assert session.work.completed_control_physics_steps == 5
    assert session.work.possible_failed_physics_steps == 5
    for operation in (session.episode, lambda: session.step((0.0,) * 4)):
        with pytest.raises((RuntimeError, ValueError)):
            operation()
    assert len(environment.commands) == 2
    session.close()


@pytest.mark.parametrize(
    "output",
    [
        (np.zeros((1, 2), np.float32), np.zeros(1), np.zeros(1, bool), np.zeros(1, bool), {}),
        (np.full((1, 3), np.nan), np.zeros(1), np.zeros(1, bool), np.zeros(1, bool), {}),
        (np.zeros((1, 3)), np.array([np.inf]), np.zeros(1, bool), np.zeros(1, bool), {}),
        (np.zeros((1, 3)), np.array([1], dtype=np.int64), np.zeros(1, bool), np.zeros(1, bool), {}),
        (np.zeros((1, 3)), 1.0, np.zeros(1, bool), np.zeros(1, bool), {}),
        (np.zeros((1, 3)), np.zeros(1), np.array([1]), np.zeros(1, bool), {}),
        (np.zeros((1, 3)), np.zeros(1), np.zeros(1, bool), np.array([0]), {}),
        (np.zeros((1, 3)), np.zeros(1), False, False, {}),
        (np.zeros((1, 3)), np.zeros(1), np.zeros(1, bool), np.zeros(1, bool)),
    ],
)
def test_should_poison_and_charge_untrusted_step_output(fake_environment, output):
    environment, _ = fake_environment
    session = open_episode()
    environment.output_override = output
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        session.step((0.0,) * 4)
    assert session.work.step_attempts == 1
    assert session.work.completed_steps == 1
    assert session.work.completed_control_physics_steps == 5
    assert session.work.possible_failed_physics_steps == 0
    with pytest.raises((RuntimeError, ValueError)):
        session.episode()
    with pytest.raises((RuntimeError, ValueError)):
        session.step((0.0,) * 4)
    assert len(environment.commands) == 1
    session.close()


@pytest.mark.parametrize(
    "name,value",
    [
        ("num_envs", 2),
        ("obs_mode", "rgb"),
        ("control_mode", "pd_ee_delta_pose"),
        ("device", SimpleNamespace(type="cuda")),
        ("sim_freq", 101),
        ("control_freq", 0),
        ("agent", SimpleNamespace(uid="other_robot")),
        (
            "single_action_space",
            SimpleNamespace(
                low=np.full(4, -2, np.float32), high=np.ones(4, np.float32), shape=(4,)
            ),
        ),
        (
            "single_action_space",
            SimpleNamespace(
                low=np.full(3, -1, np.float32), high=np.ones(3, np.float32), shape=(3,)
            ),
        ),
    ],
)
def test_should_close_and_refuse_mismatched_native_contract(fake_environment, name, value):
    environment, _ = fake_environment
    setattr(environment, name, value)
    with pytest.raises((ValueError, RuntimeError)):
        open_episode()
    assert environment.close_calls == 1
    assert environment.commands == []


@pytest.mark.parametrize("bad_reset", [np.zeros((1, 2), dtype=np.int64), np.array([[np.nan]])])
def test_should_close_on_invalid_reset_observation(fake_environment, bad_reset):
    environment, _ = fake_environment
    environment.reset_observation = bad_reset
    with pytest.raises(ValueError):
        open_episode()
    assert environment.close_calls == 1
    assert environment.commands == []


def test_should_close_on_reset_exception(fake_environment):
    environment, _ = fake_environment
    environment.reset_error = RuntimeError("reset failed")
    with pytest.raises(RuntimeError, match="reset failed"):
        open_episode()
    assert environment.close_calls == 1


def test_should_close_context_on_caller_exception(fake_environment):
    environment, _ = fake_environment
    with pytest.raises(RuntimeError, match="caller failure"):
        with open_episode():
            raise RuntimeError("caller failure")
    assert environment.close_calls == 1


@pytest.mark.parametrize("truncated", [False, True])
def test_should_bound_storage_at_official_horizon_without_inventing_completion(
    fake_environment, truncated
):
    environment, _ = fake_environment
    environment.steps = 50
    environment.terminated = False
    environment.truncated = truncated
    session = open_episode()
    for _ in range(49):
        session.step((0.0,) * 4)
    if truncated:
        last = session.step((0.0,) * 4)
        packet = session.episode()
        assert len(packet.transitions) == 50
        assert last.truncated and not last.terminated
    else:
        with pytest.raises(ValueError, match="time-limit"):
            session.step((0.0,) * 4)
        with pytest.raises(ValueError):
            session.episode()
    with pytest.raises(ValueError):
        session.step((0.0,) * 4)
    assert len(environment.commands) == 50
    assert session.work.step_attempts == session.work.completed_steps == 50
    assert session.work.completed_control_physics_steps == 250
    assert session.work.possible_failed_physics_steps == 0
    session.close()


def test_should_keep_live_spec_independent_of_returned_spec_mutation(fake_environment):
    environment, _ = fake_environment
    environment.steps = 1
    session = open_episode()
    visible_spec = session.spec
    object.__setattr__(visible_spec, "lower_bounds", (-999.0,) * 4)
    with pytest.raises(ValueError):
        session.step((-2.0, 0.0, 0.0, 0.0))
    assert environment.commands == []
    session.step((0.0,) * 4)
    assert session.episode().spec.lower_bounds == (-1.0,) * 4
    session.close()
