"""MiniGrid public-observation and charged-step boundaries without optional deps."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.core.demonstration_episode import (
    EpisodeReplaySchema,
    decode_episode,
    encode_episode,
    episode_arrays,
)
from src.infra import minigrid_memory as memory


def observation(*, direction: int = 2) -> dict[str, Any]:
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    for x in range(7):
        for y in range(7):
            image[x, y] = ((x + y) % 11, y % 6, x % 3)
    return dict(image=image, direction=direction, mission=memory.MISSION)


def result(*, terminated=False, truncated=False, reward=0.0, next_observation=None):
    return (
        observation() if next_observation is None else next_observation,
        reward,
        terminated,
        truncated,
        {"hidden_target": "must not enter learner data"},
    )


class Environment:
    def __init__(self, *, initial=None, results=None, reset_error=None):
        self.action_space = SimpleNamespace(n=7)
        self.initial = observation() if initial is None else initial
        self.results = list(results or [])
        self.reset_error = reset_error
        self.reset_seeds: list[int] = []
        self.actions: list[int] = []
        self.close_calls = 0

    @property
    def unwrapped(self):
        raise AssertionError("private environment access is not permitted")

    @property
    def grid(self):
        raise AssertionError("private grid access is not permitted")

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        if self.reset_error is not None:
            raise self.reset_error
        return self.initial, {"hidden_target": "must not enter learner data"}

    def step(self, action):
        self.actions.append(action)
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.close_calls += 1


@pytest.fixture
def boundary(monkeypatch):
    environments: list[Environment] = []
    limits: list[int] = []

    def create(*, environment=None, **options):
        owned = Environment() if environment is None else environment

        def factory(max_steps):
            environments.append(owned)
            limits.append(max_steps)
            return owned

        monkeypatch.setattr(memory, "_make_memory_environment", factory)
        return memory.MiniGridMemory(17, **options), owned

    return SimpleNamespace(create=create, environments=environments, limits=limits)


def test_should_encode_every_public_pixel_channel_in_c_order_and_direction_last():
    public = observation(direction=1)
    public["image"][6, 6] = (10, 5, 2)
    expected = []
    for x in range(7):
        for y in range(7):
            for channel, divisor in enumerate((10, 5, 2)):
                expected.append(2 * int(public["image"][x, y, channel]) / divisor - 1)
    expected.append(-1 / 3)

    encoded = memory.encode_memory_observation(public)

    assert type(encoded) is tuple and len(encoded) == 148
    assert all(type(value) is float and -1 <= value <= 1 for value in encoded)
    np.testing.assert_allclose(encoded, expected, rtol=0, atol=2e-16)
    assert encoded[144:147] == (1.0, 1.0, 1.0)
    public["image"].fill(0)
    assert encoded[144:147] == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("direction,expected", [(0, -1.0), (3, 1.0)])
def test_should_preserve_direction_extrema(direction, expected):
    assert memory.encode_memory_observation(observation(direction=direction))[-1] == expected


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "dtype",
        "list",
        "object",
        "color",
        "state",
        "negative_signed",
        "bool_direction",
        "numpy_direction",
        "direction",
        "mission",
        "missing",
        "extra",
    ],
)
def test_should_refuse_incomplete_or_changed_observation_contract(fault):
    public = observation()
    if fault == "shape":
        public["image"] = np.zeros((3, 7, 7), dtype=np.uint8)
    elif fault == "dtype":
        public["image"] = public["image"].astype(np.float64)
    elif fault == "list":
        public["image"] = public["image"].tolist()
    elif fault in ("object", "color", "state"):
        channel = ("object", "color", "state").index(fault)
        public["image"][0, 0, channel] = (11, 6, 3)[channel]
    elif fault == "negative_signed":
        public["image"] = np.full((7, 7, 3), -1, dtype=np.int8)
    elif fault == "bool_direction":
        public["direction"] = True
    elif fault == "numpy_direction":
        public["direction"] = np.int64(2)
    elif fault == "direction":
        public["direction"] = 4
    elif fault == "mission":
        public["mission"] += " hidden cue"
    elif fault == "missing":
        del public["direction"]
    else:
        public["hidden_target"] = 1

    with pytest.raises(ValueError):
        memory.encode_memory_observation(public)


def test_should_reset_once_with_explicit_seed_and_keep_only_public_initial_data(boundary):
    adapter, environment = boundary.create(max_steps=7, context="memory-A", episode_id="A-17")

    assert environment.reset_seeds == [17]
    assert boundary.limits == [7]
    assert adapter.observation == memory.encode_memory_observation(environment.initial)
    report = adapter.report
    assert report["seed"] == 17 and report["max_steps"] == 7
    assert report["context"] == "memory-A" and report["episode_id"] == "A-17"
    assert report["observation_dim"] == 148 and report["num_classes"] == 7
    assert report["reset_calls"] == 1
    assert report["environment_steps_attempted"] == report["environment_steps"] == 0
    assert report["failure_work_may_be_unobserved"] is False
    assert set(report["initial_observation"]) == {"image", "direction", "mission"}
    assert "hidden_target" not in repr(report)
    assert report["status"] == "running" and not adapter.terminal


@pytest.mark.parametrize(
    "options",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**31},
        {"max_steps": True},
        {"max_steps": 0},
        {"max_steps": 17},
        {"role": "validation"},
        {"episode_id": ""},
        {"episode_id": "é" * 33},
        {"context": " memory"},
    ],
)
def test_should_refuse_invalid_constructor_before_allocating_environment(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid constructor allocated an environment")

    monkeypatch.setattr(memory, "_make_memory_environment", forbidden)
    arguments: dict[str, Any] = dict(seed=17)
    arguments.update(options)
    with pytest.raises(ValueError):
        memory.MiniGridMemory(**arguments)


def test_should_forward_all_seven_native_actions_once_including_ineffective_actions(boundary):
    environment = Environment(results=[result(terminated=index == 6) for index in range(7)])
    adapter, _ = boundary.create(environment=environment, max_steps=7)
    initial = adapter.observation

    for action in range(7):
        assert adapter.legal_actions == tuple(range(7))
        step = adapter.step(action)
        assert step["observation"] == initial  # Native no-ops are not repaired or retried.
        assert step["terminated"] is (action == 6)
        assert step["truncated"] is False

    assert environment.actions == list(range(7))
    assert adapter.terminal and adapter.legal_actions == ()
    assert adapter.report["environment_steps_attempted"] == 7
    assert adapter.report["environment_steps"] == 7
    assert [row["action"] for row in adapter.report["transitions"]] == list(range(7))
    assert all(row["attempted"] and row["applied"] for row in adapter.report["transitions"])


@pytest.mark.parametrize("action", [True, -1, 7, 0.0, np.int64(2), "forward"])
def test_should_refuse_invalid_action_without_charging_or_calling_environment(boundary, action):
    adapter, environment = boundary.create()
    before = adapter.report
    with pytest.raises(ValueError):
        adapter.step(action)
    assert adapter.report == before
    assert environment.actions == []


def test_should_detach_reset_step_and_report_evidence_from_mutable_native_data(boundary):
    following = observation(direction=3)
    environment = Environment(
        results=[result(next_observation=following, terminated=True, reward=0.5)]
    )
    adapter, _ = boundary.create(environment=environment)
    initial = adapter.observation
    environment.initial["image"].fill(0)
    assert adapter.observation == initial
    returned = adapter.step(2)
    saved = adapter.report
    following["image"].fill(0)
    following["direction"] = 0
    returned["observation"] = ()
    returned["reward"] = -1
    leaked = adapter.report
    leaked["transitions"][0]["raw_next_observation"]["image"][0][0][0] = 99
    leaked["initial_observation"]["image"].clear()
    leaked["versions"].clear()
    assert adapter.report == saved
    assert adapter.observation == saved["transitions"][0]["next_observation"]
    assert saved["total_return"] == 0.5
    assert saved["transitions"][0]["observation"] == initial


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_should_keep_native_end_flags_separate_and_refuse_post_end_steps(
    boundary, terminated, truncated
):
    environment = Environment(results=[result(terminated=terminated, truncated=truncated)])
    adapter, _ = boundary.create(environment=environment)
    step = adapter.step(6)
    assert step["terminated"] is terminated and step["truncated"] is truncated
    assert adapter.terminal and adapter.legal_actions == ()
    before = adapter.report
    with pytest.raises(ValueError):
        adapter.step(0)
    assert adapter.report == before and environment.actions == [6]
    assert before["status"] == "complete"
    if truncated:
        with pytest.raises(ValueError, match="termination"):
            adapter.to_demonstration()


def test_should_not_invent_terminal_demonstration_for_running_episode(boundary):
    adapter, _ = boundary.create(environment=Environment(results=[result()]))
    adapter.step(2)
    with pytest.raises(ValueError, match="termination"):
        adapter.to_demonstration()


@pytest.mark.parametrize("role", ["train", "final_test", "probe_train", "probe_test"])
def test_should_preserve_failed_terminal_script_and_role_in_existing_owned_episode(boundary, role):
    adapter, _ = boundary.create(
        environment=Environment(results=[result(), result(terminated=True)]),
        role=role,
        episode_id="scripted-zero-reward",
        context="memory",
    )
    adapter.step(0)
    adapter.step(2)
    episode = adapter.to_demonstration()
    assert episode.role == role and episode.num_classes == 7
    assert episode.episode_id == "scripted-zero-reward" and episode.context == "memory"
    assert [row.action for row in episode.transitions] == [0, 2]
    assert [row.terminal for row in episode.transitions] == [False, True]
    assert [row.reward for row in episode.transitions] == [0.0, 0.0]
    features, labels = episode_arrays(episode)
    assert features.shape == (2, 148)
    np.testing.assert_array_equal(labels, [[0.0], [2.0]])
    schema = EpisodeReplaySchema(148, 2, 7)
    if role == "train":
        encoded, dummy = encode_episode(episode, schema)
        assert decode_episode(encoded, dummy, schema) == episode
    else:
        with pytest.raises(ValueError, match="training"):
            encode_episode(episode, schema)


def test_should_preserve_unknown_step_attempt_and_primary_exception_without_retry(boundary):
    class MutatingEnvironment(Environment):
        def step(self, action):
            # A raised call is not proof that the underlying environment did no work.
            self.initial["image"][0, 0, 0] = 10
            return super().step(action)

    environment = MutatingEnvironment(results=[RuntimeError("native state may have changed")])
    adapter, _ = boundary.create(environment=environment)
    with pytest.raises(RuntimeError, match="native state may have changed"):
        adapter.step(2)
    report = adapter.report
    assert report["status"] == "failed"
    assert report["environment_steps_attempted"] == 1 and report["environment_steps"] == 0
    assert report["transitions"][0]["attempted"] is True
    assert report["transitions"][0]["applied"] is None
    assert report["failure_work_may_be_unobserved"] is True
    assert environment.initial["image"][0, 0, 0] == 10
    assert report["transitions"][0]["next_observation"] is None
    assert adapter.legal_actions == ()
    with pytest.raises(ValueError):
        adapter.step(2)
    assert environment.actions == [2] and adapter.report == report
    adapter.close()
    assert adapter.report["reason"] == report["reason"]


@pytest.mark.parametrize("fault", ["observation", "reward", "flag", "result_shape"])
def test_should_charge_returned_native_step_before_payload_validation_and_refuse_retry(
    boundary, fault
):
    malformed = list(result(reward=0.25, terminated=True))
    if fault == "observation":
        malformed[0] = observation()
        malformed[0]["mission"] = "different task"
    elif fault == "reward":
        malformed[1] = float("nan")
    elif fault == "flag":
        malformed[2] = 1
    else:
        malformed.pop()
    adapter, environment = boundary.create(environment=Environment(results=[tuple(malformed)]))
    with pytest.raises(ValueError):
        adapter.step(4)
    report = adapter.report
    assert report["status"] == "failed"
    assert report["environment_steps_attempted"] == report["environment_steps"] == 1
    assert report["transitions"][0]["applied"] is True
    assert report["failure_work_may_be_unobserved"] is False
    if fault == "observation":
        assert report["total_return"] == 0.25
        assert report["terminated"] is True and report["truncated"] is False
        assert report["transitions"][0]["next_observation"] is None
    with pytest.raises(ValueError):
        adapter.step(4)
    with pytest.raises(ValueError):
        adapter.to_demonstration()
    assert environment.actions == [4] and adapter.report == report


def test_should_refuse_native_missing_timeout_instead_of_synthesizing_termination(boundary):
    adapter, environment = boundary.create(environment=Environment(results=[result()]), max_steps=1)
    with pytest.raises(ValueError, match="max_steps"):
        adapter.step(2)
    assert environment.actions == [2]
    report = adapter.report
    assert report["environment_steps"] == 1 and report["status"] == "failed"
    assert report["terminated"] is report["truncated"] is False
    assert not adapter.terminal and adapter.legal_actions == ()
    with pytest.raises(ValueError):
        adapter.to_demonstration()


@pytest.mark.parametrize("fault", ["reset", "observation", "action_space"])
def test_should_close_failed_constructor_once_without_automatic_reset_retry(boundary, fault):
    environment = Environment(
        reset_error=RuntimeError("reset failed") if fault == "reset" else None
    )
    if fault == "observation":
        environment.initial["direction"] = 4
    elif fault == "action_space":
        environment.action_space.n = 6
    with pytest.raises((ValueError, RuntimeError)):
        boundary.create(environment=environment)
    assert environment.close_calls == 1
    assert environment.reset_seeds == ([] if fault == "action_space" else [17])
    assert environment.actions == []


def test_should_close_context_once_and_refuse_steps_without_losing_completed_records(boundary):
    adapter, environment = boundary.create(
        environment=Environment(results=[result(terminated=True)])
    )
    with adapter as entered:
        assert entered is adapter
        adapter.step(2)
    saved = deepcopy(adapter.report)
    adapter.close()
    assert environment.close_calls == 1
    assert saved["closed"] is True and saved["status"] == "complete"
    assert adapter.to_demonstration().transitions[0].terminal
    with pytest.raises(ValueError):
        adapter.step(2)
    assert adapter.report == saved


def test_should_close_running_context_when_caller_fails(boundary):
    adapter, environment = boundary.create()
    with pytest.raises(RuntimeError, match="caller failure"):
        with adapter:
            raise RuntimeError("caller failure")
    assert environment.close_calls == 1 and adapter.legal_actions == ()
    assert not adapter.terminal
    assert adapter.report["environment_steps_attempted"] == 0
    with pytest.raises(ValueError):
        adapter.step(0)


def test_should_check_exact_optional_dependency_versions_before_import_or_environment(monkeypatch):
    imports = []
    monkeypatch.setattr(memory.metadata, "version", lambda name: "unsupported")
    monkeypatch.setattr(memory, "import_module", lambda name: imports.append(name))
    with pytest.raises(ValueError, match="pinned"):
        memory._make_memory_environment(16)
    assert imports == []


def test_should_create_actual_registered_memory_id_without_rendering_or_resetting_in_factory(
    monkeypatch,
):
    environment = Environment()
    calls = []
    imports = []

    def make(*args, **kwargs):
        calls.append((args, kwargs))
        return environment

    def import_optional(name):
        imports.append(name)
        return SimpleNamespace(make=make)

    monkeypatch.setattr(memory.metadata, "version", lambda name: memory.VERSIONS[name])
    monkeypatch.setattr(memory, "import_module", import_optional)
    assert memory._make_memory_environment(9) is environment
    assert imports == ["gymnasium", "minigrid"]
    assert calls == [(("MiniGrid-MemoryS7-v0",), {"max_steps": 9, "render_mode": None})]
    assert environment.reset_seeds == [] and environment.actions == []
