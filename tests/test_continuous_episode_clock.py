"""Public clock representation checks, using synthetic records and fake native IO."""

from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core import continuous_episode_clock as clock
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.infra import maniskill_continuous as port


def source_episode(role="train"):
    spec = ContinuousActionSpec(2, ("m",), (-1.0,), (1.0,))
    return ContinuousEpisode(
        "owned-episode",
        "context-a",
        role,
        spec,
        (
            ContinuousTransition(0, (10.0, 20.0), (0.25,), 0.0, (11.0, 21.0), False, False),
            ContinuousTransition(1, (11.0, 21.0), (-0.5,), 1.0, (12.0, 22.0), True, True),
        ),
    )


@pytest.mark.parametrize(
    "horizon,elapsed",
    [(0, True), (257, True), (True, True), (50.0, True), (50, 1), (50, None), (50, np.bool_(True))],
)
def test_should_refuse_invalid_clock_configuration(horizon, elapsed):
    with pytest.raises(ValueError):
        clock.EpisodeClock(horizon, elapsed)


@pytest.mark.parametrize("horizon", [1, 50, 256])
@pytest.mark.parametrize("elapsed", [False, True])
def test_should_append_hand_computed_clock_including_terminal_next_observation(horizon, elapsed):
    configuration = clock.EpisodeClock(horizon, elapsed)
    for step in (0, 1, horizon):
        result = clock.clock_observation((10.0, -200.25), step, configuration)
        assert result == (10.0, -200.25, step / horizon if elapsed else 0.0)
        assert type(result) is tuple and all(type(value) is float for value in result)


@pytest.mark.parametrize("step", [-1, 51, True, 1.0, np.int64(1)])
def test_should_refuse_non_native_or_out_of_horizon_steps(step):
    with pytest.raises(ValueError):
        clock.clock_observation((1.0,), step, clock.EpisodeClock(50, True))


@pytest.mark.parametrize(
    "raw",
    [[], [1.0], (), (True,), (float("nan"),), (float("inf"),), (np.float64(1.0),), (1.0,) * 1024],
)
def test_should_refuse_invalid_raw_observations_without_flattening_or_clipping(raw):
    with pytest.raises(ValueError):
        clock.clock_observation(raw, 0, clock.EpisodeClock(50, True))


def test_should_accept_maximum_raw_dimension_and_preserve_large_finite_values():
    raw = (1e308,) + (0.0,) * 1022
    encoded = clock.clock_observation(raw, 256, clock.EpisodeClock(256, True))
    assert len(encoded) == 1024 and encoded[:-1] == raw and encoded[-1] == 1.0


def test_should_copy_and_revalidate_clock_and_spec():
    original = clock.EpisodeClock(50, True)
    owned = clock._clock(original)
    assert owned == original and owned is not original
    spec = ContinuousActionSpec(3, ("m", "N"), (-2.0, 0.0), (3.0, 1.0))
    extended = clock.clock_spec(spec, original)
    assert extended.observation_dim == 4
    assert (extended.action_units, extended.lower_bounds, extended.upper_bounds) == (
        spec.action_units,
        spec.lower_bounds,
        spec.upper_bounds,
    )
    object.__setattr__(original, "horizon", 0)
    assert owned.horizon == 50
    with pytest.raises(ValueError):
        clock.clock_spec(spec, original)
    with pytest.raises(ValueError):
        clock._clock(object())


def test_should_refuse_spec_whose_clock_would_exceed_dimension_bound():
    spec = ContinuousActionSpec(1024, ("m",), (-1.0,), (1.0,))
    with pytest.raises(ValueError):
        clock.clock_spec(spec, clock.EpisodeClock(50, True))


@pytest.mark.parametrize("elapsed", [False, True])
def test_should_project_owned_complete_training_episode_with_exact_seams(elapsed):
    source = source_episode()
    before = repr(source)
    projected = clock.clock_training_episode(source, clock.EpisodeClock(2, elapsed))
    assert projected is not source and repr(source) == before
    assert (projected.episode_id, projected.context, projected.role) == (
        source.episode_id,
        source.context,
        "train",
    )
    assert projected.spec == replace(source.spec, observation_dim=3)
    for index, (raw, encoded) in enumerate(
        zip(source.transitions, projected.transitions, strict=True)
    ):
        assert encoded.step == raw.step
        assert encoded.observation[:-1] == raw.observation
        assert encoded.next_observation[:-1] == raw.next_observation
        assert encoded.observation[-1] == (index / 2 if elapsed else 0.0)
        assert encoded.next_observation[-1] == ((index + 1) / 2 if elapsed else 0.0)
        assert (encoded.action, encoded.reward, encoded.terminated, encoded.truncated) == (
            raw.action,
            raw.reward,
            raw.terminated,
            raw.truncated,
        )
    assert projected.transitions[0].next_observation == projected.transitions[1].observation
    assert (
        projected.transitions[-1].terminated is True and projected.transitions[-1].truncated is True
    )
    object.__setattr__(source.transitions[0], "observation", (999.0, 20.0))
    assert projected.transitions[0].observation == (10.0, 20.0, 0.0)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_non_training_roles_before_any_projection(monkeypatch, role):
    monkeypatch.setattr(
        clock, "clock_observation", lambda *_: pytest.fail("non-TRAIN episode reached projection")
    )
    with pytest.raises(ValueError):
        clock.clock_training_episode(source_episode(role), clock.EpisodeClock(50, True))


def test_should_refuse_over_horizon_or_corrupted_episode_before_projection(monkeypatch):
    monkeypatch.setattr(
        clock, "clock_observation", lambda *_: pytest.fail("invalid episode reached projection")
    )
    with pytest.raises(ValueError):
        clock.clock_training_episode(source_episode(), clock.EpisodeClock(1, True))
    damaged = source_episode()
    object.__setattr__(damaged.transitions[0], "next_observation", (999.0, 21.0))
    with pytest.raises(ValueError):
        clock.clock_training_episode(damaged, clock.EpisodeClock(50, True))


class FakeEnvironment:
    def __init__(self):
        self.unwrapped = self
        self.num_envs, self.obs_mode, self.control_mode = 1, "state", port.CONTROL_MODE
        self.device, self.agent = SimpleNamespace(type="cpu"), SimpleNamespace(uid="panda")
        self.sim_freq, self.control_freq = 100, 20
        self.single_action_space = SimpleNamespace(
            shape=(4,), low=np.full(4, -1.0, dtype=np.float32), high=np.ones(4, dtype=np.float32)
        )
        self.commands, self.reset_calls = [], []
        self.close_calls = 0
        self.close_error = self.step_error = None

    def reset(self, *, seed):
        self.reset_calls.append(seed)
        return np.array([[2.0, 3.0, 4.0]], dtype=np.float32), {}

    def step(self, command):
        self.commands.append(command.copy())
        if self.step_error is not None:
            raise self.step_error
        step = len(self.commands)
        return (
            np.array([[2.0 + step, 3.0, 4.0]], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([False]),
            np.array([step == 2]),
            {},
        )

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    native = FakeEnvironment()
    creations = []

    def factory():
        creations.append(True)
        return native

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "clock-trace.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    return native, creations, output


def model(dimension=4, role="train"):
    spec = ContinuousActionSpec(dimension, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    transform = ObservationStandardization((0.0,) * dimension, (1.0,) * (dimension - 1) + (0.5,), 2)
    owner = NumpyContinuousBCLearner(
        ContinuousBCPolicy(
            spec,
            hidden_dim=2,
            training_role=role,
            observation_standardization=transform,
        )
    )
    state = owner.snapshot_state()
    parameters = tuple(np.zeros_like(value) for value in state.parameters)
    parameters[0][-1, 0], parameters[2][0, 0] = 1.0, 1.0
    owner.restore_state(
        replace(
            state,
            parameters=parameters,
            velocity=tuple(np.full_like(value, 0.125) for value in state.velocity),
            completed_updates=2,
            training_examples=4,
        )
    )
    return owner


def read_receipt(output):
    return json.loads(output.read_text(encoding="utf8"))


def assert_same_state(before, after):
    assert before.policy == after.policy
    assert before.completed_updates == after.completed_updates
    assert before.training_examples == after.training_examples
    for group in ("parameters", "velocity"):
        for first, second in zip(getattr(before, group), getattr(after, group), strict=True):
            np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("elapsed", [False, True])
def test_should_predict_from_public_clock_with_frozen_normalizer_and_export_native_trace(
    boundary, monkeypatch, elapsed
):
    native, creations, output = boundary
    owner = model()
    before = owner.snapshot_state()
    calls = []
    original = NumpyContinuousBCLearner.predict

    def counted(child, features):
        calls.append((child, features.copy()))
        return original(child, features)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", counted)
    configuration = clock.EpisodeClock(50, elapsed)
    assert smoke.main(learner=owner, observation_clock=configuration) == 0
    result = read_receipt(output)
    assert creations == [True] and native.reset_calls == [17] and native.close_calls == 1
    assert len(calls) == 2 and all(child is not owner for child, _ in calls)
    expected_clocks = [0.0, 1 / 50 if elapsed else 0.0]
    for index, (_, values) in enumerate(calls):
        np.testing.assert_array_equal(values, [[2.0 + index, 3.0, 4.0, expected_clocks[index]]])
        expected_action = np.array(
            [[np.tanh(np.tanh(2 * expected_clocks[index])), 0.0, 0.0, 0.0]], dtype=np.float32
        )
        np.testing.assert_allclose(native.commands[index], expected_action, rtol=1e-7, atol=1e-8)
    assert result["observation_clock"] == {"horizon": 50, "elapsed": elapsed}
    assert result["clock_values"] == expected_clocks
    assert result["clock_configuration_exact"] is True
    assert result["clock_encoding_attempts"] == result["clock_encoding_returns"] == 2
    assert result["clock_encoded_values"] == 8
    assert result["prediction_attempts"] == result["prediction_returns"] == 2
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is True
    assert result["complete"] is True and result["final_success"] is False
    assert result["episode"]["spec"]["observation_dim"] == 3
    assert all(
        len(row["observation"]) == len(row["next_observation"]) == 3
        for row in result["episode"]["transitions"]
    )
    assert_same_state(before, owner.snapshot_state())


@pytest.mark.parametrize("bad", ["missing_model", "probe_policy", "horizon", "foreign", "corrupt"])
def test_should_refuse_invalid_clock_rollout_configuration_before_scene(boundary, bad):
    _, creations, output = boundary
    owner, configuration = model(), clock.EpisodeClock(50, True)
    if bad == "missing_model":
        owner = None
    elif bad == "probe_policy":
        owner = model(role="probe_train")
    elif bad == "horizon":
        configuration = clock.EpisodeClock(49, True)
    elif bad == "foreign":
        configuration = object()
    else:
        object.__setattr__(configuration, "elapsed", 1)
    assert smoke.main(learner=owner, observation_clock=configuration) == 1
    result = read_receipt(output)
    assert creations == []
    assert result["creation_attempts"] == result["creation_returns"] == 0
    assert result["prediction_attempts"] == result["clock_encoding_attempts"] == 0
    assert result["complete"] is False and result["work"] is None


def test_should_match_encoded_spec_before_encoding_or_prediction(boundary, monkeypatch):
    native, creations, output = boundary
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "predict",
        lambda *_: pytest.fail("mismatched encoded spec reached learner"),
    )
    assert smoke.main(learner=model(3), observation_clock=clock.EpisodeClock(50, True)) == 1
    result = read_receipt(output)
    assert creations == [True] and native.commands == [] and native.close_calls == 1
    assert result["prediction_attempts"] == result["clock_encoding_attempts"] == 0
    assert result["complete"] is False


def test_should_copy_clock_configuration_before_native_work(boundary, monkeypatch):
    native, _, output = boundary
    configuration = clock.EpisodeClock(50, True)
    original_predict = NumpyContinuousBCLearner.predict

    def mutate_caller_configuration(child, features):
        object.__setattr__(configuration, "elapsed", False)
        return original_predict(child, features)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", mutate_caller_configuration)
    assert smoke.main(learner=model(), observation_clock=configuration) == 0
    result = read_receipt(output)
    assert configuration.elapsed is False
    assert result["observation_clock"] == {"horizon": 50, "elapsed": True}
    assert result["clock_values"] == [0.0, 1 / 50]
    assert result["clock_configuration_exact"] is True and len(native.commands) == 2


def test_should_refuse_private_clock_mutation_before_control(boundary, monkeypatch):
    native, _, output = boundary
    original_encode = clock.clock_observation

    def corrupt_private_configuration(observation, step, configuration):
        result = original_encode(observation, step, configuration)
        object.__setattr__(configuration, "horizon", 51)
        return result

    monkeypatch.setattr(clock, "clock_observation", corrupt_private_configuration)
    assert smoke.main(learner=model(), observation_clock=clock.EpisodeClock(50, True)) == 1
    result = read_receipt(output)
    assert native.commands == [] and native.close_calls == 1
    assert result["clock_configuration_exact"] is False
    assert result["clock_encoding_attempts"] == result["clock_encoding_returns"] == 1
    assert result["clock_encoded_values"] == 4
    assert result["complete"] is False and result["work"]["step_attempts"] == 0


def test_should_preserve_failed_encoding_attempt_without_prediction_or_control(
    boundary, monkeypatch
):
    native, _, output = boundary

    def broken(*_):
        raise ValueError("clock encoding failed")

    monkeypatch.setattr(clock, "clock_observation", broken)
    assert smoke.main(learner=model(), observation_clock=clock.EpisodeClock(50, True)) == 1
    result = read_receipt(output)
    assert result["clock_encoding_attempts"] == 1 and result["clock_encoding_returns"] == 0
    assert result["clock_encoded_values"] == 0 and result["clock_values"] == []
    assert result["prediction_attempts"] == 0 and native.commands == []
    assert native.close_calls == 1 and "clock encoding failed" in result["error"]


def test_should_keep_encoded_work_when_prediction_fails_before_control(boundary, monkeypatch):
    native, _, output = boundary

    def broken(*_):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", broken)
    assert smoke.main(learner=model(), observation_clock=clock.EpisodeClock(50, True)) == 1
    result = read_receipt(output)
    assert result["clock_encoding_attempts"] == result["clock_encoding_returns"] == 1
    assert result["clock_encoded_values"] == 4 and result["clock_values"] == [0.0]
    assert result["prediction_attempts"] == 1 and result["prediction_returns"] == 0
    assert native.commands == [] and native.close_calls == 1


def test_should_detect_clock_mutation_during_native_control_before_next_encoding(
    boundary, monkeypatch
):
    native, _, output = boundary
    original_encode, original_step = clock.clock_observation, native.step
    configurations = []

    def capture(observation, step, configuration):
        configurations.append(configuration)
        return original_encode(observation, step, configuration)

    def mutate_after_control(command):
        result = original_step(command)
        object.__setattr__(configurations[-1], "elapsed", False)
        return result

    monkeypatch.setattr(clock, "clock_observation", capture)
    monkeypatch.setattr(native, "step", mutate_after_control)
    assert smoke.main(learner=model(), observation_clock=clock.EpisodeClock(50, True)) == 1
    result = read_receipt(output)
    assert len(native.commands) == 1 and result["work"]["completed_steps"] == 1
    assert result["clock_encoding_attempts"] == result["prediction_attempts"] == 1
    assert result["clock_configuration_exact"] is False and result["complete"] is False


def test_should_recheck_clock_after_cleanup_and_clear_stale_success_flag(boundary, monkeypatch):
    native, _, output = boundary
    original_encode, original_close = clock.clock_observation, native.close
    configurations = []

    def capture(observation, step, configuration):
        configurations.append(configuration)
        return original_encode(observation, step, configuration)

    def mutate_during_cleanup():
        original_close()
        object.__setattr__(configurations[-1], "elapsed", "corrupt")

    monkeypatch.setattr(clock, "clock_observation", capture)
    monkeypatch.setattr(native, "close", mutate_during_cleanup)
    assert smoke.main(learner=model(), observation_clock=clock.EpisodeClock(50, True)) == 1
    result = read_receipt(output)
    assert result["complete"] is False and result["clock_error"]
    assert result["clock_configuration_exact"] is None
    assert len(result["episode"]["transitions"]) == 2 and native.close_calls == 1


def test_should_reserve_exclusive_output_before_admitting_clock_work(boundary, monkeypatch):
    native, creations, output = boundary
    output.write_bytes(b"preserved prior attempt\n")
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "predict",
        lambda *_: pytest.fail("existing output reached inference"),
    )
    with pytest.raises(FileExistsError):
        smoke.main(learner=model(), observation_clock=clock.EpisodeClock(50, True))
    assert output.read_bytes() == b"preserved prior attempt\n"
    assert creations == [] and native.commands == [] and native.close_calls == 0
