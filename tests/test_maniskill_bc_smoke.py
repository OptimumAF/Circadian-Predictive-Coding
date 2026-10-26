"""Frozen CPU learner integration through a fake native environment, no simulator."""

from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_demonstration import ContinuousActionSpec
from src.infra import maniskill_continuous as port


class FakeEnvironment:
    """Only boundary shapes, commands, terminal flags and work are modeled."""

    def __init__(self, *, steps=2, terminated=True, truncated=False, rewards=(0.0, 1.0)):
        self.unwrapped = self
        self.num_envs = 1
        self.obs_mode = "state"
        self.control_mode = "pd_ee_delta_pos"
        self.device = SimpleNamespace(type="cpu")
        self.sim_freq, self.control_freq = 100, 20
        self.single_action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            shape=(4,),
        )
        self.agent = SimpleNamespace(uid="panda")
        self.steps, self.terminated, self.truncated = steps, terminated, truncated
        self.rewards = rewards
        self.commands = []
        self.reset_calls = []
        self.close_calls = 0
        self.step_error = self.close_error = None

    def reset(self, *, seed):
        self.reset_calls.append(seed)
        return np.array([[0.2, -0.3, 0.4]], dtype=np.float32), {}

    def step(self, command):
        self.commands.append(command.copy())
        if self.step_error is not None:
            raise self.step_error
        index = len(self.commands)
        done = index == self.steps
        return (
            np.array([[0.2 + index, -0.3, 0.4]], dtype=np.float32),
            np.array([self.rewards[min(index - 1, len(self.rewards) - 1)]], dtype=np.float32),
            np.array([done and self.terminated], dtype=np.bool_),
            np.array([done and self.truncated], dtype=np.bool_),
            {},
        )

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = FakeEnvironment()
    creations = []

    def factory():
        creations.append(True)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "trace.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    return environment, creations, output


def learner(**changes):
    spec = ContinuousActionSpec(3, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    options = dict(spec=spec, hidden_dim=2, seed=17)
    options.update(changes)
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(**options))
    initial = model.snapshot_state()
    weights = tuple(np.zeros_like(value) for value in initial.parameters)
    weights[3][:] = [0.2, -0.3, 0.4, -0.5]
    model.restore_state(
        replace(
            initial,
            parameters=weights,
            velocity=tuple(np.full_like(value, 0.125) for value in initial.velocity),
            completed_updates=32,
            training_examples=928,
        )
    )
    return model


def assert_same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for group in ("parameters", "velocity"):
        for left, right in zip(getattr(first, group), getattr(second, group), strict=True):
            np.testing.assert_array_equal(left, right)


def receipt(output):
    return json.loads(output.read_text(encoding="utf8"))


def test_should_preserve_zero_policy_default_without_prediction_work(boundary):
    environment, creations, output = boundary
    assert smoke.main() == 0
    result = receipt(output)
    assert creations == [True] and environment.reset_calls == [17]
    assert environment.close_calls == 1
    assert result["policy"] == "constant_zero_normalized_4axis"
    assert result["complete"] is True
    assert result["optimizer_updates"] == 0
    assert result["prediction_attempts"] == result["prediction_returns"] == 0
    assert result["validated_prediction_rows"] == result["validated_prediction_bytes"] == 0
    for command in environment.commands:
        np.testing.assert_array_equal(command, np.zeros((1, 4), dtype=np.float32))


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_should_send_actual_frozen_predictions_and_preserve_owner_state(
    boundary, monkeypatch, terminated, truncated
):
    environment, creations, output = boundary
    environment.terminated, environment.truncated = terminated, truncated
    model = learner()
    before = model.snapshot_state()
    original = NumpyContinuousBCLearner.predict
    calls = []

    def counted(child, features):
        calls.append((child, features.copy()))
        return original(child, features)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", counted)
    assert smoke.main(learner=model) == 0
    result = receipt(output)
    assert creations == [True] and len(environment.commands) == 2
    assert len(calls) == 2 and all(child is not model for child, _ in calls)
    assert calls[0][0] is calls[1][0]
    np.testing.assert_array_equal(
        calls[0][1], np.array([[0.2, -0.3, 0.4]], dtype=np.float32).astype(np.float64)
    )
    np.testing.assert_array_equal(
        calls[1][1], np.array([[1.2, -0.3, 0.4]], dtype=np.float32).astype(np.float64)
    )
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    for index, command in enumerate(environment.commands):
        assert command.dtype == np.float32 and command.shape == (1, 4)
        np.testing.assert_allclose(command, expected, rtol=0.0, atol=0.0)
        assert result["episode"]["transitions"][index]["action"] == command[0].tolist()
    final = result["episode"]["transitions"][-1]
    assert final["terminated"] is terminated and final["truncated"] is truncated
    assert result["complete"] is True and result["optimizer_updates"] == 0
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is True
    assert result["source_updates"] == 32 and result["source_training_rows"] == 928
    assert result["creation_attempts"] == result["creation_returns"] == 1
    assert result["prediction_attempts"] == result["prediction_returns"] == 2
    assert result["validated_prediction_rows"] == 2
    assert result["validated_prediction_bytes"] == 64
    assert result["work"] == dict(
        step_attempts=2,
        completed_steps=2,
        completed_control_physics_steps=10,
        possible_failed_physics_steps=0,
    )
    assert environment.close_calls == 1
    assert_same_state(before, model.snapshot_state())


@pytest.mark.parametrize("rewards", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
def test_should_keep_sparse_success_separate_from_complete_trace(boundary, rewards):
    environment, _, output = boundary
    environment.rewards = rewards
    assert smoke.main(learner=learner()) == 0
    result = receipt(output)
    assert result["complete"] is True
    assert result["sparse_return"] == sum(rewards)
    assert result["any_success"] is any(value == 1.0 for value in rewards)
    assert result["final_success"] is (rewards[-1] == 1.0)


@pytest.mark.parametrize("bad", ["probe_policy", "units", "bounds", "foreign"])
def test_should_refuse_invalid_policy_before_scene_creation(boundary, bad):
    _, creations, output = boundary
    if bad == "foreign":
        model = object()
    elif bad == "probe_policy":
        model = learner(training_role="probe_train")
    else:
        spec = ContinuousActionSpec(3, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
        spec = (
            replace(spec, action_units=("m",) * 4)
            if bad == "units"
            else replace(spec, lower_bounds=(-2.0,) * 4)
        )
        model = learner(spec=spec)
    assert smoke.main(learner=model) == 1
    result = receipt(output)
    assert creations == []
    assert result["creation_attempts"] == result["creation_returns"] == 0
    assert result["prediction_attempts"] == 0 and result["work"] is None
    assert result["complete"] is False and result["error"]


def test_should_refuse_actual_observation_spec_mismatch_before_predict(boundary, monkeypatch):
    environment, creations, output = boundary
    model = learner(spec=ContinuousActionSpec(2, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4))
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "predict",
        lambda *_: pytest.fail("mismatched spec reached prediction"),
    )
    assert smoke.main(learner=model) == 1
    result = receipt(output)
    assert creations == [True] and environment.commands == []
    assert result["prediction_attempts"] == 0 and result["creation_returns"] == 1
    assert environment.close_calls == 1 and result["complete"] is False


@pytest.mark.parametrize(
    "bad", ["list", "float32", "flat", "batch", "axis", "nan", "inf", "bounds"]
)
def test_should_refuse_bad_predictions_before_control_and_keep_work_receipt(
    boundary, monkeypatch, bad
):
    environment, _, output = boundary
    values = np.zeros((1, 4), dtype=np.float64)
    if bad == "list":
        values = values.tolist()
    elif bad == "float32":
        values = values.astype(np.float32)
    elif bad == "flat":
        values = values.ravel()
    elif bad == "batch":
        values = np.zeros((2, 4), dtype=np.float64)
    elif bad == "axis":
        values = np.zeros((1, 3), dtype=np.float64)
    elif bad in ("nan", "inf", "bounds"):
        values[0, 0] = {"nan": np.nan, "inf": np.inf, "bounds": 1.00000001}[bad]
    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", lambda *_: values)
    assert smoke.main(learner=learner()) == 1
    result = receipt(output)
    assert environment.commands == [] and environment.close_calls == 1
    assert result["prediction_attempts"] == result["prediction_returns"] == 1
    assert result["validated_prediction_rows"] == result["validated_prediction_bytes"] == 0
    assert result["work"]["step_attempts"] == 0 and result["complete"] is False


def test_should_record_failed_prediction_attempt_without_control(boundary, monkeypatch):
    environment, _, output = boundary

    def broken(*_):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", broken)
    assert smoke.main(learner=learner()) == 1
    result = receipt(output)
    assert result["prediction_attempts"] == 1 and result["prediction_returns"] == 0
    assert result["validated_prediction_rows"] == 0
    assert environment.commands == [] and environment.close_calls == 1
    assert "prediction failed" in result["error"]


@pytest.mark.parametrize(
    "component",
    [
        *(f"parameter-{index}" for index in range(4)),
        *(f"velocity-{index}" for index in range(4)),
        "completed_updates",
        "training_examples",
        "policy",
    ],
)
def test_should_refuse_full_child_state_mutation_before_control(boundary, monkeypatch, component):
    environment, _, output = boundary
    model = learner()
    before = model.snapshot_state()
    original = NumpyContinuousBCLearner.predict

    def corrupt(child, features):
        prediction = original(child, features)
        if component.startswith("parameter-"):
            child._state.parameters[int(component[-1])].flat[0] += 0.01
        elif component.startswith("velocity-"):
            child._state.velocity[int(component[-1])].flat[0] += 0.01
        elif component == "policy":
            object.__setattr__(child._policy, "learning_rate", 0.02)
        else:
            object.__setattr__(child._state, component, getattr(child._state, component) + 1)
        return prediction

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", corrupt)
    assert smoke.main(learner=model) == 1
    result = receipt(output)
    assert environment.commands == [] and environment.close_calls == 1
    assert result["prediction_attempts"] == result["prediction_returns"] == 1
    assert result["work"]["step_attempts"] == 0 and result["complete"] is False
    assert_same_state(before, model.snapshot_state())


def test_should_detect_owner_mutation_even_when_fork_is_unchanged(boundary, monkeypatch):
    environment, _, output = boundary
    model = learner()
    original = NumpyContinuousBCLearner.predict

    def corrupt(child, features):
        prediction = original(child, features)
        model._state.velocity[3][0] += 0.01
        return prediction

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", corrupt)
    assert smoke.main(learner=model) == 1
    assert environment.commands == [] and environment.close_calls == 1
    result = receipt(output)
    assert result["complete"] is False
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is False


def test_should_clear_prior_success_flags_when_later_corrupt_state_cannot_be_observed(
    boundary, monkeypatch
):
    environment, _, output = boundary
    model = learner()
    before = model.snapshot_state()
    original = NumpyContinuousBCLearner.predict
    calls = []

    def corrupt_second_prediction(child, features):
        prediction = original(child, features)
        calls.append(True)
        if len(calls) == 2:
            child._state.parameters[0].flat[0] = np.nan
        return prediction

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", corrupt_second_prediction)
    assert smoke.main(learner=model) == 1
    result = receipt(output)
    assert result["prediction_attempts"] == result["prediction_returns"] == 2
    assert result["validated_prediction_rows"] == 1
    assert len(environment.commands) == result["work"]["completed_steps"] == 1
    assert result["complete"] is False and result["state_error"]
    assert result["frozen_state_exact"] is None and result["original_owner_exact"] is None
    assert environment.close_calls == 1
    assert_same_state(before, model.snapshot_state())


def test_should_keep_native_failure_work_and_close_once(boundary):
    environment, _, output = boundary
    environment.step_error = RuntimeError("native step failed")
    model = learner()
    before = model.snapshot_state()
    assert smoke.main(learner=model) == 1
    result = receipt(output)
    assert result["prediction_attempts"] == result["prediction_returns"] == 1
    assert result["validated_prediction_rows"] == 1
    assert result["work"] == dict(
        step_attempts=1,
        completed_steps=0,
        completed_control_physics_steps=0,
        possible_failed_physics_steps=5,
    )
    assert "native step failed" in result["error"] and environment.close_calls == 1
    assert_same_state(before, model.snapshot_state())


def test_should_record_constructor_failure_without_claiming_returned_port_or_work(
    boundary, monkeypatch
):
    environment, _, output = boundary

    def broken():
        raise RuntimeError("construction failed")

    monkeypatch.setattr(port, "_make_environment", broken)
    assert smoke.main(learner=learner()) == 1
    result = receipt(output)
    assert result["creation_attempts"] == 1 and result["creation_returns"] == 0
    assert result["prediction_attempts"] == 0 and result["work"] is None
    assert "construction failed" in result["error"]
    assert environment.commands == [] and environment.close_calls == 0


def test_should_preserve_complete_trace_but_fail_when_cleanup_raises(boundary):
    environment, _, output = boundary
    environment.close_error = RuntimeError("cleanup failed")
    assert smoke.main(learner=learner()) == 1
    result = receipt(output)
    assert len(result["episode"]["transitions"]) == 2
    assert result["complete"] is False and result["cleanup_error"] == "cleanup failed"
    assert environment.close_calls == 1
    assert result["work"]["completed_steps"] == 2
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is True


def test_should_detect_state_change_during_control_before_admitting_next_prediction(
    boundary, monkeypatch
):
    environment, _, output = boundary
    original_predict = NumpyContinuousBCLearner.predict
    original_step = environment.step
    children = []

    def capture(child, features):
        children.append(child)
        return original_predict(child, features)

    def mutate_after_control(command):
        returned = original_step(command)
        children[-1]._state.velocity[0].flat[0] += 0.01
        return returned

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", capture)
    monkeypatch.setattr(environment, "step", mutate_after_control)
    assert smoke.main(learner=learner()) == 1
    result = receipt(output)
    assert result["prediction_attempts"] == result["prediction_returns"] == 1
    assert len(environment.commands) == 1 and result["work"]["completed_steps"] == 1
    assert result["frozen_state_exact"] is False and result["original_owner_exact"] is True


def test_should_recheck_state_after_cleanup_and_keep_changed_state_evidence(boundary, monkeypatch):
    environment, _, output = boundary
    model = learner()
    original_close = environment.close

    def mutate_during_close():
        original_close()
        model._state.velocity[1].flat[0] += 0.01

    monkeypatch.setattr(environment, "close", mutate_during_close)
    assert smoke.main(learner=model) == 1
    result = receipt(output)
    assert result["complete"] is False and result["state_error"]
    assert len(result["episode"]["transitions"]) == 2 and environment.close_calls == 1
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is False


@pytest.mark.parametrize("final_flag", [True, False])
def test_should_obey_fifty_step_bound_without_silently_accepting_missing_terminal_flag(
    boundary, final_flag
):
    environment, _, output = boundary
    environment.steps = 50 if final_flag else 51
    environment.terminated, environment.truncated = False, True
    environment.rewards = (0.0,)
    assert smoke.main(learner=learner()) == (0 if final_flag else 1)
    result = receipt(output)
    assert len(environment.commands) == 50 and environment.close_calls == 1
    assert result["prediction_attempts"] == result["prediction_returns"] == 50
    assert result["validated_prediction_rows"] == 50
    assert result["work"]["completed_steps"] == 50
    assert result["work"]["completed_control_physics_steps"] == 250
    assert result["complete"] is final_flag
    if final_flag:
        assert len(result["episode"]["transitions"]) == 50
        assert result["final_success"] is False
    else:
        assert "episode" not in result
        assert "time-limit flag is missing" in result["error"]


def test_should_reserve_exclusive_output_before_any_creation_or_inference(boundary, monkeypatch):
    environment, creations, output = boundary
    original_bytes = b"preserved failed attempt\n"
    output.write_bytes(original_bytes)
    model = learner()
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "predict",
        lambda *_: pytest.fail("existing output reached inference"),
    )
    with pytest.raises(FileExistsError):
        smoke.main(learner=model)
    assert output.read_bytes() == original_bytes
    assert creations == [] and environment.commands == [] and environment.close_calls == 0
