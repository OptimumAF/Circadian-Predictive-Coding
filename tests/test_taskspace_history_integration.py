"""Fixed physical taskspace through real learners and bounded fake native IO.

Five true D12/H64 initializations5120 normaldraws;15sourceSGD30rows plus40
privateSGD80rows, no statistical fits. Fifteen fake scenes30predictions/controls,
zero physics. Cached fault tests use RNG-free forks and retain paid reports.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core import pick_cube_action_history as history
from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_state_quaternion_robot_learning import (
    QuaternionEnvironment,
    episodes as old_episodes,
    observation,
)


def forbidden(*args, **kwargs):
    pytest.fail("invalid taskspace request admitted model/native work")


def raw(index, offset=0.0):
    values = list(observation(index))
    values[18] = float(index > 0)
    values[36:42] = [offset - 0.02 + index * 0.005, 0.01, -0.1, 0.03, -0.04, 0.15]
    return tuple(values)


def episodes():
    return tuple(
        replace(
            episode,
            transitions=tuple(
                replace(
                    row,
                    observation=raw(row.step, offset),
                    next_observation=raw(row.step + 1, offset),
                )
                for row in episode.transitions
            ),
        )
        for episode, offset in zip(old_episodes(), (0.0, 0.02), strict=True)
    )


class TaskspaceEnvironment(QuaternionEnvironment):
    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)

    def step(self, command):
        returned = super().step(command)
        return (np.asarray([raw(len(self.commands))], dtype=np.float64), *returned[1:])


@pytest.fixture(
    scope="module",
    params=[
        ("bp", "executed"),
        ("pc", "executed"),
        ("neutral-cpc", "executed"),
        ("active-cpc", "executed"),
        ("bp", "zero"),
    ],
)
def learned(request, tmp_path_factory):
    family, mode = request.param
    recipe = PickCubeTaskSpaceHistoryRecipe(mode=mode)
    warm, new = episodes()
    captured: Any = dict(predictions=[], result=None)
    original_train = entry.run_continuous_learnability
    cls = bp.NumpyContinuousBCLearner if family == "bp" else pc.NumpyContinuousPCLearner
    original_predict = cls.predict

    def train(*args, **options):
        result = original_train(*args, **options)
        captured["result"] = result
        return result

    def predict(owner, features):
        if len(features) == 1:
            captured["predictions"].append(features.copy())
        return original_predict(owner, features)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(entry, "run_continuous_learnability", train)
        patch.setattr(cls, "predict", predict)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        patch.setattr(native, "_make_environment", TaskspaceEnvironment)
        patch.setattr(smoke, "version", lambda _: "fake-taskspace")
        report = entry.run_learning_rollout(
            tmp_path_factory.mktemp("taskspace") / "case.json",
            warm,
            new,
            learner_family=family,
            phase_updates=1,
            a_only_control=True,
            action_history_recipe=recipe,
        )
    assert report["complete"] is True, report
    return recipe, warm, new, report, captured


def test_should_train_all_four_families_without_empirical_fit_or_statistics(learned):
    recipe, _, _, report, captured = learned
    trained = report["training"]
    assert trained["complete"] is True
    assert trained["action_history_recipe"] == asdict(recipe)
    assert (
        trained["work"]["fit_calls"]
        == trained["work"]["fit_rows"]
        == trained["work"]["fit_feature_values"]
        == 0
    )
    assert trained["work"]["statistics_numeric_bytes_per_owner"] == 0
    candidate = captured["result"].candidate
    base = candidate.policy.base if hasattr(candidate.policy, "base") else candidate.policy
    assert base.spec.observation_dim == 12 and base.observation_standardization is None
    assert candidate.snapshot_state().completed_updates == 3
    assert candidate.snapshot_state().training_examples == 6


def test_should_predict_fixed_world_units_and_causal_actual_history_while_preserving_raw42(learned):
    recipe, _, _, report, captured = learned
    native_report = report["rollout"]
    assert native_report["action_history_exact"] is report["candidate_unchanged"] is True
    assert native_report["episode"]["spec"]["observation_dim"] == 42
    values = captured["predictions"]
    assert len(values) == 2 and all(value.shape == (1, 12) for value in values)
    assert tuple(values[0][0]) == history.action_history_observation(
        raw(0), history.ActionHistoryState(), recipe
    )
    first = tuple(native_report["episode"]["transitions"][0]["action"])
    state = history.advance_action_history(history.ActionHistoryState(), first)
    assert tuple(values[1][0]) == history.action_history_observation(raw(1), state, recipe)
    assert native_report["action_history"]["work"]["model_values_emitted"] == 24


@pytest.fixture(scope="module")
def corrected(learned, tmp_path_factory):
    recipe, _, _, _, captured = learned
    source = captured["result"].candidate.fork()
    original = source.snapshot_state()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "_make_environment", TaskspaceEnvironment)
        patch.setattr(smoke, "version", lambda _: "fake-taskspace-correction")
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        patch.setattr(
            queries,
            "_geometry",
            lambda port: PickCubeGeometry(
                (0.0, 0.0, 0.08) if not port._records else (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.2),
            ),
        )
        summary = smoke.run_corrective_adaptation(
            source,
            output_directory=tmp_path_factory.mktemp("taskspace-correction") / "case",
            visual_observation=False,
            readiness_corrections=True,
            acquisition_only=True,
            readiness_roll_in=ReadinessRollInPlan(1),
            action_history_recipe=recipe,
        )
    assert summary["complete"] is True, summary
    assert demo._same_state(original, source.snapshot_state())
    return source, summary


def test_should_apply_arrived_raw_corrections_privately_then_return_as_pure_student(corrected):
    source, summary = corrected
    assert source.snapshot_state().completed_updates == 3
    update = summary["update"]
    assert update["work"]["fit_calls"] == 0 and update["work"]["training_rows"] == 16
    assert update["input_projection"]["source_spec"]["observation_dim"] == 42
    assert update["input_projection"]["model_spec"]["observation_dim"] == 12
    assert (
        update["policy"].get(
            "observation_standardization",
            update["policy"].get("base", {}).get("observation_standardization"),
        )
        is None
    )
    assert len(update["applied_updates"]) == 8
    assert all(
        r["observed_at"] == 2 and r["arrived_at"] == r["applied_at"] == 3
        for r in update["applied_updates"]
    )
    assert summary["rollouts"][0]["action_history_exact"] is True
    assert summary["candidate_state_exact"] is True


@pytest.mark.parametrize("fault", ["mode", "dimension", "frame", "setup"])
def test_should_refuse_changed_cached_recipe_before_sdk(learned, monkeypatch, tmp_path, fault):
    recipe, warm, new, _, captured = learned
    result = deepcopy(captured["result"])
    key, value = {
        "mode": ("mode", "zero" if recipe.mode == "executed" else "executed"),
        "dimension": ("feature_dim", 12.0),
        "frame": ("displacement_frame", "root"),
        "setup": ("setup", "rotated_panda"),
    }[fault]
    result.report["action_history_recipe"][key] = value
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    report = entry.run_learning_rollout(
        tmp_path / "refused.json",
        warm,
        new,
        phase_updates=1,
        learner_family=result.report.get("learner_family", "bp"),
        a_only_control=True,
        action_history_recipe=recipe,
    )
    assert report["complete"] is False and report["rollout_attempted"] is False
    assert report["training"] == result.report


@pytest.mark.parametrize("fault", ["floor", "rgb", "expected"])
def test_should_refuse_empirical_transform_requests_before_constructor(monkeypatch, fault):
    recipe = PickCubeTaskSpaceHistoryRecipe()
    warm, new = (history.action_history_training_episode(value, recipe) for value in episodes())
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    options: Any = {
        "floor": dict(scale_floors=(0.1,) * 12),
        "rgb": dict(rgb_range_recipe=object()),
        "expected": dict(expected_standardization=object()),
    }[fault]
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(
            warm, new, phase_updates=1, action_history_recipe=recipe, **options
        )


@pytest.mark.parametrize("mode", ["executed", "zero"])
def test_should_forward_explicit_cli_taskspace_recipe_without_training(monkeypatch, tmp_path, mode):
    calls = []
    monkeypatch.setattr(
        entry, "_read_teacher", lambda path, seed, **options: episodes()[int(seed == 19)]
    )

    def run(*args, **options):
        calls.append(options)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_learning_rollout", run)
    assert (
        entry.main(
            [
                "--output",
                str(tmp_path / "cli.json"),
                "--warm-teacher",
                "unused17",
                "--new-teacher",
                "unused19",
                "--taskspace-history",
                mode,
            ]
        )
        == 0
    )
    assert calls[0]["action_history_recipe"] == PickCubeTaskSpaceHistoryRecipe(mode=mode)


def test_should_refuse_world_recipe_under_changed_native_controller_without_predictions(
    learned, monkeypatch, tmp_path
):
    recipe, _, _, _, captured = learned
    source = captured["result"].candidate.fork()
    monkeypatch.setattr(native, "_make_environment", TaskspaceEnvironment)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-taskspace-refusal")
    monkeypatch.setattr(smoke, "CONTROL_MODE", "unsupported_controller")
    monkeypatch.setattr(type(source), "predict", forbidden)
    result = smoke.run_frozen_rollout(
        tmp_path / "bad-controller.json",
        learner=source,
        visual_observation=False,
        action_history_recipe=recipe,
    )
    assert result.report["complete"] is False
    assert result.report["prediction_attempts"] == 0
