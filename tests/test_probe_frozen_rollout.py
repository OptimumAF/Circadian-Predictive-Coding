"""Reserved reset24 scores frozen policies through deterministic fake native IO.

Four cached H2 sources are synthetic states, not claims about trained control.
No fitting, optimizer, teacher, or simulator is used by this module.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app.action_history_encoding import ActionHistoryEncoding
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import assert_state
from test_state_quaternion_robot_learning import QuaternionEnvironment


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def forbidden(*args, **kwargs):
    pytest.fail("reserved scoring admitted forbidden learner, label, or native work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        SGD=0,
        prediction_calls=0,
        prediction_rows=0,
        fake_controls=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    predicts = (bp.NumpyContinuousBCLearner.predict, pc.NumpyContinuousPCLearner.predict)
    step = native.ManiSkillPickCubeEpisode.step

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (policy.spec.observation_dim + 4)
        return initialize(owner, policy)

    def predicted(original):
        def predict(owner, features):
            counts["prediction_calls"] += 1
            counts["prediction_rows"] += len(features)
            return original(owner, features)

        return predict

    def stepped(port, command):
        counts["fake_controls"] += 1
        return step(port, command)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted(predicts[0]))
        patch.setattr(pc.NumpyContinuousPCLearner, "predict", predicted(predicts[1]))
        patch.setattr(native.ManiSkillPickCubeEpisode, "step", stepped)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(smoke, "PickCubeCorrectionQuery", forbidden)
        yield counts
    if os.environ.get("NPE_FROZEN_LEDGER"):
        with Path(os.environ["NPE_FROZEN_LEDGER"]).open("x", encoding="utf8") as handle:
            json.dump(counts, handle, sort_keys=True)
            handle.write("\n")
    assert counts["initializations"] == 4 and counts["normal_draws"] == 128
    assert counts["prediction_calls"] <= 50 and counts["prediction_rows"] <= 50
    assert counts["fake_controls"] <= 50


@pytest.fixture(scope="module")
def owners(ledger):
    spec = ContinuousActionSpec(12, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    result: dict[str, Any] = {}
    owner: Any
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(spec, hidden_dim=2, seed=17)
        chemistry = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        owner = (
            bp.NumpyContinuousBCLearner(base)
            if family == "bp"
            else pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=chemistry))
        )
        initial = owner.snapshot_state()
        parameters = tuple(np.zeros_like(value) for value in initial.parameters)
        parameters[3][:] = (0.1, -0.2, 0.3, -0.4)
        owner.restore_state(replace(initial, parameters=parameters))
        result[family] = owner
    return result


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = QuaternionEnvironment()
    creations = []

    def sparse_reward():
        environment.bad_reward = np.array([float(len(environment.commands) == 2)], dtype=np.float64)

    def factory():
        creations.append(True)
        return environment

    environment.after_step = sparse_reward
    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-reserved-probe")
    return environment, creations, tmp_path


@pytest.mark.parametrize("family", FAMILIES)
def test_should_score_reserved_test24_with_each_frozen_family_and_native32_history(
    owners, boundary, monkeypatch, family
):
    source = owners[family].fork()
    before = source.snapshot_state()
    environment, creations, folder = boundary
    features = []
    original = type(source).predict

    def predict(owner, values):
        features.append(values.copy())
        return original(owner, values)

    monkeypatch.setattr(type(source), "predict", predict)
    outcome = smoke.run_frozen_rollout(
        folder / "test24.json",
        learner=source,
        probe_test_seed=24,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    report = outcome.report
    assert report["complete"] is True, report.get("error", report.get("action_history_error"))
    assert outcome.corrections is None and "corrections" not in report
    assert report["role"] == report["episode"]["role"] == "probe_test"
    assert report["context"] == report["episode"]["context"] == "pick-cube-reserved-probe-23-24"
    assert report["episode"]["episode_id"] == "pick-cube-probe-test-smoke-24"
    assert report["seed"] == report["probe_test_seed"] == 24 and report["scoring_only"] is True
    assert "guard_seed" not in report and report["optimizer_updates"] == 0
    assert (
        report["frozen_state_exact"]
        is report["original_owner_exact"]
        is report["action_history_exact"]
        is True
    )
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert (
        report["sparse_return"] == 1.0 and report["any_success"] is report["final_success"] is True
    )
    assert creations == [True] and environment.reset_calls == [24] and environment.close_calls == 1
    assert len(features) == 2 and tuple(features[0][0, -5:]) == (0.0,) * 5
    sent = tuple(float(value) for value in environment.commands[0][0])
    assert environment.commands[0].dtype == np.dtype("float32")
    assert tuple(features[1][0, -5:]) == (*sent, 1 / 6)
    assert report["episode"]["transitions"][0]["action"] == sent
    assert_state(before, source.snapshot_state())


def test_should_repeat_the_identical_reserved_reset_for_two_independent_frozen_scores(
    owners, monkeypatch, tmp_path
):
    environments = []
    original = owners["bp"].snapshot_state()
    for phase in ("before_probe", "after_probe"):
        environment = QuaternionEnvironment()
        environments.append(environment)
        monkeypatch.setattr(native, "_make_environment", lambda: environment)
        monkeypatch.setattr(smoke, "version", lambda _: "fake-paired-endpoints")
        outcome = smoke.run_frozen_rollout(
            tmp_path / (phase + ".json"),
            learner=owners["bp"].fork(),
            probe_test_seed=24,
            action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        )
        assert outcome.report["complete"] is True
        assert outcome.corrections is None
    assert [environment.reset_calls for environment in environments] == [[24], [24]]
    assert all(environment.close_calls == 1 for environment in environments)
    assert_state(original, owners["bp"].snapshot_state())


def test_should_preserve_default_train_schema_when_probe_test_seed_is_none(owners, boundary):
    environment, _, folder = boundary
    outcome = smoke.run_frozen_rollout(
        folder / "train.json",
        learner=owners["bp"].fork(),
        probe_test_seed=None,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert outcome.report["complete"] is True
    assert outcome.report["role"] == "train" and environment.reset_calls == [17]
    assert "probe_test_seed" not in outcome.report and "scoring_only" not in outcome.report


@pytest.mark.parametrize("seed", [17, 19, 23, 29, 31, 37, True, 24.0, "24"])
def test_should_refuse_any_other_probe_test_seed_before_model_or_native_work(
    owners, boundary, monkeypatch, seed
):
    _, creations, folder = boundary
    monkeypatch.setattr(smoke, "_prepare_learner", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    outcome = smoke.run_frozen_rollout(
        folder / "refused.json",
        learner=owners["bp"].fork(),
        probe_test_seed=seed,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["prediction_attempts"] == outcome.report["creation_attempts"] == 0
    assert creations == []


@pytest.mark.parametrize(
    "option",
    [
        "guard_seed",
        "train_seed",
        "actor",
        "observation_clock",
        "visual_observation",
        "corrective_queries",
        "readiness_corrections",
        "observed_grasp_corrections",
        "goal_consistent_corrections",
        "readiness_roll_in",
        "camera_context",
        "mass_context",
        "lighting_context",
        "observation_projection",
        "action_decoding_recipe",
        "monitor_joint_positions",
        "missing_recipe",
        "zero_recipe",
        "history47",
        "missing_learner",
    ],
)
def test_should_refuse_incompatible_reserved_scoring_modes_before_work(
    owners, boundary, monkeypatch, option
):
    _, creations, folder = boundary
    options: dict[str, Any] = dict(
        learner=owners["bp"].fork(),
        probe_test_seed=24,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    if option in ("guard_seed", "train_seed"):
        options[option] = 31 if option == "guard_seed" else 19
    elif option in ("missing_recipe", "missing_learner"):
        options["learner" if option == "missing_learner" else "action_history_recipe"] = None
    elif option in ("zero_recipe", "history47"):
        options["action_history_recipe"] = (
            PickCubeActionHistoryRecipe()
            if option == "history47"
            else PickCubeTaskSpaceHistoryRecipe(mode="zero")
        )
    else:
        options[option] = (
            True
            if option
            in (
                "visual_observation",
                "corrective_queries",
                "readiness_corrections",
                "observed_grasp_corrections",
                "goal_consistent_corrections",
                "monitor_joint_positions",
            )
            else object()
        )
    monkeypatch.setattr(smoke, "_prepare_learner", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    outcome = smoke.run_frozen_rollout(folder / "refused.json", **options)
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["prediction_attempts"] == outcome.report["creation_attempts"] == 0
    assert creations == []


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "identity",
        "seed",
        "report",
        "history_binding",
        "source_at_close",
        "identity_at_close",
    ],
)
def test_should_refuse_corrupted_reserved_identity_and_keep_the_native_paid_prefix(
    owners, boundary, monkeypatch, fault
):
    source = owners["bp"].fork()
    before = source.snapshot_state()
    environment, _, folder = boundary
    original_step = native.ManiSkillPickCubeEpisode.step
    original_close = native.ManiSkillPickCubeEpisode.close
    original_encode = ActionHistoryEncoding.encode
    original_rollout = smoke._rollout
    captured: dict[str, Any] = {}

    def step(port, command):
        record = original_step(port, command)
        if fault == "role":
            port._role = "probe_train"
        elif fault == "identity":
            port._episode_id = "pick-cube-probe-teacher-24"
        elif fault == "seed":
            port._seed = 23
        return record

    def close(port):
        original_close(port)
        if fault == "source_at_close":
            source._state.parameters[3][0] += 0.125
        elif fault == "identity_at_close":
            port._episode_id = "pick-cube-probe-teacher-24"

    def encode(encoding, raw):
        returned = original_encode(encoding, raw)
        if fault == "history_binding":
            encoding.native_binding = (
                23,
                "pick-cube-reserved-probe-23-24",
                "pick-cube-probe-teacher-23",
            )
        return returned

    def rollout(*args, **kwargs):
        captured["report"] = args[4]
        return original_rollout(*args, **kwargs)

    ticks: list[None] = []

    def clock():
        ticks.append(None)
        if len(ticks) == 2 and fault == "report":
            captured["report"]["probe_test_seed"] = 23
        return float(len(ticks))

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "close", close)
    monkeypatch.setattr(ActionHistoryEncoding, "encode", encode)
    monkeypatch.setattr(smoke, "_rollout", rollout)
    monkeypatch.setattr(smoke, "time", SimpleNamespace(monotonic=clock))
    outcome = smoke.run_frozen_rollout(
        folder / "failed.json",
        learner=source,
        probe_test_seed=24,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    report = outcome.report
    assert report["complete"] is False and outcome.corrections is None
    assert report["action_history_exact"] is None
    paid = (
        0
        if fault == "history_binding"
        else 2
        if fault in ("report", "source_at_close", "identity_at_close")
        else 1
    )
    assert (
        report["prediction_attempts"]
        == len(environment.commands)
        == report["work"]["completed_steps"]
        == paid
    )
    assert environment.close_calls == 1
    if fault == "source_at_close":
        assert source._state.parameters[3][0] == before.parameters[3][0] + 0.125
    else:
        assert_state(before, source.snapshot_state())
