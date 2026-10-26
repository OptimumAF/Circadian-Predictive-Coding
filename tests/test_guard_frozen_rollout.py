"""Guard scoring through real frozen learners and deterministic fake native IO.

Four cached H2/D12 initializations supply all cases; forks draw no randomness.
No fixture fits, updates, queries a teacher, or starts a simulator.
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
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import assert_state
from test_state_quaternion_robot_learning import QuaternionEnvironment


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def forbidden(*args, **kwargs):
    pytest.fail("guard scoring admitted forbidden model, teacher, or native work")


@pytest.fixture(scope="module", autouse=True)
def work_ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        predictions=0,
        prediction_rows=0,
        fake_controls=0,
        updates=0,
        fits=0,
        teacher_queries=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    bp_predict, pc_predict = (
        bp.NumpyContinuousBCLearner.predict,
        pc.NumpyContinuousPCLearner.predict,
    )
    native_step = native.ManiSkillPickCubeEpisode.step

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (policy.spec.observation_dim + 4)
        return initialize(owner, policy)

    def predicted(original):
        def predict(owner, features):
            counts["predictions"] += 1
            counts["prediction_rows"] += len(features)
            return original(owner, features)

        return predict

    def stepped(port, action):
        counts["fake_controls"] += 1
        return native_step(port, action)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted(bp_predict))
        patch.setattr(pc.NumpyContinuousPCLearner, "predict", predicted(pc_predict))
        patch.setattr(native.ManiSkillPickCubeEpisode, "step", stepped)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(smoke, "PickCubeCorrectionQuery", forbidden)
        yield counts
    if os.environ.get("CPG_LEDGER"):
        with Path(os.environ["CPG_LEDGER"]).open("x", encoding="utf8") as handle:
            json.dump(counts, handle, sort_keys=True)
            handle.write("\n")
    assert counts["initializations"] == 4 and counts["normal_draws"] == 128
    assert counts["predictions"] <= 128 and counts["prediction_rows"] <= 128
    assert counts["fake_controls"] <= 128


@pytest.fixture(scope="module")
def owners(work_ledger):
    spec = ContinuousActionSpec(12, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    result: dict[str, Any] = {}
    owner: Any
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(spec, hidden_dim=2, seed=17)
        if family == "bp":
            owner = bp.NumpyContinuousBCLearner(base)
        else:
            chemistry = (
                None
                if family == "pc"
                else CoupledCircadianConfig.matched_pc_control()
                if family == "neutral-cpc"
                else CoupledCircadianConfig()
            )
            owner = pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=chemistry))
        # Why: deterministic nonrepresentable commands expose float32 history;
        # this is an explicit synthetic frozen state, not a claimed trained model.
        initial = owner.snapshot_state()
        parameters = tuple(np.zeros_like(value) for value in initial.parameters)
        parameters[3][:] = (0.1, -0.2, 0.3, -0.4)
        owner.restore_state(replace(initial, parameters=parameters))
        result[family] = owner
    return result


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = QuaternionEnvironment()

    # This reused kinematics fixture has arbitrary rewards (25, 50). Match the
    # native sparse task contract explicitly rather than changing score assertions.
    def sparse_reward():
        environment.bad_reward = np.array([float(len(environment.commands) == 2)], dtype=np.float64)

    environment.after_step = sparse_reward
    creations = []

    def factory():
        creations.append(True)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-guard-boundary")
    return environment, creations, tmp_path / "guard.json"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [31, 37])
def test_should_score_both_guard_resets_with_frozen_family_and_native32_history(
    owners, boundary, monkeypatch, family, seed
):
    owner = owners[family].fork()
    before = owner.snapshot_state()
    environment, creations, output = boundary
    owner_type = type(owner)
    original_predict = owner_type.predict
    features = []

    def predict(model, values):
        features.append(values.copy())
        return original_predict(model, values)

    monkeypatch.setattr(owner_type, "predict", predict)
    outcome = smoke.run_frozen_rollout(
        output,
        learner=owner,
        guard_seed=seed,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    report = outcome.report
    assert report["complete"] is True, report.get("error", report.get("action_history_error"))
    assert outcome.corrections is None
    assert report["role"] == report["episode"]["role"] == "inner_guard"
    assert report["context"] == report["episode"]["context"] == f"pick-cube-inner-guard-{seed}"
    assert report["seed"] == report["guard_seed"] == seed and report["scoring_only"] is True
    assert report["episode"]["episode_id"] == f"pick-cube-guard-smoke-{seed}"
    assert report["episode"]["spec"]["observation_dim"] == 42
    assert report["optimizer_updates"] == 0 and "corrections" not in report
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["sparse_return"] == 1.0
    assert report["any_success"] is report["final_success"] is True
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    assert report["action_history_exact"] is True
    assert creations == [True] and environment.reset_calls == [seed]
    assert environment.close_calls == 1 and len(environment.commands) == 2
    assert len(features) == 2 and all(value.shape == (1, 12) for value in features)
    assert tuple(features[0][0, -5:]) == (0.0,) * 5
    sent = tuple(float(value) for value in environment.commands[0][0])
    assert environment.commands[0].dtype == np.dtype("float32")
    assert tuple(features[1][0, -5:]) == (*sent, 1 / 6)
    assert report["episode"]["transitions"][0]["action"] == sent
    assert report["action_history"]["work"]["committed_controls"] == 2
    assert_state(before, owner.snapshot_state())
    assert json.loads(output.read_text(encoding="utf8"))["episode"]["role"] == "inner_guard"


@pytest.mark.parametrize("seed", [17, 19])
def test_should_preserve_train_default_schema_and_explicit_other_train_reset(
    owners, boundary, seed
):
    environment, _, output = boundary
    options: dict[str, Any] = {} if seed == 17 else dict(train_seed=19)
    outcome = smoke.run_frozen_rollout(
        output,
        learner=owners["bp"].fork(),
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        **options,
    )
    report = outcome.report
    assert report["complete"] is True, report.get("error")
    assert report["role"] == report["episode"]["role"] == "train"
    assert report["seed"] == seed and environment.reset_calls == [seed]
    assert report["episode"]["episode_id"] == f"pick-cube-smoke-{seed}"
    assert "guard_seed" not in report and "scoring_only" not in report
    assert outcome.corrections is None


@pytest.mark.parametrize("seed", [17, 19, 23, 24, 29, 32, True, 31.0, "31"])
def test_should_refuse_non_guard_seed_before_prediction_or_environment(
    owners, boundary, monkeypatch, seed
):
    _, creations, output = boundary
    monkeypatch.setattr(smoke, "_prepare_learner", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    outcome = smoke.run_frozen_rollout(
        output,
        learner=owners["bp"].fork(),
        guard_seed=seed,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["prediction_attempts"] == outcome.report["creation_attempts"] == 0
    assert creations == []


@pytest.mark.parametrize("kind", ["missing", "foreign", "nearest", "probe_train"])
def test_should_require_supported_native_train_owner_before_environment(
    owners, boundary, monkeypatch, kind
):
    _, creations, output = boundary
    owner: Any = owners["bp"].fork()
    if kind == "missing":
        owner = None
    elif kind == "foreign":
        owner = object()
    elif kind == "nearest":
        # The guard admission must reject this unsupported class before reading
        # prototypes or preparing a learner; no nearest initializer is needed.
        owner = object.__new__(smoke.NumpyNearestContinuousBCLearner)
    else:
        changed = replace(owner.policy, training_role="probe_train")
        owner._policy = changed
        owner._state = replace(owner._state, policy=changed)
    if kind != "probe_train":
        monkeypatch.setattr(smoke, "_prepare_learner", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    outcome = smoke.run_frozen_rollout(
        output, learner=owner, guard_seed=31, action_history_recipe=PickCubeTaskSpaceHistoryRecipe()
    )
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["prediction_attempts"] == outcome.report["creation_attempts"] == 0
    assert creations == []


@pytest.mark.parametrize(
    "option",
    [
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
        "monitor_joint_positions",
        "action_decoding_recipe",
        "missing_recipe",
        "zero_recipe",
        "history47",
    ],
)
def test_should_refuse_forbidden_guard_modes_before_model_or_native_work(
    owners, boundary, monkeypatch, option
):
    _, creations, output = boundary
    options: dict[str, Any] = dict(
        learner=owners["bp"].fork(),
        guard_seed=31,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    if option == "train_seed":
        options[option] = 19
    elif option == "missing_recipe":
        options["action_history_recipe"] = None
    elif option == "zero_recipe":
        options["action_history_recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    elif option == "history47":
        options["action_history_recipe"] = PickCubeActionHistoryRecipe()
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
    outcome = smoke.run_frozen_rollout(output, **options)
    assert outcome.report["complete"] is False and outcome.corrections is None
    assert outcome.report["prediction_attempts"] == outcome.report["creation_attempts"] == 0
    assert creations == []


def test_should_bind_guard_history_identity_from_birth_and_verify_native_episode(owners, boundary):
    _, _, output = boundary
    recipe = PickCubeTaskSpaceHistoryRecipe()
    encoding = ActionHistoryEncoding(recipe, guard_seed=31)
    assert encoding.native_binding == (31, "pick-cube-inner-guard-31", "pick-cube-guard-smoke-31")
    assert encoding.native_role == "inner_guard"
    assert ActionHistoryEncoding(recipe).native_role is None
    assert ActionHistoryEncoding(recipe, train_seed=17).native_role == "train"
    with pytest.raises(ValueError):
        ActionHistoryEncoding(recipe, train_seed=17, guard_seed=31)
    outcome = smoke.run_frozen_rollout(
        output, learner=owners["bp"].fork(), guard_seed=31, action_history_recipe=recipe
    )
    assert outcome.report["complete"] is True
    data = outcome.report["episode"]
    episode = ContinuousEpisode(
        data["episode_id"],
        data["context"],
        data["role"],
        ContinuousActionSpec(**data["spec"]),
        tuple(ContinuousTransition(**record) for record in data["transitions"]),
    )
    for record in episode.transitions:
        encoding.encode(record.observation)
        encoding.commit(record, record.action)
    encoding.verify_episode(episode)
    for changed_episode in (
        replace(episode, role="train"),
        replace(episode, context="pick-cube"),
        replace(episode, episode_id="pick-cube-smoke-17"),
    ):
        with pytest.raises(ValueError):
            encoding.verify_episode(changed_episode)
    # A coherent replacement of the public reset tuple cannot renew its authority.
    encoding.native_binding = (17, "pick-cube", "pick-cube-smoke-17")
    with pytest.raises(ValueError):
        encoding.report()


@pytest.mark.parametrize(
    "fault",
    [
        "port_role",
        "history_binding",
        "report_namespace",
        "source_at_close",
        "identity_at_close",
        "late_owner_getter",
    ],
)
def test_should_refuse_late_namespace_or_owner_mutation_and_preserve_paid_prefix(
    owners, boundary, monkeypatch, fault
):
    owner = owners["bp"].fork()
    before = owner.snapshot_state()
    environment, _, output = boundary
    captured: dict[str, Any] = {}
    original_rollout = smoke._rollout
    original_step = native.ManiSkillPickCubeEpisode.step
    original_encode = ActionHistoryEncoding.encode
    original_close = native.ManiSkillPickCubeEpisode.close
    original_snapshot = owner.snapshot_state

    def rollout(*args, **options):
        captured["report"] = args[4]
        return original_rollout(*args, **options)

    def step(port, action):
        captured["port"] = port
        returned = original_step(port, action)
        if fault == "port_role":
            port._role = "train"
        return returned

    def encode(encoding, raw):
        captured["encoding"] = encoding
        returned = original_encode(encoding, raw)
        if fault == "history_binding":
            encoding.native_binding = (17, "pick-cube", "pick-cube-smoke-17")
        return returned

    def close(port):
        original_close(port)
        if fault == "source_at_close":
            owner._state.parameters[3][0] += 0.125
        elif fault == "identity_at_close":
            port._episode_id = "pick-cube-smoke-17"

    def snapshot():
        returned = original_snapshot()
        if fault == "late_owner_getter" and environment.close_calls:
            captured["port"]._role = "train"
        return returned

    ticks: list[None] = []

    def clock():
        ticks.append(None)
        if len(ticks) == 2 and fault == "report_namespace":
            captured["report"]["guard_seed"] = 37
        return float(len(ticks))

    monkeypatch.setattr(smoke, "_rollout", rollout)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    monkeypatch.setattr(ActionHistoryEncoding, "encode", encode)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "close", close)
    monkeypatch.setattr(owner, "snapshot_state", snapshot)
    monkeypatch.setattr(smoke, "time", SimpleNamespace(monotonic=clock))
    outcome = smoke.run_frozen_rollout(
        output, learner=owner, guard_seed=31, action_history_recipe=PickCubeTaskSpaceHistoryRecipe()
    )
    report = outcome.report
    assert report["complete"] is False and outcome.corrections is None
    assert report["action_history_exact"] is None
    paid = 0 if fault == "history_binding" else 1 if fault == "port_role" else 2
    assert len(environment.commands) == report["work"]["completed_steps"] == paid
    assert report["prediction_attempts"] == paid
    assert environment.close_calls == 1
    if fault == "source_at_close":
        assert owner._state.parameters[3][0] == before.parameters[3][0] + 0.125
    else:
        assert_state(before, original_snapshot())
