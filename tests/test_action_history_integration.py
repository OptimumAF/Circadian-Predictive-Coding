"""Tiny actual history streams and frozen fake IO; source-count budget recorded separately."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
import struct
from typing import Any

import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app.action_history_encoding import ActionHistoryEncoding
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import ContinuousTransition
from src.core.observation_standardization import ScaleFlooredObservationStandardization
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import assert_state
from test_state_quaternion_robot_learning import QuaternionEnvironment, episodes, observation

FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
MODES = ("executed", "zero")


def forbidden(*args, **kwargs):
    pytest.fail("refused history request admitted model, fit or native work")


@pytest.fixture(scope="module", params=[(f, m) for f in FAMILIES for m in MODES])
def learned(request, tmp_path_factory):
    family, mode = request.param
    recipe = history.PickCubeActionHistoryRecipe(mode=mode, canonical_quaternions=True)
    warm, new = episodes()
    before_inputs = asdict(warm), asdict(new)
    captured: dict[str, Any] = dict(predictions=[], fits=[], initial_parameters=[])
    environment = QuaternionEnvironment()
    original_train = entry.run_continuous_learnability
    original_fit = demo.fit_training_standardization
    original_init = bp.NumpyContinuousBCLearner.__init__
    owner_type = bp.NumpyContinuousBCLearner if family == "bp" else pc.NumpyContinuousPCLearner
    original_predict = owner_type.predict

    def initialize(owner, policy):
        original_init(owner, policy)
        captured["initial_parameters"].append(tuple(v.copy() for v in owner._state.parameters))

    def train(a, b, **options):
        captured["training_inputs"] = a, b
        result = original_train(a, b, **options)
        captured["result"] = result
        assert result.candidate is not None, result.report.get("error")
        captured["before_native"] = result.candidate.snapshot_state()
        return result

    def fit(value, **options):
        captured["fits"].append((asdict(value), options))
        return original_fit(value, **options)

    def predict(owner, features):
        if len(features) == 1:
            captured["predictions"].append(features.copy())
        return original_predict(owner, features)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(entry, "run_continuous_learnability", train)
        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialize)
        patch.setattr(owner_type, "predict", predict)
        patch.setattr(native, "_make_environment", lambda: environment)
        patch.setattr(smoke, "version", lambda _: "fake-history-correctness")
        report = entry.run_learning_rollout(
            tmp_path_factory.mktemp("history-stream") / "case.json",
            warm,
            new,
            learner_family=family,
            phase_updates=1,
            a_only_control=True,
            action_history_recipe=recipe,
            diagnostic_stage_bindings=(
                ReadinessStageBinding(warm, ("lift", "align")),
                ReadinessStageBinding(new, ("lift", "align")),
            ),
        )
    assert (asdict(warm), asdict(new)) == before_inputs
    return family, recipe, report, captured, environment, warm, new


def test_should_fit_prefix_once_and_leave_history_scale_fixed_for_every_family(learned):
    _, recipe, report, captured, _, warm, _ = learned
    assert report["complete"] is True, report.get("error")
    model_warm, _ = captured["training_inputs"]
    assert model_warm.spec.observation_dim == 47
    assert len(captured["fits"]) == 1
    assert captured["fits"] == [(asdict(model_warm), dict(action_history_recipe=recipe))]
    candidate = captured["result"].candidate
    policy = candidate.policy.base if hasattr(candidate.policy, "base") else candidate.policy
    stats = policy.observation_standardization
    assert stats.means[0] == 0.5 and stats.scales[0] == 0.5 and stats.fitted_rows == 2
    assert stats.means[22:26] == (0.5, -0.5, 0.5, -0.5)
    assert stats.means[-5:] == (0.0,) * 5 and stats.scales[-5:] == (1.0,) * 5
    assert report["training"]["work"]["fit_feature_values"] == 84
    assert report["training"]["work"]["fixed_history_dimensions"] == 5
    assert report["training"]["work"]["training_rows"] == 6
    if "training_rows_by_context" in report["training"]["work"]:
        assert report["training"]["work"]["training_rows_by_context"] == dict(warm=6, new=0)
    assert all(r["episode_id"] == warm.episode_id for r in report["training"]["applied_updates"])
    assert len(captured["initial_parameters"]) == 1
    assert captured["initial_parameters"][0][0].shape == (47, 64)


def test_should_serve_prior_executed_native32_and_preserve_raw_trace_and_owners(learned):
    _, recipe, report, captured, environment, _, _ = learned
    native_report = report["rollout"]
    assert report["candidate_unchanged"] is native_report["action_history_exact"] is True
    assert native_report["episode"]["spec"]["observation_dim"] == 42
    assert native_report["episode"]["transitions"][0]["observation"][22:26] == (
        -0.5,
        0.5,
        -0.5,
        0.5,
    )
    values = captured["predictions"]
    assert len(values) == 2 and all(v.shape == (1, 47) for v in values)
    assert tuple(values[0][0, -5:]) == (0.0,) * 5
    first_sent = tuple(float(v) for v in environment.commands[0][0])
    expected_tail = (
        (*first_sent, 1 / 6 if first_sent[-1] < 0 else 0.0)
        if recipe.mode == "executed"
        else (0.0,) * 5
    )
    assert tuple(values[1][0, -5:]) == expected_tail
    assert tuple(values[1][0, 22:26]) == (0.5, -0.5, 0.5, -0.5)
    assert native_report["action_history"]["work"] == dict(
        encoding_attempts=2,
        encoding_returns=2,
        committed_controls=2,
        source_values_validated=84,
        model_values_emitted=94,
        model_numeric_bytes_emitted=752,
    )
    assert native_report["optimizer_updates"] == 0 and len(environment.commands) == 2
    assert environment.reset_calls == [17] and environment.close_calls == 1
    assert_state(captured["before_native"], captured["result"].candidate.snapshot_state())


@pytest.mark.parametrize(
    "fault", ["missing", "other_mode", "float_dimension", "bool_count", "tail_mean"]
)
def test_should_refuse_cached_training_recipe_or_scaling_before_sdk(
    learned, monkeypatch, tmp_path, fault
):
    _, recipe, _, captured, _, warm, new = learned
    result = demo.ContinuousLearnabilityResult(
        deepcopy(captured["result"].report), captured["result"].candidate.fork()
    )
    if fault == "missing":
        del result.report["action_history_recipe"]
    elif fault == "other_mode":
        result.report["action_history_recipe"]["mode"] = (
            "zero" if recipe.mode == "executed" else "executed"
        )
    elif fault == "float_dimension":
        result.report["action_history_recipe"]["source_dim"] = 42.0
    elif fault == "bool_count":
        result.report["action_history_recipe"]["closing_limit"] = True
    else:
        owner: Any = result.candidate
        base = owner.policy.base if hasattr(owner.policy, "base") else owner.policy
        stats = replace(
            base.observation_standardization,
            means=base.observation_standardization.means[:-1] + (0.5,),
        )
        changed = replace(base, observation_standardization=stats)
        policy = replace(owner.policy, base=changed) if hasattr(owner.policy, "base") else changed
        owner._policy = policy
        owner._state = replace(owner._state, policy=policy)
        result.report["policy"] = asdict(policy)
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    actual = entry.run_learning_rollout(
        tmp_path / "refused.json",
        warm,
        new,
        learner_family=learned[0],
        phase_updates=1,
        a_only_control=True,
        action_history_recipe=recipe,
    )
    assert actual["complete"] is False and actual["rollout_attempted"] is False
    assert actual["training"] == result.report


@pytest.mark.parametrize("mode", MODES)
def test_should_keep_declared_prefix_floors_and_fixed_tail_without_extra_model(mode):
    recipe = history.PickCubeActionHistoryRecipe(mode=mode)
    encoded = history.action_history_training_episode(episodes()[0], recipe)
    stats = bp.fit_training_standardization(
        encoded, action_history_recipe=recipe, scale_floors=(2.0,) * 42
    )
    assert type(stats) is ScaleFlooredObservationStandardization
    assert stats.means[0] == 0.5 and stats.scales[0] == 0.5
    assert stats.effective_scales[0] == 2.0
    assert (
        stats.means[-5:] == (0.0,) * 5
        and stats.scales[-5:] == stats.effective_scales[-5:] == (1.0,) * 5
    )
    assert stats.scale_floors == (2.0,) * 42 + (1.0,) * 5


@pytest.mark.parametrize("fault", ["tail", "role", "rgb", "floors"])
def test_should_refuse_invalid_fit_before_array_math(monkeypatch, fault):
    recipe = history.PickCubeActionHistoryRecipe()
    encoded = history.action_history_training_episode(episodes()[0], recipe)
    options: dict[str, Any] = dict(action_history_recipe=recipe)
    if fault == "tail":
        object.__setattr__(
            encoded.transitions[0], "observation", encoded.transitions[0].observation[:-1] + (0.5,)
        )
    elif fault == "role":
        encoded = replace(encoded, role="probe_train")
    elif fault == "rgb":
        options["rgb_range_recipe"] = object()
    else:
        options["scale_floors"] = (1.0,) * 47
    monkeypatch.setattr(bp.observations, "fit_observation_standardization", forbidden)
    with pytest.raises(ValueError):
        bp.fit_training_standardization(encoded, **options)


def test_should_bind_quantized_return_and_refuse_coherent_ledger_change_without_models():
    recipe = history.PickCubeActionHistoryRecipe()
    ledger = ActionHistoryEncoding(recipe)
    requested = (0.1, -0.2, 0.3, -1e-50)
    raw = observation()
    ledger.encode(raw)
    sent = tuple(struct.unpack("!f", struct.pack("!f", v))[0] for v in requested)
    record = ContinuousTransition(0, raw, sent, 0.0, observation(1), False, True)
    ledger.commit(record, requested)
    assert ledger.state.previous_action == sent and ledger.state.closing_count == 0
    ledger.validate_executions((record,), observation(1))
    ledger.state = history.ActionHistoryState((0.5, 0.0, 0.0, -1.0), 1)
    ledger.check(ledger.report())  # Public coherence cannot supply execution authority.
    with pytest.raises(ValueError):
        ledger.validate_executions((record,), observation(1))


def test_should_refuse_second_encoding_after_uncertain_control_without_models():
    ledger = ActionHistoryEncoding(history.PickCubeActionHistoryRecipe())
    ledger.encode(observation())
    with pytest.raises(ValueError):
        ledger.encode(observation(1))
    assert ledger.report()["work"]["committed_controls"] == 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("canonical", [False, True])
def test_should_forward_cli_recipe_from_explicit_selection_without_training(
    monkeypatch, tmp_path, mode, canonical
):
    calls = []
    monkeypatch.setattr(
        entry,
        "_read_readiness_teacher",
        lambda path, seed, **kw: (
            episodes()[0 if seed == 17 else 1],
            dict(stages=("lift", "align")),
        ),
    )

    def rollout(*args, **kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_learning_rollout", rollout)
    argv = [
        "--output",
        str(tmp_path / "cli.json"),
        "--warm-teacher",
        "warm.json",
        "--new-teacher",
        "new.json",
        "--readiness-teachers",
        "--readiness-stage-residuals",
        "--action-history",
        mode,
    ]
    if canonical:
        argv += ["--canonical-state-quaternions"]
    assert entry.main(argv) == 0
    assert calls[0]["action_history_recipe"] == history.PickCubeActionHistoryRecipe(
        mode=mode, canonical_quaternions=canonical
    )
    assert "canonical_state_quaternions" not in calls[0]


def test_should_refuse_visual_history_before_inputs_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "_admit_inputs", forbidden)
    with pytest.raises(ValueError):
        entry.run_learning_rollout(
            tmp_path / "absent.json",
            *episodes(),
            visual_observation=True,
            action_history_recipe=history.PickCubeActionHistoryRecipe(),
        )
    assert not (tmp_path / "absent.json").exists()


@pytest.fixture(scope="module")
def failure_source():
    recipe = history.PickCubeActionHistoryRecipe(canonical_quaternions=True)
    warm, new = (history.action_history_training_episode(value, recipe) for value in episodes())
    result = demo.run_continuous_learnability(
        warm, new, phase_updates=1, a_only_control=True, action_history_recipe=recipe
    )
    assert result.candidate is not None, result.report.get("error")
    return recipe, result.candidate


@pytest.mark.parametrize(
    "fault",
    [
        "step",
        "return",
        "timer_metadata",
        "timer_counter",
        "timer_raw",
        "timer_raise",
        "observation_owner",
        "admission_owner",
        "coherent_history",
    ],
)
def test_should_refuse_unobserved_or_corrupted_history_and_keep_paid_work(
    failure_source, monkeypatch, tmp_path, fault
):
    recipe, source = failure_source
    owner = source.fork()
    before = owner.snapshot_state()
    environment = QuaternionEnvironment()
    captured: dict[str, Any] = {}
    original_rollout, original_step = smoke._rollout, native.ManiSkillPickCubeEpisode.step
    original_observe = vars(native.ManiSkillPickCubeEpisode)["observation"].fget
    original_prepare, original_encode = smoke._prepare_learner, ActionHistoryEncoding.encode

    def rollout(*args, **kwargs):
        captured["report"] = args[4]
        return original_rollout(*args, **kwargs)

    def step(port, action):
        record = original_step(port, action)
        return replace(record, action=(0.0,) * 4) if fault == "return" else record

    def observe(port):
        raw = original_observe(port)
        if fault == "observation_owner":
            owner._state.parameters[0][0, 0] += 0.01
        return raw

    def prepare(value):
        if fault == "admission_owner":
            owner._state.parameters[0][0, 0] += 0.01
        return original_prepare(value)

    def encode(ledger, raw):
        result = original_encode(ledger, raw)
        if fault == "coherent_history":
            ledger.state = history.ActionHistoryState((0.5, 0.0, 0.0, -1.0), 1)
        return result

    clock_calls: list[None] = []

    def clock():
        clock_calls.append(None)
        if len(clock_calls) == 2:
            if fault == "timer_raise":
                raise RuntimeError("elapsed observer failed after paid controls")
            if fault == "timer_metadata":
                captured["report"]["action_history"]["recipe"]["source_dim"] = 42.0
            if fault == "timer_counter":
                captured["report"]["action_history"]["work"]["committed_controls"] = 2.0
            if fault == "timer_raw":
                captured["report"]["episode"]["transitions"][0]["reward"] = 0.5
        return float(len(clock_calls))

    if fault == "step":
        environment.after_step = lambda: (_ for _ in ()).throw(
            RuntimeError("uncertain native control")
        )
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "observation", property(observe))
    monkeypatch.setattr(smoke, "_prepare_learner", prepare)
    monkeypatch.setattr(ActionHistoryEncoding, "encode", encode)
    monkeypatch.setattr(smoke, "_rollout", rollout)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-history-refusal")
    monkeypatch.setattr(smoke, "time", SimpleNamespace(monotonic=clock))
    output = tmp_path / (fault + ".json")
    result = smoke.run_frozen_rollout(
        output, learner=owner, action_history_recipe=recipe, monitor_joint_positions=True
    )
    report = result.report
    assert output.exists() and report["complete"] is False and result.corrections is None
    assert report["action_history_exact"] is None
    expected_predictions = (
        1
        if fault in ("step", "return")
        else (0 if fault in ("observation_owner", "admission_owner", "coherent_history") else 2)
    )
    assert report["prediction_attempts"] == expected_predictions
    assert len(environment.commands) == expected_predictions
    if fault in ("step", "return"):
        assert report["action_history"]["pending"] is True
        assert report["action_history"]["work"]["committed_controls"] == 0
    if fault == "step":
        assert report["work"]["step_attempts"] == 1 and report["work"]["completed_steps"] == 0
    if fault not in ("observation_owner", "admission_owner"):
        assert_state(before, owner.snapshot_state())
    else:
        assert report["original_owner_exact"] is False or "owner changed" in report["error"]


def test_should_keep_json_wrapper_default_dispatch_without_history_keyword(monkeypatch):
    calls = []

    def run(*args, **options):
        calls.append(options)
        return demo.ContinuousLearnabilityResult(dict(complete=False), None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    assert demo.run_continuous_learnability_demo(*episodes()) == dict(complete=False)
    assert calls == [dict(expected_standardization=None)]


def test_should_finish_history_capture_without_querying_disposed_native_scene(
    failure_source, monkeypatch, tmp_path
):
    recipe, source = failure_source
    owner = source.fork()
    before = owner.snapshot_state()
    environment = QuaternionEnvironment()
    original_episode = native.ManiSkillPickCubeEpisode.episode

    def episode(port):
        if port._closed:
            pytest.fail("post-close public episode read could touch disposed SDK robot")
        return original_episode(port)

    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "episode", episode)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-disposed-scene")
    outcome = smoke.run_frozen_rollout(
        tmp_path / "closed.json",
        learner=owner,
        action_history_recipe=recipe,
        monitor_joint_positions=True,
    )
    assert outcome.report["complete"] is outcome.report["action_history_exact"] is True
    assert outcome.report["work"]["completed_steps"] == 2
    assert environment.close_calls == 1 and len(environment.commands) == 2
    assert_state(before, owner.snapshot_state())
