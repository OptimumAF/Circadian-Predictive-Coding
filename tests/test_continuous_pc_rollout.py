"""Frozen trained PC/CPC models at the existing fake robot boundary.

No simulator or learned-skill claim. Six tiny initializations and nine source
updates precede all rollout checks; each case then owns an RNG-free source fork.
Full module: 39 cases, 12 PC/BP constructor entries, 200 normal values, nine
source updates/18 rows, at most45 energy entries/36 latent proposals. Direct
prediction math covers43 rows plus18 source-candidate forward rows; one scripted
prediction does not return. Fake IO has29 scene/reset/close attempts,26 controls,
and nine accepted joint samples/81 coordinates. No fit or actual native work.
"""

from dataclasses import asdict, replace
import json
import sys
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_episode_clock import EpisodeClock
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_public_vectors import public_vector_projection
from src.infra import maniskill_continuous as native
from test_maniskill_bc_smoke import FakeEnvironment
from test_maniskill_joint_positions import StateEnvironment
from test_maniskill_rgb_observations import RGBEnvironment


FAMILIES = ("pc", "neutral-cpc", "active-cpc")


def assert_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    if first.wake_state is None:
        assert second.wake_state is None
    else:
        assert first.wake_state.wake_updates == second.wake_state.wake_updates
        np.testing.assert_array_equal(
            first.wake_state.chemical_state, second.wake_state.chemical_state
        )
        np.testing.assert_array_equal(
            first.wake_state.plasticity_state, second.wake_state.plasticity_state
        )


def changed_wake(state):
    wake, config = state.wake_state, state.policy.circadian
    chemicals = wake.chemical_state + 0.01
    gates = np.clip(np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1.0)
    return replace(
        state, wake_state=replace(wake, chemical_state=chemicals, plasticity_state=gates)
    )


@pytest.fixture(scope="module")
def sources():
    values, oracle = {}, {}
    for name, dimension, updates in [
        ("pc", 3, 2),
        ("neutral-cpc", 3, 2),
        ("active-cpc", 3, 2),
        ("clock", 4, 1),
        ("projection", 21, 1),
        ("joint", 42, 1),
    ]:
        config = (
            None
            if name == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if name == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        declared = ContinuousActionSpec(dimension, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
        owner = pc.NumpyContinuousPCLearner(
            pc.ContinuousPCPolicy(
                bp.ContinuousBCPolicy(declared, hidden_dim=2, seed=17), circadian=config
            )
        )
        features = np.zeros((2, dimension))
        features[:, :3] = [[0.2, -0.3, 0.4], [1.2, -0.3, 0.4]]
        targets = np.array([[0.2, -0.2, 0.4, -1.0], [-0.3, 0.5, -0.2, 1.0]])
        for _ in range(updates):
            owner.train_batch(features, targets)
        assert any(np.any(array) for array in owner.snapshot_state().velocity)
        if config is not None:
            assert owner.wake_updates == updates and np.any(owner.get_chemical_state() > 0)
        values[name] = owner
        if name in FAMILIES:
            raw = np.array([[0.2, -0.3, 0.4], [1.2, -0.3, 0.4]], dtype=np.float32).astype(
                np.float64
            )
            oracle[name] = np.vstack([owner.predict(raw[index : index + 1]) for index in range(2)])
    return values, oracle


@pytest.fixture(autouse=True)
def frozen_only(sources, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("frozen rollout attempted training, latent inference, chemistry or RNG")

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "train_batch", forbidden)
    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", forbidden)
    monkeypatch.setattr(pc.energy, "evaluate_continuous_pc", forbidden)
    monkeypatch.setattr(pc.energy, "feedforward_continuous_latents", forbidden)
    monkeypatch.setattr(pc.cpc, "advance_continuous_wake", forbidden)
    monkeypatch.setattr(bp.np.random, "default_rng", forbidden)


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment, creations = FakeEnvironment(), []

    def factory(**options):
        creations.append(options)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-pc-boundary")
    return environment, creations, tmp_path / "trace.json"


def run(output, **options):
    result = smoke.run_frozen_rollout(output, **options)
    assert json.loads(output.read_text(encoding="utf8")) == json.loads(json.dumps(result.report))
    return result


@pytest.mark.parametrize("family", FAMILIES)
def test_should_send_real_trained_frozen_pc_predictions_and_report_full_native_state(
    boundary, sources, monkeypatch, family
):
    environment, creations, output = boundary
    owner = sources[0][family].fork()
    before, calls = owner.snapshot_state(), []
    original = pc.NumpyContinuousPCLearner.predict

    def predict(child, features):
        calls.append(child)
        return original(child, features)

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", predict)
    result = run(output, learner=owner)
    report = result.report
    assert report["complete"] is True and result.corrections is None
    assert creations == [{}] and environment.reset_calls == [17] and environment.close_calls == 1
    assert len(calls) == 2 and calls[0] is calls[1] and calls[0] is not owner
    for actual, expected in zip(environment.commands, sources[1][family], strict=True):
        np.testing.assert_array_equal(actual, expected[None].astype(np.float32))
    assert report["policy"] == "frozen_continuous_pc"
    assert report["learner_policy"] == asdict(before.policy)
    assert report["source_update_definition"] == CONTINUOUS_PC_ENERGY_ID
    assert report["source_updates"] == 2 and report["source_training_rows"] == 4
    assert report["source_wake_updates"] == (None if family == "pc" else 2)
    assert report["continuous_parameter_momentum_numeric_bytes_per_owner"] == 320
    assert report["continuous_wake_numeric_bytes_per_owner"] == (0 if family == "pc" else 32)
    assert report["continuous_prediction_math"] == "feedforward_without_latent_inference_or_updates"
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["validated_prediction_rows"] == 2 and report["validated_prediction_bytes"] == 64
    assert report["derived_returned_prediction_dense_multiply_accumulates"] == 28
    assert (
        report["optimizer_updates"] == 0
        and report["frozen_state_exact"]
        and report["original_owner_exact"]
    )
    assert_state(before, owner.snapshot_state())
    assert_state(before, calls[0].snapshot_state())


def test_should_keep_negative_task_outcome_complete_without_training_or_forced_success(
    boundary, sources
):
    environment, _, output = boundary
    environment.rewards = (0.0, 0.0)
    report = run(output, learner=sources[0]["active-cpc"].fork()).report
    assert report["complete"] is True
    assert (
        report["sparse_return"] == 0 and report["any_success"] is report["final_success"] is False
    )


@pytest.mark.parametrize(
    "bad", ["foreign", "subclass", "role", "units", "bounds", "corrections_with_clock"]
)
def test_should_refuse_unadmitted_pc_sources_before_fork_sdk_or_queries(
    boundary, sources, monkeypatch, bad
):
    _, creations, output = boundary
    owner: Any = sources[0]["active-cpc"].fork()
    options: dict[str, Any] = {}
    if bad == "foreign":
        owner = object()
    elif bad == "subclass":

        class DerivedPC(pc.NumpyContinuousPCLearner):
            pass

        owner = object.__new__(DerivedPC)
    elif bad == "corrections_with_clock":
        options["corrective_queries"] = True
        options["observation_clock"] = EpisodeClock(50, True)
    else:
        state = owner.snapshot_state()
        base = state.policy.base
        if bad == "role":
            base = replace(base, training_role="probe_train")
        elif bad == "units":
            base = replace(base, spec=replace(base.spec, action_units=("m",) * 4))
        else:
            base = replace(base, spec=replace(base.spec, lower_bounds=(-2.0,) * 4))
        replacement = replace(state.policy, base=base)
        owner._policy, owner._state = replacement, replace(state, policy=replacement)
    monkeypatch.setattr(
        pc.NumpyContinuousPCLearner,
        "fork",
        lambda *_: pytest.fail("unadmitted PC source was forked"),
    )
    monkeypatch.setattr(
        smoke, "version", lambda *_: pytest.fail("unadmitted PC source queried SDK")
    )
    monkeypatch.setattr(
        smoke,
        "PickCubeCorrectionQuery",
        lambda *_: pytest.fail("PC source opened corrective query"),
    )
    result = run(output, learner=owner, **options)
    assert result.report["complete"] is False and result.corrections is None
    assert result.report["creation_attempts"] == 0 and creations == []


@pytest.mark.parametrize("bad", ["alias", "foreign"])
def test_should_refuse_nonowned_or_foreign_private_fork_before_any_native_callback(
    boundary, sources, monkeypatch, bad
):
    _, creations, output = boundary
    owner = sources[0]["pc"].fork()

    class ForeignFork:
        def snapshot_state(self):
            pytest.fail("foreign PC fork snapshot callback executed")

    monkeypatch.setattr(owner, "fork", lambda: owner if bad == "alias" else ForeignFork())
    monkeypatch.setattr(smoke, "version", lambda *_: pytest.fail("bad fork queried SDK"))
    result = run(output, learner=owner)
    assert result.report["complete"] is False and creations == []
    assert result.report["prediction_attempts"] == 0


@pytest.mark.parametrize("bad", ["float_count", "float32"])
def test_should_refuse_equal_valued_invalid_snapshot_returns_after_prediction(
    boundary, sources, monkeypatch, bad
):
    environment, _, output = boundary
    owner = sources[0]["active-cpc"].fork()
    changed: list[pc.NumpyContinuousPCLearner] = []
    original_snapshot, original_predict = (
        pc.NumpyContinuousPCLearner.snapshot_state,
        pc.NumpyContinuousPCLearner.predict,
    )

    def snapshot(child):
        state = original_snapshot(child)
        if changed and child is changed[0]:
            if bad == "float_count":
                malformed_rows: Any = float(state.training_examples)
                state = replace(state, training_examples=malformed_rows)
            else:
                state = replace(
                    state,
                    parameters=(state.parameters[0].astype(np.float32), *state.parameters[1:]),
                )
        return state

    def predict(child, features):
        result = original_predict(child, features)
        changed[:] = [child]
        return result

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", predict)
    report = run(output, learner=owner).report
    assert report["complete"] is False and environment.commands == []
    assert report["prediction_attempts"] == report["prediction_returns"] == 1
    assert report["validated_prediction_rows"] == 0
    assert report["frozen_state_exact"] is not True


@pytest.mark.parametrize("bad", ["throw", "dtype", "shape", "rows", "nan", "bounds", "list"])
def test_should_keep_prediction_attempts_and_refuse_invalid_return_before_control(
    boundary, sources, monkeypatch, bad
):
    environment, _, output = boundary
    owner = sources[0]["active-cpc"].fork()
    before = owner.snapshot_state()
    original = pc.NumpyContinuousPCLearner.predict

    def malformed(child, features):
        if bad == "throw":
            raise FloatingPointError("unreturned PC prediction")
        result = original(child, features)
        if bad == "dtype":
            return result.astype(np.float32)
        if bad == "shape":
            return result[:, :3]
        if bad == "rows":
            return np.vstack([result, result])
        if bad == "nan":
            result[0, 0] = np.nan
        if bad == "bounds":
            result[0, 0] = 1.1
        return result.tolist() if bad == "list" else result

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", malformed)
    result = run(output, learner=owner)
    report = result.report
    assert report["complete"] is False and result.corrections is None
    assert environment.commands == [] and environment.close_calls == 1
    assert report["prediction_attempts"] == 1
    assert report["prediction_returns"] == int(bad != "throw")
    assert report["derived_returned_prediction_dense_multiply_accumulates"] == 14 * int(
        bad != "throw"
    )
    assert report["validated_prediction_rows"] == report["validated_prediction_bytes"] == 0
    assert_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "target,phase",
    [
        ("source", "before_control"),
        ("child", "before_control"),
        ("source", "cleanup"),
        ("child", "cleanup"),
    ],
)
def test_should_refuse_coherent_wake_replacement_even_when_weights_and_actions_are_unchanged(
    boundary, sources, monkeypatch, target, phase
):
    environment, _, output = boundary
    owner = sources[0]["active-cpc"].fork()
    initial = owner.snapshot_state()
    children: list[pc.NumpyContinuousPCLearner] = []
    original_predict, original_close = pc.NumpyContinuousPCLearner.predict, environment.close

    def corrupt():
        victim = owner if target == "source" else children[0]
        victim.restore_state(changed_wake(victim.snapshot_state()))

    def predict(child, features):
        children[:] = [child]
        result = original_predict(child, features)
        if phase == "before_control":
            corrupt()
        return result

    def close():
        original_close()
        if phase == "cleanup":
            corrupt()

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", predict)
    monkeypatch.setattr(environment, "close", close)
    result = run(output, learner=owner)
    report = result.report
    assert report["complete"] is False and result.corrections is None
    assert len(environment.commands) == (0 if phase == "before_control" else 2)
    assert report["prediction_returns"] == (1 if phase == "before_control" else 2)
    assert report["original_owner_exact"] is (target != "source")
    assert report["frozen_state_exact"] is (target != "child")
    victim = owner if target == "source" else children[0]
    after = victim.snapshot_state()
    for a, b in zip(
        initial.parameters + initial.velocity, after.parameters + after.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    assert after.completed_updates == initial.completed_updates
    assert after.wake_state is not None and initial.wake_state is not None
    assert not np.array_equal(after.wake_state.chemical_state, initial.wake_state.chemical_state)


def test_should_refuse_actual_port_spec_mismatch_before_any_forward(boundary, sources):
    environment, creations, output = boundary
    result = run(output, learner=sources[0]["clock"].fork())
    assert result.report["complete"] is False
    assert creations == [{}] and environment.close_calls == 1 and environment.commands == []
    assert result.report["prediction_attempts"] == result.report["prediction_returns"] == 0


def test_should_preserve_native_raw_trace_with_opt_in_elapsed_clock(boundary, sources):
    _, _, output = boundary
    owner = sources[0]["clock"].fork()
    before = owner.snapshot_state()
    report = run(output, learner=owner, observation_clock=EpisodeClock(50, True)).report
    assert report["complete"] is True and report["clock_configuration_exact"] is True
    assert report["clock_values"] == [0.0, 0.02]
    assert all(len(row["observation"]) == 3 for row in report["episode"]["transitions"])
    assert report["derived_returned_prediction_dense_multiply_accumulates"] == 32
    assert_state(before, owner.snapshot_state())


def test_should_feed_projected_public_vectors_to_pc_while_retaining_raw_rgb(
    boundary, sources, monkeypatch
):
    _, _, output = boundary
    environment, options = RGBEnvironment(), []

    def factory(**kwargs):
        options.append(kwargs)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    owner = sources[0]["projection"].fork()
    before = owner.snapshot_state()
    report = run(
        output,
        learner=owner,
        visual_observation=True,
        observation_projection=public_vector_projection(),
    ).report
    assert report["complete"] is True and report["projection_exact"] is True
    assert options == [dict(visual_observation=True)]
    assert report["visual_recipe_exact"] is True
    assert all(len(row["observation"]) == 789 for row in report["episode"]["transitions"])
    assert report["projection_work"]["observation_returns"] == 2
    assert report["projection_work"]["projected_values"] == 42
    assert report["derived_returned_prediction_dense_multiply_accumulates"] == 100
    assert_state(before, owner.snapshot_state())


def test_should_validate_joint_samples_without_changing_trained_cpc(boundary, sources, monkeypatch):
    _, _, output = boundary
    environment = StateEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    owner = sources[0]["joint"].fork()
    before = owner.snapshot_state()
    report = run(output, learner=owner, monitor_joint_positions=True).report
    assert report["complete"] is True and report["joint_position_exact"] is True
    assert report["joint_position_summary"]["samples"] == 3
    assert report["joint_position_validation_samples"] == 3
    assert report["joint_position_validation_coordinates"] == 27
    assert report["derived_returned_prediction_dense_multiply_accumulates"] == 184
    assert_state(before, owner.snapshot_state())


@pytest.mark.parametrize("bad", ["throw", "wake"])
def test_should_refuse_late_joint_callbacks_and_retain_paid_controls_without_releasing_batch(
    boundary, sources, monkeypatch, bad
):
    _, _, output = boundary
    environment = StateEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    owner = sources[0]["joint"].fork()
    original = native.ManiSkillPickCubeEpisode.__dict__["joint_position_summary"].fget
    injected: list[bool] = []

    def late(port):
        value = original(port)
        if environment.close_calls and not injected:
            injected.append(True)
            if bad == "throw":
                raise RuntimeError("late joint metadata failed")
            owner.restore_state(changed_wake(owner.snapshot_state()))
        return value

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "joint_position_summary", property(late))
    result = run(output, learner=owner, monitor_joint_positions=True)
    report = result.report
    assert injected == [True] and report["complete"] is False
    assert result.corrections is None and report["joint_position_exact"] is None
    assert len(environment.commands) == report["work"]["completed_steps"] == 2
    assert report["prediction_returns"] == 2
    assert report["joint_position_work"]["samples"] == 3
    if bad == "wake":
        assert report["original_owner_exact"] is False


def test_should_refuse_existing_output_before_fork_or_native_and_keep_original_bytes(
    boundary, sources, monkeypatch
):
    _, creations, output = boundary
    output.write_bytes(b"preserved original")
    owner = sources[0]["pc"].fork()
    monkeypatch.setattr(owner, "fork", lambda: pytest.fail("existing output forked model"))
    with pytest.raises(FileExistsError):
        smoke.run_frozen_rollout(output, learner=owner)
    assert output.read_bytes() == b"preserved original" and creations == []


def test_should_keep_default_zero_policy_schema_free_of_pc_specific_metadata(boundary):
    _, _, output = boundary
    report = run(output).report
    assert report["complete"] is True and report["policy"] == "constant_zero_normalized_4axis"
    assert "continuous_prediction_math" not in report
    assert "continuous_wake_numeric_bytes_per_owner" not in report
    assert "source_wake_updates" not in report


def test_should_refuse_wake_change_from_raw_observation_getter_before_attempting_prediction(
    boundary, sources, monkeypatch
):
    environment, _, output = boundary
    owner = sources[0]["active-cpc"].fork()
    original = native.ManiSkillPickCubeEpisode.__dict__["observation"].fget
    injected: list[bool] = []

    def observe(port):
        value = original(port)
        if not injected:
            owner.restore_state(changed_wake(owner.snapshot_state()))
            injected.append(True)
        return value

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "observation", property(observe))
    report = run(output, learner=owner).report
    assert injected == [True] and report["complete"] is False
    assert report["prediction_attempts"] == report["prediction_returns"] == 0
    assert environment.commands == [] and report["original_owner_exact"] is False


def test_should_observe_private_wake_again_after_clean_source_snapshot_callback(
    boundary, sources, monkeypatch
):
    environment, _, output = boundary
    owner = sources[0]["active-cpc"].fork()
    source_snapshot, original_predict = owner.snapshot_state, pc.NumpyContinuousPCLearner.predict
    children: list[pc.NumpyContinuousPCLearner] = []
    injected: list[bool] = []

    def predict(child, features):
        result = original_predict(child, features)
        children[:] = [child]
        return result

    def snapshot():
        result = source_snapshot()
        if children and not injected:
            child = children[0]
            child.restore_state(changed_wake(child.snapshot_state()))
            injected.append(True)
        return result

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", predict)
    monkeypatch.setattr(owner, "snapshot_state", snapshot)
    report = run(output, learner=owner).report
    assert injected == [True] and report["complete"] is False
    assert environment.commands == []
    assert report["prediction_attempts"] == report["prediction_returns"] == 1
    assert report["frozen_state_exact"] is not True
    assert report["original_owner_exact"] is True


def test_should_refuse_postprediction_clock_callback_wake_change_before_control(
    boundary, sources, monkeypatch
):
    environment, _, output = boundary
    owner = sources[0]["clock"].fork()
    original = smoke._check_clock
    injected: list[bool] = []

    def clock_check(clock, signature, report):
        original(clock, signature, report)
        if report["prediction_returns"] and not injected:
            owner.restore_state(changed_wake(owner.snapshot_state()))
            injected.append(True)

    monkeypatch.setattr(smoke, "_check_clock", clock_check)
    report = run(output, learner=owner, observation_clock=EpisodeClock(50, True)).report
    assert injected == [True] and report["complete"] is False
    assert report["prediction_returns"] == report["validated_prediction_rows"] == 1
    assert report["clock_configuration_exact"] is True
    assert environment.commands == [] and report["original_owner_exact"] is False


def test_should_refuse_valid_visual_recipe_getter_mutating_wake_after_prediction(
    boundary, sources, monkeypatch
):
    _, _, output = boundary
    environment = RGBEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda **_options: environment)
    owner = sources[0]["projection"].fork()
    original_predict, original_recipe = (
        pc.NumpyContinuousPCLearner.predict,
        native.ManiSkillPickCubeEpisode.__dict__["observation_recipe"].fget,
    )
    predicted: list[bool] = []
    injected: list[bool] = []

    def predict(child, features):
        result = original_predict(child, features)
        predicted.append(True)
        return result

    def recipe(port):
        result = original_recipe(port)
        if predicted and not injected:
            owner.restore_state(changed_wake(owner.snapshot_state()))
            injected.append(True)
        return result

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", predict)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "observation_recipe", property(recipe))
    report = run(
        output,
        learner=owner,
        visual_observation=True,
        observation_projection=public_vector_projection(),
    ).report
    assert injected == [True] and report["complete"] is False
    assert report["prediction_returns"] == 1 and environment.commands == []
    assert report["original_owner_exact"] is False


def test_should_route_existing_main_to_exact_pc_learner(boundary, sources, monkeypatch):
    _, _, output = boundary
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    assert smoke.main(learner=sources[0]["pc"].fork()) == 0
    report = json.loads(output.read_text(encoding="utf8"))
    assert report["complete"] is True and report["policy"] == "frozen_continuous_pc"
    assert report["prediction_returns"] == 2


def test_should_require_frozen_statistics_for_pc_corrective_pipeline_before_creating_output(
    boundary, sources
):
    _, _, output = boundary
    destination = output.parent / "corrections"
    with pytest.raises(ValueError, match="statistics"):
        smoke.run_corrective_adaptation(sources[0]["pc"].fork(), output_directory=destination)
    assert not destination.exists()
