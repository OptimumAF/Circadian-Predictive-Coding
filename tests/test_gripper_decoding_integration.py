"""Four trained tiny MSE families, explicit decoding, and frozen fake serving.

Prospective:4 true initializers128normal draws/4 source updates4rows/0fits;
PC at most15 energy entries12 latent proposals. Six fake scenes11 predictions
and returned controls; no simulator/physics or corrective/private updates.
"""

from dataclasses import asdict, replace
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core import continuous_pc as energy
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_episode_clock import EpisodeClock
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.pick_cube_action_decoding import PickCubeBinaryGripperRecipe
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_taskspace_memory_control import TaskspaceEnvironment, raw


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
HISTORY = PickCubeTaskSpaceHistoryRecipe()


def assert_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    if type(first) is pc.ContinuousPCSnapshot:
        if first.wake_state is None:
            assert second.wake_state is None
        else:
            assert second.wake_state is not None
            assert first.wake_state.wake_updates == second.wake_state.wake_updates
            np.testing.assert_array_equal(
                first.wake_state.chemical_state, second.wake_state.chemical_state
            )
            np.testing.assert_array_equal(
                first.wake_state.plasticity_state, second.wake_state.plasticity_state
            )


def forbidden(*args, **kwargs):
    pytest.fail("frozen decoding requested a fit, update, latent solve or native creation")


@pytest.fixture(scope="module")
def sources():
    declared = ContinuousActionSpec(12, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    features = np.array([[0.0, 0.5, -0.25, 0.125, -1.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = np.array([[0.25, -0.5, 0.125, -1.0]])
    records: dict[str, Any] = {}
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(declared, hidden_dim=2, seed=17)
        owner: Any
        if family == "bp":
            owner = bp.NumpyContinuousBCLearner(base)
        else:
            config = (
                None
                if family == "pc"
                else CoupledCircadianConfig.matched_pc_control()
                if family == "neutral-cpc"
                else CoupledCircadianConfig()
            )
            owner = pc.NumpyContinuousPCLearner(
                pc.ContinuousPCPolicy(
                    base,
                    inference_steps=4,
                    inference_policy=CoupledInferencePolicy(min_steps=1),
                    circadian=config,
                )
            )
        owner.train_batch(features, targets)
        assert (
            owner.snapshot_state().completed_updates
            == owner.snapshot_state().training_examples
            == 1
        )
        records[family] = owner
    return records


@pytest.fixture(scope="module")
def rollouts(sources, tmp_path_factory):
    records = {}
    observation_property = vars(native.ManiSkillPickCubeEpisode)["observation"]
    assert type(observation_property) is property
    assert observation_property.fget is not None
    get_observation = observation_property.fget
    for family, selected, late in [
        *((family, True, False) for family in FAMILIES),
        ("bp", False, False),
        ("bp", True, True),
    ]:
        source = sources[family].fork()
        before = source.snapshot_state()
        scene = TaskspaceEnvironment()
        captured: dict[str, Any] = dict(predictions=[], inputs=[], inference=None, mutations=0)
        cls = bp.NumpyContinuousBCLearner if family == "bp" else pc.NumpyContinuousPCLearner
        predict = cls.predict
        recipe = PickCubeBinaryGripperRecipe()

        def predicting(owner, features):
            returned = predict(owner, features)
            captured["inference"] = owner
            captured["inputs"].append(features.copy())
            captured["predictions"].append(returned.copy())
            return returned

        def observation(port):
            value = get_observation(port)
            if late and port._records and captured["mutations"] == 0:
                object.__setattr__(recipe, "definition", "late request mutation")
                captured["mutations"] += 1
            return value

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native, "_make_environment", lambda: scene)
            patch.setattr(native.ManiSkillPickCubeEpisode, "observation", property(observation))
            patch.setattr(smoke, "version", lambda _: "fake-decoded-gripper")
            patch.setattr(cls, "predict", predicting)
            patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
            patch.setattr(pc.NumpyContinuousPCLearner, "train_batch", forbidden)
            patch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", forbidden)
            patch.setattr(energy, "feedforward_continuous_latents", forbidden)
            patch.setattr(energy, "evaluate_continuous_pc", forbidden)
            patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
            options: dict[str, Any] = {"action_decoding_recipe": recipe} if selected else {}
            outcome = smoke.run_frozen_rollout(
                tmp_path_factory.mktemp("decoded-gripper") / "trace.json",
                learner=source,
                action_history_recipe=HISTORY,
                **options,
            )
        assert_state(before, source.snapshot_state())
        records[family, selected, late] = outcome, captured, scene, before
    return records


@pytest.mark.parametrize("family", FAMILIES)
def test_should_freeze_all_trained_state_and_record_proposal_decision_and_actual_command_separately(
    rollouts, family
):
    outcome, captured, scene, before = rollouts[family, True, False]
    report = outcome.report
    assert report["complete"] is True, report
    assert report["action_decoding_exact"] is True
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert report["source_updates"] == report["source_training_rows"] == 1
    assert report["optimizer_updates"] == 0 and outcome.corrections is None
    assert_state(before, captured["inference"].snapshot_state())
    assert len(scene.commands) == len(captured["predictions"]) == 2
    controls = report["action_decoding"]["controls"]
    for i, (prediction, control, command) in enumerate(
        zip(captured["predictions"], controls, scene.commands, strict=True)
    ):
        proposed = tuple(float(value) for value in prediction[0])
        selected = (*proposed[:3], -1.0 if proposed[3] < 0.0 else 1.0)
        assert control["step"] == i and control["control_attempted"] is True
        assert tuple(control["prediction"]) == proposed
        assert tuple(control["selected_command"]) == selected
        np.testing.assert_array_equal(command[0], np.asarray(selected, dtype=np.float32))
        assert tuple(control["returned_action"]) == tuple(float(value) for value in command[0])
        assert -1.0 < proposed[3] < 1.0
    first_native = tuple(float(value) for value in scene.commands[0][0])
    assert tuple(captured["inputs"][0][0][-5:]) == (0.0,) * 5
    assert tuple(captured["inputs"][1][0][-5:]) == (*first_native, float(first_native[3] < 0.0) / 6)
    assert first_native[3] != float(captured["predictions"][0][0, 3])
    assert report["episode"]["spec"]["observation_dim"] == 42
    assert tuple(report["episode"]["transitions"][0]["observation"]) == raw(0)
    assert [tuple(row["action"]) for row in report["episode"]["transitions"]] == [
        tuple(command[0]) for command in scene.commands
    ]
    assert report["action_decoding"]["work"] == dict(
        decode_attempts=2,
        decode_returns=2,
        prediction_values=8,
        copied_xyz_values=6,
        gripper_decisions=2,
        control_attempts=2,
        control_returns=2,
        validated_control_returns=2,
        returned_action_values=8,
    )


def test_should_preserve_default_soft_gripper_commands_and_omit_decoding_fields(rollouts):
    outcome, captured, scene, before = rollouts["bp", False, False]
    assert outcome.report["complete"] is True
    assert "action_decoding" not in outcome.report and "action_decoding_exact" not in outcome.report
    for predicted, command in zip(captured["predictions"], scene.commands, strict=True):
        np.testing.assert_array_equal(command, predicted.astype(np.float32))
    first = tuple(float(value) for value in scene.commands[0][0])
    assert tuple(captured["inputs"][1][0][-5:]) == (*first, float(first[3] < 0.0) / 6)
    assert_state(before, captured["inference"].snapshot_state())


def test_should_retain_the_first_paid_control_when_a_late_getter_changes_the_original_recipe(
    rollouts,
):
    outcome, captured, scene, _ = rollouts["bp", True, True]
    report = outcome.report
    assert report["complete"] is False and report["error"] is not None
    assert captured["mutations"] == 1 and len(captured["predictions"]) == len(scene.commands) == 1
    assert report["action_decoding_exact"] is None
    assert report["action_decoding"]["recipe"] == asdict(PickCubeBinaryGripperRecipe())
    assert report["action_decoding"]["work"]["control_returns"] == 1
    assert report["work"]["completed_steps"] == report["work"]["step_attempts"] == 1
    assert scene.close_calls == 1 and outcome.corrections is None


@pytest.mark.parametrize(
    "fault", ["history", "zero", "queries", "roll_in", "clock", "visual", "mixed", "none"]
)
def test_should_refuse_unsupported_decoding_routes_before_environment_creation(
    sources, monkeypatch, tmp_path, fault
):
    owner = sources["bp"].fork()
    options: dict[str, Any] = dict(
        learner=owner,
        action_history_recipe=HISTORY,
        action_decoding_recipe=PickCubeBinaryGripperRecipe(),
    )
    if fault == "history":
        options.pop("action_history_recipe")
    elif fault == "zero":
        options["action_history_recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    elif fault == "queries":
        options["corrective_queries"] = True
    elif fault == "roll_in":
        from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan

        options["readiness_roll_in"] = ReadinessRollInPlan(1)
    elif fault == "clock":
        options["observation_clock"] = EpisodeClock(50, True)
    elif fault == "visual":
        options["visual_observation"] = True
    elif fault == "mixed":
        snapshot = owner.snapshot_state()
        policy = replace(owner.policy, binary_gripper_axis=3)
        owner._policy = policy
        owner.restore_state(replace(snapshot, policy=policy))
    else:
        options["learner"] = None
    monkeypatch.setattr(native, "_make_environment", forbidden)
    result = smoke.run_frozen_rollout(tmp_path / "refused.json", **options)
    assert result.report["complete"] is False and result.report["error"] is not None
    assert result.report["creation_attempts"] == result.report["prediction_attempts"] == 0


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_the_execution_recipe_only_when_selected_without_repeating_native_work(
    sources, rollouts, monkeypatch, tmp_path, selected
):
    cached, _, _, before = rollouts["bp", selected, False]
    forwarded: dict[str, Any] = {}
    recipe = PickCubeBinaryGripperRecipe()
    native = deepcopy(cached.report)
    # This is a forwarding stub, not a new joint-telemetry test or native result.
    # The helper requires the existing joint-proof flag; cached serving used none.
    native["joint_position_exact"] = True

    def frozen(output, **options):
        forwarded.update(options)
        return smoke.FrozenRolloutResult(native, None)

    monkeypatch.setattr(entry, "run_frozen_rollout", frozen)
    outer: dict[str, Any] = {"action_history_recipe": asdict(HISTORY)}
    options: dict[str, Any] = {"action_decoding_recipe": recipe} if selected else {}
    entry._frozen_return(
        outer,
        sources["bp"],
        before,
        tmp_path / "unused.json",
        False,
        action_history_recipe=HISTORY,
        **options,
    )
    assert outer["candidate_unchanged"] is True
    assert forwarded["monitor_joint_positions"] is True
    if selected:
        assert forwarded["action_decoding_recipe"] is recipe
        assert outer["action_decoding_recipe"] == asdict(recipe)
        assert outer["rollout"]["action_decoding_exact"] is True
    else:
        assert "action_decoding_recipe" not in forwarded
        assert "action_decoding_recipe" not in outer
