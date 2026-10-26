"""Observed contact, not a closing intention, authorizes readiness transport.

Prospective per platform: five D12/H2/A4 parameter initializations (160 normal
values), zero fits; five warm1/private8 paths =45 native updates/90 rows.
The binary BP pipeline and direct MSE BP paths perform18 backwards/684 gradient
values,22 forwards/40 rows including four frozen predictions. Three PC paths
add27 updates/54 rows, at most135 energies/108 latent proposals,27 candidate
feedforwards/54 rows. Total49 normalizations/94 rows/1128 coordinates.
At most four fake scenes, five controls, four predictions and five geometry
attempts; no physics. Other cases use pure records, cached reports or forks.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core import continuous_bc as core
from src.core.continuous_pc import CONTINUOUS_PC_ENERGY_ID
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_readiness_corrections import (
    OBSERVED_GRASP_PROVENANCE,
    PROVENANCE,
    ObservedGraspCorrectionBatch,
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
    copy_readiness_correction_batch,
    readiness_correction_definition,
    readiness_correction_provenance,
)
from src.core.pick_cube_readiness_roll_in import ReadinessRollInPlan
from src.core.pick_cube_readiness_teacher import (
    DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    ObservedGraspPickCubeTeacher,
    ReadinessPickCubeTeacher,
)
from src.core.pick_cube_state_corrections import state_quaternion_correction_view
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_joint_positions import StateEnvironment


RECIPE = PickCubeTaskSpaceHistoryRecipe()
OFF_HOVER = PickCubeGeometry((-0.3, 0.0, 0.25), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2))


def raw(index):
    values = [0.0] * 42
    values[0], values[7], values[8], values[18] = index / 10, 0.02, 0.02, float(index == 0)
    values[22], values[33] = 1.0, -1.0
    values[36:42] = (0.05 + index * 0.025, -0.025, 0.0125, -0.1, 0.025, 0.05)
    return tuple(values)


def batch():
    spec = ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    actions = ((0.5, 0.0, 0.0, 0.5), (0.0, -0.25, 0.0, -0.5))
    episode = ContinuousEpisode(
        "pick-cube-smoke-19",
        "pick-cube-train-19",
        "train",
        spec,
        tuple(
            ContinuousTransition(i, raw(i), action, float(i == 1), raw(i + 1), i == 1, False)
            for i, action in enumerate(actions)
        ),
    )
    return ObservedGraspCorrectionBatch(
        19,
        episode,
        (
            ReadinessCorrectionLabel(0, raw(0), (0.5, 0.0, 0.5, -1.0), "lift"),
            ReadinessCorrectionLabel(1, raw(1), (0.25, 0.5, 0.0, 1.0), "align"),
        ),
        (True, False),
    )


def assert_state(first, second):
    assert first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    wake = getattr(first, "wake_state", None)
    if wake is not None:
        assert second.wake_state.wake_updates == wake.wake_updates
        np.testing.assert_array_equal(wake.chemical_state, second.wake_state.chemical_state)
        np.testing.assert_array_equal(wake.plasticity_state, second.wake_state.plasticity_state)
    else:
        assert getattr(second, "wake_state", None) is None


def forbidden(*args, **kwargs):
    pytest.fail("invalid or cached observed-grasp case admitted additional work")


@pytest.mark.parametrize("near_goal", [False, True])
def test_should_transport_observed_grasp_off_hover_without_a_prior_closing_schedule(near_goal):
    geometry = replace(OFF_HOVER, goal=(0.01, 0.0, 0.0)) if near_goal else OFF_HOVER
    teacher = ObservedGraspPickCubeTeacher()
    decision = teacher.decide(geometry, grasped=True, executed_gripper=1.0)
    assert decision.stage == ("settle" if near_goal else "lift")
    assert decision.action == ((0.0, 0.0, 0.0, -1.0) if near_goal else (1.0, 0.0, 1.0, -1.0))
    assert teacher.steps == 1
    if near_goal:
        assert (
            teacher.decide(replace(geometry, goal=(0.025, 0.0, 0.0)), grasped=True).stage
            == "settle"
        )
        assert (
            teacher.decide(replace(geometry, goal=(0.031, 0.0, 0.0)), grasped=True).stage == "lift"
        )


def test_should_not_promote_six_executed_closes_to_contact_and_should_recover_after_loss():
    teacher = ObservedGraspPickCubeTeacher()
    hover = PickCubeGeometry((0.0, 0.0, 0.08), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2))
    at_cube = replace(hover, tcp=(0.0, 0.0, 0.0))
    assert teacher.decide(hover, grasped=False).stage == "descend"
    assert all(
        teacher.decide(at_cube, grasped=False, executed_gripper=-1.0).stage == "grasp"
        for _ in range(8)
    )
    assert teacher.decide(OFF_HOVER, grasped=True, executed_gripper=1.0).stage == "lift"
    recovered = teacher.decide(OFF_HOVER, grasped=False, executed_gripper=-1.0)
    assert recovered.stage == "align" and recovered.action[3] == 1.0


@pytest.mark.parametrize("feedback", [None, 1, "true", np.bool_(True)])
def test_should_require_current_native_boolean_feedback_atomically(feedback):
    teacher = ObservedGraspPickCubeTeacher()
    with pytest.raises(TypeError):
        getattr(teacher, "decide")(OFF_HOVER)
    with pytest.raises(ValueError):
        teacher.decide(OFF_HOVER, grasped=feedback)
    with pytest.raises(ValueError):
        teacher.decide(OFF_HOVER, grasped=True, executed_gripper=float("nan"))
    assert teacher.steps == 0


def test_should_stop_at_the_same_fifty_decision_horizon_without_time_based_transport():
    teacher = ObservedGraspPickCubeTeacher()
    assert all(teacher.decide(OFF_HOVER, grasped=False).stage == "align" for _ in range(50))
    with pytest.raises(ValueError):
        teacher.decide(OFF_HOVER, grasped=True)
    assert teacher.steps == 50


def test_should_preserve_the_existing_self_executing_readiness_teacher_and_family():
    teacher = ReadinessPickCubeTeacher()
    hover = PickCubeGeometry((0.0, 0.0, 0.08), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2))
    assert teacher.decide(hover).stage == "descend"
    at_cube = replace(hover, tcp=(0.0, 0.0, 0.0))
    assert [teacher.decide(at_cube).stage for _ in range(7)] == ["grasp"] * 6 + ["lift"]
    observed = batch()
    legacy = ReadinessCorrectionBatch(19, observed.episode, observed.labels)
    assert readiness_correction_provenance(legacy) == PROVENANCE
    assert readiness_correction_definition(legacy) == DEFINITION
    assert "observed_grasp" not in asdict(copy_readiness_correction_batch(legacy))


@pytest.mark.parametrize("view_kind", ["copy", "canonical42", "history47", "taskspace12"])
def test_should_preserve_feedback_family_arrival_targets_and_actual_actions_through_views(
    view_kind,
):
    original = batch()
    before = asdict(original)
    if view_kind == "copy":
        model = copy_readiness_correction_batch(original)
    elif view_kind == "canonical42":
        model = state_quaternion_correction_view(original, state_quaternion_projection()).model
    else:
        recipe = PickCubeActionHistoryRecipe() if view_kind == "history47" else RECIPE
        model = action_history_correction_view(original, recipe).model
    assert type(model) is ObservedGraspCorrectionBatch and model is not original
    assert model.observed_grasp == (True, False)
    assert readiness_correction_provenance(model) == OBSERVED_GRASP_PROVENANCE
    assert readiness_correction_definition(model) == OBSERVED_GRASP_DEFINITION
    assert tuple(row.action for row in model.episode.transitions) == tuple(
        row.action for row in original.episode.transitions
    )
    assert [
        (label.stage, label.observed_at, label.labels_arrived_at) for label in model.labels
    ] == [
        ("lift", 0, 1),
        ("align", 2, 3),
    ]
    with pytest.raises(ValueError, match="arrived"):
        model.training_pairs(now=2)
    assert model.training_pairs(now=3)[1] == original.training_pairs(now=3)[1]
    if view_kind in ("history47", "taskspace12"):
        assert model.labels[1].observation[-5:] == (0.5, 0.0, 0.0, 0.5, 0.0)
        assert model.labels[1].observation[-5:-1] != model.labels[0].target
    object.__setattr__(model.labels[0], "target", (0.0, 0.0, 0.0, -1.0))
    assert asdict(original) == before


@pytest.mark.parametrize(
    "damage",
    [
        "list",
        "length",
        "integer",
        "mismatch",
        "current_bool",
        "next_fraction",
        "true_approach",
        "false_transport",
    ],
)
def test_should_refuse_inconsistent_feedback_flags_and_transport_semantics_before_copy(damage):
    value = batch()
    feedback: Any = value.observed_grasp
    labels = value.labels
    if damage == "list":
        feedback = list(feedback)
    elif damage == "length":
        feedback = feedback[:1]
    elif damage == "integer":
        feedback = (1, False)
    elif damage == "mismatch":
        feedback = (False, False)
    elif damage in ("current_bool", "next_fraction"):
        row = value.episode.transitions[0]
        field = "observation" if damage == "current_bool" else "next_observation"
        changed = list(getattr(row, field))
        changed[18] = True if damage == "current_bool" else 0.5
        object.__setattr__(row, field, tuple(changed))
    elif damage == "true_approach":
        labels = (ReadinessCorrectionLabel(0, raw(0), (0.0, 0.0, 0.0, -1.0), "grasp"), labels[1])
    else:
        labels = (labels[0], ReadinessCorrectionLabel(1, raw(1), (0.0, 0.0, 0.0, -1.0), "settle"))
    with pytest.raises(ValueError):
        ObservedGraspCorrectionBatch(19, value.episode, labels, feedback)


@pytest.fixture(scope="module")
def source():
    view = action_history_correction_view(batch(), RECIPE)
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            view.model.episode.spec,
            hidden_dim=2,
            seed=17,
            binary_gripper_axis=3,
        )
    )
    features, targets = view.model.training_pairs(now=3)
    owner.train_batch(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    assert any(np.any(item != 0.0) for item in owner.snapshot_state().velocity)
    return owner


@pytest.fixture(scope="module")
def direct_families():
    """Each family warms once, then consumes the same arrived observed labels."""
    records: dict[str, Any] = {}
    view = action_history_correction_view(batch(), RECIPE)
    features, targets = view.model.training_pairs(now=3)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    runtime_factory = demo.ActorShadowRuntime
    for family in ("bp", "pc", "neutral-cpc", "active-cpc"):
        base = bp.ContinuousBCPolicy(view.model.episode.spec, hidden_dim=2, seed=17)
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
            owner = pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=config))
        owner.train_batch(x, y)
        initial = owner.snapshot_state()
        captured: dict[str, Any] = {}

        def runtime(*args, **options):
            value = runtime_factory(*args, **options)
            captured["runtime"] = value
            return value

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(demo, "ActorShadowRuntime", runtime)
            patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
            result = demo.run_continuous_correction_update(
                owner,
                batch(),
                now=3,
                action_history_recipe=RECIPE,
            )
        assert result.report["complete"] is True, result.report
        records[family] = (owner, initial, result, captured["runtime"])
    return records


@pytest.mark.parametrize("family", ["bp", "pc", "neutral-cpc", "active-cpc"])
def test_should_preserve_observed_family_in_mse_pc_and_circadian_private_updates(
    direct_families, family
):
    owner, initial, result, runtime = direct_families[family]
    report = result.report
    assert report["provenance"] == OBSERVED_GRASP_PROVENANCE
    assert report["teacher_definition"] == OBSERVED_GRASP_DEFINITION
    assert report["observed_grasp"] == [True, False]
    assert report["source_updates"] == 1 and report["source_training_rows"] == 2
    assert report["input_projection"]["model_spec"]["observation_dim"] == 12
    assert report["work"]["optimizer_updates"] == 8 and report["work"]["training_rows"] == 16
    assert report["work"]["unique_correction_rows"] == 2
    assert report["work"]["repeated_training_rows"] == 14
    assert report["work"]["fit_calls"] == report["work"]["reference_replay_rows"] == 0
    assert "binary_gripper_axis" not in report and "training_objective" not in report
    receipts = report["applied_updates"]
    assert len(receipts) == 8
    definition = core.OBJECTIVE if family == "bp" else CONTINUOUS_PC_ENERGY_ID
    for index, receipt in enumerate(receipts):
        assert receipt["diagnostic"]["definition"] == definition
        assert receipt["episode_id"] == "pick-cube-smoke-19"
        assert receipt["sample_id"] == f"corrective-batch-{index}"
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["update_number"] == index + 1
    state = result.candidate.snapshot_state()
    assert state.policy == initial.policy
    assert (state.completed_updates, state.training_examples) == (9, 18)
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    assert_state(state, runtime.candidate_snapshot().state)
    if family == "bp":
        assert report["protocol_id"] == "continuous_correction_8_v1"
        assert "native_updates" not in report
    else:
        assert report["protocol_id"] == "continuous_pc_correction_8_v1"
        assert len(report["native_updates"]) == 8
        for receipt, returned in zip(receipts, report["native_updates"], strict=True):
            assert (
                returned["training_rows"] == 2 and returned["diagnostic"] == receipt["diagnostic"]
            )
        for key in report["native_updates"][0]["work"]:
            total = sum(row["work"][key] for row in report["native_updates"])
            assert report["work"]["native_update_counts"][key] == total
            assert report["work"]["native_event_row_weights"][key] == total * 2
        if family == "pc":
            assert state.wake_state is None
        else:
            assert initial.wake_state.wake_updates == 1 and state.wake_state.wake_updates == 9
    if family == "neutral-cpc":
        plain = direct_families["pc"][2]
        plain_state = plain.candidate.snapshot_state()
        assert replace(state.policy, circadian=None) == plain_state.policy
        for a, b in zip(
            state.parameters + state.velocity,
            plain_state.parameters + plain_state.velocity,
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(plain.report["native_updates"], report["native_updates"], strict=True):
            assert a["diagnostic"] == b["diagnostic"]
            assert a["task_loss"] == b["task_loss"] and a["inference"] == b["inference"]
            assert a["work"] == {key: b["work"][key] for key in a["work"]}


class TaskspaceEnvironment(StateEnvironment):
    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)

    def step(self, command):
        returned = super().step(command)
        return (np.asarray([raw(len(self.commands))], dtype=np.float64), *returned[1:])


@pytest.fixture(scope="module")
def pipeline(source, tmp_path_factory):
    owner, initial = source.fork(), source.snapshot_state()
    environments, outcomes, updates, calls = [], [], [], []
    rollout, update = smoke.run_frozen_rollout, smoke.run_continuous_correction_update

    def factory():
        value = TaskspaceEnvironment()
        environments.append(value)
        return value

    def frozen(*args, **options):
        calls.append(options)
        result = rollout(*args, **options)
        outcomes.append(result)
        return result

    def updating(*args, **options):
        result = update(*args, **options)
        updates.append(result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(queries, "_geometry", lambda _: OFF_HOVER)
        patch.setattr(smoke, "version", lambda _: "fake-observed-grasp")
        patch.setattr(smoke, "run_frozen_rollout", frozen)
        patch.setattr(smoke, "run_continuous_correction_update", updating)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        summary = smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path_factory.mktemp("observed-grasp") / "case",
            visual_observation=False,
            readiness_corrections=True,
            observed_grasp_corrections=True,
            acquisition_only=True,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
        )
    assert summary["complete"] is True, summary
    return owner, initial, summary, environments, outcomes, updates[0], calls


def test_should_collect_current_contact_update_eight_times_and_return_only_the_frozen_student(
    pipeline,
):
    owner, initial, summary, environments, outcomes, update, calls = pipeline
    assert [row["train_seed"] for row in calls] == [17, 17]
    assert calls[0]["observed_grasp_corrections"] is True
    assert "observed_grasp_corrections" not in calls[1] and "corrective_queries" not in calls[1]
    assert outcomes[1].corrections is None
    actual = outcomes[0].corrections
    assert type(actual) is ObservedGraspCorrectionBatch and actual.observed_grasp == (True, False)
    assert [label.stage for label in actual.labels] == ["lift", "align"]
    assert [label.target[3] for label in actual.labels] == [-1.0, 1.0]
    assert all(abs(row.action[3]) < 1.0 for row in actual.episode.transitions)
    assert len(environments) == 2 and all(
        len(env.commands) == 2 and env.close_calls == 1 for env in environments
    )
    assert summary["observed_grasp_corrections"] is True
    report = update.report
    assert report["provenance"] == OBSERVED_GRASP_PROVENANCE
    assert report["teacher_definition"] == OBSERVED_GRASP_DEFINITION
    assert report["observed_grasp"] == [True, False]
    assert report["training_objective"] == core.MIXED_OBJECTIVE
    assert report["work"]["optimizer_updates"] == 8 and report["work"]["training_rows"] == 16
    assert report["work"]["binary_gripper_loss_rows"] == 16 and report["work"]["fit_calls"] == 0
    assert report["input_projection"]["model_spec"]["observation_dim"] == 12
    assert len(report["applied_updates"]) == 8
    assert all(
        row["diagnostic"]["definition"] == core.MIXED_OBJECTIVE for row in report["applied_updates"]
    )
    state = update.candidate.snapshot_state()
    assert (state.completed_updates, state.training_examples) == (9, 18)
    assert state.policy == initial.policy
    assert_state(initial, owner.snapshot_state())
    assert all(
        outcome.report["frozen_state_exact"] is True
        and outcome.report["action_history_exact"] is True
        for outcome in outcomes
    )


@pytest.mark.parametrize("fault", ["no_readiness", "visual", "roll_in"])
def test_should_refuse_unsupported_observed_modes_before_fork_or_scene(
    source, monkeypatch, tmp_path, fault
):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    options: dict[str, Any] = dict(readiness_corrections=True, visual_observation=False)
    if fault == "no_readiness":
        options["readiness_corrections"] = False
    elif fault == "visual":
        options["visual_observation"] = True
    else:
        options["readiness_roll_in"] = ReadinessRollInPlan(1)
    with pytest.raises(ValueError):
        smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path / "refused",
            observed_grasp_corrections=True,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            **options,
        )
    assert not (tmp_path / "refused").exists()


def test_should_refuse_late_public_getter_flag_changes_before_geometry_and_remain_poisoned(
    monkeypatch,
):
    environment = TaskspaceEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(queries, "_geometry", forbidden)
    port = native.ManiSkillPickCubeEpisode("pick-cube-smoke-19", "pick-cube-train-19", seed=19)
    query = queries.PickCubeCorrectionQuery(
        19, readiness_corrections=True, observed_grasp_corrections=True
    )
    original_spec: Any = native.ManiSkillPickCubeEpisode.spec

    def changed(current):
        result = original_spec.fget(current)
        values = list(current._observation)
        values[18] = 0.0
        current._observation = tuple(values)
        return result

    try:
        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "spec", property(changed))
        with pytest.raises(ValueError):
            query.query(port, 0, port.observation)
        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "spec", original_spec)
        port._observation = raw(0)
        with pytest.raises(ValueError):
            query.query(port, 0, port.observation)
    finally:
        port.close()
    report = query.report()
    assert report["geometry_attempts"] == report["teacher_actions"] == 0
    assert report["observed_grasp"] == [] and report["labels"] == []
    assert environment.commands == [] and environment.close_calls == 1


def test_should_retain_paid_feedback_and_control_when_geometry_changes_the_current_flag(
    monkeypatch,
):
    environment = TaskspaceEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(queries, "_geometry", lambda _: OFF_HOVER)
    port = native.ManiSkillPickCubeEpisode("pick-cube-smoke-19", "pick-cube-train-19", seed=19)
    query = queries.PickCubeCorrectionQuery(
        19, readiness_corrections=True, observed_grasp_corrections=True
    )
    try:
        query.query(port, 0, port.observation)
        port.step((0.5, 0.0, 0.0, 0.5))
        expected = port.observation

        def changed(current):
            values = list(current._observation)
            values[18] = 1.0
            current._observation = tuple(values)
            return OFF_HOVER

        monkeypatch.setattr(queries, "_geometry", changed)
        with pytest.raises(ValueError):
            query.query(port, 1, expected)
        port._observation = expected
        with pytest.raises(ValueError):
            query.query(port, 1, expected)
        with pytest.raises(ValueError):
            query.finish(port.episode())
    finally:
        port.close()
    report = query.report()
    assert report["geometry_attempts"] == report["geometry_returns"] == 2
    assert report["teacher_actions"] == report["teacher_decision_steps"] == 1
    assert report["observed_grasp"] == [True] and report["labels"][0]["stage"] == "lift"
    assert report["batch"] is None and len(environment.commands) == 1


@pytest.mark.parametrize("fault", ["legacy_batch", "legacy_update"])
def test_should_refuse_selected_family_substitution_before_further_private_or_native_work(
    pipeline, monkeypatch, tmp_path, fault
):
    source, initial, _, _, outcomes, update, _ = pipeline
    collection = outcomes[0]
    if fault == "legacy_batch":
        value = collection.corrections
        legacy = ReadinessCorrectionBatch(value.train_seed, value.episode, value.labels)
        collection = smoke.FrozenRolloutResult(deepcopy(collection.report), legacy)
    calls = []

    def frozen(*args, **options):
        calls.append(options)
        assert len(calls) == 1
        return collection

    def updating(*args, **options):
        if fault == "legacy_batch":
            forbidden()
        report = deepcopy(update.report)
        report.update(provenance=PROVENANCE, teacher_definition=DEFINITION)
        return replace(update, report=report, candidate=update.candidate.fork())

    monkeypatch.setattr(smoke, "run_frozen_rollout", frozen)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", updating)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    result = smoke.run_corrective_adaptation(
        source.fork(),
        output_directory=tmp_path / fault,
        visual_observation=False,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        acquisition_only=True,
        action_history_recipe=RECIPE,
        binary_gripper_axis=3,
    )
    assert result["complete"] is False and len(calls) == 1
    if fault == "legacy_update":
        assert result["update"]["work"]["optimizer_updates"] == 8
    assert_state(initial, source.snapshot_state())
