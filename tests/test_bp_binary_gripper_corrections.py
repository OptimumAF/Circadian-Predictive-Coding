"""Explicit existing BP mixed objective in arrived private corrections.

One D12/H2 BP initializer/32 normal values, zero fits; warm1 + replay12 +
pipeline8 + paid-objective-failure1 + final-getter-failure12 =34 SGD/60 rows.
34 gradient calls/1292 values;38 forwards/64 rows include four fake predictions.
Two scenes/four controls/two readiness queries, no joint queries or physics.
All other cases use cached state, RNG-free forks or before-work type probes.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.app.continuous_stream_rehearsal import ContinuousReplayPlan
from src.core import continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_action_history import (
    ActionHistoryState,
    action_history_observation,
    action_history_training_episode,
    advance_action_history,
)
from src.core.pick_cube_action_history_corrections import action_history_correction_view
from src.core.pick_cube_readiness_corrections import (
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_joint_positions import StateEnvironment


RECIPE = PickCubeTaskSpaceHistoryRecipe()


def raw(index):
    values = [0.0] * 42
    values[0], values[7], values[8], values[18] = index / 10, 0.02, 0.02, float(index > 0)
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
    return ReadinessCorrectionBatch(
        19,
        episode,
        (
            ReadinessCorrectionLabel(0, raw(0), (0.0, 0.0, 0.0, -1.0), "grasp"),
            ReadinessCorrectionLabel(1, raw(1), (0.25, 0.5, 0.0, 1.0), "align"),
        ),
    )


def reference(gripper=-1.0):
    spec = batch().episode.spec
    episode = ContinuousEpisode(
        "reference-A",
        "pick-cube",
        "train",
        spec,
        (ContinuousTransition(0, raw(0), (0.25, -0.5, 0.0, gripper), 1.0, raw(1), True, False),),
    )
    return action_history_training_episode(episode, RECIPE)


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


def changed_policy(owner, **options):
    state = owner.snapshot_state()
    policy = replace(state.policy, **options)
    owner._policy = policy
    owner.restore_state(replace(state, policy=policy))
    return owner


def forbidden(*args, **kwargs):
    pytest.fail("invalid or cached mixed correction admitted additional work")


@pytest.fixture(scope="module")
def source():
    view = action_history_correction_view(batch(), RECIPE)
    owner = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(view.model.episode.spec, hidden_dim=2, seed=17, binary_gripper_axis=3)
    )
    features, targets = view.model.training_pairs(now=3)
    owner.train_batch(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    assert any(np.any(value != 0.0) for value in owner.snapshot_state().velocity)
    return owner


@pytest.fixture(scope="module")
def replayed(source):
    owner = source.fork()
    captured: dict[str, Any] = {}
    initial = owner.snapshot_state()
    factory, train = demo.ActorShadowRuntime, bp.NumpyContinuousBCLearner.train_batch
    calls = []

    def runtime(*args, **options):
        value = factory(*args, **options)
        captured.update(runtime=value, budget=options["budget"])
        return value

    def training(model, features, targets):
        calls.append((features.copy(), targets.copy()))
        return train(model, features, targets)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", training)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        result = demo.run_continuous_correction_update(
            owner,
            batch(),
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            replay_plan=ContinuousReplayPlan(2, 4),
            reference_episodes=(reference(),),
        )
    assert result.report["complete"] is True, result.report
    return owner, initial, result, captured, calls


def test_should_continue_full_mixed_policy_momentum_and_counters_with_separate_replay_charge(
    replayed,
):
    owner, initial, result, captured, _ = replayed
    assert result.candidate is not None
    state = result.candidate.snapshot_state()
    assert state.policy == initial.policy and state.policy.binary_gripper_axis == 3
    assert state.policy.observation_standardization is None
    assert state.completed_updates == 13 and state.training_examples == 22
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)
    assert_state(state, captured["runtime"].candidate_snapshot().state)
    report = result.report
    assert (
        report["binary_gripper_axis"] == 3 and report["training_objective"] == core.MIXED_OBJECTIVE
    )
    assert report["work"]["optimizer_updates"] == 12 and report["work"]["training_rows"] == 20
    assert report["work"]["binary_gripper_loss_rows"] == 20 and report["work"]["fit_calls"] == 0
    assert report["work"]["reference_replay_rows"] == report["replay"]["charged_exposures"] == 4
    assert (
        captured["budget"].updates_completed == 8
        and captured["budget"].replay_examples_completed == 0
    )
    assert len(report["applied_updates"]) == 8 and len(report["applied_consolidations"]) == 4
    assert all(
        item["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
        for item in report["applied_updates"] + report["applied_consolidations"]
    )


def test_should_validate_labels_not_soft_student_commands_and_keep_actual_causal_inputs(replayed):
    _, _, result, _, calls = replayed
    view = action_history_correction_view(batch(), RECIPE)
    wake_x, wake_y = view.model.training_pairs(now=3)
    bank_x = tuple(row.observation for row in reference().transitions)
    bank_y = tuple(row.action for row in reference().transitions)
    assert len(calls) == 12
    for index, (features, targets) in enumerate(calls):
        x, y = (bank_x, bank_y) if index % 3 == 2 else (wake_x, wake_y)
        np.testing.assert_array_equal(features, x)
        np.testing.assert_array_equal(targets, y)
    assert wake_x[1][7:] == (0.5, 0.0, 0.0, 0.5, 0.0)
    assert wake_y[0][3] == -1.0 and batch().episode.transitions[0].action[3] == 0.5
    assert [len(x) for x, _ in calls] == [2, 2, 1] * 4
    assert all(
        row["observed_at"] == 2 and row["arrived_at"] == row["applied_at"] == 3
        for row in result.report["applied_updates"]
    )


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

    def geometry(port):
        return PickCubeGeometry(
            (0.0, 0.0, 0.08) if not port._records else (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.2),
        )

    def frozen(*args, **options):
        calls.append(options)
        returned = rollout(*args, **options)
        outcomes.append(returned)
        return returned

    def updating(*args, **options):
        assert options["binary_gripper_axis"] == 3
        returned = update(*args, **options)
        updates.append(returned)
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(queries, "_geometry", geometry)
        patch.setattr(smoke, "version", lambda _: "fake-binary-correction")
        patch.setattr(smoke, "run_frozen_rollout", frozen)
        patch.setattr(smoke, "run_continuous_correction_update", updating)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        summary = smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path_factory.mktemp("mixed-corrections") / "case",
            visual_observation=False,
            readiness_corrections=True,
            acquisition_only=True,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
        )
    assert summary["complete"] is True, summary
    assert_state(initial, owner.snapshot_state())
    return owner, initial, summary, environments, outcomes, updates[0], calls


def test_should_collect_raw_readiness_update_privately_and_return_pure_soft_student(pipeline):
    owner, initial, summary, environments, outcomes, result, calls = pipeline
    assert [item["train_seed"] for item in calls] == [17, 17]
    assert all("binary_gripper_axis" not in item for item in calls)
    assert calls[0]["corrective_queries"] is True and "corrective_queries" not in calls[1]
    assert outcomes[0].corrections is not None and outcomes[1].corrections is None
    raw_batch = outcomes[0].corrections
    assert (
        raw_batch.episode.spec.observation_dim == 42
        and result.candidate.policy.spec.observation_dim == 12
    )
    assert tuple(label.target[3] for label in raw_batch.labels) == (1.0, -1.0)
    assert all(abs(row.action[3]) < 1.0 for row in raw_batch.episode.transitions)
    assert (
        result.candidate.snapshot_state().completed_updates == 9
        and result.candidate.snapshot_state().training_examples == 18
    )
    assert summary["update"]["work"]["binary_gripper_loss_rows"] == 16
    assert summary["update"]["work"]["fit_calls"] == 0
    assert len(environments) == 2 and all(
        len(value.commands) == 2 and value.close_calls == 1 for value in environments
    )
    assert all(
        outcome.report["action_history_exact"] is True
        and outcome.report["frozen_state_exact"] is True
        for outcome in outcomes
    )
    assert outcomes[1].report["learner_policy"]["binary_gripper_axis"] == 3
    assert outcomes[1].report["source_updates"] == 9
    first = tuple(raw_batch.episode.transitions[0].action)
    state = advance_action_history(ActionHistoryState(), first)
    model = action_history_correction_view(raw_batch, RECIPE).model
    assert model.episode.transitions[1].observation == action_history_observation(
        raw_batch.episode.transitions[1].observation, state, RECIPE
    )
    assert_state(initial, owner.snapshot_state())


@pytest.mark.parametrize("axis", [True, 3.0, "3", 1])
def test_should_refuse_ambiguous_or_wrong_axis_before_private_fork(source, monkeypatch, axis):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch(), now=3, action_history_recipe=RECIPE, binary_gripper_axis=axis
        )


@pytest.mark.parametrize("kind", ["pc", "nearest", "none", "foreign"])
def test_should_refuse_non_bp_owner_before_any_model_callback(kind):
    source = None if kind == "none" else object()
    if kind in ("pc", "nearest"):
        cls = (
            demo.NumpyContinuousPCLearner if kind == "pc" else demo.NumpyNearestContinuousBCLearner
        )
        source = object.__new__(cls)
        setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(source, batch(), now=3, binary_gripper_axis=3)


@pytest.mark.parametrize("selected", [False, True])
def test_should_bind_owned_policy_to_omitted_or_requested_likelihood_before_work(
    source, monkeypatch, selected
):
    owner = (
        source.fork() if not selected else changed_policy(source.fork(), binary_gripper_axis=None)
    )
    monkeypatch.setattr(owner, "fork", forbidden)
    options: dict[str, Any] = dict(binary_gripper_axis=3) if selected else {}
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch(), now=3, action_history_recipe=RECIPE, **options
        )


@pytest.mark.parametrize("which", ["correction", "replay"])
def test_should_refuse_nonbinary_arrived_labels_or_teacher_reference_targets_before_fork(
    source, monkeypatch, which
):
    owner, corrections = source.fork(), batch()
    options: dict[str, Any] = {}
    if which == "correction":
        object.__setattr__(corrections.labels[0], "target", (0.0, 0.0, 0.0, -0.5))
    else:
        options.update(replay_plan=ContinuousReplayPlan(2, 4), reference_episodes=(reference(0.5),))
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(demo, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner,
            corrections,
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            **options,
        )


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_should_refuse_unreleased_reference_bank_before_private_work(source, monkeypatch, role):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner,
            batch(),
            now=3,
            action_history_recipe=RECIPE,
            binary_gripper_axis=3,
            replay_plan=ContinuousReplayPlan(2, 4),
            reference_episodes=(replace(reference(), role=role),),
        )


def test_should_refuse_unarrived_raw_batch_before_projection_or_forks(source, monkeypatch):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(demo, "action_history_correction_view", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch(), now=2, action_history_recipe=RECIPE, binary_gripper_axis=3
        )


def test_should_keep_paid_ordinary_input_mixed_receipt_when_returned_objective_is_substituted(
    source, monkeypatch
):
    owner = changed_policy(
        source.fork(),
        observation_standardization=ObservationStandardization((0.0,) * 12, (1.0,) * 12, 2),
    )
    initial = owner.snapshot_state()
    ordinary = action_history_correction_view(batch(), RECIPE).model
    factory = demo.ActorShadowRuntime
    captured = []

    def runtime(*args, **options):
        value = factory(*args, **options)
        captured.append(value)
        ready = value.train_ready

        def wrong(*args, **kwargs):
            returned = ready(*args, **kwargs)
            return tuple(
                replace(item, diagnostic=TrainingDiagnostic(core.OBJECTIVE, item.diagnostic.value))
                for item in returned
            )

        monkeypatch.setattr(value, "train_ready", wrong)
        return value

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(owner, ordinary, now=3, binary_gripper_axis=3)
    assert result.candidate is None and result.report["complete"] is False
    assert (
        result.report["progress"]["updates_completed"] == 1
        and len(result.report["applied_updates"]) == 1
    )
    assert result.report["applied_updates"][0]["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
    assert result.report["events"][0]["applied"][0]["diagnostic"]["definition"] == core.OBJECTIVE
    assert captured[0]._budget.updates_completed == 1
    assert_state(initial, owner.snapshot_state())


@pytest.mark.parametrize("selected", [False, True])
def test_should_refuse_driver_missing_selection_or_selected_mse_before_collection(
    source, monkeypatch, tmp_path, selected
):
    owner = (
        source.fork() if not selected else changed_policy(source.fork(), binary_gripper_axis=None)
    )
    monkeypatch.setattr(smoke, "run_frozen_rollout", forbidden)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", forbidden)
    options: dict[str, Any] = dict(binary_gripper_axis=3) if selected else {}
    with pytest.raises(ValueError):
        smoke.run_corrective_adaptation(
            owner,
            output_directory=tmp_path / "no-work",
            visual_observation=False,
            readiness_corrections=True,
            acquisition_only=True,
            action_history_recipe=RECIPE,
            **options,
        )
    assert not (tmp_path / "no-work").exists()


def test_should_keep_paid_update_and_stop_before_return_when_cached_candidate_changes_objective(
    pipeline, monkeypatch, tmp_path
):
    owner, _, _, _, outcomes, original, _ = pipeline
    candidate = changed_policy(original.candidate.fork(), binary_gripper_axis=None)
    report = deepcopy(original.report)
    report["policy"] = asdict(candidate.policy)
    substituted = ContinuousLearnabilityResult(report, candidate)
    calls = []

    def collection(*args, **options):
        calls.append(options)
        assert len(calls) == 1
        return outcomes[0]

    monkeypatch.setattr(smoke, "run_frozen_rollout", collection)
    monkeypatch.setattr(
        smoke, "run_continuous_correction_update", lambda *args, **options: substituted
    )
    result = smoke.run_corrective_adaptation(
        owner.fork(),
        output_directory=tmp_path / "bad-candidate",
        visual_observation=False,
        readiness_corrections=True,
        acquisition_only=True,
        action_history_recipe=RECIPE,
        binary_gripper_axis=3,
    )
    assert result["complete"] is False and len(calls) == 1
    assert result["update"] == report and result["update"]["work"]["optimizer_updates"] == 8


def test_should_refuse_final_budget_report_getter_source_mutation_after_all_twelve_paid_updates(
    source, monkeypatch
):
    owner = source.fork()
    captured: dict[str, Any] = dict(runtime=None, readout=None, armed=False, reads=0, changed=False)
    initial = owner.snapshot_state()
    factory, helper_factory, fork, timer = (
        demo.ActorShadowRuntime,
        demo.ContinuousStreamRehearsal,
        owner.fork,
        demo.monotonic,
    )
    original_report: Any = helper_factory.report

    def runtime(*args, **options):
        value = factory(*args, **options)
        captured["runtime"] = value
        return value

    def forking():
        child = fork()
        if captured["runtime"] is not None and captured["readout"] is None:
            captured["readout"] = child
        return child

    def clock():
        value = timer()
        if captured["readout"] is not None and captured["readout"]._state.completed_updates == 13:
            captured["armed"] = True
        return value

    def observed(helper):
        value = original_report.fget(helper)
        if captured["armed"]:
            captured["reads"] += 1
            # First is the final before-update guard's report; second is the
            # last report publication, after the intervening native-owner check.
            if captured["reads"] == 2:
                state = owner.snapshot_state()
                state.parameters[0][0, 0] += 0.125
                owner.restore_state(state)
                captured["changed"] = True
        return value

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(owner, "fork", forking)
    monkeypatch.setattr(demo, "monotonic", clock)
    monkeypatch.setattr(helper_factory, "report", property(observed))
    result = demo.run_continuous_correction_update(
        owner,
        batch(),
        now=3,
        action_history_recipe=RECIPE,
        binary_gripper_axis=3,
        replay_plan=ContinuousReplayPlan(2, 4),
        reference_episodes=(reference(),),
    )
    assert captured["changed"] is True and captured["reads"] >= 2
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 8
    assert (
        len(result.report["applied_updates"]) == 8
        and len(result.report["applied_consolidations"]) == 4
    )
    assert (
        result.report["replay"]["charged_exposures"] == 4
        and result.report["replay"]["committed_updates"] == 4
    )
    assert not demo._same_state(initial, owner.snapshot_state())
