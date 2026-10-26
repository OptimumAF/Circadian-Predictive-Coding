"""Canonical correction views and the existing TRAIN17 acquisition composition.

One D42/H2 BP initialization draws92 normal values, then deterministic owned
state/velocity are restored without fitting. Four eight-update runs plus one
post-paid observer failure use33SGD/66rows:33 forward/backward calls,3234 gradient
values/25872B,12672 denseMACs and33 normalizations/66rows/2772coordinates.
The two-scene fake pipeline adds
four predictions/controls, two reset/close calls and two geometry queries; no
simulator, additional initializer, fitting, probes, replay or model selection.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_corrections import PickCubeCorrectionBatch, PickCubeCorrectionLabel
from src.core.pick_cube_readiness_corrections import (
    PROVENANCE,
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
)
from src.core.pick_cube_state_corrections import state_quaternion_correction_view
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_continuous_robot_learning import assert_state
from test_state_quaternion_robot_learning import QuaternionEnvironment, observation


def batch(seed=17):
    spec = ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    value = ContinuousEpisode(
        f"pick-cube-smoke-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        spec,
        tuple(
            ContinuousTransition(
                i,
                observation(i),
                (0.0, 0.0, 0.0, -0.75),
                float(i == 1),
                observation(i + 1),
                i == 1,
                False,
            )
            for i in range(2)
        ),
    )
    labels = (
        ReadinessCorrectionLabel(
            0, value.transitions[0].observation, (0.0, 0.0, -0.8, 1.0), "descend"
        ),
        ReadinessCorrectionLabel(
            1, value.transitions[1].observation, (0.0, 0.0, 0.0, -1.0), "grasp"
        ),
    )
    return ReadinessCorrectionBatch(seed, value, labels)


def fixed_batch():
    source = batch()
    return PickCubeCorrectionBatch(
        17,
        source.episode,
        tuple(
            PickCubeCorrectionLabel(i, row.observation, (0.5, 0.0, 1.0, 1.0), "align")
            for i, row in enumerate(source.episode.transitions)
        ),
    )


def test_should_own_raw_and_canonical_readiness_pairing_without_releasing_future_labels():
    source = batch()
    before = asdict(source)
    projection = state_quaternion_projection()
    view = state_quaternion_correction_view(source, projection)
    assert type(view.source) is type(view.model) is ReadinessCorrectionBatch
    assert asdict(view.source) == before and asdict(source) == before
    assert view.projection == projection and view.projection is not projection
    assert not hasattr(view, "source_recipe")
    assert view.source.episode.transitions[0].observation[22:26] == (-0.5, 0.5, -0.5, 0.5)
    assert all(
        row.observation[22:26] == (0.5, -0.5, 0.5, -0.5) for row in view.model.episode.transitions
    )
    assert view.model.episode.transitions[-1].next_observation[32:36] == (0.0, 0.0, 1.0, 0.0)
    assert [
        (label.stage, label.target, label.observed_at, label.labels_arrived_at)
        for label in view.model.labels
    ] == [
        (label.stage, label.target, label.observed_at, label.labels_arrived_at)
        for label in source.labels
    ]
    assert [row.action for row in view.model.episode.transitions] == [
        row.action for row in source.episode.transitions
    ]
    with pytest.raises(ValueError):
        view.model.training_pairs(now=2)
    assert view.model.training_pairs(now=3)[1] == source.training_pairs(now=3)[1]
    assert state_quaternion_correction_view(view.model, projection).model == view.model
    object.__setattr__(view.model.labels[0], "observation", (99.0,) * 42)
    assert asdict(source) == before and asdict(view.source) == before
    with pytest.raises(FrozenInstanceError):
        setattr(view.work, "observation_projections", 5)


def test_should_count_both_transition_inputs_and_terminal_outputs_once_per_view():
    view = state_quaternion_correction_view(batch(), state_quaternion_projection())
    assert asdict(view.work) == dict(
        observation_projections=4,
        source_values_validated=168,
        projected_values=168,
        projected_numeric_bytes=1344,
        projected_label_observation_values=84,
        projected_label_target_values=8,
        projected_label_observation_numeric_bytes=672,
        projected_label_target_numeric_bytes=64,
        quaternion_unit_checks=8,
        quaternion_components_checked=32,
    )


@pytest.mark.parametrize(
    "fault", ["fixed_family", "role", "context", "dimension", "pairing", "unit", "recipe"]
)
def test_should_refuse_unbound_raw_evidence_or_recipe_without_role_conversion(fault):
    source: Any = batch()
    projection: Any = state_quaternion_projection()
    if fault == "fixed_family":
        source = fixed_batch()
    elif fault == "role":
        object.__setattr__(source.episode, "role", "probe_train")
    elif fault == "context":
        object.__setattr__(source.episode, "context", "pick-cube-mass-x2-train-17")
    elif fault == "dimension":
        object.__setattr__(source.episode, "spec", replace(source.episode.spec, observation_dim=41))
    elif fault == "pairing":
        object.__setattr__(source.labels[0], "observation", source.labels[1].observation)
    elif fault == "unit":
        values = list(source.episode.transitions[-1].next_observation)
        values[32:36] = (0.0,) * 4
        object.__setattr__(source.episode.transitions[-1], "next_observation", tuple(values))
    else:
        projection = asdict(projection)
    with pytest.raises(ValueError):
        state_quaternion_correction_view(source, projection)


@pytest.fixture(scope="module")
def source():
    spec = batch().episode.spec
    statistics = ObservationStandardization((0.0,) * 42, (1.0,) * 42, 2)
    model = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(spec, hidden_dim=2, seed=17, observation_standardization=statistics)
    )
    initial = model.snapshot_state()
    parameters = cast(bp.Parameters, tuple(np.zeros_like(value) for value in initial.parameters))
    parameters[3][3] = -1.0
    model.restore_state(
        replace(
            initial,
            parameters=parameters,
            velocity=cast(
                bp.Parameters, tuple(np.full_like(value, 0.125) for value in initial.velocity)
            ),
            completed_updates=4,
            training_examples=8,
        )
    )
    return model


@pytest.fixture(scope="module")
def completed(source):
    owner, raw = source.fork(), batch()
    original = owner.snapshot_state()
    trains, runtimes = [], []
    train, runtime_factory = bp.NumpyContinuousBCLearner.train_batch, demo.ActorShadowRuntime
    with pytest.MonkeyPatch.context() as patch:

        def recorded_train(model, features, targets):
            trains.append((features.copy(), targets.copy()))
            return train(model, features, targets)

        def runtime(model, **options):
            value = runtime_factory(model, **options)
            runtimes.append((value, options))
            return value

        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", recorded_train)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        result = demo.run_continuous_correction_update(
            owner, raw, now=3, observation_projection=state_quaternion_projection()
        )
    return owner, original, raw, result, trains, runtimes[0]


def test_should_update_canonical_inputs_with_arrived_teacher_targets_and_continue_full_momentum(
    completed,
):
    owner, initial, raw, result, trains, (runtime, options) = completed
    assert result.report["complete"] is True and result.candidate is not owner
    expected = state_quaternion_correction_view(
        raw, state_quaternion_projection()
    ).model.training_pairs(now=3)
    assert len(trains) == 8
    for features, targets in trains:
        assert features.tolist() == [list(value) for value in expected[0]]
        assert targets.tolist() == [list(value) for value in expected[1]]
        assert targets.tolist() != [list(row.action) for row in raw.episode.transitions]
    final = result.candidate.snapshot_state()
    assert final.completed_updates == 12 and final.training_examples == 24
    assert (
        final.policy == initial.policy
        and final.policy.observation_standardization == initial.policy.observation_standardization
    )
    assert_state(initial, owner.snapshot_state())
    assert_state(initial, runtime.actor.snapshot_state().state)
    assert_state(final, runtime.candidate_snapshot().state)
    assert all(np.all(value == 0.125) for value in initial.velocity)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(final.velocity, initial.velocity, strict=True)
    )
    assert (
        options["budget"].updates_completed == 8
        and options["budget"].replay_examples_completed == 0
    )


def test_should_preserve_raw_encoding_provenance_clocks_and_selected_update_work(completed):
    _, _, raw, result, _, _ = completed
    report = result.report
    metadata = report["input_projection"]
    assert metadata["source_encoding"] == "official_state42_raw"
    assert metadata["model_encoding"] == state_quaternion_projection().definition
    assert metadata["source_episode_id"] == raw.episode.episode_id
    assert (
        metadata["source_spec"]["observation_dim"]
        == metadata["model_spec"]["observation_dim"]
        == 42
    )
    assert "source_recipe" not in metadata
    assert report["provenance"] == PROVENANCE
    assert report["work"]["fit_calls"] == 0 and report["work"]["training_rows"] == 16
    for index, receipt in enumerate(report["applied_updates"]):
        assert receipt["episode_id"] == "pick-cube-smoke-17"
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["update_number"] == index + 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("now", [True, -1, 2, 3.0, None])
def test_should_check_raw_arrival_before_projection_forks_or_private_work(source, monkeypatch, now):
    owner = source.fork()
    monkeypatch.setattr(demo, "state_quaternion_correction_view", forbidden)
    monkeypatch.setattr(owner, "fork", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch(), now=now, observation_projection=state_quaternion_projection()
        )


def forbidden(*args, **kwargs):
    pytest.fail("invalid canonical correction request admitted more work")


@pytest.mark.parametrize("callback", ["view", "policy"])
def test_should_refuse_source_changed_by_projection_callback_before_fork(
    source, monkeypatch, callback
):
    owner = source.fork()
    original = demo.state_quaternion_correction_view

    def changed(raw, projection):
        view = original(raw, projection)
        state = owner.snapshot_state()
        owner._state = replace(state, completed_updates=5, training_examples=10)
        return view

    if callback == "view":
        monkeypatch.setattr(demo, "state_quaternion_correction_view", changed)
    else:
        original_property: Any = bp.NumpyContinuousBCLearner.policy
        assert original_property.fget is not None

        def policy(model):
            value = original_property.fget(model)
            if model is owner:
                state = model._state
                model._state = replace(state, completed_updates=5, training_examples=10)
            return value

        monkeypatch.setattr(bp.NumpyContinuousBCLearner, "policy", property(policy))
    monkeypatch.setattr(owner, "fork", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        demo.run_continuous_correction_update(
            owner, batch(), now=3, observation_projection=state_quaternion_projection()
        )


@pytest.mark.parametrize("fault", ["readout", "budget"])
def test_should_keep_eight_paid_receipts_but_refuse_mutation_by_final_elapsed_observer(
    source, monkeypatch, fault
):
    owner = source.fork()
    initial = owner.snapshot_state()
    captured: dict[str, Any] = {}
    original_restore, original_timer, original_runtime = (
        bp.NumpyContinuousBCLearner.restore_state,
        demo.monotonic,
        demo.ActorShadowRuntime,
    )

    def runtime(model, **options):
        captured["budget"] = options["budget"]
        return original_runtime(model, **options)

    def restore(model, state):
        old_updates = model.snapshot_state().completed_updates
        returned = original_restore(model, state)
        if old_updates == 4 and state.completed_updates == 12:
            captured["readout"] = model
        return returned

    def timer():
        if "readout" in captured and not captured.get("mutated"):
            if fault == "readout":
                captured["readout"]._state.parameters[0][0, 0] += 0.25
            else:
                captured["budget"].updates_completed = 9
            captured["mutated"] = True
        return original_timer()

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "restore_state", restore)
    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(demo, "monotonic", timer)
    result = demo.run_continuous_correction_update(
        owner, batch(), now=3, observation_projection=state_quaternion_projection()
    )
    assert captured["mutated"] is True
    assert result.report["complete"] is False and result.candidate is None
    assert len(result.report["applied_updates"]) == 8
    assert result.report["progress"]["updates_completed"] == 8
    assert "work" not in result.report
    assert_state(initial, owner.snapshot_state())


def test_should_refuse_coherent_candidate_mutation_after_label_before_any_update(
    source, monkeypatch
):
    owner = source.fork()
    original = demo.ActorShadowRuntime.record_label

    def record(runtime, value):
        returned = original(runtime, value)
        runtime._candidate._state.parameters[0][0, 0] += 0.25
        return returned

    monkeypatch.setattr(demo.ActorShadowRuntime, "record_label", record)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    result = demo.run_continuous_correction_update(
        owner, batch(), now=3, observation_projection=state_quaternion_projection()
    )
    assert result.report["complete"] is False and result.candidate is None
    assert result.report["applied_updates"] == []
    assert result.report["progress"]["updates_completed"] == 0


def test_should_keep_one_paid_update_and_refuse_late_public_snapshot_candidate_change(
    source, monkeypatch
):
    owner = source.fork()
    initial = owner.snapshot_state()
    original = demo.ActorShadowRuntime.candidate_snapshot

    def snapshot(runtime):
        returned = original(runtime)
        if returned.state.completed_updates == 5:
            runtime._candidate._state.parameters[0][0, 0] += 0.25
        return returned

    monkeypatch.setattr(demo.ActorShadowRuntime, "candidate_snapshot", snapshot)
    result = demo.run_continuous_correction_update(
        owner, batch(), now=3, observation_projection=state_quaternion_projection()
    )
    assert result.report["complete"] is False and result.candidate is None
    assert len(result.report["applied_updates"]) == 1
    assert result.report["progress"]["updates_completed"] == 1
    assert len(result.report["events"]) == 1 and "work" not in result.report
    assert_state(initial, owner.snapshot_state())


@pytest.fixture(scope="module")
def acquisition(source, tmp_path_factory):
    directory = tmp_path_factory.mktemp("canonical-acquisition") / "case"
    owner = source.fork()
    initial = owner.snapshot_state()
    environments, returned, updates = [], [], []
    original_rollout, original_update = (
        smoke.run_frozen_rollout,
        smoke.run_continuous_correction_update,
    )
    with pytest.MonkeyPatch.context() as patch:

        def factory(**options):
            assert options == {}
            value = QuaternionEnvironment()
            environments.append(value)
            return value

        def geometry(port):
            return PickCubeGeometry(
                (0.0, 0.0, 0.08) if not port._records else (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.2),
            )

        def rollout(*args, **options):
            value = original_rollout(*args, **options)
            returned.append(value)
            return value

        def update(*args, **options):
            value = original_update(*args, **options)
            updates.append(value)
            return value

        patch.setattr(native, "_make_environment", factory)
        patch.setattr(queries, "_geometry", geometry)
        patch.setattr(smoke, "version", lambda _: "fake-canonical-acquisition")
        patch.setattr(smoke, "run_frozen_rollout", rollout)
        patch.setattr(smoke, "run_continuous_correction_update", update)
        summary = smoke.run_corrective_adaptation(
            owner,
            output_directory=directory,
            visual_observation=False,
            readiness_corrections=True,
            acquisition_only=True,
            observation_projection=state_quaternion_projection(),
        )
    assert summary["complete"] is True, summary
    return summary, owner, initial, environments, returned, updates[0], directory


def test_should_collect_raw_readiness17_then_update_canonical_features_and_return_once_to17(
    acquisition,
):
    summary, owner, initial, environments, returned, updated, directory = acquisition
    assert summary["complete"] is summary["source_state_exact"] is True
    assert summary["protocol_id"] == "corrective_robot_train_17_acquisition_v1"
    assert [value.reset_calls for value in environments] == [[17], [17]]
    assert all(len(value.commands) == 2 and value.close_calls == 1 for value in environments)
    assert len(summary["rollouts"]) == 1 and summary["rollouts"][0]["seed"] == 17
    assert not (directory / "updated-19.json").exists()
    collection = returned[0]
    assert type(collection.corrections) is ReadinessCorrectionBatch
    assert collection.corrections.train_seed == 17
    assert collection.corrections.episode.transitions[0].observation[22] == -0.5
    assert collection.corrections.labels[0].observation[22] == -0.5
    assert [label.stage for label in collection.corrections.labels] == ["descend", "grasp"]
    assert all(row.action[3] < 0 for row in collection.corrections.episode.transitions)
    assert collection.report["projection_exact"] is returned[1].report["projection_exact"] is True
    assert collection.report["corrections"]["teacher_decision_steps"] == 2
    assert updated.report["progress"]["updates_completed"] == 8
    assert (
        returned[1].report["source_updates"] == 12
        and returned[1].report["source_training_rows"] == 24
    )
    assert_state(initial, owner.snapshot_state())


@pytest.mark.parametrize(
    "options",
    [
        dict(acquisition_only=1),
        dict(readiness_corrections=False),
        dict(observation_projection=None),
        dict(visual_observation=True),
    ],
)
def test_should_refuse_incomplete_acquisition_mode_before_directory_or_native(
    source, monkeypatch, tmp_path, options
):
    monkeypatch.setattr(smoke, "run_frozen_rollout", forbidden)
    owner = source.fork()
    target = tmp_path / "absent"
    arguments: dict[str, Any] = dict(
        visual_observation=False,
        readiness_corrections=True,
        acquisition_only=True,
        observation_projection=state_quaternion_projection(),
    )
    arguments.update(options)
    with pytest.raises(ValueError):
        smoke.run_corrective_adaptation(owner, output_directory=target, **arguments)
    assert not target.exists()


@pytest.mark.parametrize("selected", [None, False])
def test_should_preserve_default19_collection_kwargs_protocol_and_paid_failure(
    source, monkeypatch, tmp_path, selected
):
    calls = []

    def rollout(*args, **options):
        calls.append(options)
        return smoke.FrozenRolloutResult(
            dict(complete=False, error="retained paid collection"), None
        )

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", forbidden)
    options = {} if selected is None else dict(acquisition_only=selected)
    summary = smoke.run_corrective_adaptation(
        source.fork(), output_directory=tmp_path / "legacy", visual_observation=False, **options
    )
    assert (
        summary["complete"] is False
        and summary["protocol_id"] == "corrective_robot_train_19_17_19_v1"
    )
    assert calls[0]["train_seed"] == 19 and calls[0]["corrective_queries"] is True
    assert "readiness_corrections" not in calls[0] and "observation_projection" not in calls[0]
    assert summary["update"] is None


def test_should_refuse_source_mutated_by_first_public_snapshot_before_creating_directory(
    source, monkeypatch, tmp_path
):
    owner = source.fork()
    original = owner.snapshot_state

    def snapshot():
        returned = original()
        owner._state.parameters[0][0, 0] += 0.25
        return returned

    monkeypatch.setattr(owner, "snapshot_state", snapshot)
    monkeypatch.setattr(smoke, "run_frozen_rollout", forbidden)
    target = tmp_path / "absent"
    with pytest.raises((ValueError, RuntimeError)):
        smoke.run_corrective_adaptation(
            owner,
            output_directory=target,
            visual_observation=False,
            readiness_corrections=True,
            acquisition_only=True,
            observation_projection=state_quaternion_projection(),
        )
    assert not target.exists()


@pytest.mark.parametrize("fault", ["prior_projection", "final_clock"])
def test_should_keep_cached_paid_update_but_refuse_changed_prior_or_final_owner(
    acquisition, source, monkeypatch, tmp_path, fault
):
    # Reuse detached reports from the real fixture; this stub spends no new SGD.
    _, _, _, _, returned, completed_update, _ = acquisition
    owner = source.fork()
    collection = deepcopy(returned[0])
    frozen = deepcopy(returned[1])
    candidate = completed_update.candidate.fork()
    paid = demo.ContinuousLearnabilityResult(deepcopy(completed_update.report), candidate)
    calls = []

    def rollout(*args, **options):
        calls.append(options["train_seed"])
        return collection if len(calls) == 1 else frozen

    def update(*args, **options):
        if fault == "prior_projection":
            collection.report["projection_exact"] = False
        return paid

    original_timer = smoke.time.monotonic
    clock_calls = []

    def timer():
        clock_calls.append(True)
        if len(clock_calls) == 2:
            owner._state.parameters[0][0, 0] += 0.25
        return original_timer()

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    if fault == "final_clock":
        monkeypatch.setattr(smoke.time, "monotonic", timer)
    summary = smoke.run_corrective_adaptation(
        owner,
        output_directory=tmp_path / "cached",
        visual_observation=False,
        readiness_corrections=True,
        acquisition_only=True,
        observation_projection=state_quaternion_projection(),
    )
    assert summary["complete"] is False
    assert summary["update"]["progress"]["updates_completed"] == 8
    assert len(summary["update"]["applied_updates"]) == 8
    if fault == "prior_projection":
        assert calls == [17] and summary["rollouts"] == []
    else:
        assert calls == [17, 17] and len(summary["rollouts"]) == 1
