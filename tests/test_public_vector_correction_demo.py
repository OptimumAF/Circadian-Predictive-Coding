"""Raw789 queries and owned public21 correction updates, without native work.

Per invocation: one shared8-update fixture, one8-update fake-IO pipeline,
one paid1-update failure and one8-update final-poll failure:25 SGD/50rows.
All other cases initialize frozen synthetic sources but perform zero SGD.
"""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_correction_demo as demo
from src.adapters import numpy_continuous_bc as bc_adapter
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_episode_clock import EpisodeClock
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_corrections import PickCubeCorrectionBatch
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_maniskill_bc_smoke import assert_same_state, learner
from test_maniskill_rgb_observations import RGBEnvironment, packet, public_feature_oracle
from test_pick_cube_corrections import labels_for
from test_pick_cube_public_vectors import raw_episode


def source(dimension=21, **changes):
    spec = replace(raw_episode(19).spec, observation_dim=dimension)
    options = dict(
        spec=spec,
        observation_standardization=ObservationStandardization(
            (0.0,) * dimension, (1.0,) * dimension, 2
        ),
    )
    options.update(changes)
    return learner(**options)


def raw_batch():
    episode = replace(raw_episode(19), episode_id="pick-cube-smoke-19")
    return PickCubeCorrectionBatch(19, episode, labels_for(episode))


def forbidden(*_args, **_kwargs):
    pytest.fail("invalid input admitted downstream work")


@pytest.fixture(scope="module")
def completed():
    """Exactly8 SGD/16rows, shared; source starts32/928 with momentum."""
    owner, batch = source(), raw_batch()
    before, original = owner.snapshot_state(), asdict(batch)
    captured = dict(runtimes=[], trains=[])
    runtime_factory, train = demo.ActorShadowRuntime, NumpyContinuousBCLearner.train_batch
    with pytest.MonkeyPatch.context() as patch:

        def runtime(model, **options):
            value = runtime_factory(model, **options)
            captured["runtimes"].append((value, options))
            return value

        def update(model, features, targets):
            captured["trains"].append((model, features.copy(), targets.copy()))
            return train(model, features, targets)

        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", update)
        patch.setattr(bc_adapter, "fit_training_standardization", forbidden)
        result = demo.run_continuous_correction_update(
            owner, batch, now=3, observation_projection=public_vector_projection()
        )
    assert result.report["complete"] is True and asdict(batch) == original
    return owner, batch, before, result, captured


def test_should_train_only_owned_projected_pairs_and_continue_full_source_optimizer(completed):
    owner, batch, before, result, captured = completed
    runtime, options = captured["runtimes"][0]
    assert len(captured["runtimes"]) == 1 and len(captured["trains"]) == 8
    assert options["budget"].updates_completed == options["budget"].budget.max_training_updates == 8
    assert options["budget"].replay_examples_completed == 0
    observations, targets = batch.training_pairs(now=3)
    for model, features, labels in captured["trains"]:
        assert model is runtime._candidate
        assert features.shape == (2, 21) and features.dtype == np.float64
        np.testing.assert_array_equal(features, np.asarray([row[:21] for row in observations]))
        np.testing.assert_array_equal(labels, np.asarray(targets))
    after = result.candidate.snapshot_state()
    assert after.completed_updates == before.completed_updates + 8 == 40
    assert after.training_examples == before.training_examples + 16 == 944
    assert after.policy == before.policy
    assert any(
        not np.array_equal(a, b) for a, b in zip(after.velocity, before.velocity, strict=True)
    )
    assert_same_state(after, runtime.candidate_snapshot().state)
    assert_same_state(before, owner.snapshot_state())
    assert_same_state(before, runtime.actor.snapshot_state().state)
    exported = result.candidate.snapshot_state()
    exported.parameters[0][0, 0] += 100.0
    exported.velocity[0][0, 0] -= 100.0
    assert_same_state(after, result.candidate.snapshot_state())
    assert batch.episode.spec.observation_dim == 789 and len(batch.labels[0].observation) == 789


def test_should_keep_original_source_labels_times_ids_and_all_eight_paid_receipts(completed):
    _, batch, before, result, _ = completed
    report = result.report
    assert report["source_updates"] == before.completed_updates
    assert report["source_training_rows"] == before.training_examples
    assert report["observed_at"] == 2 and report["labels_arrived_at"] == 3
    for index, (event, receipt) in enumerate(
        zip(report["events"], report["applied_updates"], strict=True)
    ):
        assert event["success_validated"] is event["update_attempted"] is True
        assert event["applied"] == [receipt]
        assert (
            receipt["sample_id"] == f"corrective-batch-{index}"
            and receipt["event_id"] == f"corrective-label-{index}"
        )
        assert receipt["episode_id"] == batch.episode.episode_id
        assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
        assert receipt["update_number"] == index + 1
        assert receipt["actor_version"] == "actor-0" and receipt["learner_version"] == "candidate-0"
        assert np.isfinite(receipt["diagnostic"]["value"])
    assert len(report["events"]) == len(report["applied_updates"]) == 8
    assert report["progress"] == dict(updates_completed=8, replay_examples_completed=0, next_tick=4)
    json.dumps(report, allow_nan=False)


def test_should_account_separate_one_time_projection_and_repeated21d_optimizer_work(completed):
    report = completed[3].report
    expected = dict(
        optimizer_updates=8,
        training_rows=16,
        unique_correction_rows=2,
        repeated_training_rows=14,
        reference_replay_rows=0,
        fit_calls=0,
        forward_passes=8,
        forward_rows=16,
        backward_passes=8,
        backward_rows=16,
        normalization_calls=8,
        normalization_rows=16,
        normalization_values=336,
        gradient_values=448,
        gradient_numeric_bytes=3584,
        dense_multiply_accumulates=1728,
        model_numeric_bytes_per_owner=896,
        statistics_numeric_bytes_per_owner=336,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=3200,
    )
    for name, value in expected.items():
        assert report["work"][name] == value, name
    metadata = report["input_projection"]
    assert metadata["projection"] == asdict(public_vector_projection())
    assert metadata["source_recipe"] == asdict(pick_cube_rgb_recipe())
    assert metadata["source_episode_id"] == completed[1].episode.episode_id
    assert metadata["source_spec"]["observation_dim"] == 789
    assert metadata["model_spec"]["observation_dim"] == 21
    assert metadata["work"] == dict(
        observation_projections=4,
        source_values_validated=3156,
        projected_values=84,
        projected_numeric_bytes=672,
        projected_label_observation_values=42,
        projected_label_target_values=8,
        projected_label_observation_numeric_bytes=336,
        projected_label_target_numeric_bytes=64,
    )


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environments, options = [], []

    def factory(**kwargs):
        environment = RGBEnvironment()
        environments.append(environment)
        options.append(kwargs)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-public21-corrections")
    monkeypatch.setattr(
        queries,
        "_geometry",
        lambda _: PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2)),
    )
    return environments, options, tmp_path


def test_should_query_exact_raw_prediction_packet_and_keep_student_controls(boundary, monkeypatch):
    environments, options, directory = boundary
    owner, predictions, queried = source(), [], []
    before = owner.snapshot_state()
    predict, query = NumpyContinuousBCLearner.predict, queries.PickCubeCorrectionQuery.query

    def forward(child, values):
        result = predict(child, values)
        predictions.append((values.copy(), result.copy()))
        return result

    def label(query_owner, port, step, observation):
        queried.append((step, observation, port.observation))
        return query(query_owner, port, step, observation)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", forward)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(queries.PickCubeCorrectionQuery, "query", label)
    result = smoke.run_frozen_rollout(
        directory / "collection.json",
        learner=owner,
        train_seed=19,
        visual_observation=True,
        corrective_queries=True,
        observation_projection=public_vector_projection(),
    )
    assert result.report["complete"] is True and type(result.corrections) is PickCubeCorrectionBatch
    assert options == [dict(visual_observation=True)]
    environment = environments[0]
    assert environment.reset_calls == [19] and environment.close_calls == 1
    assert len(queried) == len(predictions) == len(environment.commands) == 2
    for index, (
        (values, prediction),
        (step, observation, current),
        command,
        row,
        correction,
    ) in enumerate(
        zip(
            predictions,
            queried,
            environment.commands,
            result.report["episode"]["transitions"],
            result.corrections.labels,
            strict=True,
        )
    ):
        assert step == index and observation == current == tuple(row["observation"])
        assert len(observation) == len(correction.observation) == 789
        assert observation == correction.observation == tuple(public_feature_oracle(packet(index)))
        assert values.shape == (1, 21) and values.dtype == np.float64
        np.testing.assert_array_equal(values[0], observation[:21])
        np.testing.assert_array_equal(command, prediction.astype(np.float32))
        assert tuple(row["action"]) == tuple(float(value) for value in command[0])
        assert correction.target != tuple(row["action"])
    assert result.corrections.episode.spec.observation_dim == 789
    assert result.report["projection_work"]["source_values_validated"] == 1578
    assert result.report["projection_work"]["projected_values"] == 42
    assert (
        result.report["corrections"]["geometry_attempts"]
        == result.report["corrections"]["geometry_returns"]
        == 2
    )
    assert result.report["optimizer_updates"] == 0
    assert_same_state(before, owner.snapshot_state())
    preserved = result.corrections.training_pairs(now=3)
    result.report["episode"]["transitions"][0]["observation"] = (100.0,) * 789
    assert result.corrections.training_pairs(now=3) == preserved


@pytest.mark.parametrize("failure", ["raw_geometry", "owner", "cleanup"])
def test_should_refuse_changed_raw_pairing_owner_or_cleanup_and_retain_partial_query_work(
    boundary, monkeypatch, failure
):
    environments, _, directory = boundary
    owner = source()
    if failure == "raw_geometry":

        def geometry(port):
            port._observation = (port.observation[0] + 1.0,) + port.observation[1:]
            return PickCubeGeometry((0.0, 0.0, 0.0), (0.05, 0.0, 0.02), (0.1, 0.0, 0.2))

        monkeypatch.setattr(queries, "_geometry", geometry)
    elif failure == "owner":
        original = queries.PickCubeCorrectionQuery.query

        def query(*args, **kwargs):
            result = original(*args, **kwargs)
            state = owner.snapshot_state()
            state.velocity[0][0, 0] += 1.0
            owner.restore_state(state)
            return result

        monkeypatch.setattr(queries.PickCubeCorrectionQuery, "query", query)
    else:
        factory = native._make_environment

        def failing_factory(**kwargs):
            environment = factory(**kwargs)
            environment.close_error = RuntimeError("late RGB cleanup failure")
            return environment

        monkeypatch.setattr(native, "_make_environment", failing_factory)
    result = smoke.run_frozen_rollout(
        directory / "failed.json",
        learner=owner,
        train_seed=19,
        visual_observation=True,
        corrective_queries=True,
        observation_projection=public_vector_projection(),
    )
    assert result.report["complete"] is False and result.corrections is None
    assert result.report["corrections"]["batch"] is None
    environment = environments[0]
    assert environment.close_calls == 1
    if failure == "cleanup":
        assert len(environment.commands) == len(result.report["corrections"]["labels"]) == 2
        assert "cleanup_error" in result.report
    else:
        assert environment.commands == []
        assert result.report["prediction_returns"] == 1
        assert result.report["corrections"]["geometry_attempts"] == 1
        assert result.report["corrections"]["geometry_returns"] == 1
        assert len(result.report["corrections"]["labels"]) == (1 if failure == "owner" else 0)


@pytest.mark.parametrize(
    "options",
    [
        dict(visual_observation=False),
        dict(observation_clock=EpisodeClock(50, True)),
        dict(actor=object()),
        dict(camera_context=pick_cube_camera_context(), train_seed=17),
        dict(corrective_queries=1),
    ],
)
def test_should_keep_projection_query_incompatibilities_before_fork_or_scene(
    boundary, monkeypatch, options
):
    environments, _, directory = boundary
    owner = source()
    monkeypatch.setattr(owner, "fork", forbidden)
    arguments = dict(
        learner=owner,
        train_seed=19,
        visual_observation=True,
        corrective_queries=True,
        observation_projection=public_vector_projection(),
    )
    arguments.update(options)
    result = smoke.run_frozen_rollout(directory / "refused.json", **arguments)
    assert result.report["complete"] is False and result.corrections is None
    assert environments == []


@pytest.mark.parametrize("now", [True, -1, 2, 3.0, None])
def test_should_require_label_arrival_before_projection_or_any_model_fork(monkeypatch, now):
    owner, batch = source(), raw_batch()
    monkeypatch.setattr(demo, "public_vector_correction_view", forbidden)
    monkeypatch.setattr(NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch, now=now, observation_projection=public_vector_projection()
        )


@pytest.mark.parametrize(
    "damage",
    [
        "model_view_with_flag",
        "raw_without_flag",
        "raw_owner",
        "range",
        "terminal_range",
        "label_pair",
        "policy",
    ],
)
def test_should_revalidate_original_raw_batch_and_policy_before_fork(monkeypatch, damage):
    owner, batch = source(), raw_batch()
    projection = public_vector_projection()
    if damage == "model_view_with_flag":
        from src.core.pick_cube_public_corrections import public_vector_correction_view

        batch = public_vector_correction_view(
            batch, projection, source_recipe=pick_cube_rgb_recipe()
        ).model
    elif damage == "raw_without_flag":
        projection = None
    elif damage == "raw_owner":
        owner = source(dimension=789)
    elif damage == "range":
        row = batch.episode.transitions[0]
        invalid = row.observation[:-1] + (1.01,)
        object.__setattr__(row, "observation", invalid)
        object.__setattr__(batch.labels[0], "observation", invalid)
    elif damage == "terminal_range":
        last = batch.episode.transitions[-1]
        object.__setattr__(last, "next_observation", last.next_observation[:-1] + (-0.01,))
    elif damage == "label_pair":
        object.__setattr__(
            batch.labels[0], "observation", (100.0,) + batch.labels[0].observation[1:]
        )
    else:
        owner = source(binary_gripper_axis=3)
    monkeypatch.setattr(NumpyContinuousBCLearner, "fork", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_correction_update(
            owner, batch, now=3, observation_projection=projection
        )


def test_should_retain_one_paid_projected_update_when_receipt_return_fails(monkeypatch):
    """ExactlyONE SGD/two rows; no second update after the failed return."""
    owner, batch = source(), raw_batch()
    before, captured = owner.snapshot_state(), []
    original = demo.ActorShadowRuntime

    def runtime(model, **options):
        value = original(model, **options)
        captured.append((value, options))
        ready = value.train_ready

        def failed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if applied:
                raise RuntimeError("projected update postcommit failure")
            return applied

        monkeypatch.setattr(value, "train_ready", failed)
        return value

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(
        owner, batch, now=3, observation_projection=public_vector_projection()
    )
    runtime, options = captured[0]
    assert result.candidate is None and result.report["complete"] is False
    assert "postcommit failure" in result.report["error"]
    assert (
        result.report["progress"]["updates_completed"] == options["budget"].updates_completed == 1
    )
    assert len(result.report["applied_updates"]) == len(result.report["events"]) == 1
    assert (
        result.report["events"][0]["applied"] == []
        and result.report["events"][0]["success_validated"] is False
    )
    (receipt,) = result.report["applied_updates"]
    assert receipt["observed_at"] == 2 and receipt["arrived_at"] == receipt["applied_at"] == 3
    assert result.report["input_projection"]["work"]["observation_projections"] == 4
    assert runtime.candidate_snapshot().state.completed_updates == before.completed_updates + 1
    assert_same_state(before, owner.snapshot_state())
    assert_same_state(before, runtime.actor.snapshot_state().state)


def test_should_refuse_final_empty_poll_state_change_and_preserve_eight_permanent_charges(
    monkeypatch,
):
    """ExactlyEIGHT SGD/16rows; no candidate is released after corruption."""
    owner, before, captured = source(), None, []
    before = owner.snapshot_state()
    original = demo.ActorShadowRuntime

    def runtime(model, **options):
        value = original(model, **options)
        captured.append((value, options))
        ready = value.train_ready

        def changed(*args, **kwargs):
            applied = ready(*args, **kwargs)
            if not applied and options["budget"].updates_completed == 8:
                state = value._candidate.snapshot_state()
                state.velocity[0][0, 0] += 0.25
                value._candidate.restore_state(state)
            return applied

        monkeypatch.setattr(value, "train_ready", changed)
        return value

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_correction_update(
        owner, raw_batch(), now=3, observation_projection=public_vector_projection()
    )
    runtime, options = captured[0]
    assert result.candidate is None and result.report["complete"] is False
    assert "final correction empty poll" in result.report["error"]
    assert len(result.report["applied_updates"]) == len(result.report["events"]) == 8
    assert all(event["success_validated"] is True for event in result.report["events"])
    assert (
        result.report["progress"]["updates_completed"] == options["budget"].updates_completed == 8
    )
    assert "work" not in result.report
    assert_same_state(before, owner.snapshot_state())
    assert_same_state(before, runtime.actor.snapshot_state().state)


def test_should_collect_raw_then_update_and_use_public21_candidate_in_both_contexts(
    boundary, monkeypatch
):
    """Exactly8 SGD/16rows through the existing collector/update/rollout path."""
    environments, options, directory = boundary
    owner, captured = source(), []
    before = owner.snapshot_state()
    original = smoke.run_continuous_correction_update

    def update(model, batch, *, now, **kwargs):
        assert batch.episode.spec.observation_dim == 789 and len(batch.labels[0].observation) == 789
        result = original(model, batch, now=now, **kwargs)
        captured.append((batch, now, kwargs, result, result.candidate.snapshot_state()))
        return result

    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    monkeypatch.setattr(bc_adapter, "fit_training_standardization", forbidden)
    output = directory / "pipeline"
    summary = smoke.run_corrective_adaptation(
        owner, output_directory=output, observation_projection=public_vector_projection()
    )
    assert summary["complete"] is summary["source_state_exact"] is True
    assert summary["observation_projection"] == asdict(public_vector_projection())
    assert [environment.reset_calls for environment in environments] == [[19], [17], [19]]
    assert options == [dict(visual_observation=True)] * 3
    assert all(
        environment.close_calls == 1 and len(environment.commands) == 2
        for environment in environments
    )
    assert_same_state(before, owner.snapshot_state())
    ((batch, now, kwargs, result, final_state),) = captured
    assert now == 3 and batch.train_seed == 19
    assert kwargs == dict(observation_projection=public_vector_projection())
    assert final_state.completed_updates == 40 and final_state.training_examples == 944
    assert result.report["progress"]["updates_completed"] == 8
    assert_same_state(final_state, result.candidate.snapshot_state())
    assert {path.name for path in output.iterdir()} == {
        "collection.json",
        "update.json",
        "updated-17.json",
        "updated-19.json",
        "summary.json",
    }
    for index, name in enumerate(("collection.json", "updated-17.json", "updated-19.json")):
        report = json.loads((output / name).read_text())
        assert (
            report["complete"]
            is report["projection_exact"]
            is report["frozen_state_exact"]
            is report["original_owner_exact"]
            is True
        )
        assert report["episode"]["spec"]["observation_dim"] == 789
        assert report["learner_policy"]["spec"]["observation_dim"] == 21
        assert report["source_updates"] == (32 if index == 0 else 40)
        assert report["source_training_rows"] == (928 if index == 0 else 944)
        assert report["optimizer_updates"] == 0
        assert ("corrections" in report) is (index == 0)
        assert report["projection_work"]["source_values_validated"] == 1578
        assert report["projection_work"]["projected_values"] == 42
    assert any(
        not np.array_equal(before_command, after_command)
        for before_command, after_command in zip(
            environments[0].commands, environments[1].commands, strict=True
        )
    )
    update_report = json.loads((output / "update.json").read_text())
    assert update_report == json.loads(json.dumps(result.report))
    assert update_report["input_projection"]["source_spec"]["observation_dim"] == 789
    assert update_report["input_projection"]["model_spec"]["observation_dim"] == 21
    assert [row["seed"] for row in summary["rollouts"]] == [17, 19]
    assert (
        summary["guard_reads"]
        == summary["probe_reads"]
        == summary["final_reads"]
        == summary["promotion_calls"]
        == 0
    )


@pytest.mark.parametrize(
    "damage",
    ["no_projection", "wrong_dimension", "nonvisual", "bad_projection", "no_stats", "binary"],
)
def test_should_admit_projection_and_source_policy_before_creating_pipeline_directory(
    boundary, damage
):
    environments, _, directory = boundary
    projection, visual, changes = public_vector_projection(), True, {}
    if damage == "no_projection":
        projection = None
    elif damage == "wrong_dimension":
        changes["dimension"] = 789
    elif damage == "nonvisual":
        visual = False
    elif damage == "bad_projection":
        projection = asdict(projection)
    elif damage == "no_stats":
        changes["observation_standardization"] = None
    else:
        changes["binary_gripper_axis"] = 3
    output = directory / "refused"
    with pytest.raises(ValueError):
        smoke.run_corrective_adaptation(
            source(**changes),
            output_directory=output,
            visual_observation=visual,
            observation_projection=projection,
        )
    assert not output.exists() and environments == []


def test_should_refuse_existing_pipeline_directory_before_collection(boundary):
    environments, _, directory = boundary
    output = directory / "existing"
    output.mkdir()
    (output / "evidence").write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        smoke.run_corrective_adaptation(
            source(), output_directory=output, observation_projection=public_vector_projection()
        )
    assert environments == [] and (output / "evidence").read_bytes() == b"preserved"


@pytest.mark.parametrize("failure", ["owner", "trace"])
def test_should_validate_late_episode_getters_before_releasing_raw_corrections(
    boundary, monkeypatch, failure
):
    environments, _, directory = boundary
    owner, mutations = source(), []
    original = native.ManiSkillPickCubeEpisode.episode

    def late_episode(port):
        episode = original(port)
        if environments and environments[0].close_calls:
            if failure == "owner" and not mutations:
                state = owner.snapshot_state()
                state.velocity[3][0] += 0.5
                owner.restore_state(state)
                mutations.append(True)
            elif failure == "trace":
                first = episode.transitions[0]
                episode = replace(
                    episode,
                    transitions=(replace(first, reward=0.0, action=(0.25,) + first.action[1:]),)
                    + episode.transitions[1:],
                )
        return episode

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "episode", late_episode)
    result = smoke.run_frozen_rollout(
        directory / "late.json",
        learner=owner,
        train_seed=19,
        visual_observation=True,
        corrective_queries=True,
        observation_projection=public_vector_projection(),
    )
    assert result.report["complete"] is False and result.corrections is None
    assert result.report["corrections"]["batch"] is None
    assert "correction_error" in result.report
    if failure == "trace":
        assert "retained executed trace" in result.report["correction_error"]
    assert len(environments[0].commands) == len(result.report["corrections"]["labels"]) == 2
    assert result.report["corrections"]["geometry_attempts"] == 2
    assert environments[0].close_calls == 1


@pytest.mark.parametrize("failure", ["collection", "update", "first_rollout"])
def test_should_stop_projected_pipeline_at_failed_stage_without_losing_returned_paid_work(
    boundary, monkeypatch, failure
):
    """Zero SGD: a synthetic update result exercises pipeline refusal/forwarding."""
    environments, _, directory = boundary
    owner, observed = source(), []
    before = owner.snapshot_state()
    original_rollout = smoke.run_frozen_rollout

    def rollout(output, **options):
        if failure == "collection" or (
            failure == "first_rollout" and output.name == "updated-17.json"
        ):
            factory = native._make_environment

            def failing_factory(**kwargs):
                environment = factory(**kwargs)
                environment.close_error = RuntimeError("projected stage cleanup failed")
                return environment

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(native, "_make_environment", failing_factory)
                return original_rollout(output, **options)
        return original_rollout(output, **options)

    def update(model, batch, *, now, **options):
        observed.append((batch, now, options))
        paid = 1 if failure == "update" else 8
        report = dict(
            complete=failure != "update",
            applied_updates=[dict(update_number=index + 1) for index in range(paid)],
            progress=dict(updates_completed=paid),
        )
        if failure == "update":
            report["error"] = "synthetic paid update result"
        if failure == "update":
            return demo.ContinuousLearnabilityResult(report, None)
        child = model.fork()
        state = child.snapshot_state()
        child.restore_state(
            replace(
                state,
                completed_updates=state.completed_updates + 8,
                training_examples=state.training_examples + 8 * len(batch.labels),
            )
        )
        return demo.ContinuousLearnabilityResult(report, child)

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", update)
    output = directory / "failed-pipeline"
    report = smoke.run_corrective_adaptation(
        owner, output_directory=output, observation_projection=public_vector_projection()
    )
    assert report["complete"] is False and report["source_state_exact"] is True
    assert_same_state(before, owner.snapshot_state())
    assert (output / "collection.json").exists() and (output / "summary.json").exists()
    assert not (output / "updated-19.json").exists()
    if failure == "collection":
        assert observed == [] and report["update"] is None and report["rollouts"] == []
        assert len(environments) == 1
    else:
        assert observed[0][2] == dict(observation_projection=public_vector_projection())
        paid = 1 if failure == "update" else 8
        assert report["update"]["progress"]["updates_completed"] == paid
        assert report["update"]["applied_updates"] == [
            dict(update_number=index + 1) for index in range(paid)
        ]
        assert (output / "update.json").exists()
        assert len(environments) == (1 if failure == "update" else 2)
        assert len(report["rollouts"]) == (0 if failure == "update" else 1)
